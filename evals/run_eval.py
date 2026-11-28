# -*- coding: utf-8 -*-
"""批量评测：跑某个版本的题目集，输出「通过率 / 平均步数 / 平均 token / 平均耗时」。

目录约定：
    eval_task/<版本>/tasks.jsonl   题目
    eval_task/<版本>/runs/<时间戳>/ 结果（results.jsonl + summary.json）
    eval_task/<版本>/notes.md      这一版改了什么
    eval_task/<版本>/baseline.json 定稿基线（--baseline 时写入）

用法（项目根目录）：
    python -m evals.run_eval                    # 用最新版本
    python -m evals.run_eval --set v1           # 指定版本
    python -m evals.run_eval --limit 3          # 只跑前 3 条（调试）
    python -m evals.run_eval --only dirty_03    # 只跑一条
    python -m evals.run_eval --verbose          # 显示每步工具调用
    python -m evals.run_eval --baseline         # 本次结果记为该版本基线

判定方式（v1，简单可解释）：
    从回答里抽出所有数字，要求 expect_numbers 里每个数字都能在 tol 容差内找到。
    【局限】agent 把所有数字都列出来、但结论说反了，也会判通过。
    【升级方向】先解析它最终给出的排名顺序，再和预期顺序比对。
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from data_agent.agent_chat import SYSTEM_PROMPT, run
from data_agent.config import ROOT, set_session

EVAL_TASK_DIR = ROOT / "eval_task"


# ---------------------------------------------------------------------------
# 版本（目录）管理
# ---------------------------------------------------------------------------
def list_sets() -> list[Path]:
    """列出所有含 tasks.jsonl 的版本目录，按名字排序（v1 < v2 < ...）。"""
    if not EVAL_TASK_DIR.is_dir():
        return []
    return sorted(
        p for p in EVAL_TASK_DIR.iterdir()
        if p.is_dir() and (p / "tasks.jsonl").exists()
    )


def resolve_set(name: str | None) -> Path:
    """解析要跑的版本目录。不指定时用最后一个（最新的）。"""
    sets = list_sets()
    if not sets:
        sys.exit(f"没有找到任何版本目录（需要 {EVAL_TASK_DIR}/<版本>/tasks.jsonl）")
    if name is None:
        return sets[-1]
    d = EVAL_TASK_DIR / name
    if not (d / "tasks.jsonl").exists():
        sys.exit(f"找不到版本 {name}，现有：{[p.name for p in sets]}")
    return d


def eval_fingerprint(set_dir: Path) -> dict:
    """版本指纹：版本名 + 题目内容哈希 + 当前代码 commit。

    为什么要有 tasks_sha：
        题目改了通过率必然变化，不看指纹就分不清"agent 变差"还是"题变难"。
        有了哈希，两次结果是否可比就是可判定的。
    """
    tasks_sha = hashlib.sha256((set_dir / "tasks.jsonl").read_bytes()).hexdigest()[:8]

    commit = "unknown"
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            commit = proc.stdout.strip()
    except Exception:
        pass

    return {"eval_version": set_dir.name, "tasks_sha": tasks_sha, "git_commit": commit}


def load_tasks(set_dir: Path) -> list[dict]:
    """读 jsonl：一行一个 JSON 对象，空行跳过。"""
    f = set_dir / "tasks.jsonl"
    lines = f.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# 单题执行与判定
# ---------------------------------------------------------------------------
def files_header(task: dict) -> str:
    """数据文件说明。多文件时逐行列出，让模型知道要分别加载。"""
    files = task["files"]
    if len(files) == 1:
        return f"数据文件：{files[0]}"
    return "数据文件：\n" + "\n".join(f"- {f}" for f in files)


def task_turns(task: dict) -> list[str]:
    """该任务的各轮用户输入。

    多轮题写成 "turns": ["第一问", "追问一", ...]；
    单轮题只有 "ask"，等价于一轮。
    """
    if task.get("turns"):
        return [str(t) for t in task["turns"]]
    return [task.get("ask", "")]


def build_messages(task: dict) -> list[dict]:
    """只构造 system 起始消息。用户轮次由调用方逐轮 append。

    为什么不再在这里拼第一问：
        多轮任务要在同一个 messages 上连续追加，
        所以「起始消息」和「用户轮次」必须分开构造。
    """
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def merge_stats(total: dict, part: dict) -> None:
    """把一轮的指标累加进总计（多轮任务用）。

    steps 用 `or 1`：每轮至少一次 API 调用，缺失时按 1 记，避免总数被低估。
    """
    total["tokens"] = (total.get("tokens") or 0) + (part.get("tokens") or 0)
    total["steps"] = (total.get("steps") or 0) + (part.get("steps") or 1)
    total["cache_hit"] = (total.get("cache_hit") or 0) + (part.get("cache_hit") or 0)
    total["cache_miss"] = (total.get("cache_miss") or 0) + (part.get("cache_miss") or 0)
    total["compressed"] = (total.get("compressed") or 0) + (part.get("compressed") or 0)


def extract_numbers(text: str) -> list[float]:
    """从文本里抽出所有数字。

    必须先去千分位逗号：模型常写 "36,193"，直接正则会抽出 36 和 193 两个数。
    规则：只删「前后都是数字」的逗号（半角 , 和全角 ，），不碰句子里的普通逗号。

    还要统一负号：模型常用 Unicode 减号/连字符（− U+2212、– U+2013、— U+2014），
    不归一化的话 r = −0.7762 会被整段漏掉（v4 就因此把一道对的题判成错的）。
    """
    cleaned = text or ""
    for dash in ("\u2212", "\u2013", "\u2014", "\uff0d"):   # − – — －
        cleaned = cleaned.replace(dash, "-")
    cleaned = re.sub(r"(?<=\d)[,，](?=\d)", "", cleaned)
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", cleaned)]


def check_answer(answer: str, task: dict) -> tuple[bool, list]:
    """返回 (是否通过, 缺了什么)。

    支持两种期望，可以单用也可以叠加：
        expect_numbers   答案里必须出现这些数字（带 tol 容差）
        expect_text_any  答案里必须出现其中任意一个短语
                         —— 用于"数据里没有这东西，该明说没有"的题
    """
    tol = task.get("tol", 0.01)
    text = answer or ""
    found = extract_numbers(text)

    missing: list = [
        want for want in task.get("expect_numbers", [])
        if not any(abs(num - want) <= tol for num in found)
    ]

    phrases = task.get("expect_text_any")
    if phrases and not any(p in text for p in phrases):
        missing.append(f"缺关键词之一 {phrases}")

    # 正则期望：用于"格式严格"的题（如要求 markdown 表格、保留 2 位小数）
    # 所有 pattern 都必须匹配才算通过
    for pat in task.get("expect_regex", []):
        if not re.search(pat, text):
            missing.append(f"格式不符 /{pat}/")

    return (not missing), missing


CODE_PREVIEW_CHARS = 1500      # 轨迹里保留的代码长度（分析「它在查什么」需要看代码）


def extract_trajectory(messages: list) -> list:
    """抽出工具调用序列：结构化参数 + 是否报错 + 结果大小。

    【曾经有个 bug】旧版把 arguments（JSON 字符串）直接截到 400 字符，
    截断后 JSON 不合法，下游 json.loads 全部失败 → 代码预览显示为空。
    正确做法：**先解析 JSON，再按字段截断**。

    额外记两个字段，用于判断「哪一步是白走的」：
        errored      这一步是否报错（traceback / 错误 / 超时）
        result_chars 工具返回了多少字符
    """
    # tool_call_id -> 该次工具的结果文本（tool 消息里带着对应关系）
    results = {
        m.get("tool_call_id"): (m.get("content") or "")
        for m in messages if m.get("role") == "tool"
    }

    traj = []
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments") or "{}"

            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    parsed = {"_value": parsed}
            except Exception:
                parsed = {"_raw": raw[:300]}       # 真解析不了才退回原字符串

            args = {}
            for k, v in parsed.items():
                text = v if isinstance(v, str) else str(v)
                limit = CODE_PREVIEW_CHARS if k == "code" else 200
                args[k] = text[:limit] + ("…" if len(text) > limit else "")

            res = results.get(tc.get("id"), "")
            low = res.lower()
            traj.append({
                "tool": name,
                "args": args,
                "errored": ("traceback" in low)
                           or res.startswith("错误")
                           or res.startswith("执行超时"),
                "result_chars": len(res),
            })
    return traj


def run_once(tasks: list, fingerprint: dict, args, print_detail: bool = True):
    """跑一轮完整评测，返回 (逐题明细, 汇总)。"""
    rows = []
    for task in tasks:
        set_session(f"eval_{task['id']}")          # 每条任务一个独立会话（隔离数据集）
        stats: dict = {}
        started = time.perf_counter()

        turns = task_turns(task)
        messages = build_messages(task)
        answer = ""

        # 默认吞掉 agent 打印的工具调用日志，让评测输出清爽；--verbose 时照常显示
        sink = sys.stdout if args.verbose else io.StringIO()
        try:
            with contextlib.redirect_stdout(sink):
                for i, q in enumerate(turns):
                    if i == 0:
                        q = f"{files_header(task)}\n\n问题：{q}"
                    messages.append({"role": "user", "content": q})
                    turn_stats: dict = {}
                    answer = run(messages, task.get("max_steps", 10), stats=turn_stats)
                    merge_stats(stats, turn_stats)     # 多轮时累加，单轮等价于直接取
        except Exception as exc:                       # 单条失败不能拖垮整轮评测
            answer = f"[异常] {exc!r}"

        elapsed = time.perf_counter() - started
        ok, missing = check_answer(answer, task)

        rows.append({
            "id": task["id"],
            "ok": ok,
            "missing": missing,
            "turns": len(turns),
            "steps": stats.get("steps"),
            "tokens": stats.get("tokens"),
            "compressed": stats.get("compressed"),
            "cache_hit": stats.get("cache_hit"),
            "cache_miss": stats.get("cache_miss"),
            "seconds": round(elapsed, 2),
            "answer": (answer or "")[:500],
            "trajectory": extract_trajectory(messages),   # 工具调用序列，用于事后分析
        })

        if print_detail:
            hit, miss = stats.get("cache_hit") or 0, stats.get("cache_miss") or 0
            cache_note = f" 缓存{hit / (hit + miss):.0%}" if (hit + miss) else ""
            comp_note = f" 压{stats.get('compressed')}条" if stats.get("compressed") else ""
            mt_note = f" ({len(turns)}轮)" if len(turns) > 1 else ""
            mark = "✓" if ok else "✗"
            print(f"{mark} {task['id']:<12}{mt_note:<7} 步数={stats.get('steps', '-'):<3} "
                  f"token={stats.get('tokens', '-'):<6}{cache_note:<9}{comp_note:<8}"
                  f"{elapsed:5.1f}s   {task.get('note', '')}")
            if not ok:
                print(f"    缺少 {missing}；回答开头：{(answer or '')[:80]!r}")

    # ---------- 汇总 ----------
    total = len(rows)
    passed = sum(1 for r in rows if r["ok"])
    steps = [r["steps"] for r in rows if r["steps"]]
    tokens = [r["tokens"] for r in rows if r["tokens"]]
    secs = [r["seconds"] for r in rows]

    # 缓存命中率：命中/（命中+未命中）。命中部分单价低一个数量级，
    # 所以这个比例直接决定实际花销——同样的 token 数，命中率高就便宜得多。
    hits = sum(r["cache_hit"] for r in rows if r.get("cache_hit"))
    misses = sum(r["cache_miss"] for r in rows if r.get("cache_miss"))
    cache_rate = (hits / (hits + misses)) if (hits + misses) else None
    compressed = sum(r.get("compressed") or 0 for r in rows)

    summary = {
        "task_count_total": len(tasks),
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "mean_steps": round(statistics.mean(steps), 2) if steps else None,
        "mean_tokens": round(statistics.mean(tokens)) if tokens else None,
        "mean_seconds": round(statistics.mean(secs), 2),
        "cache_hit_tokens": hits,
        "cache_miss_tokens": misses,
        "cache_hit_rate": round(cache_rate, 4) if cache_rate is not None else None,
        "compressed_messages": compressed,
        "failed_ids": [r["id"] for r in rows if not r["ok"]],
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="批量评测数据分析 agent")
    ap.add_argument("--set", help="版本目录名，默认最新")
    ap.add_argument("--limit", type=int, help="只跑前 N 条")
    ap.add_argument("--only", type=str, help="只跑指定 id")
    ap.add_argument("--ids", type=str, help="只跑这几个 id（逗号分隔）")
    ap.add_argument("--multi", action="store_true", help="只跑多轮任务（测上下文管理时用）")
    ap.add_argument("--repeat", type=int, default=1,
                    help="整轮重复 N 次取中位数（单次 token 波动可达 ±26%，A/B 对比必须 ≥3）")
    ap.add_argument("--verbose", action="store_true", help="显示每步的工具调用")
    ap.add_argument("--baseline", action="store_true", help="把本次结果写成该版本的 baseline.json")
    args = ap.parse_args()

    set_dir = resolve_set(args.set)
    results_dir = set_dir / "runs"
    fingerprint = eval_fingerprint(set_dir)

    tasks = load_tasks(set_dir)
    task_count_total = len(tasks)
    if args.only:
        tasks = [t for t in tasks if t["id"] == args.only]
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        tasks = [t for t in tasks if t["id"] in want]
    if args.multi:
        tasks = [t for t in tasks if t.get("turns")]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        sys.exit("没有匹配的任务")

    multi = sum(1 for t in tasks if t.get("turns"))
    print(f"版本 {fingerprint['eval_version']}  "
          f"题目指纹 {fingerprint['tasks_sha']}  代码 {fingerprint['git_commit']}")
    print(f"共 {len(tasks)} 条任务（该版本共 {task_count_total} 条，其中多轮 {multi} 条），"
          f"重复 {args.repeat} 次")
    print("-" * 72)

    runs = []
    for rep in range(args.repeat):
        if args.repeat > 1:
            print(f"\n===== 第 {rep + 1}/{args.repeat} 轮 =====")
        # 只有最后一轮打印逐题明细，避免刷屏
        rows, summ = run_once(tasks, fingerprint, args,
                              print_detail=(args.repeat == 1 or rep == args.repeat - 1))
        runs.append({"rows": rows, "summary": summ})

    last = runs[-1]
    rows, summary = last["rows"], last["summary"]

    print("-" * 72)
    if args.repeat == 1:
        print(f"通过率    {summary['passed']}/{summary['total']} = {summary['pass_rate']:.1%}")
        if summary.get("mean_steps") is not None:
            print(f"平均步数  {summary['mean_steps']:.1f}")
        if summary.get("mean_tokens") is not None:
            print(f"平均 token {summary['mean_tokens']:.0f}")
        if summary.get("cache_hit_rate") is not None:
            print(f"缓存命中率 {summary['cache_hit_rate']:.1%}  "
                  f"（命中 {summary['cache_hit_tokens']:,} / 未命中 {summary['cache_miss_tokens']:,}）")
        if summary.get("compressed_messages"):
            print(f"历史压缩  共 {summary['compressed_messages']} 条工具结果被压缩")
        print(f"平均耗时  {summary['mean_seconds']:.1f}s")
    else:
        # 多次重复：取中位数，并列出每次的通过率（便于看波动）
        rates = [r["summary"]["pass_rate"] for r in runs]
        tok = [r["summary"]["mean_tokens"] for r in runs if r["summary"].get("mean_tokens")]
        stp = [r["summary"]["mean_steps"] for r in runs if r["summary"].get("mean_steps")]
        print(f"各次通过率  " + "  ".join(f"{x:.1%}" for x in rates))
        print(f"通过率均值  {statistics.mean(rates):.1%}")
        if stp:
            print(f"步数中位数  {statistics.median(stp):.1f}   (各次 {[round(x,1) for x in stp]})")
        if tok:
            print(f"token 中位数 {statistics.median(tok):.0f}   (各次 {[round(x) for x in tok]})")
        summary = {**summary, "repeat": args.repeat,
                   "pass_rate_mean": round(statistics.mean(rates), 4),
                   "median_tokens": round(statistics.median(tok)) if tok else None,
                   "median_steps": round(statistics.median(stp), 2) if stp else None,
                   "per_repeat": [{"pass_rate": r["summary"]["pass_rate"],
                                   "mean_tokens": r["summary"].get("mean_tokens"),
                                   "mean_steps": r["summary"].get("mean_steps")}
                                  for r in runs]}

    # ---------- 落盘 ----------
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = results_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "results.jsonl"
    out_file.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )

    summary = {"run": stamp, **fingerprint, **summary}
    summary_file = out_dir / "summary.json"
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n明细已写入 {out_file}")
    print(f"汇总已写入 {summary_file}")

    if args.baseline:
        baseline_file = set_dir / "baseline.json"
        baseline_file.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"基线已写入 {baseline_file}（这一份建议提交进 git）")


if __name__ == "__main__":
    main()
