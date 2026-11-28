# -*- coding: utf-8 -*-
"""朴素基线：不用 agent，直接把 CSV 内容塞进 prompt，让模型一次答完。

这是为了回答面试里最尖锐的那一问：
    「你的 agent 比『直接把表格丢给 ChatGPT』强在哪？」

做法：
    1. 读 CSV（能塞多少塞多少，超出就截断，并明确告知模型）
    2. 一次 API 调用，不给任何工具
    3. 用和 agent 完全相同的判据（evals.run_eval.check_answer）打分

诚实说明这个基线的局限：
    大文件必然被截断，模型看不到全量数据。它答错不代表"模型笨"，
    而代表"把数据塞进上下文这种做法在稍大的数据上根本不成立"——
    这恰恰是要证明的结论。

用法（项目根目录，注意是 -m）：
    python -m evals.baseline_naive --set v3
    python -m evals.baseline_naive --set v3 --limit 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from openai import OpenAI

from data_agent.config import API_KEY, BASE_URL, MODEL, ROOT
from evals.run_eval import check_answer, eval_fingerprint, load_tasks, resolve_set

MAX_CHARS_PER_TASK = 15000      # 单个任务塞进 prompt 的字符上限（约 4–5k token）

SYSTEM_NAIVE = "你是数据分析助手。只依据提供的数据内容回答，不要编造。"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def read_csv_text(path: Path, max_chars: int) -> tuple[str, bool]:
    """读 CSV 文本，支持 GBK 回退。返回 (文本, 是否被截断)。"""
    raw = path.read_bytes()
    for enc in ("utf-8", "gbk"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", "replace")

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + f"\n……（内容已截断，原文共 {len(text)} 字符）"
    return text, truncated


def build_naive_messages(task: dict) -> tuple[list[dict], bool]:
    """把文件内容拼成一条 user 消息。多文件任务平分字符预算。"""
    files = task["files"]
    budget = MAX_CHARS_PER_TASK // max(1, len(files))

    blocks, any_truncated = [], False
    for rel in files:
        text, cut = read_csv_text(ROOT / rel, budget)
        any_truncated |= cut
        blocks.append(f"--- 文件：{rel} ---\n{text}")

    content = (
        "\n\n".join(blocks)
        + f"\n\n问题：{task['ask']}\n\n"
        + "请直接给出答案和依据。注意：如果内容被截断，你看到的数据可能不完整。"
    )
    return [
        {"role": "system", "content": SYSTEM_NAIVE},
        {"role": "user", "content": content},
    ], any_truncated


def main() -> None:
    ap = argparse.ArgumentParser(description="朴素基线：把 CSV 塞进 prompt 直接问")
    ap.add_argument("--set", help="版本目录名，默认最新")
    ap.add_argument("--limit", type=int, help="只跑前 N 条")
    args = ap.parse_args()

    set_dir = resolve_set(args.set)
    fp = eval_fingerprint(set_dir)
    tasks = load_tasks(set_dir)
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"朴素基线（无工具，CSV 塞进 prompt）  版本 {fp['eval_version']}  "
          f"指纹 {fp['tasks_sha']}  共 {len(tasks)} 题")
    print("-" * 72)

    rows = []
    for task in tasks:
        messages, truncated = build_naive_messages(task)
        started = time.perf_counter()
        try:
            resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
            answer = resp.choices[0].message.content or ""
            usage = resp.usage
        except Exception as exc:                      # 上下文超限等异常要记下来，不能中断
            answer, usage = f"[异常] {exc!r}", None

        elapsed = time.perf_counter() - started
        ok, missing = check_answer(answer, task)
        tokens = usage.total_tokens if usage else None

        rows.append({"id": task["id"], "ok": ok, "missing": missing,
                     "tokens": tokens, "truncated": truncated, "seconds": round(elapsed, 2)})

        mark = "✓" if ok else "✗"
        cut = "[截断] " if truncated else ""
        print(f"{mark} {task['id']:<22} token={tokens if tokens else '-':<7}{cut}"
              f"{elapsed:5.1f}s   {task.get('note', '')[:30]}")
        if not ok:
            print(f"    缺：{missing}")

    total = len(rows)
    passed = sum(1 for r in rows if r["ok"])
    toks = [r["tokens"] for r in rows if r["tokens"]]
    print("-" * 72)
    print(f"朴素基线通过率 {passed}/{total} = {passed / total:.1%}")
    if toks:
        print(f"平均 token {statistics.mean(toks):.0f}")
    print(f"被截断的题数 {sum(1 for r in rows if r['truncated'])}/{total}")

    # ---------- 落盘，便于和 agent 的结果对比（目录名带 -naive 后缀） ----------
    out_dir = set_dir / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "-naive")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    summary = {
        "run": out_dir.name, "kind": "naive", **fp,
        "total": total, "passed": passed, "pass_rate": round(passed / total, 4),
        "mean_tokens": round(statistics.mean(toks)) if toks else None,
        "truncated_tasks": sum(1 for r in rows if r["truncated"]),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已写入 {out_dir}")


if __name__ == "__main__":
    main()
