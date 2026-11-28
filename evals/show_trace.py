# -*- coding: utf-8 -*-
"""可视化 agent 的执行轨迹：每一步调了什么工具、写了什么代码、有没有报错。

两种数据来源：
    --set v5 [--run <目录名>]   读一次评测 run 的 results.jsonl（每题一个卡片）
    --session <会话名>          读 runs/sessions/<会话名>/history.json（一次对话）

用法（项目根目录）：
    python -m evals.show_trace --set v5                 # 最新一次 run，生成 HTML
    python -m evals.show_trace --session demo           # 看某个会话干了什么
    python -m evals.show_trace --set v5 --text          # 直接打印到终端
    python -m evals.show_trace --set v5 --only dirty_03

输出：runs/traces/<名字>-<时间戳>.html，路径会打印出来，用浏览器打开即可。
"""
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

from data_agent.config import ROOT
from evals.run_eval import extract_trajectory, list_sets

TRACE_DIR = ROOT / "runs" / "traces"


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def latest_run(set_dir: Path, name: str | None) -> Path:
    runs = [p for p in (set_dir / "runs").iterdir()
            if p.is_dir() and not p.name.endswith("-naive")]
    if name:
        return set_dir / "runs" / name
    return sorted(runs, key=lambda p: p.name)[-1]


def load_run_items(set_dir: Path, run_dir: Path) -> list[dict]:
    """一次评测 run → 每个任务一个卡片。"""
    f = run_dir / "results.jsonl"
    if not f.exists():
        raise SystemExit(f"找不到 {f}")
    items = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        items.append({
            "id": r["id"],
            "ok": r.get("ok"),
            "steps": r.get("steps"),
            "tokens": r.get("tokens"),
            "answer": r.get("answer") or "",
            "traj": r.get("trajectory") or [],
        })
    return items


def load_session_items(session_id: str) -> list[dict]:
    """一个交互式会话的历史 → 单个卡片。

    会话存档里是完整的 OpenAI messages 列表，所以直接用同一套 extract_trajectory。
    """
    path = ROOT / "runs" / "sessions" / session_id / "history.json"
    if not path.exists():
        raise SystemExit(f"找不到会话历史 {path}")
    messages = json.loads(path.read_text(encoding="utf-8"))

    # 最后一轮 assistant 的文本就是最终回答
    answer = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            answer = m["content"]
            break

    return [{
        "id": f"会话 {session_id}",
        "ok": None,
        "steps": len(extract_trajectory(messages)),
        "tokens": None,
        "answer": answer,
        "traj": extract_trajectory(messages),
    }]


# ---------------------------------------------------------------------------
# 渲染：终端
# ---------------------------------------------------------------------------
def step_args_text(args) -> str:
    """把参数渲染成可读文本（代码原样，其他字段一行）。"""
    if isinstance(args, str):                     # 旧格式：直接是字符串
        return args
    parts = []
    for k, v in (args or {}).items():
        if k == "code":
            parts.append(str(v))
        else:
            parts.append(f"{k} = {v}")
    return "\n".join(parts)


def render_text(items: list[dict], title: str) -> str:
    out = [f"=== {title} ==="]
    for it in items:
        head = f"\n--- {it['id']}"
        if it.get("ok") is not None:
            head += "  ✓" if it["ok"] else "  ✗"
        if it.get("steps"):
            head += f"  {it['steps']} 步"
        if it.get("tokens"):
            head += f"  {it['tokens']} token"
        out.append(head)

        for i, s in enumerate(it["traj"], 1):
            flag = "  [报错]" if s.get("errored") else ""
            out.append(f"  {i:>2}. {s.get('tool')}{flag}  (返回 {s.get('result_chars', 0)} 字符)")
            body = step_args_text(s.get("args"))
            for line in body.splitlines()[:12]:
                out.append(f"        {line}")
            if len(body.splitlines()) > 12:
                out.append("        ……")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 渲染：HTML
# ---------------------------------------------------------------------------
CSS = """
:root{--bg:#0f1115;--card:#171a21;--fg:#e6e8ee;--dim:#8b93a7;--line:#262b36;
--ok:#3ecf8e;--bad:#f2555a;--code:#0b0e13;--warn:#e0a03a}
@media (prefers-color-scheme:light){:root{--bg:#f6f7f9;--card:#fff;--fg:#1c2028;
--dim:#5d6675;--line:#e3e6ec;--code:#f2f4f8;--ok:#187c53;--bad:#c0272d;--warn:#a06700}}
*{box-sizing:border-box}
body{margin:0;padding:28px;background:var(--bg);color:var(--fg);
font:15px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
h1{font-size:20px;margin:0 0 6px}
.meta{color:var(--dim);font-size:13px;margin-bottom:20px}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:22px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px 16px;min-width:110px}
.stat b{display:block;font-size:20px;font-weight:600}
.stat span{color:var(--dim);font-size:12px}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;
margin-bottom:12px;overflow:hidden}
summary{cursor:pointer;padding:12px 16px;font-weight:600;list-style:none;
display:flex;gap:10px;align-items:center;flex-wrap:wrap}
summary::-webkit-details-marker{display:none}
summary:hover{background:rgba(128,128,128,.06)}
.tag{font-size:12px;font-weight:500;color:var(--dim);border:1px solid var(--line);
border-radius:999px;padding:1px 9px}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
td,th{border-top:1px solid var(--line);padding:9px 12px;vertical-align:top;text-align:left}
th{font-weight:600;color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
td.n{color:var(--dim);width:40px}
pre{margin:0;background:var(--code);border-radius:8px;padding:10px 12px;
overflow-x:auto;font:12.5px/1.55 ui-monospace,Consolas,monospace;white-space:pre-wrap}
.arg{color:var(--dim);font-family:ui-monospace,Consolas,monospace;font-size:12.5px}
.answer{padding:12px 16px;border-top:1px solid var(--line);color:var(--dim);
font-size:13px;white-space:pre-wrap;max-height:260px;overflow:auto}
"""


def render_html(items: list[dict], title: str, subtitle: str) -> str:
    total_steps = sum(len(it["traj"]) for it in items)
    err_steps = sum(1 for it in items for s in it["traj"] if s.get("errored"))
    passed = sum(1 for it in items if it.get("ok"))
    judged = sum(1 for it in items if it.get("ok") is not None)

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        f"<style>{CSS}</style>",
        f"<h1>{html.escape(title)}</h1>",
        f"<div class='meta'>{html.escape(subtitle)}</div>",
        "<div class='stats'>",
        f"<div class='stat'><b>{len(items)}</b><span>任务/会话</span></div>",
        f"<div class='stat'><b>{total_steps}</b><span>工具调用步</span></div>",
        f"<div class='stat'><b class='{'warn' if err_steps else 'ok'}'>{err_steps}</b>"
        f"<span>报错步</span></div>",
    ]
    if judged:
        parts.append(f"<div class='stat'><b>{passed}/{judged}</b><span>通过</span></div>")
    parts.append("</div>")

    for it in items:
        if it.get("ok") is None:
            mark = ""
        else:
            mark = "<span class='ok'>✓</span>" if it["ok"] else "<span class='bad'>✗</span>"
        tags = [f"<span class='tag'>{len(it['traj'])} 步</span>"]
        if it.get("tokens"):
            tags.append(f"<span class='tag'>{it['tokens']:,} token</span>")
        nerr = sum(1 for s in it["traj"] if s.get("errored"))
        if nerr:
            tags.append(f"<span class='tag warn'>{nerr} 步报错</span>")

        parts.append("<details open><summary>"
                     f"{mark} <span>{html.escape(str(it['id']))}</span>{''.join(tags)}"
                     "</summary>")

        if not it["traj"]:
            parts.append("<table><tr><td class='n'>—</td><td>没有工具调用</td></tr></table>")
        else:
            parts.append("<table><tr><th>#</th><th>工具 / 代码</th><th>结果</th></tr>")
            for i, s in enumerate(it["traj"], 1):
                args = step_args_text(s.get("args"))
                code_block = ""
                if args.strip():
                    code_block = f"<pre>{html.escape(args)}</pre>"
                tool = html.escape(str(s.get("tool")))
                badge = " <span class='bad'>[报错]</span>" if s.get("errored") else ""
                parts.append(
                    f"<tr><td class='n'>{i}</td>"
                    f"<td><div class='arg'>{tool}{badge}</div>{code_block}</td>"
                    f"<td class='arg'>{s.get('result_chars', 0)} 字符</td></tr>"
                )
            parts.append("</table>")

        ans = (it.get("answer") or "").strip()
        if ans:
            parts.append(f"<div class='answer'>{html.escape(ans[:1200])}</div>")
        parts.append("</details>")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="可视化 agent 执行轨迹")
    ap.add_argument("--set", help="评测版本名，默认最新")
    ap.add_argument("--run", help="指定 run 目录名")
    ap.add_argument("--session", help="改为渲染某个交互式会话（runs/sessions/<名字>）")
    ap.add_argument("--only", help="只看某个任务 id")
    ap.add_argument("--text", action="store_true", help="直接打印到终端，不生成 HTML")
    args = ap.parse_args()

    if args.session:
        items = load_session_items(args.session)
        title = f"会话轨迹 · {args.session}"
        subtitle = "来自 runs/sessions/<会话>/history.json"
        stem = f"session-{args.session}"
    else:
        sets = list_sets()
        if not sets:
            raise SystemExit("没有任何评测版本目录")
        set_dir = (ROOT / "eval_task" / args.set) if args.set else sets[-1]
        run_dir = latest_run(set_dir, args.run)
        items = load_run_items(set_dir, run_dir)
        title = f"评测轨迹 · {set_dir.name} / {run_dir.name}"
        subtitle = "每一步的工具调用、生成的代码、是否报错"
        stem = f"{set_dir.name}-{run_dir.name}"

    if args.only:
        items = [it for it in items if it["id"] == args.only]
        if not items:
            raise SystemExit(f"找不到任务 {args.only}")

    if args.text:
        print(render_text(items, title))
        return

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    # run 目录名本身已带时间戳，不必再叠一个；会话会不断更新，才需要加时间戳
    fname = (f"session-{args.session}-{time.strftime('%Y%m%d-%H%M%S')}.html"
             if args.session else f"{stem}.html")
    out = TRACE_DIR / fname
    out.write_text(render_html(items, title, subtitle), encoding="utf-8")
    print(f"已生成：{out}")
    print(f"用浏览器打开即可查看 {len(items)} 个任务、"
          f"{sum(len(it['traj']) for it in items)} 步执行细节。")


if __name__ == "__main__":
    main()
