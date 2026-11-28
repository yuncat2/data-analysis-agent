# -*- coding: utf-8 -*-
"""查看评测历史：按版本分组，列出每次 run 的通过率、成本和可比性。

用法（项目根目录）：
    python -m evals.history

「备注」列说明这次结果能不能和该版本的当前题目直接比较：
    可比         指纹与当前一致，且跑了全部任务
    只跑了 N 条  用 --limit/--only 跑的部分任务，通过率不代表全量
    题目不同     题目改过了，通过率不能跨这两次比较
    无指纹       早期的 run（加版本指纹之前）
"""
from __future__ import annotations

import json
from pathlib import Path

from evals.run_eval import EVAL_TASK_DIR, eval_fingerprint, list_sets, load_tasks


def summarize(d: Path) -> dict:
    """读一次 run。有 summary.json 就用它，没有就从 results.jsonl 现算（兼容旧记录）。"""
    sj = d / "summary.json"
    if sj.exists():
        return json.loads(sj.read_text(encoding="utf-8"))

    rows = []
    rj = d / "results.jsonl"
    if rj.exists():
        rows = [json.loads(x) for x in rj.read_text(encoding="utf-8").splitlines() if x.strip()]

    steps = [r["steps"] for r in rows if r.get("steps")]
    toks = [r["tokens"] for r in rows if r.get("tokens")]
    passed = sum(1 for r in rows if r.get("ok"))
    hits = sum(r.get("cache_hit") or 0 for r in rows)
    misses = sum(r.get("cache_miss") or 0 for r in rows)
    return {
        "run": d.name,
        "eval_version": "—",
        "tasks_sha": "—",
        "git_commit": "—",
        "total": len(rows),
        "passed": passed,
        "pass_rate": (passed / len(rows)) if rows else 0.0,
        "mean_steps": round(sum(steps) / len(steps), 2) if steps else None,
        "mean_tokens": round(sum(toks) / len(toks)) if toks else None,
        "cache_hit_rate": (hits / (hits + misses)) if (hits + misses) else None,
    }


def main() -> None:
    sets = list_sets()
    if not sets:
        raise SystemExit(f"没有找到任何版本目录：{EVAL_TASK_DIR}")

    print(f"评测目录：{EVAL_TASK_DIR}\n")

    for set_dir in sets:
        fp = eval_fingerprint(set_dir)
        task_count = len(load_tasks(set_dir))
        print(f"═══ {set_dir.name} ═══  指纹 {fp['tasks_sha']}  代码 {fp['git_commit']}  "
              f"共 {task_count} 条任务")

        runs_dir = set_dir / "runs"
        runs = sorted(p for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.is_dir() else []
        if not runs:
            print("  （还没有跑过）\n")
            continue

        print(f"  {'run':<16}{'题数':>4}{'通过率':>8}{'步数':>7}{'token':>8}{'命中率':>8}  备注")
        print("  " + "-" * 76)
        for d in runs:
            s = summarize(d)
            total = s.get("total") or 0
            sha = s.get("tasks_sha") or "—"
            steps = f"{s['mean_steps']:.1f}" if s.get("mean_steps") else "—"
            toks = f"{s['mean_tokens']}" if s.get("mean_tokens") else "—"
            rate = f"{s['cache_hit_rate']:.0%}" if s.get("cache_hit_rate") is not None else "—"

            if s.get("kind") == "naive":
                note = "朴素基线（无工具）"
            elif sha == "—":
                note = "无指纹（旧记录）"
            elif sha != fp["tasks_sha"]:
                note = "题目不同"
            elif total < task_count:
                note = f"只跑了 {total} 条"
            else:
                note = "可比"

            print(f"  {s['run']:<16}{total:>4}{s['pass_rate']:>8.1%}"
                  f"{steps:>7}{toks:>8}{rate:>8}  {note}")
        print()


if __name__ == "__main__":
    main()
