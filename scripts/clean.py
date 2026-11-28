# -*- coding: utf-8 -*-
"""清理项目里的临时产物。

跑评测会不断产生临时会话目录和 run 记录，久了会乱。这个脚本把它们收掉。

用法（项目根目录）：
    python scripts/clean.py                    # 清零风险的：eval 临时会话 + __pycache__
    python scripts/clean.py --dry-run          # 只报告会删什么，不动手
    python scripts/clean.py --prune-runs 3     # 另外把每个评测版本只留最近 3 个 run

哪些东西是"零风险可删"：
    runs/sessions/eval_*   评测时每条任务建的临时会话（下次跑评测会重建）
    **/__pycache__         Python 字节码缓存

哪些**不能删**：
    runs/sessions/<真会话>  你自己的交互历史（demo / m2 / multi …）
    eval_task/*/baseline.json  版本基线（已进 git，删了也没法恢复对齐）
    eval_task/*/tasks.jsonl    题目（进 git）
    runs/traces/*.html        轨迹可视化产物
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "runs" / "sessions"
EVAL_TASK = ROOT / "eval_task"


def human(n: int) -> str:
    return f"{n / 1024 / 1024:.2f} MB" if n > 1024 * 1024 else f"{n / 1024:.0f} KB"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def clean_eval_sessions(dry: bool) -> tuple[int, int]:
    """删 runs/sessions/eval_* —— 评测临时会话。"""
    if not SESSIONS.is_dir():
        return 0, 0
    targets = [p for p in SESSIONS.iterdir() if p.is_dir() and p.name.startswith("eval_")]
    total = sum(dir_size(p) for p in targets)
    if not dry:
        for p in targets:
            shutil.rmtree(p, ignore_errors=True)
    return len(targets), total


def clean_pycache(dry: bool) -> tuple[int, int]:
    """删所有 __pycache__。"""
    targets = [p for p in ROOT.rglob("__pycache__") if p.is_dir()]
    total = sum(dir_size(p) for p in targets)
    if not dry:
        for p in targets:
            shutil.rmtree(p, ignore_errors=True)
    return len(targets), total


def prune_runs(keep: int, dry: bool) -> tuple[int, int]:
    """每个评测版本只保留最近 keep 个 run 目录（按名字排序，时间戳天然有序）。"""
    if not EVAL_TASK.is_dir():
        return 0, 0
    removed, total = 0, 0
    for set_dir in sorted(p for p in EVAL_TASK.iterdir() if p.is_dir()):
        runs_dir = set_dir / "runs"
        if not runs_dir.is_dir():
            continue
        runs = sorted((p for p in runs_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
        old = runs[:-keep] if keep > 0 else runs
        for p in old:
            total += dir_size(p)
            removed += 1
            if not dry:
                shutil.rmtree(p, ignore_errors=True)
        if old:
            print(f"  {set_dir.name}: 保留 {len(runs) - len(old)}，删 {len(old)}")
    return removed, total


def main() -> None:
    ap = argparse.ArgumentParser(description="清理临时产物")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不删除")
    ap.add_argument("--prune-runs", type=int, metavar="N",
                    help="每个评测版本只保留最近 N 个 run")
    args = ap.parse_args()

    tag = "[dry-run] " if args.dry_run else ""
    print(f"{tag}清理开始\n")

    n, size = clean_eval_sessions(args.dry_run)
    print(f"  评测临时会话 runs/sessions/eval_*   {n:>3} 个   {human(size)}")

    n, size = clean_pycache(args.dry_run)
    print(f"  __pycache__                        {n:>3} 个   {human(size)}")

    if args.prune_runs:
        print(f"\n  精简 eval_task/*/runs（每个版本保留最近 {args.prune_runs} 个）:")
        n, size = prune_runs(args.prune_runs, args.dry_run)
        print(f"  合计删除                            {n:>3} 个   {human(size)}")

    print("\n完成。" + ("（未实际删除）" if args.dry_run else ""))


if __name__ == "__main__":
    main()
