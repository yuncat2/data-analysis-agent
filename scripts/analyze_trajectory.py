# -*- coding: utf-8 -*-
"""分析评测轨迹：统计工具使用频次和 run_python 里的操作模式。

用途：为「把反复生成的代码固化成工具函数」提供数据依据。
    频次高的操作 → 值得做成专门函数；只出现一次的 → 不值得。

用法（项目根目录）：
    python scripts/analyze_trajectory.py             # 最新版本的最近一次 run
    python scripts/analyze_trajectory.py --set v4
    python scripts/analyze_trajectory.py --set v4 --run 20261112-190500
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_TASK = ROOT / "eval_task"

# pandas / numpy 常见操作的正则。命中就记一次。
PATTERNS = {
    "groupby + 聚合": r"\.groupby\(",
    "agg 多聚合": r"\.agg\(",
    "mean": r"\.mean\(",
    "sum": r"\.sum\(",
    "median": r"\.median\(",
    "count": r"\.count\(",
    "std": r"\.std\(",
    "min/max": r"\.(?:min|max)\(",
    "排序 sort_values": r"\.sort_values\(",
    "取前 N nlargest": r"\.n(?:largest|smallest)\(",
    "合并 merge": r"\.merge\(",
    "条件筛选 query": r"\.query\(",
    "位置索引 loc/iloc": r"\.(?:loc|iloc)\[",
    "字符串 .str": r"\.str\.",
    "缺失值处理": r"\.(?:isna|notna|dropna|fillna)\(",
    "透视表 pivot": r"\.pivot_table\(",
    "相关 corr": r"\.corr\(",
    "去重": r"\.(?:duplicated|drop_duplicates)\(",
    "类型转换 astype": r"\.astype\(",
    "日期解析 to_datetime": r"to_datetime\(",
    "列名清洗 strip": r"\.strip\(\)",
    "唯一值 unique/nunique": r"\.(?:unique|nunique)\b",
    "分箱 cut/qcut": r"\b(?:pd\.)?(?:cut|qcut)\(",
    "统计检验 ttest": r"\.ttest|ttest_ind|mannwhitneyu",
}


def latest_run(set_dir: Path) -> Path:
    """取该版本最近一次「非朴素基线」的 run。"""
    runs = [p for p in (set_dir / "runs").iterdir() if p.is_dir()]
    runs = [p for p in runs if not p.name.endswith("-naive")]
    if not runs:
        raise SystemExit(f"{set_dir.name} 还没有可分析的 run")
    return sorted(runs, key=lambda p: p.name)[-1]


def load_rows(run_dir: Path) -> list[dict]:
    f = run_dir / "results.jsonl"
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]


def extract_code(args) -> str:
    """取出 run_python 的代码。

    兼容两种格式：
        dict  —— 新格式（run_eval 已解析并截断字段）
        str   —— 旧格式（历史 run 里存的 JSON 字符串，可能是被截断的）
    """
    if isinstance(args, dict):
        return args.get("code", "") or ""
    try:
        return json.loads(args or "{}").get("code", "") or ""
    except Exception:
        return args or ""


def main() -> None:
    ap = argparse.ArgumentParser(description="分析评测轨迹里的工具与操作频次")
    ap.add_argument("--set", help="版本名，默认最新")
    ap.add_argument("--run", help="指定 run 目录名")
    ap.add_argument("--min-count", type=int, default=3,
                    help="频次达到多少才建议固化成函数（默认 3）")
    args = ap.parse_args()

    sets = sorted(p for p in EVAL_TASK.iterdir() if p.is_dir() and (p / "tasks.jsonl").exists())
    set_dir = EVAL_TASK / args.set if args.set else sets[-1]
    run_dir = (set_dir / "runs" / args.run) if args.run else latest_run(set_dir)
    rows = load_rows(run_dir)

    print(f"分析：{set_dir.name} / {run_dir.name}   共 {len(rows)} 题\n")

    # ---------- 1. 工具使用频次 ----------
    tool_counter = Counter()
    per_task_ops = {}
    op_counter = Counter()
    no_run_python = []

    for r in rows:
        traj = r.get("trajectory") or []
        ops = set()
        for step in traj:
            tool_counter[step.get("tool")] += 1
            if step.get("tool") == "run_python":
                code = extract_code(step.get("args", ""))
                for name, pat in PATTERNS.items():
                    if re.search(pat, code):
                        op_counter[name] += 1
                        ops.add(name)
        per_task_ops[r["id"]] = ops
        if "run_python" not in [s.get("tool") for s in traj]:
            no_run_python.append(r["id"])

    print("=== 工具使用频次 ===")
    for name, n in tool_counter.most_common():
        print(f"  {name:<14}{n:>4} 次")
    print(f"\n  只用 load_data 就答完的题：{len(no_run_python)} 道")
    if no_run_python:
        print(f"    {', '.join(no_run_python)}")

    # ---------- 1b. 步数构成与「无效步」 ----------
    # 注意：errored 只有新格式的 run 才有；旧记录会全部算作「未报错」。
    total_steps = sum(len(r.get("trajectory") or []) for r in rows)
    err_steps = sum(1 for r in rows for s in (r.get("trajectory") or []) if s.get("errored"))
    with_err = [r["id"] for r in rows
                if any(s.get("errored") for s in (r.get("trajectory") or []))]
    has_flag = any("errored" in s for r in rows for s in (r.get("trajectory") or []))

    print("\n=== 步数构成（只算工具调用步；每轮的「最终回答」不是工具步）===")
    print(f"  工具调用步合计 {total_steps}")
    if has_flag:
        rate = err_steps / total_steps if total_steps else 0
        print(f"  其中报错步     {err_steps}  ({rate:.0%})  ← 这些是「无效步」的主要来源")
        print(f"  有报错的任务   {len(with_err)} 道")
        if with_err:
            print(f"    {', '.join(with_err[:10])}")
    else:
        print("  （该 run 是旧格式，没有 errored 字段，无法统计报错步）")

    # ---------- 2. 操作频次 ----------
    print("\n=== run_python 里的操作频次（按次数排序）===")
    print(f"  {'操作':<22}{'出现题数':>8}   建议")
    print("  " + "-" * 52)
    for name, n in op_counter.most_common():
        hint = "★ 可固化成函数" if n >= args.min_count else ""
        print(f"  {name:<22}{n:>8}   {hint}")

    # ---------- 3. 候选函数清单 ----------
    candidates = [n for n, c in op_counter.most_common() if c >= args.min_count]
    print(f"\n=== 候选函数（出现 ≥ {args.min_count} 次）共 {len(candidates)} 个 ===")
    for i, name in enumerate(candidates, 1):
        print(f"  {i}. {name}")

    # ---------- 4. 每题明细 ----------
    print("\n=== 每题用到的操作 ===")
    for r in rows:
        mark = "✓" if r["ok"] else "✗"
        ops = sorted(per_task_ops.get(r["id"], []))
        print(f"  {mark} {r['id']:<22} 步数={r.get('steps')}  {' | '.join(ops) if ops else '(无 run_python)'}")


if __name__ == "__main__":
    main()
