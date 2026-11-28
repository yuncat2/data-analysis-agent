# -*- coding: utf-8 -*-
"""对比两次评测结果。

用法（项目根目录）：
    python -m evals.compare                     # 自动取最近两次
    python -m evals.compare <run1> <run2>       # 指定 run 目录名，如 20261104-192412
    python -m evals.compare <路径1> <路径2>     # 也可以给完整路径（可跨版本对比）

它回答三个问题：
    1. 两次结果可比吗？（题目指纹/版本是否一致）
    2. 通过率、步数、token 怎么变了？
    3. 哪几道题从「过」变成「挂」了（回归）？哪几道修好了？
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.run_eval import list_sets


def all_run_dirs() -> list[Path]:
    """收集所有版本下的 run 目录，按时间戳名排序。"""
    out: list[Path] = []
    for s in list_sets():
        runs_dir = s / "runs"
        if runs_dir.is_dir():
            out.extend(p for p in runs_dir.iterdir() if p.is_dir())
    return sorted(out, key=lambda p: p.name)


def find_run(name: str) -> Path:
    """先当路径解析，找不到再在所有版本里按目录名搜。"""
    p = Path(name)
    if p.is_dir():
        return p
    for d in all_run_dirs():
        if d.name == name:
            return d
    sys.exit(f"找不到 run：{name}")


def load_run(d: Path) -> dict:
    """读一次 run 的 summary + 每题明细。summary 缺失时从 results.jsonl 现算。"""
    results = {}
    rj = d / "results.jsonl"
    if rj.exists():
        for line in rj.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                results[row["id"]] = row

    sj = d / "summary.json"
    if sj.exists():
        summary = json.loads(sj.read_text(encoding="utf-8"))
    else:
        # 兼容旧记录：从 results.jsonl 现算
        total = len(results)
        passed = sum(1 for r in results.values() if r.get("ok"))
        steps = [r["steps"] for r in results.values() if r.get("steps")]
        toks = [r["tokens"] for r in results.values() if r.get("tokens")]
        summary = {
            "run": d.name,
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total else 0,
            "mean_steps": round(sum(steps) / len(steps), 2) if steps else None,
            "mean_tokens": round(sum(toks) / len(toks)) if toks else None,
        }

    return {"name": d.name, "dir": str(d), "summary": summary, "results": results}


def fmt_delta(before, after, unit="") -> str:
    if before is None or after is None:
        return "—"
    delta = after - before
    if abs(delta) < 1e-9:
        return "不变"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}{unit}"


def main() -> None:
    runs = all_run_dirs()
    if len(sys.argv) >= 3:
        a, b = find_run(sys.argv[1]), find_run(sys.argv[2])
    elif len(runs) >= 2:
        a, b = runs[-2], runs[-1]
    else:
        sys.exit(f"至少需要两次 run 才能对比，当前只有 {len(runs)} 次")

    old, new = load_run(a), load_run(b)
    so, sn = old["summary"], new["summary"]

    print(f"对比：{old['name']}  →  {new['name']}\n" + "=" * 72)

    # ---------- 1. 可比性 ----------
    sha_o, sha_n = so.get("tasks_sha"), sn.get("tasks_sha")
    ver_o, ver_n = so.get("eval_version"), sn.get("eval_version")
    if sha_o and sha_n and sha_o != sha_n:
        print("⚠ 两次的题目指纹不同 —— 通过率不能直接比较！")
        print(f"   旧 {ver_o}/{sha_o}   新 {ver_n}/{sha_n}")
        print("   题目变了，先看「新增/消失的任务」，再谈涨跌。\n")
    elif sha_n:
        print(f"✅ 题目指纹一致（{sha_n}，版本 {ver_n}），结果可比\n")
    else:
        print("ℹ 这两次都没有版本指纹（早期记录），无法判断题目是否一致\n")

    # ---------- 2. 总体指标 ----------
    print(f"{'指标':<12}{'旧':>10}{'新':>12}{'变化':>12}")
    print("-" * 46)
    print(f"{'通过率':<12}{so['pass_rate']:>10.1%}{sn['pass_rate']:>12.1%}"
          f"{fmt_delta(so['pass_rate'] * 100, sn['pass_rate'] * 100, '%'):>12}")
    print(f"{'平均步数':<12}{(so.get('mean_steps') or 0):>10.1f}{(sn.get('mean_steps') or 0):>12.1f}"
          f"{fmt_delta(so.get('mean_steps'), sn.get('mean_steps')):>12}")
    print(f"{'平均token':<12}{(so.get('mean_tokens') or 0):>10.0f}{(sn.get('mean_tokens') or 0):>12.0f}"
          f"{fmt_delta(so.get('mean_tokens'), sn.get('mean_tokens')):>12}")

    hr_o, hr_n = so.get("cache_hit_rate"), sn.get("cache_hit_rate")
    o_txt = f"{hr_o:.1%}" if hr_o is not None else "—"
    n_txt = f"{hr_n:.1%}" if hr_n is not None else "—"
    d_txt = fmt_delta(hr_o * 100, hr_n * 100, "%") if (hr_o is not None and hr_n is not None) else "—"
    print(f"{'缓存命中率':<12}{o_txt:>10}{n_txt:>12}{d_txt:>12}")

    # ---------- 3. 逐题变化 ----------
    ro, rn = old["results"], new["results"]
    only_old = sorted(set(ro) - set(rn))
    only_new = sorted(set(rn) - set(ro))
    regressions = [i for i in sorted(set(ro) & set(rn)) if ro[i].get("ok") and not rn[i].get("ok")]
    fixed = [i for i in sorted(set(ro) & set(rn)) if not ro[i].get("ok") and rn[i].get("ok")]

    print()
    if only_new:
        print(f"新增任务 {len(only_new)}：{', '.join(only_new)}")
    if only_old:
        print(f"移除任务 {len(only_old)}：{', '.join(only_old)}")
    if regressions:
        print(f"⚠ 回归（过 → 挂）{len(regressions)}：{', '.join(regressions)}")
    if fixed:
        print(f"✅ 修好（挂 → 过）{len(fixed)}：{', '.join(fixed)}")
    if not (regressions or fixed or only_new or only_old):
        print("题目集合与逐题结果都没有变化")


if __name__ == "__main__":
    main()
