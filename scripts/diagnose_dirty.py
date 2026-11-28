# -*- coding: utf-8 -*-
"""脏数据测试集的离线自检（不调 API、不花钱、2 秒跑完）。

作用：在测 agent 之前，先证明「测试集本身是对的」——
manifest 声称每级比上一级多破坏了什么，这里逐条核实，
并给出「不洗数据会怎么失败」的实际报错，避免把测试集 bug 当成 agent bug。

用法：
    python scripts/diagnose_dirty.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DIRTY = ROOT / "data" / "dirty"
CLEAN = ROOT / "data" / "鸢尾花.csv"

ANS = "petal_length"
GRP = "species"
TOL = 1e-6


def read_like_agent(p: Path) -> pd.DataFrame:
    """完全复刻 tools/tabular.py 的读取策略（含 GBK 回退）。"""
    try:
        return pd.read_csv(p)
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="gbk")


def inspect(df: pd.DataFrame) -> dict:
    """对照 manifest，逐条核实污染是否真的存在。"""
    cols = list(df.columns)
    padded_cols = [c for c in cols if c != c.strip()]
    ans_col = next((c for c in cols if c.strip() == ANS), None)
    facts = {
        "shape": f"{df.shape[0]}x{df.shape[1]}",
        "列名带空格": len(padded_cols),
        "合计行": int((df[GRP].astype(str).str.strip() == "合计").sum()) if GRP in cols else -1,
    }
    if ans_col is not None:
        col = df[ans_col]
        facts["答案列dtype"] = str(col.dtype)
        s = col.astype(str)
        facts["带' cm'后缀"] = int(s.str.contains(r"\bcm\b", regex=True).sum())
        facts["是object"] = "是" if col.dtype == object else "否"
    if "sepal_length" in cols or "sepal_length " in cols:
        c = df[[x for x in cols if x.strip() == "sepal_length"][0]]
        facts["小数逗号"] = int(c.astype(str).str.contains(",", regex=False).sum())
    wc = [x for x in cols if x.strip() == "sepal_width"]
    if wc:
        c = df[wc[0]].astype(str)
        facts["字面'缺失'"] = int((c == "缺失").sum())
        facts["首尾空格单元格"] = int((c != c.str.strip()).sum())
    return facts


def naive_attempt(df: pd.DataFrame) -> str:
    """模拟「agent 没做任何清洗，直接 groupby 求均值」的结果。"""
    try:
        cols = {c.strip(): c for c in df.columns}
        g = df.groupby(GRP)[cols[ANS]].mean()
        return "未报错 → " + ", ".join(f"{k}={v:.3f}" for k, v in g.items())
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"


def oracle(df: pd.DataFrame) -> dict:
    """标准清洗流程 = agent 最少要做对的事。"""
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    d[GRP] = d[GRP].astype(str).str.strip()
    d = d[d[GRP] != "合计"]
    for c in d.columns:
        if c == GRP:
            continue
        s = d[c].astype(str).str.strip()
        s = (s.str.replace(r"\s*cm$", "", regex=True)
               .str.replace(",", ".", regex=False)
               .replace({"缺失": None, "": None}))
        d[c] = pd.to_numeric(s, errors="coerce")
    return d.groupby(GRP)[ANS].mean().round(6).to_dict()


def main() -> int:
    gt = json.loads((DIRTY / "ground_truth.json").read_text(encoding="utf-8"))
    expect = gt["group_means"]
    clean = pd.read_csv(CLEAN)

    print(f"干净数据基线: {len(clean)} 行")
    base = clean.groupby(GRP)[ANS].mean().round(6).to_dict()
    print(f"  均值 = {base}")
    print(f"  与 ground_truth 一致: {base == {k: round(v, 6) for k, v in expect.items()}}\n")

    files = sorted(DIRTY.glob("dirty_*.csv"))
    ok_all = True
    for p in files:
        df = read_like_agent(p)
        facts = inspect(df)
        got = oracle(df)
        match = all(abs(got.get(k, 1e9) - v) < TOL for k, v in expect.items()) and len(got) == len(expect)

        print("=" * 74)
        print(f"{p.name}")
        print(f"  污染核实: " + " | ".join(f"{k}={v}" for k, v in facts.items()))
        print(f"  不清洗直接算 → {naive_attempt(df)}")
        print(f"  标准清洗后 → {got}")
        print(f"  {'✅ 与标准答案一致' if match else '❌ 与标准答案不一致（测试集或 oracle 有问题）'}")
        ok_all &= match

    print("=" * 74)
    print("结论: " + ("四个文件都可用作评测基准 ✅" if ok_all else "存在不一致，先修测试集 ❌"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
