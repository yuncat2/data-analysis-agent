# -*- coding: utf-8 -*-
"""生成脏数据测试集（W1 验收用）。

用法：
    python scripts/make_dirty_data.py

输出到 data/dirty/：
    dirty_1_light.csv    只有单位后缀污染
    dirty_2_medium.csv   + 小数逗号 + 真缺失
    dirty_3_heavy.csv    + 合计行 + 列名尾随空格 + 单元格前后空格
    dirty_4_gbk.csv      medium 的内容，GBK 编码（专门测 tabular.py 的编码回退）
    ground_truth.json    标准答案
    manifest.json        每个文件破坏了什么

三条设计原则（面试被问「你的测试数据怎么设计的」时，就讲这三条）：

1. **只污染格式，不污染数值。** 清洗正确后，答案必须与干净数据完全一致（容差 1e-6）。
   如果造数据时把数值也改了，标准答案就跟着变，"答错了"和"没洗干净"就分不清了。

2. **难度分级，每个文件只比上一个多一类问题。**
   这样才能定位"它到底死在第几类问题上"，而不是笼统地知道"它不行"。

3. **污染列的选择是有讲究的。**
   答案列（petal_length）只做格式污染；真缺失（N/A）只放在答案用不到的列（sepal_width）。
   否则缺失值会让均值发生变化，标准答案就不再固定，评测失去可判定性。

固定随机种子，任何一次运行结果都一样。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "鸢尾花.csv"
OUT = ROOT / "data" / "dirty"

SEED = 42
GROUP_COL = "species"
ANSWER_COL = "petal_length"


def build_ground_truth(clean: pd.DataFrame) -> dict:
    g = clean.groupby(GROUP_COL)[ANSWER_COL]
    means = {k: round(float(v), 6) for k, v in g.mean().items()}
    counts = {k: int(v) for k, v in g.count().items()}
    ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "question": f"按 {GROUP_COL} 分组，{ANSWER_COL} 的均值排名",
        "group_col": GROUP_COL,
        "answer_col": ANSWER_COL,
        "n_rows_clean": int(len(clean)),
        "group_means": means,
        "group_counts": counts,
        "expected_ranking": [k for k, _ in ranked],
        "tolerance": 1e-6,
        "note": "只污染格式、不污染数值，因此四个脏文件的正确答案都与此完全相同"
        "（heavy 会多出一个『合计』组，必须被丢弃，不计入排名）",
    }


# ---------------------------------------------------------------------------
# 逐级污染
# ---------------------------------------------------------------------------
def corrupt_light(base: pd.DataFrame, rng: random.Random) -> tuple[pd.DataFrame, dict]:
    """单位后缀：petal_length 若干行写成 '1.4 cm' → 整列被 pandas 读成 object。"""
    df = base.copy().astype(object)
    rows = rng.sample(range(len(df)), k=20)
    for i in rows:
        df.at[i, ANSWER_COL] = f"{float(base.at[i, ANSWER_COL]):.1f} cm"
    return df, {
        "name": "light",
        "corruptions": [f"{ANSWER_COL} 列 {len(rows)} 行加上 ' cm' 单位后缀"],
        "expected_failure_if_unfixed": f"{ANSWER_COL} 被读成 object，groupby 求均值时报错或得到字符串拼接结果",
        "rows_touched": {ANSWER_COL: rows},
    }


def corrupt_medium(base: pd.DataFrame, rng: random.Random) -> tuple[pd.DataFrame, dict]:
    """light + 小数逗号（欧洲格式）+ 真缺失（放在答案列之外）。"""
    df, meta = corrupt_light(base, rng)

    comma_rows = [i for i in rng.sample(range(len(df)), k=15)]
    for i in comma_rows:
        df.at[i, "sepal_length"] = f"{float(base.at[i, 'sepal_length']):.1f}".replace(".", ",")

    na_rows = rng.sample(range(len(df)), k=10)
    for i in na_rows:
        # 注意：不能用 'N/A'、'NULL'、'None'——它们都在 pandas 的默认 na_values 里，
        # 会被自动转成 NaN，这样的"缺失"测试一点压力都没有。用中文'缺失'才真的需要清洗。
        df.at[i, "sepal_width"] = "缺失"

    meta["name"] = "medium"
    meta["corruptions"] += [
        f"sepal_length 列 {len(comma_rows)} 行改用小数逗号（如 '5,1'）",
        f"sepal_width 列 {len(na_rows)} 行改为 '缺失'（真缺失，故意选了不进 na_values 的写法）",
    ]
    meta["expected_failure_if_unfixed"] += "；sepal_length 读成 str；'缺失' 不会被识别为缺失，缺失统计和 nunique 全部失真"
    meta["rows_touched"].update({"sepal_length": comma_rows, "sepal_width_na": na_rows})
    return df, meta


def corrupt_heavy(base: pd.DataFrame, rng: random.Random) -> tuple[pd.DataFrame, dict]:
    """medium + 合计行 + 列名尾随空格 + 单元格前后空格。"""
    df, meta = corrupt_medium(base, rng)

    # 单元格前后空格：不清洗就会被当成新值（必须在改列名之前做，否则列名找不到）
    pad_rows = [i for i in rng.sample(range(len(base)), k=12) if i not in meta["rows_touched"]["sepal_width_na"]]
    for i in pad_rows:
        df.at[i, "sepal_width"] = f"  {df.at[i, 'sepal_width']}  "

    # 合计行：species='合计'，数值列给求和值
    total_row = {c: round(float(base[c].sum()), 1) for c in base.select_dtypes("number").columns}
    total_row[GROUP_COL] = "合计"
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    # 列名尾随空格：会让 df['sepal_length'] 直接 KeyError
    df.columns = [f"{c} " if c != GROUP_COL else c for c in df.columns]

    meta["name"] = "heavy"
    meta["corruptions"] += [
        "末尾追加 1 行『合计』（species='合计'，数值列为求和）",
        "所有数值列列名加了尾随空格（如 'sepal_length '）",
        f"{len(pad_rows)} 行存在单元格前后空格",
    ]
    meta["expected_failure_if_unfixed"] += "；若不丢合计行，排名里会多出一个数值巨大的『合计』组；列名带空格会导致 KeyError"
    meta["rows_touched"]["whitespace"] = pad_rows
    return df, meta


def corrupt_gbk(base: pd.DataFrame, rng: random.Random) -> tuple[pd.DataFrame, dict]:
    """medium 的内容 + 一行『合计』，然后整体用 GBK 编码存盘。

    为什么必须带中文：如果文件是纯 ASCII，GBK 和 UTF-8 字节完全一样，
    这个"编码测试"就是空的。加一行中文『合计』才能让 utf-8 解码真的失败。
    """
    df, meta = corrupt_medium(base, rng)
    total_row = {c: round(float(base[c].sum()), 1) for c in base.select_dtypes("number").columns}
    total_row[GROUP_COL] = "合计"
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    meta["name"] = "gbk"
    meta["corruptions"] = list(meta["corruptions"]) + [
        "末尾追加 1 行『合计』（同时保证文件含非 ASCII 字节）",
        "整个文件用 GBK 编码保存",
    ]
    meta["expected_failure_if_unfixed"] = (
        "utf-8 解码失败 → 若 tabular.py 的 GBK 回退失效，load_data 直接报错"
    )
    return df, meta


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(SRC)

    gt = build_ground_truth(clean)
    (OUT / "ground_truth.json").write_text(
        json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rng = random.Random(SEED)
    light, m1 = corrupt_light(clean, rng)
    rng = random.Random(SEED)
    medium, m2 = corrupt_medium(clean, rng)
    rng = random.Random(SEED)
    heavy, m3 = corrupt_heavy(clean, rng)
    rng = random.Random(SEED)
    gbk, m4 = corrupt_gbk(clean, rng)

    light.to_csv(OUT / "dirty_1_light.csv", index=False, encoding="utf-8")
    medium.to_csv(OUT / "dirty_2_medium.csv", index=False, encoding="utf-8")
    heavy.to_csv(OUT / "dirty_3_heavy.csv", index=False, encoding="utf-8")
    gbk.to_csv(OUT / "dirty_4_gbk.csv", index=False, encoding="gbk")

    manifest = {
        "source": str(SRC.name),
        "seed": SEED,
        "files": [m1, m2, m3, m4],
        "answer_column_note": f"{ANSWER_COL} 只做格式污染，所以四个文件的标准答案与干净数据一致",
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"标准答案（{GROUP_COL} 分组，{ANSWER_COL} 均值）:")
    for k in gt["expected_ranking"]:
        print(f"  {k:<16} {gt['group_means'][k]:>8}  (n={gt['group_counts'][k]})")
    print(f"\n已生成 {len(manifest['files'])} 个文件 → {OUT}")
    for m in manifest["files"]:
        print(f"  {m['name']:<7} {len(m['corruptions'])} 类污染")


if __name__ == "__main__":
    main()
