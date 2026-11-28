# -*- coding: utf-8 -*-
"""生成评测任务：标准答案由 pandas 现算，绝不手抄、绝不问模型。

用法（项目根目录）：
    python scripts/gen_tasks.py                    # v1 的题 + 6 道新题 → eval_task/v2/tasks.jsonl
    python scripts/gen_tasks.py --base eval_task/v1/tasks.jsonl --out eval_task/v2/tasks.jsonl
    python scripts/gen_tasks.py --out /tmp/t.jsonl  # 只看看生成结果

为什么必须由脚本算：
    手抄标准答案一定会出错，而标准答案一错，后面所有评测结论都是假的
    （v1 第一版就因为漏了千分位处理，把 3 道对的题判成错的）。

为什么 v2 = v1 的题 + 新题：
    新题压难度，但不能丢掉 v1 的回归题——否则以后改坏了脏数据处理，
    你没有任何一道题能报警。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 需要跨文件或复杂口径的计算，写成函数（仍然由 pandas 现算，绝不手抄）
# ---------------------------------------------------------------------------
def _gdp_china_rank(_df=None) -> list:
    """2020 年中国人均 GDP 在「真正的国家」里的排名。

    这份数据混着 High income / Low income 等 4 行收入分组（iso3 为空），
    不排除就会多算一名 → 错答 95，正确答案 94。
    """
    d = pd.read_csv(ROOT / "eval_data/ny_gdp_pcap_cd/data.csv")
    d = d[(d["year"] == 2020) & d["value"].notna()]
    d = d[d["iso3"].notna() & (d["iso3"].astype(str).str.len() == 3)]
    d = d.sort_values("value", ascending=False).reset_index(drop=True)
    return [int(d[d["country"] == "China"].index[0] + 1)]


def _le_china_rank(_df=None) -> list:
    """2020 年中国预期寿命的排名。

    三层陷阱：全部行 68 名 / 只排除 Code 为空 67 名 / 再排除 OWID 区域码 65 名。
    """
    d = pd.read_csv(ROOT / "eval_data/owid_life_expectancy/data.csv")
    d = d[(d["Year"] == 2020) & d["Life expectancy"].notna()]
    d = d[d["Code"].notna() & (d["Code"].astype(str).str.len() == 3)]
    d = d.sort_values("Life expectancy", ascending=False).reset_index(drop=True)
    return [int(d[d["Entity"] == "China"].index[0] + 1)]


def _two_ranks(_df=None) -> list:
    """同一会话里查两份数据的排名 —— 专门测「多数据集并存」"""
    return _gdp_china_rank() + _le_china_rank()


def _ecb_2022_peak(_df) -> list:
    """2022 年 HICP 同比最高那个月的数值"""
    d = _df[_df["period"].astype(str).str.startswith("2022")]
    return [round(float(d.loc[d["value"].idxmax(), "value"]), 4)]


HICP = "eval_data/ecb_hicp/data.csv"
LE = "eval_data/owid_life_expectancy/data.csv"
GDP = "eval_data/ny_gdp_pcap_cd/data.csv"
POLICY = "eval_data/ecb_policy_rate/data.csv"
MTCARS = "eval_data/r_mtcars/data.csv"
TOOTH = "eval_data/r_toothgrowth/data.csv"
AIRQ = "eval_data/uci_airquality/data.csv"
PLANT = "eval_data/r_plantgrowth/data.csv"


def _strat_p(df) -> list:
    """v6 方法题：整体 Welch p 与「固定 dose=1.0」的 Welch p。

    这两个数就是「混杂」的证据：整体 0.060（不显著）→ 固定剂量 0.001（显著）。
    用 Welch 而不是 Student：即使方差不齐也稳；两者在这份数据上差 <0.001，
    所以容差 0.001 能同时容纳。
    """
    from scipy import stats
    oj = df[df["supp"] == "OJ"]["len"]
    vc = df[df["supp"] == "VC"]["len"]
    overall = stats.ttest_ind(oj, vc, equal_var=False).pvalue
    d1 = df[df["dose"] == 1.0]
    dose1 = stats.ttest_ind(d1[d1["supp"] == "OJ"]["len"],
                            d1[d1["supp"] == "VC"]["len"],
                            equal_var=False).pvalue
    return [round(float(overall), 4), round(float(dose1), 5)]



def _read(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def _real_countries_2020(rel: str) -> pd.DataFrame:
    """取 2020 年「真实国家」行：value 非空，iso3 是三位国家码（排掉收入分组等汇总行）。"""
    d = _read(rel)
    d = d[(d["year"] == 2020) & d["value"].notna()]
    return d[d["iso3"].notna() & (d["iso3"].astype(str).str.len() == 3)]


def _le_growth_cn(_=None) -> list:
    s = _read(LE).query("Entity == 'China'").set_index("Year")["Life expectancy"].dropna()
    return [round(float(s.loc[2020] - s.loc[2000]), 4)]


def _gdp_median_2020(_=None) -> list:
    return [round(float(_real_countries_2020(GDP)["value"].median()), 4)]


def _gdp_top1_2020(_=None) -> list:
    return [round(float(_real_countries_2020(GDP)["value"].max()), 2)]


def _ecb_range_2022(_=None) -> list:
    d = _read(HICP)
    d = d[d["period"].astype(str).str.startswith("2022")]
    return [round(float(d["value"].max() - d["value"].min()), 4)]


def _ecb_policy_2022_mean(_=None) -> list:
    d = _read(POLICY)
    d = d[d["period"].astype(str).str.startswith("2022")]
    return [round(float(d["value"].mean()), 4)]


def _ecb_vs_hicp_2022(_=None) -> list:
    """2022 年通胀均值 − 政策利率均值（跨两个数据集）"""
    hicp = _read(HICP)
    hicp = hicp[hicp["period"].astype(str).str.startswith("2022")]
    pol = _read(POLICY)
    pol = pol[pol["period"].astype(str).str.startswith("2022")]
    return [round(float(hicp["value"].mean() - pol["value"].mean()), 4)]


def _mtcars_corr(_=None) -> list:
    d = _read(MTCARS)
    return [round(float(d["mpg"].corr(d["hp"])), 4)]


# ---------------------------------------------------------------------------
# 新加的 6 道题。每道题的 expect_numbers 都由 build(df) 现算。
# ---------------------------------------------------------------------------
SPECS = [
    {
        "id": "mtcars_cyl_mpg",
        "file": "eval_data/r_mtcars/data.csv",
        "ask": "按气缸数（cyl）分组，mpg 的平均值分别是多少？请给出具体数字",
        "tol": 0.01,
        "max_steps": 10,
        "note": "简单基线：小数据集 + 单维度分组",
        "build": lambda df: [round(float(v), 4)
                             for v in df.groupby("cyl")["mpg"].mean().sort_values(ascending=False)],
    },
    {
        "id": "toothgrowth_supp_len",
        "file": "eval_data/r_toothgrowth/data.csv",
        "ask": "按补充方式（supp）分组，牙齿长度 len 的平均值分别是多少？",
        "tol": 0.01,
        "max_steps": 10,
        "note": "双类别分组，数值接近，考察精度",
        "build": lambda df: [round(float(v), 4)
                             for v in df.groupby("supp")["len"].mean()],
    },
    {
        "id": "life_exp_2020_max",
        "file": "eval_data/owid_life_expectancy/data.csv",
        "ask": "2020 年预期寿命（Life expectancy）最高的数值是多少？请给出具体数字",
        "tol": 0.01,
        "max_steps": 12,
        "note": "长表筛选；注意有 15 行是地区汇总（Code 为空），不该当成国家",
        "build": lambda df: [round(float(
            df[(df["Year"] == 2020) & df["Code"].notna()]["Life expectancy"].max()
        ), 4)],
    },
    {
        "id": "ecb_hicp_2022_mean",
        "file": "eval_data/ecb_hicp/data.csv",
        "ask": "2022 年全年 value 的平均值是多少？",
        "tol": 0.01,
        "max_steps": 12,
        "note": "时间序列按年筛选再聚合，考察 period 的字符串处理",
        "build": lambda df: [round(float(
            df[df["period"].astype(str).str.startswith("2022")]["value"].mean()
        ), 4)],
    },
    {
        "id": "china_gdp_2020",
        "file": "eval_data/ny_gdp_pcap_cd/data.csv",
        "ask": "China（中国）2020 年的人均 GDP 数值是多少？",
        "tol": 1.0,
        "max_steps": 12,
        "note": "精确筛选（国家 + 年份），考察不瞎猜",
        "build": lambda df: [round(float(
            df[(df["country"] == "China") & (df["year"] == 2020)]["value"].iloc[0]
        ), 4)],
    },
    {
        "id": "absent_entity",
        "file": "eval_data/ny_gdp_pcap_cd/data.csv",
        "ask": "这份数据里有 Atlantis 这个国家的数据吗？如果没有，请直接说明没有，不要编造数字。",
        "max_steps": 10,
        "note": "★ 测「不编造」：数据里不存在，正确行为是明说没有",
        "expect_text_any": ["没有", "不包含", "不存在", "未找到", "无记录", "找不到"],
    },
    # ===== v3：歧义口径 / 多数据集 / 不编造 =====
    {
        "id": "china_gdp_rank",
        "file": "eval_data/ny_gdp_pcap_cd/data.csv",
        "ask": "2020 年 China（中国）的人均 GDP 在数据里的所有国家中排第几？"
               "注意：数据里混有少量收入分组等汇总行（iso3 为空），需要排除后才算排名。",
        "tol": 0,
        "max_steps": 14,
        "note": "★ 歧义口径：不排除汇总行会得到 95，正确是 94",
        "build": _gdp_china_rank,
    },
    {
        "id": "china_le_rank",
        "file": "eval_data/owid_life_expectancy/data.csv",
        "ask": "2020 年中国的预期寿命（Life expectancy）在数据里的所有国家中排第几？"
               "注意：数据里混有地区、收入分组等汇总行，它们的 Code 不是三位国家代码，需要排除后排名。",
        "tol": 0,
        "max_steps": 16,
        "note": "★ 三层陷阱：全含 68 / 只排 NaN 67 / 排对 65",
        "build": _le_china_rank,
    },
    {
        "id": "ecb_peak_month",
        "file": "eval_data/ecb_hicp/data.csv",
        "ask": "2022 年全年 value 最高的月份是哪个？请给出月份和数值",
        "tol": 0.01,
        "max_steps": 14,
        "note": "★ 口径：period 是月度字符串，要按 2022 前缀筛再取最大",
        "expect_text_any": ["2022-10", "2022 年 10", "2022年10"],
        "build": _ecb_2022_peak,
    },
    {
        "id": "gdp_total_absent",
        "file": "eval_data/ny_gdp_pcap_cd/data.csv",
        "ask": "这份数据里，China（中国）2020 年的 GDP 总量是多少？",
        "max_steps": 12,
        "note": "★ 测「不编造」：数据只有「人均」GDP，没有总量，正确行为是说明这一点",
        "expect_text_any": ["人均", "没有总量", "不包含", "不提供", "无法确定"],
    },
    {
        "id": "two_datasets_rank",
        "files": ["eval_data/ny_gdp_pcap_cd/data.csv", "eval_data/owid_life_expectancy/data.csv"],
        "ask": "分别告诉我：2020 年 China 的人均 GDP 排名、以及 2020 年中国的预期寿命排名。"
               "两份数据都要排除汇总行后再排名。",
        "tol": 0,
        "max_steps": 20,
        "note": "★ 多数据集：需要两次 load_data + 两次排名",
        "build": _two_ranks,
    },
    # ===== v4：隐含陷阱 / 多步计算 / 跨数据集 / 拒绝预测 / 格式严格 =====
    {
        "id": "le_rank_implicit",
        "file": LE,
        "ask": "2020 年中国的预期寿命在数据中的国家里排第几？",
        "tol": 0,
        "max_steps": 16,
        "note": "★★ 隐含陷阱：不提「汇总行」，正确 65；只排 NaN 会得 67，全含 68",
        "build": _le_china_rank,
    },
    {
        "id": "gdp_rank_implicit",
        "file": GDP,
        "ask": "2020 年 China 的人均 GDP 在数据中的国家里排第几？",
        "tol": 0,
        "max_steps": 14,
        "note": "★★ 隐含陷阱：不提「收入分组」，正确 94，不排除得 95",
        "build": _gdp_china_rank,
    },
    {
        "id": "le_growth_cn",
        "file": LE,
        "ask": "中国（China）的预期寿命从 2000 年到 2020 年增长了几年？给出具体数值",
        "tol": 0.01,
        "max_steps": 14,
        "note": "多步：同表两年相减，需要精确定位",
        "build": _le_growth_cn,
    },
    {
        "id": "le_future_2025",
        "file": LE,
        "ask": "中国 2025 年的预期寿命是多少？",
        "max_steps": 14,
        "note": "★ 拒绝预测：数据只到 2023，2025 没有数据，正确行为是说无法确定",
        "expect_text_any": ["没有", "无法确定", "不包含", "未覆盖", "超过", "超出", "截止", "截至"],
    },
    {
        "id": "gdp_median_2020",
        "file": GDP,
        "ask": "2020 年各国人均 GDP 的中位数是多少？",
        "tol": 1.0,
        "max_steps": 16,
        "note": "口径：要中位数不是均值（均值 16365，中位数 6131）；还要排掉汇总行",
        "build": _gdp_median_2020,
    },
    {
        "id": "gdp_top1_2020",
        "file": GDP,
        "ask": "2020 年人均 GDP 最高的国家是哪个？数值是多少？",
        "tol": 1.0,
        "max_steps": 14,
        "note": "取最大值（Monaco），需要正确处理数值",
        "build": _gdp_top1_2020,
    },
    {
        "id": "ecb_range_2022",
        "file": HICP,
        "ask": "2022 年全年通胀率的最高值与最低值相差多少？",
        "tol": 0.01,
        "max_steps": 14,
        "note": "多步：先筛年，再求极差（10.6 − 5.1）",
        "build": _ecb_range_2022,
    },
    {
        "id": "ecb_policy_2022_mean",
        "file": POLICY,
        "ask": "2022 年欧央行主要再融资利率的平均值是多少？",
        "tol": 0.01,
        "max_steps": 14,
        "note": "小数据集（2022 年仅 4 条），考察筛选正确",
        "build": _ecb_policy_2022_mean,
    },
    {
        "id": "ecb_vs_hicp_2022",
        "files": [HICP, POLICY],
        "ask": "2022 年的平均通胀率减去 2022 年的平均政策利率，结果是多少？"
               "两份数据都要先按 2022 年筛选。",
        "tol": 0.01,
        "max_steps": 20,
        "note": "★ 跨数据集计算：两次 load_data + 各自聚合 + 相减",
        "build": _ecb_vs_hicp_2022,
    },
    {
        "id": "mtcars_corr",
        "file": MTCARS,
        "ask": "在 mtcars 数据里，mpg 与 hp 的皮尔逊相关系数是多少？",
        "tol": 0.001,
        "max_steps": 12,
        "note": "统计量计算，考察是否用 corr 而不是心算",
        "build": _mtcars_corr,
    },
    {
        "id": "strict_format_table",
        "file": GDP,
        "ask": "用 markdown 表格列出 2020 年人均 GDP 最高的 3 个国家及其数值，"
               "数值保留 2 位小数，表格必须包含「排名」「国家」「人均GDP」三列。",
        "tol": 1.0,
        "max_steps": 16,
        "note": "★ 格式严格：必须 markdown 表格 + 2 位小数",
        "expect_regex": [r"\|.*\|", r"\d+\.\d{2}(?!\d)"],
        "build": lambda _: [round(float(v), 2) for v in
                            _real_countries_2020(GDP)["value"].nlargest(3)],
    },
    {
        "id": "gdp_precise",
        "file": GDP,
        "ask": "China（中国）2020 年的人均 GDP 精确到小数点后 4 位是多少？",
        "tol": 0.001,
        "max_steps": 12,
        "note": "★ 精度：容差收紧到 0.001，四舍五入到 2 位会失败",
        "build": lambda _: [round(float(
            _read(GDP).query("country == 'China' and year == 2020")["value"].iloc[0]
        ), 4)],
    },
    # ===== v5：多轮追问（测上下文管理 / 指代 / 长历史） =====
    # 判据只看最后一轮的答案；中间轮次的作用是把历史撑长、制造指代。
    {
        "id": "mt_iris_01",
        "file": "data/鸢尾花.csv",
        "turns": ["各品种 petal_length 的均值排名，给出具体数字",
                  "那 setosa 的 petal_length 最小值是多少？"],
        "tol": 0.01,
        "max_steps": 8,
        "note": "★ 多轮 + 指代：「那 setosa」依赖上一轮的品种集合",
        "build": lambda df: [round(float(
            df[df["species"] == "Iris-setosa"]["petal_length"].min()), 4)],
    },
    {
        "id": "mt_sales_01",
        "file": "data/sample.csv",
        "turns": ["各区域销售额总额排名，给出前三名和具体数字",
                  "华东区域有多少条订单记录？（按记录条数算，不是按金额非空计数）"],
        "tol": 0,
        "max_steps": 8,
        "note": "★ 口径陷阱：该区 9 条记录，但 amount 有 1 个缺失 → 按金额计数会得 8",
        "build": lambda df: [int((df["region"] == "华东").sum())],
    },
    {
        "id": "mt_mtcars_01",
        "file": MTCARS,
        "turns": ["按气缸数（cyl）分组，mpg 的平均值分别是多少",
                  "那 8 缸的车里面，mpg 最低的是多少？"],
        "tol": 0.01,
        "max_steps": 8,
        "note": "★ 多轮 + 指代：「那 8 缸」要延续上一轮的分组维度",
        "build": lambda df: [round(float(df[df["cyl"] == 8]["mpg"].min()), 4)],
    },
    {
        "id": "mt_ecb_01",
        "file": HICP,
        "turns": ["2022 年全年 value 的平均值是多少？",
                  "那 2022 年 value 最高的那个月，数值是多少？"],
        "tol": 0.01,
        "max_steps": 8,
        "note": "★ 多轮：第二轮复用了第一轮的时间筛选条件",
        "build": lambda df: [round(float(
            df[df["period"].astype(str).str.startswith("2022")]["value"].max()), 4)],
    },
    {
        "id": "mt_multi_01",
        "files": ["data/鸢尾花.csv", "data/sample.csv"],
        "turns": ["这两份数据各有多少行？",
                  "行数较少的那个，各区域销售额总额排名第一的是哪个区，金额多少？"],
        "tol": 1.0,
        "max_steps": 10,
        "note": "★ 多轮 + 多数据集：「行数较少的那个」需要在两个数据集中定向",
        # 注意：build 拿到的是 files[0]（鸢尾花），所以这里自己读第二份
        "build": lambda _: [round(float(
            _read("data/sample.csv").groupby("region")["amount"].sum().max()), 2)],
    },
    {
        "id": "mt_long_01",
        "file": "data/鸢尾花.csv",
        "turns": [
            "各品种 petal_length 的均值排名，给出具体数字",
            "那 petal_length 的最大值呢？",
            "按 species 分组，sepal_width 的标准差分别是多少？",
            "这份数据里有缺失值吗？",
            "最后问一个：Iris-versicolor 的 petal_width 中位数是多少？",
        ],
        "tol": 0.01,
        "max_steps": 8,
        "note": "★ 长对话（5 轮）：专门用来触发历史压缩，并检验压缩是否损害指代",
        "build": lambda df: [round(float(
            df[df["species"] == "Iris-versicolor"]["petal_width"].median()), 4)],
    },
    # ===== v6：方法选择 / 统计判断（考"该不该这么算"，不是"算得准不准"） =====
    # 判据都是「该方法的正确数字 + 必须说明的前提」。选错方法数字就不同，蒙不过去。
    # 约束：沙箱里只有 pandas/numpy/scipy，**没有 statsmodels**，
    #       所以题目必须能用 scipy 做出来（两因素 ANOVA 这类做不了，别出）。
    {
        "id": "method_stratify_tooth",
        "file": TOOTH,
        "ask": "ToothGrowth 里直接比较 OJ 与 VC 两组的 len，p 值是多少？"
               "再只看 dose=1.0 的记录比较这两组，p 值是多少？"
               "然后解释：为什么按剂量分开后结论会变？",
        "tol": 0.001,
        "max_steps": 14,
        "note": "★★ 混杂/分层：整体 p≈0.060 不显著，固定剂量后显著 → 必须指出 dose 是混杂因素",
        "expect_regex": [r"混杂|分层|控制|剂量|dose"],
        "build": _strat_p,
    },
    {
        "id": "method_skew_airq",
        "file": AIRQ,
        "ask": "airquality 数据里 pm2.5 的「典型水平」用哪个统计量表示更合适？"
               "给出该统计量的数值，并说明你依据什么判断。",
        "tol": 0.01,
        "max_steps": 12,
        "note": "★★ 偏态：均值 98.61 被长尾拉高，中位数 72 才代表典型水平；偏度 1.80",
        "expect_regex": [r"中位数|median", r"偏|右偏|长尾|skew"],
        "build": lambda df: [round(float(df["pm2.5"].dropna().median()), 2)],
    },
    {
        "id": "method_anova_plant",
        "file": PLANT,
        "ask": "plantgrowth 的三组 weight 均值是否存在显著差异？"
               "请说明你用的检验方法、该检验的前提是否满足（给出前提检验的结果），以及 p 值。",
        "tol": 0.001,
        "max_steps": 16,
        "note": "★★ 前提检验：正态(Shapiro p>0.45)+方差齐(Levene p=0.34) → ANOVA p=0.0159；"
                "改用 Kruskal 会得 0.0184，容差 0.001 能区分",
        "expect_regex": [r"ANOVA|方差分析|F\s*检验", r"正态|Shapiro|Levene|方差齐"],
        "build": lambda df: [round(float(
            stats.f_oneway(*[s["weight"].values for _, s in df.groupby("group")]).pvalue), 4)],
    },
]


def main() -> None:
    ap = argparse.ArgumentParser(description="生成评测任务（标准答案由 pandas 现算）")
    ap.add_argument("--base", default=str(ROOT / "eval_task" / "v1" / "tasks.jsonl"),
                    help="要继承的已有题目文件")
    ap.add_argument("--out", default=str(ROOT / "eval_task" / "v2" / "tasks.jsonl"),
                    help="输出文件")
    args = ap.parse_args()

    base_path = Path(args.base)
    out_path = Path(args.out)

    tasks: list[dict] = []
    if base_path.exists():
        tasks = [json.loads(x) for x in base_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        print(f"继承 {len(tasks)} 道已有题目 ← {base_path}")
    else:
        print(f"⚠ 基础题目文件不存在，只生成新题：{base_path}")

    existing_ids = {t["id"] for t in tasks}

    added = 0
    for spec in SPECS:
        if spec["id"] in existing_ids:
            print(f"  跳过（已存在）：{spec['id']}")
            continue

        files = spec.get("files") or [spec["file"]]
        task = {"id": spec["id"], "files": files,
                "max_steps": spec.get("max_steps", 10), "note": spec.get("note", "")}
        # 多轮题写 turns，单轮题写 ask（评测脚本两种都认）
        if spec.get("turns"):
            task["turns"] = spec["turns"]
        else:
            task["ask"] = spec["ask"]

        if "build" in spec:
            df = pd.read_csv(ROOT / files[0])       # 多文件题自己读第二份
            numbers = spec["build"](df)
            if any(pd.isna(n) for n in numbers):
                sys.exit(f"❌ {spec['id']} 算出了 NaN，检查列名或筛选条件")
            task["expect_numbers"] = numbers
            task["tol"] = spec.get("tol", 0.01)
            print(f"  + {spec['id']:<22} {files[0]:<36} 答案={numbers}")

        if "expect_text_any" in spec:
            task["expect_text_any"] = spec["expect_text_any"]
            print(f"  + {spec['id']:<22} {files[0]:<36} 期望包含任一：{spec['expect_text_any']}")

        # expect_regex 是「全部都要匹配」（AND），用于要求答案里同时出现几类表述
        if "expect_regex" in spec:
            task["expect_regex"] = spec["expect_regex"]
            print(f"  + {spec['id']:<22} {files[0]:<36} 必须同时匹配：{spec['expect_regex']}")

        tasks.append(task)
        added += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n",
        encoding="utf-8",
    )
    print(f"\n共 {len(tasks)} 道题（新增 {added} 道）→ {out_path}")


if __name__ == "__main__":
    main()
