"""表格类工具：读文件 + 数据画像。"""
import re
import traceback
from pathlib import Path

import pandas as pd

from ..config import ROOT, datasets_file, QUALITY_HINTS_ENABLED


LOAD_DATA_SCHEMA = {
    "type": "function",
    "function": {
        "name": "load_data",
        "description": (
            "加载用户的数据文件（支持 .csv / .xlsx / .xls），返回列名、类型、缺失情况、"
            "前几行样本和数值列概要。只返回画像，不返回全量数据。开始任何计算前必须先调用它。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径，可以是相对路径"},
                "name": {"type": "string",
                         "description": "数据集的名字，省略则用文件名。对比多个数据源时请起有意义的名字，如 公司A_2024"}
            },
            "required": ["path"],
        },
    },
}


def _profile_text(df: pd.DataFrame) -> str:
    lines = [f"shape: {df.shape[0]} 行 x {df.shape[1]} 列", "", "列信息:"]
    miss = df.isna().sum()
    for col in df.columns:
        lines.append(
            f"- {col} | dtype={df[col].dtype} | 缺失={int(miss[col])} | "
            f"唯一值={df[col].nunique(dropna=True)}"
        )
    lines += ["", "前 5 行:", df.head(5).to_string(max_cols=20)]
    num = df.select_dtypes("number")
    if not num.empty:
        lines += ["", "数值列概要:", num.describe().to_string()]
    return "\n".join(lines)


_INVISIBLE = ("\ufeff", "\u200b", "\u200c", "\u200d", "\u00a0")


def _normalize_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """在数据入口做「无论问什么都该做」的规范化，返回 (新 df, 做了什么的说明)。

    原则：**人能在代码里解决的，就不要丢给模型。**
    但只做语义不变的修正——否则就是替模型做了它该判断的决策。

    做：
        列名去首尾空白、去不可见字符（BOM / 零宽空格 / 不间断空格）

    刻意不做：
        去重（取决于「多少条记录 vs 多少条唯一观测」的口径）
        填缺失、剔除汇总行（同上）
        去单位转数字（会丢掉原始文本）
    """
    notes: list[str] = []
    changed: list[tuple[str, str]] = []

    new_cols = []
    for c in df.columns:
        raw = str(c)
        cleaned = raw
        for ch in _INVISIBLE:
            cleaned = cleaned.replace(ch, "")
        cleaned = cleaned.strip()
        if cleaned != raw:
            changed.append((raw, cleaned))
        new_cols.append(cleaned)

    if changed:
        df = df.copy()
        df.columns = new_cols
        show = "、".join(f"{a!r}→{b!r}" for a, b in changed[:3])
        more = f" 等 {len(changed)} 个" if len(changed) > 3 else ""
        notes.append(f"列名已自动规范化（去首尾空白/不可见字符）：{show}{more}")
        # 规范化后可能出现重名列，提醒一下（不自动改，因为可能是真重复）
        dup = {c for c in new_cols if new_cols.count(c) > 1}
        if dup:
            notes.append(f"⚠ 规范化后出现重名列 {sorted(dup)}，引用前请先改名")

    return df, notes


def _quality_hints(df: pd.DataFrame) -> list[str]:
    """在画像里主动报告「看起来不像正常观测行 / 需要清洗」的迹象。

    为什么要在工具返回里报，而不是靠模型自己发现：
        v4 里 3 道错题的根因都不是"不会写代码"，而是"没想到要检查什么"——
        数据里混着收入分组、地区、OWID 区域码等汇总行，模型直接算了。
        修法不是加工具、也不是改 prompt，而是让 load_data 把这类事实说出来。
        成本几十 token，可能省掉一步试错（一步 ≈ 3,900 token）。

    只在真的发现迹象时才输出，干净数据不会多一个字。
    """
    hints: list[str] = []

    # 注意：**「列名有前后空格」已经不在这里提示了**，因为它在数据入口被自动修掉了
    # （见 _normalize_df）。能自动修的就不该让模型操心——
    # 提示一条它还得去改的规则，等于让它多写一行代码。

    # B. 像「单位/地区代码」的列：大部分值是三位大写码，或 OWID_ 前缀的区域码
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        vals = s.dropna().astype(str)
        if vals.empty:
            continue
        iso3_ratio = vals.str.fullmatch(r"[A-Z]{3}").mean()
        owid_ratio = vals.str.match(r"^OWID_").mean()
        if iso3_ratio > 0.5 or owid_ratio > 0.2:
            miss = int(s.isna().sum())
            nonstd = int((~vals.str.fullmatch(r"[A-Z]{3}")).sum())
            hints.append(
                f"{col}：像单位代码列，其中 {miss} 行为空、{nonstd} 行不是三位大写码 "
                f"→ 这些通常是汇总/区域行，按「国家/单位」统计排名前应先排除"
            )

    # C. 分类列里出现了汇总关键词
    keywords = ("合计", "总计", "小计", "汇总", "total", "subtotal", "average", "mean")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        vals = df[col].dropna().astype(str)
        if vals.empty:
            continue
        hits = [v for v in pd.unique(vals) if any(k in v.lower() for k in keywords)]
        if hits:
            hints.append(f"{col}：出现疑似汇总行取值 {hits[:3]} → 统计前应排除这些行")

    # D. 文本列但大部分值能转成数字 → **自动报出「脏的部分」的完整分布**
    #
    # 为什么不用「if 'cm' in values / if ',' in values」这种硬编码规则：
    #   硬编码 = 我事先猜了几种写法。数据里若有第四种（'5.1$'、'约5.1'），规则全漏，
    #   模型也就不知道。而「把数字和小数点删掉、看剩下什么」是通用办法：
    #   不管剩下的是什么字符，都会出现在统计里，不需要预先想到。
    #
    # 为什么要配样例值：
    #   "4,9" 和 "4,900" 删掉数字后剩下的都是 ','，统计上分不出来。
    #   给两三个原始样例，模型才知道是小数逗号还是千分位。
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        # 排除「ID / 编号」类列：它们本来就该是文本，清洗没有意义还会误导模型
        if re.search(r"(^|_)(id|code|no|num)$", str(col).lower()):
            continue
        vals = df[col].dropna().astype(str).str.strip()
        if vals.empty:
            continue
        if df[col].nunique(dropna=True) == int(df[col].notna().sum()) and len(vals) > 5:
            continue                       # 值全唯一 + 文本 → 基本是 ID，不当数值清洗

        # 判断「是否看起来是数字列」：去掉逗号和非数字字符后，能不能转成数字
        probe = (vals.str.replace(r"[,，]", "", regex=True)
                     .str.replace(r"[^\d.\-]", "", regex=True))
        ratio = pd.to_numeric(probe, errors="coerce").notna().mean()
        if ratio <= 0.5:
            continue                       # 不是数字列（是正常类别列），不提示

        # 真正的通用做法：删掉数字/小数点/负号，剩下的就是「脏的部分」
        sig = vals.str.replace(r"[0-9.\-]", "", regex=True)
        counts = sig.value_counts()

        parts = []
        for s, n in counts.items():
            parts.append(f"{'纯数字' if s == '' else repr(s)}×{n}")
        detail = "、".join(parts[:6])
        if len(counts) > 6:
            detail += f" 等 {len(counts)} 种"

        bad = vals[sig != ""]
        samples = bad.unique()[:3].tolist() if len(bad) else []

        msg = f"{col}：是文本但 {ratio:.0%} 的值能转成数字；删掉数字后剩下：{detail}"
        if samples:
            msg += f"；样例 {samples}"
        hints.append(msg)

    # 注意：这里【故意不报告"有 N 行完全重复"】。
    #   重复行要不要处理**取决于问题**：问「多少条记录」要保留，问「多少条唯一观测」要去重。
    #   把它放进"质量提示"区块，会被模型解读成"这是待修的问题"，从而做它没被要求的去重
    #   （会直接把 heavy_rows 那类题从 150 打成 147）。
    #   原则：提示区块只放「不处理必错」的迹象；取决于口径的事实让模型自己用
    #   df.duplicated().sum() 去查。

    return hints


def load_data(path: str,name:str | None = None) -> str:
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT):
        return f"错误：不允许读取项目外的文件"
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.exists():
        return f"错误：文件不存在 {p}"

    suffix = p.suffix.lower()
    try:
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(p)
        elif suffix in {".csv", ".txt"}:
            try:
                df = pd.read_csv(p)
            except UnicodeDecodeError:
                df = pd.read_csv(p, encoding="gbk")
        else:
            return f"错误：暂不支持 {suffix}"
    except Exception:
        return "读取失败:\n" + traceback.format_exc(limit=3)

    # 在入口做「无论问什么都该做」的规范化，并把做过的事告诉模型。
    # 关键：先规范化再入库，后续所有 run_python 拿到的就是干净的列名。
    df, norm_notes = _normalize_df(df)

    store = {"datasets": {}, "last": None}

    p_store = datasets_file()
    if p_store.exists():
        store = pd.read_pickle(p_store)

    key = name or Path(path).stem
    overwritten = key in store["datasets"]  #记下是不是覆盖了同名
    store["datasets"][key] = df  # 加入新表
    store["last"] = key
    pd.to_pickle(store, p_store)

    hint=f"(覆盖同名旧数据集'{key}')" if overwritten else ""

    # 【已自动处理】：我做完了，你不用管
    fixed = ""
    if norm_notes:
        fixed = "\n\n【已自动处理，无需你干预】\n" + "\n".join(f"- {n}" for n in norm_notes)

    # 【数据质量提示】：需要你注意/自行判断
    hints = _quality_hints(df) if QUALITY_HINTS_ENABLED else []
    quality = ""
    if hints:
        quality = "\n\n【数据质量提示】\n" + "\n".join(f"- {h}" for h in hints)

    return (f"已加载为数据集 '{key}'{hint}\n可用数据集：{list(store['datasets'])}"
            f"{fixed}{quality}\n\n" + _profile_text(df))

# tools/tabular.py 里加
LIST_FILES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": (
            "列出某个目录下可用的数据文件（.csv/.xlsx/.xls/.txt）。"
            "当用户说『data 文件夹里的数据』这类模糊描述、或你不确定确切文件名时，"
            "先调用本工具，再用返回的确切路径调用 load_data。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dir_path": {"type": "string", "description": "目录路径，默认 data"}
            },
        },
    },
}


def list_files(dir_path: str = "data") -> str:
    base = (ROOT / dir_path).resolve()
    max_depth=5
    limit=20
    if not base.is_relative_to(ROOT):
        return f"错误：不允许访问项目外的项目{base}"
    if not base.is_dir():
        return f"错误：目录不存在 {base}"

    exts = {".csv", ".txt", ".xlsx", ".xls"}
    hits = []
    for x in base.rglob("*"):
        if x.is_file() and x.suffix.lower() in exts:
            rel = x.relative_to(base)
            if len(rel.parts) <= max_depth:
                hits.append(str(rel).replace("\\", "/"))

    hits.sort()
    if not hits:
        return f"{dir_path} 下（含 {max_depth} 层子目录）没有支持的数据文件"

    shown,extra = hits[:limit], len(hits)-limit
    out = [f"{dir_path} 下共 {len(hits)} 个数据文件："]
    out += [f"- {dir_path}/{h}" for h in shown]  # ← 给出可直接 load_data 的完整相对路径
    if extra > 0:
        out.append(f"……还有 {extra} 个未列出，请指定更具体的目录。")
    return "\n".join(out)