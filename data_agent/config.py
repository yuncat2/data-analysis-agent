import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent      # = starter/
load_dotenv(ROOT / ".env")


DATA_DIR = ROOT / "runs" / "sessions"
_current_dir : Path | None = None

MODEL = os.getenv("AGENT_MODEL", "deepseek-chat")
BASE_URL = os.getenv("AGENT_BASE_URL", "https://api.deepseek.com")
API_KEY = os.getenv("AGENT_API_KEY", "")

MAX_STEPS = 10
EXEC_TIMEOUT = 60
MAX_OBS_CHARS = 4000
TOKEN_BUDGET = 60000

# ---- 上下文管理 ----
# 这几个参数支持环境变量覆盖，目的是能直接做 A/B 实验而不用改代码
# （改代码会改变 git commit，让两次评测的可比性变差）。
#
# 【历史压缩默认关闭：注意这个结论的适用范围】
#   实测对象：本文件的实现方式 = **每轮无条件、逐条**把旧 tool 输出截到 300 字符
#   实测条件：v5 / mt_long_01，5 轮对话，各跑两次
#     关闭压缩：35,508 / 36,312 token，缓存命中 93%
#     激进压缩：39,427 / 37,489 token，缓存命中 70–74%
#   原因：
#     1) 单条工具输出只有几百字符，压下来省不到 1k token；
#        但信息丢失会让模型多走一步，而「多一步」要重发整个历史 ≈ 3,900 token。
#     2) 压缩改变了历史中段内容，破坏 prompt 前缀缓存，命中率 93% → 72%。
#
#   【不要由此得出"压缩没用"】成熟实现（如 DSH 的 compaction）做的是另一件事：
#     - 由 token 计量**按压力触发**，压力不够时一个字都不动
#     - 把较早的多轮压成**一条摘要**（结构真的变小），而不是逐条截断
#     - 被压掉的原文仍留在 session log 里，可回放
#     - 超大工具输出走 spill（存盘 + 上下文只留定位符）
#   本实现没有做到这三点，所以负收益是**实现问题**，不是方法问题。
COMPRESS_ENABLED = os.getenv("AGENT_COMPRESS", "0") == "1"

# ---- 数据质量提示 ----
# load_data 返回画像时，主动报告"像汇总行 / 带单位 / 列名有空格 / 有重复"等迹象。
# 目的：v4 的 3 道错题根因都是"没想到要检查什么"，而不是"不会写代码"。
# 关掉用于 A/B：$env:AGENT_QUALITY_HINTS = "0"
QUALITY_HINTS_ENABLED = os.getenv("AGENT_QUALITY_HINTS", "1") == "1"
OBS_TAIL_RATIO = float(os.getenv("AGENT_OBS_TAIL_RATIO", "0.3333"))   # 尾部保留比例
COMPRESS_TO_CHARS = int(os.getenv("AGENT_COMPRESS_TO_CHARS", "300"))  # 压到多少字符
COMPRESS_KEEP_RECENT_TOOLS = int(os.getenv("AGENT_COMPRESS_KEEP", "4"))  # 保留最近 N 条原文

def _check_sid(session_id: str) -> str:
    """校验会话名，防止 ../ 越界。"""
    if not re.fullmatch(r"[\w\-\u4e00-\u9fa5]{1,64}", session_id):
        raise ValueError(f"会话名不合法: {session_id!r}")
    return session_id


def set_session(session_id: str) -> Path:
    global _current_dir
    _check_sid(session_id)
    d = DATA_DIR / session_id
    d.mkdir(exist_ok=True, parents=True)
    (d / "output").mkdir(exist_ok=True)      # 产物目录：建会话时就建，下游不用再判存在
    _current_dir = d
    return d


def reset_session(session_id: str) -> Path:
    """清空并重建会话目录（含 output/），返回新目录。

    给界面层的「覆盖」分支用。**不要自己遍历会话目录逐个 unlink()**：
    目录里除了文件还有 output/ 子目录，对目录调 unlink 在 Windows 上会抛
    PermissionError(WinError 5)。这里用 shutil.rmtree 一次性删整棵子树，
    界面层就不需要知道会话目录里到底有什么。
    """
    _check_sid(session_id)
    d = DATA_DIR / session_id
    if d.exists():
        shutil.rmtree(d)
    return set_session(session_id)

def current_dir() -> Path:
    if _current_dir is None:
        raise RuntimeError("还没设置会话，请先调用 config.set_session(session_id)")
    return _current_dir

def output_dir() -> Path:
    """当前会话的产物目录（图、导出文件）。run_python 的子进程 cwd 就指向这里。

    为什么建会话时就建，而不是用到再建：
        路径永远存在，所有写文件的地方就不用各自 mkdir 兜底，
        能消除一整类 FileNotFoundError。
    """
    d = current_dir() / "output"
    d.mkdir(exist_ok=True)
    return d

def datasets_file() -> Path:
    return current_dir() / "datasets.pkl"

def history_file() -> Path:
    return current_dir() / "history.json"