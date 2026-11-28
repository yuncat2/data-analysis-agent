# -*- coding: utf-8 -*-
"""最小可运行的 Function-Calling Agent 骨架（第 1 周用）。

它只做一件事：让你亲眼看到「LLM 自己决定调哪个工具 → 拿到结果 → 再决定下一步」。
这不是最终架构，第 2 周会把它拆成 ingest/ tools/ sandbox.py agent.py 多文件模块。

准备：
    pip install openai pandas openpyxl python-dotenv
    把 .env.example 复制为 .env，填入 AGENT_API_KEY

运行：
    python agent_loop.py --file sample.csv --ask "各区域销售额排名，给出前3名与占比"

看什么：
    打印出的 [1] load_data(...) / [2] run_python(...) 就是 agent 的"思考轨迹"。
    读一遍 README 里「tool_calls → role:tool」的协议，80% 的新手卡点在这里。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

MODEL = os.getenv("AGENT_MODEL", "deepseek-chat")
BASE_URL = os.getenv("AGENT_BASE_URL", "https://api.deepseek.com")
API_KEY = os.getenv("AGENT_API_KEY", "")

# 所有中间产物都放这里：current.pkl（当前数据集）、step.py（生成的代码）、生成的图
WORKDIR = HERE / "_work"
WORKDIR.mkdir(exist_ok=True)
PICKLE = WORKDIR / "current.pkl"

MAX_STEPS = 10       # 超过就停，防止死循环烧钱
EXEC_TIMEOUT = 60     # 单次代码执行超时（秒）
MAX_OBS_CHARS = 4000  # 回喂给模型的观测值截断长度
MAX_TOKEN=60000

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ---------------------------------------------------------------------------
# 1. 工具定义：这段 JSON 就是"给 LLM 看的说明书"，描述写得含糊，它就永远调不对
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_data",
            "description": (
                "加载用户的数据文件（支持 .csv / .xlsx / .xls），返回数据的列名、类型、"
                "缺失情况、前几行样本和数值列概要。只返回画像，不返回全量数据。"
                "在开始任何计算之前必须先调用它。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "文件路径，可以是相对路径"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "在一个独立的 Python 进程里执行代码，用于所有计算、统计、建模、画图。"
                "执行环境里已经预置好两种变量：df（用户完整的 pandas.DataFrame）和 pd / np。"
                "必须先调用过 load_data 才能用。"
                "只返回 stdout 和 stderr，所以你需要用 print() 把结果打印出来。"
                "如果报错，阅读 traceback 后修改代码重试。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的 Python 代码，不要用 input() 或 plt.show()"}
                },
                "required": ["code"],
            },
        },
    },
]

SYSTEM_PROMPT = """你是一个严谨的数据分析助手，服务于统计专业用户。

工作规则：
1. 先调用 load_data 看清列名、类型、样本，再动手计算；不要猜数据长什么样。
2. 所有数字都必须来自 run_python 的真实执行结果，禁止凭想象编造。
3. 代码报错时，读 traceback 定位原因，修改后重试，最多 3 次；连续失败就如实说明。
4. 涉及排名、占比、同比等问题，必须实际算出数字再回答，并说明是依据哪一列算的。
5. 无法从数据中得出结论时，明确说"无法确定"并说明缺什么，不要编。
6. 解释原因时，必须区分"数据支持的"和"你的推测"，推测要明确标注为推测。
7. 最终回答用中文，先给结论，再给关键数字（保留合理小数位）。"""


# ---------------------------------------------------------------------------
# 2. 工具实现：注意每个函数都返回"字符串"，因为要回喂给模型
# ---------------------------------------------------------------------------
def _profile_text(df: pd.DataFrame) -> str:
    """把 DataFrame 压缩成模型能读的画像摘要。这是上下文管理的核心：数据不进 prompt。"""
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


def load_data(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = (HERE / p).resolve()
    if not p.exists():
        return f"错误：文件不存在 {p}。请向用户确认路径。"
    suffix = p.suffix.lower()
    try:
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(p)
        elif suffix in {".csv", ".txt"}:
            # 真实数据常有中文/混合编码，先 utf-8 再退回 gbk
            try:
                df = pd.read_csv(p)
            except UnicodeDecodeError:
                df = pd.read_csv(p, encoding="gbk")
        else:
            return f"错误：暂不支持 {suffix}。第 3 周会加图片/PDF 适配器（见 starter 目录说明）。"
    except Exception:
        return "读取失败:\n" + traceback.format_exc(limit=3)

    df.to_pickle(PICKLE)  # 存成 pickle，保留 dtype
    return _profile_text(df)


def run_python(code: str) -> str:
    if not PICKLE.exists():
        return "错误：还没有加载数据集，请先调用 load_data。"
    prelude = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 中文不乱码\n"
        "plt.rcParams['axes.unicode_minus'] = False\n"
        f"df = pd.read_pickle(r'{PICKLE}')\n"
    )
    script = WORKDIR / "step.py"
    script.write_text(prelude + textwrap.dedent(code), encoding="utf-8")

    # 关键：子进程的 stdout 被重定向成管道时，Windows 默认用 GBK 编码，
    # 我们按 UTF-8 解码就会得到乱码。强制子进程用 UTF-8 输出。
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(WORKDIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=EXEC_TIMEOUT,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        return f"执行超时（>{EXEC_TIMEOUT}s），计算可能过重或进入了死循环，请简化后重试。"

    parts = []
    if proc.stdout and proc.stdout.strip():
        parts.append("stdout:\n" + proc.stdout.strip())
    if proc.stderr and proc.stderr.strip():
        # traceback 只留最后 25 行：最有用的 KeyError/行号都在尾部，前面的框架栈是噪音
        err_lines = proc.stderr.strip().splitlines()
        if len(err_lines) > 25:
            err_lines = [f"...（省略前 {len(err_lines) - 25} 行）..."] + err_lines[-25:]
        parts.append("stderr / traceback:\n" + "\n".join(err_lines))
    if proc.returncode != 0:
        parts.append(f"[exit code {proc.returncode}]")

    # 把生成的图告诉模型：它自己看不到图，但必须知道图画到哪里去了
    pngs = sorted(p.name for p in WORKDIR.glob("*.png"))
    if pngs:
        parts.append("工作目录中的图片文件: " + ", ".join(pngs))

    text = "\n\n".join(parts) or "（无输出。记得用 print() 打印需要看的结果）"
    return text[:MAX_OBS_CHARS]


def dispatch(name: str, args: dict) -> str:
    """工具路由。加上 try 是因为模型可能给出缺字段的 JSON。"""
    try:
        if name == "load_data":
            return load_data(args["path"])
        if name == "run_python":
            return run_python(args["code"])
        return f"未知工具: {name}"
    except KeyError as exc:
        return f"参数缺失: {exc}。请检查工具参数格式。"
    except Exception:
        return "工具执行异常:\n" + traceback.format_exc(limit=3)


# ---------------------------------------------------------------------------
# 3. 主循环：agent 的全部秘密就在这里
# ---------------------------------------------------------------------------
def _short(text: str, limit: int = 160) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + " ..."


def run(data_file: str, question: str, max_steps: int = MAX_STEPS) -> None:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"数据文件: {data_file}\n\n任务: {question}"},
    ]
    total_token=0
    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0,
        )
        total_token+=resp.usage.total_tokens
        if total_token >= MAX_TOKEN:
            print(f"已消耗{total_token}，超出预算")
            return

        msg = resp.choices[0].message

        # 情况 A：模型认为可以收尾了
        if not msg.tool_calls:
            print(f"\n===== 最终回答（第 {step} 步结束）=====\n{msg.content}")
            return

        # 情况 B：模型要调工具。必须把这条 assistant 消息原样放回历史，
        # 否则下一轮协议不合法（这是新手最常见的 400 报错）。
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [call.model_dump() for call in msg.tool_calls],
            }
        )

        for call in msg.tool_calls:
            raw_args = call.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                obs = f"你的参数不是合法 JSON，请重新生成：{raw_args[:200]}"
                args = None
            if args is not None:
                print(f"[{step}] {call.function.name}({_short(args, 120)})")
                obs = dispatch(call.function.name, args)
            print(f"    -> {_short(obs)}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": obs}
            )

    print(f"\n达到最大步数 {max_steps} 仍未收敛，已停止（这是成本保护，不是 bug）。")


def main() -> None:
    parser = argparse.ArgumentParser(description="最小 function-calling 数据分析 agent")
    parser.add_argument("--file", required=True, help="数据文件路径")
    parser.add_argument("--ask", required=True, help="分析需求")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    if not API_KEY:
        sys.exit("缺少 AGENT_API_KEY。请把 .env.example 复制成 .env 并填入 key。")

    print(f"模型: {MODEL}  |  文件: {args.file}\n")
    run(args.file, args.ask, args.max_steps)


if __name__ == "__main__":
    main()
