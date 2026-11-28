"""代码执行沙箱：agent 唯一真正做事的地方。"""
import os
import subprocess
import sys
import textwrap

from .config import EXEC_TIMEOUT, MAX_OBS_CHARS, OBS_TAIL_RATIO
from .config import current_dir, datasets_file, output_dir

RUN_PYTHON_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "在一个独立的 Python 进程里执行代码，用于所有计算、统计、建模、画图。"
            "执行环境里已预置变量："
            "dfs（dict，键是数据集名，值是对应的 pandas.DataFrame）、pd、np、plt。"
            "访问数据必须写成 dfs['数据集名']，例如 dfs['销售'].groupby(...)。"
            "必须先调用过 load_data 才能用。只返回 stdout 和 stderr，需要用 print() 输出结果。"
            "报错时阅读 traceback 后修改代码重试。"
            "【保存文件】当前工作目录**就是**本会话的产物目录，直接写文件名即可，"
            "例如 plt.savefig('箱线图.png')、df.to_csv('结果.csv', index=False)。"
            "不要在前面再套一层 output/ 或 .figures/ 之类的目录，会多出一层嵌套。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的 Python 代码，不要用 input() 或 plt.show()"}
            },
            "required": ["code"],
        },
    },
}


def truncate_observation(text: str, limit: int = MAX_OBS_CHARS,
                         tail_ratio: float = OBS_TAIL_RATIO) -> str:
    """超长观测值按「头 + 尾」截断，并明确告诉模型省略了多少。

    为什么要保尾部：
        头部是数据画像 / 表头，尾部是 traceback 和汇总数字。
        只留头会把报错丢掉——而报错恰恰是自修复的输入。

    为什么必须"报告省略"：
        静默截断会让模型以为自己看全了，然后基于残缺信息下结论。
        这是"安静地算错"，比报错危险得多。
    """
    if len(text) <= limit:
        return text

    tail = max(1, int(limit * tail_ratio))
    head = max(1, limit - tail)
    omitted = len(text) - head - tail

    return (
        text[:head]
        + f"\n\n……（中间已省略 {omitted} 字符；以下是末尾 {tail} 字符）……\n\n"
        + text[-tail:]
    )


def run_python(code: str) -> str:
    if not datasets_file().exists():
        return "错误：还没有加载任何数据集，请先调用 load_data。"

    prelude = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']\n"
        "plt.rcParams['axes.unicode_minus'] = False\n"
        f"_store = pd.read_pickle(r'{datasets_file()}')\n"
        "dfs = _store['datasets']\n"
    )
    script = current_dir() / "step.py"          # 最近一次执行的代码，调试时可看
    script.write_text(prelude + textwrap.dedent(code), encoding="utf-8")

    # 记录执行前 output/ 里已有的文件，用来判断"这次新生成了什么"
    # （output/ 是累积的，不区分新旧会把上几轮留下的图也算进来）
    out = output_dir()
    before = {p.name for p in out.glob("*") if p.is_file()}

    # 子进程 stdout 是管道时 Windows 默认用 GBK，必须强制 UTF-8
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(out),                        # 工作目录 = 产物目录：模型写的相对路径文件自然落在这里
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=EXEC_TIMEOUT,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        return f"执行超时（>{EXEC_TIMEOUT}s），请简化计算。"

    parts = []
    if proc.stdout and proc.stdout.strip():
        parts.append("stdout:\n" + proc.stdout.strip())
    if proc.stderr and proc.stderr.strip():
        err_lines = proc.stderr.strip().splitlines()
        if len(err_lines) > 25:
            err_lines = [f"...（省略前 {len(err_lines) - 25} 行）..."] + err_lines[-25:]
        parts.append("stderr / traceback:\n" + "\n".join(err_lines))
    if proc.returncode != 0:
        parts.append(f"[exit code {proc.returncode}]")

    # 只报「本次新生成」的文件：output/ 是累积的，报全部会把旧图重复算进来
    new_files = sorted(p.name for p in out.glob("*")
                       if p.is_file() and p.name not in before)
    if new_files:
        parts.append("本次新生成的文件（在当前目录）: " + ", ".join(new_files))

    text = "\n\n".join(parts) or "（无输出。记得用 print() 打印需要看的结果）"
    return truncate_observation(text)