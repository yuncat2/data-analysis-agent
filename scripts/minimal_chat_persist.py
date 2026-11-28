# -*- coding: utf-8 -*-
"""最小「多轮 + 会话持久化」示例 —— 教学用，不属于项目。

跑法（项目根目录）：
    python scripts/minimal_chat_persist.py            # 默认会话 demo
    python scripts/minimal_chat_persist.py work       # 另一个会话 work

验收标准（就这一条）：
    第一次运行说「我叫张三，我在做数据分析 agent」→ 输入 /exit 退出
    第二次运行问「我叫什么？我在做什么？」→ 它答得出来

它要让你看清一件事，也是整个持久化的核心洞察：

    **磁盘上的完整历史 ≠ 每次发给模型的上下文。**

你可以永久保存全部对话（追加写入，永不删），
但每次调模型时只发「system + 最近 N 轮」。
把这两件事混在一起，是你后面一定会踩的坑：
要么上下文爆掉，要么为了省 token 把历史删了导致什么都记不住。

跑起来时盯着这行输出：
    [磁盘 X 条 / 本次发给模型 Y 条]
X 会一直涨，Y 会被 MAX_HISTORY_SENT 卡住 —— 这就是那个分离。

可用命令：/history 看磁盘条数、/raw 看文件原文、/clear 清空本会话、/exit 退出
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

client = OpenAI(
    api_key=os.getenv("AGENT_API_KEY"),
    base_url=os.getenv("AGENT_BASE_URL", "https://api.deepseek.com"),
)
MODEL = os.getenv("AGENT_MODEL", "deepseek-chat")

SYSTEM = {"role": "system", "content": "你是简洁的助手，回答控制在两句话内。"}
MAX_HISTORY_SENT = 6  # 只把最近 6 条（约 3 轮问答）发给模型

# 会话文件放在 runs/sessions/，runs/ 已在 .gitignore 里
SESSIONS = ROOT / "runs" / "sessions"
SESSIONS.mkdir(parents=True, exist_ok=True)

session_id = sys.argv[1] if len(sys.argv) > 1 else "demo"
session_file = SESSIONS / f"{session_id}.jsonl"

# 不可见字符：BOM(U+FEFF)、零宽空格(U+200B/C/D)、全角空格(U+3000)
# 它们肉眼不可见，但会让字符串比较、分组、去重全部失效。
# 真实场景来源：从 Excel/Word 复制粘贴、Windows 管道、带 BOM 的文件读入。
_INVISIBLE = "\ufeff\u200b\u200c\u200d\u3000"


def clean_input(text: str) -> str:
    for ch in _INVISIBLE:
        text = text.replace(ch, "")
    return text.strip()


# ---------------------------------------------------------------------------
# 持久化的全部技术：写入用「追加」，读回用「逐行解析」
# ---------------------------------------------------------------------------
def load_history() -> list[dict]:
    """从 jsonl 读回全部消息。文件不存在就是新会话。"""
    if not session_file.exists():
        return []
    rows = []
    for line in session_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def save_message(msg: dict) -> None:
    """追加一条消息。

    为什么用追加（append）而不是「全量重写」？
    因为全量重写时如果进程在第 3 行崩溃，文件就只剩半截，整个会话毁了。
    追加写入的话，最坏情况是丢了最后一条，前面的都还在。
    这个原则叫 append-only，是日志类存储的通用做法。
    """
    with open(session_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")


def build_context(history: list[dict]) -> list[dict]:
    """从完整历史里挑出这一次要发给模型的部分。

    这里就是那个分离：磁盘上什么都有，但只发 system + 最近几条。
    后面你可以把「最近几条之外的部分」换成一段摘要，而不是直接丢掉。
    """
    return [SYSTEM] + history[-MAX_HISTORY_SENT:]


history = load_history()
print(f"会话 [{session_id}]，磁盘上已恢复 {len(history)} 条历史")
print("命令：/history  /raw  /clear  /exit")

while True:
    q = clean_input(input("\n你: "))
    if q in {"/exit", "/quit", "q"}:
        break
    if not q:
        continue
    if q == "/history":
        print(f"磁盘 {len(history)} 条 / 本次会发 {len(build_context(history))} 条")
        continue
    if q == "/raw":
        print(session_file.read_text(encoding="utf-8") if session_file.exists() else "（空）")
        continue
    if q == "/clear":
        session_file.unlink(missing_ok=True)
        history = []
        print("本会话已清空")
        continue

    user_msg = {"role": "user", "content": q}
    history.append(user_msg)
    save_message(user_msg)  # ← 先落盘，再调模型。这样即使 API 挂了，用户这句话也不会丢

    resp = client.chat.completions.create(model=MODEL, messages=build_context(history))
    answer = resp.choices[0].message.content
    assistant_msg = {"role": "assistant", "content": answer}
    history.append(assistant_msg)
    save_message(assistant_msg)

    print(f"\nAI: {answer}")
    print(f"[磁盘 {len(history)} 条 / 本次发给模型 {len(build_context(history))} 条]")
