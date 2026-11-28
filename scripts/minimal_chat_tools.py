# -*- coding: utf-8 -*-
"""最小「多轮 + 工具」agent —— 教学用，不属于项目，不进 data_agent/。

跑法（项目根目录）：
    python scripts/minimal_chat_tools.py

它和你的 data_agent/agent.py 是同一个范式，只是把「数据分析」换成了一个假的
calculate 工具，代码压到 60 行左右。目的是让你看清三件事：

1. **工具循环跑在「一轮用户输入」内部。**
   一轮用户输入 → 可能触发 N 次工具调用 → 最后产出一条最终回答。
   这 N 次是「任务内多轮」，你项目里已经有了。

2. **步数预算必须每轮重置。**
   如果把 for step 放在最外层 while 外面，第二句话就没有额度了。
   这是多轮改造里最容易踩的坑。

3. **历史里累积的是 assistant(tool_calls) + tool 消息，不是最终答案。**
   这些 tool 消息往往是几百上千 token 的原始输出，它们才是上下文增长的主力。
   你项目里的 run_python 返回的就是 stdout + traceback，比这里大得多。

三句可以试的话：
    算一下 123 * 456 是多少
    再加 100 呢
    那 7 的 8 次方呢
"""
import ast
import json
import operator
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(
    api_key=os.getenv("AGENT_API_KEY"),
    base_url=os.getenv("AGENT_BASE_URL", "https://api.deepseek.com"),
)
MODEL = os.getenv("AGENT_MODEL", "deepseek-chat")
MAX_STEPS_PER_TURN = 5

# ---------------------------------------------------------------------------
# 一个玩具工具：安全四则运算（不是重点，只要有工具就行）
# ---------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,  # 注意：7**8 是 BinOp(op=Pow)，所以必须放进这张表，
}  #                         而不是单独判断 isinstance(node, ast.Pow)


def safe_eval(expr: str) -> float:
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError(f"不支持的语法: {ast.dump(node)}")

    return _eval(ast.parse(expr, mode="eval").body)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "计算一个算术表达式并返回数值。做任何算数都必须调用本工具，"
                "不要自己心算。支持 + - * / ** 和括号。"
                "传入的 expression 必须是纯算术式，例如 '123*456' 或 '(7**8)+100'。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "纯算术表达式，例如 '123*456'"}
                },
                "required": ["expression"],
            },
        },
    }
]

# ===== 核心：messages 在 while 外面，跨轮存活 =====
messages = [
    {
        "role": "system",
        "content": "你是简洁的助手。需要算数时必须调用 calculate 工具，不要自己心算。",
    }
]


def one_turn(user_text: str, max_steps: int = MAX_STEPS_PER_TURN) -> str:
    """处理一轮用户输入：可能触发多次工具调用，直到模型给出最终回答。"""
    messages.append({"role": "user", "content": user_text})

    # 注意：步数预算是「每轮」的，所以这个 for 在 one_turn 内部
    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message

        # 情况 A：模型认为可以收尾
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return msg.content or ""

        # 情况 B：模型要调工具。assistant 消息必须原样放回历史，否则下一轮协议不合法
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [c.model_dump() for c in msg.tool_calls],
            }
        )

        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
                obs = str(safe_eval(args["expression"]))
                print(f"    [{step}] {call.function.name}({args['expression']}) -> {obs}")
            except Exception as exc:  # 工具出错也要变成观测值回喂，而不是抛异常终止
                obs = f"工具执行错误: {exc}"
                print(f"    [{step}] 出错 -> {obs}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": obs})

    return "（本轮超出步数上限，已停止）"


while True:
    q = input("\n你: ").strip()
    if q in {"q", "quit", "exit"}:
        break
    if not q:
        continue

    answer = one_turn(q)
    print(f"\nAI: {answer}")
    # 这两个数字一起看：消息条数涨得慢，但 total_tokens 涨得快 —— 因为 tool 消息很长
    print(f"[历史消息数 {len(messages)}]")
