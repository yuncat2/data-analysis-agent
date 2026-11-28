# -*- coding: utf-8 -*-
"""最小多轮对话 —— 教学用，不属于项目，不进 data_agent/。

跑法（项目根目录）：
    python scripts/minimal_chat.py

它只做一件事：证明「多轮对话」的全部技术就是
    「messages 一直带着，每轮把新的问答追加进去」。
除了这一点，没有任何别的机制。

跑的时候观察三件事：
1. 先说「我叫张三」，再问「我叫什么」——它答得出来，说明多轮成立。
2. 每次打印的「历史消息数」一直在涨——这就是你以后所有上下文成本的来源。
    第 1 轮 messages 长度是 3，第 5 轮会到 11。真实项目里 tool 消息会让它涨得更快。
3. 输入 q 退出后重开——它不记得你了。因为 messages 活在内存里，进程一结束就没了。
    这就是「会话持久化」为什么是个独立问题。
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 复用项目根目录的 .env，不用另外配
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(
    api_key=os.getenv("AGENT_API_KEY"),
    base_url=os.getenv("AGENT_BASE_URL", "https://api.deepseek.com"),
)
MODEL = os.getenv("AGENT_MODEL", "deepseek-chat")

# ===== 核心：messages 定义在循环外，所以它的生命周期跨越多轮 =====
messages = [{"role": "system", "content": "你是简洁的助手，回答控制在两句话内。"}]

while True:
    q = input("\n你: ").strip()
    if q in {"q", "quit", "exit"}:
        break
    if not q:
        continue

    messages.append({"role": "user", "content": q})                    # 追加上一轮的输入
    resp = client.chat.completions.create(model=MODEL, messages=messages)
    answer = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})          # 追加上一轮的输出

    print(f"\nAI: {answer}")
    print(f"[历史消息数 {len(messages)} | 本次调用计费 tokens {resp.usage.total_tokens}]")
