"""多轮交互版（WORK IN PROGRESS）—— 在这个文件上改，不要动 agent.py。

分工：
    agent.py       单轮入口。评测脚本（evals/run_eval.py）要靠它批量跑，没有人工输入。
    agent_chat.py  多轮入口。给人用，支持连续追问 + 会话持久化。
两个入口是长期分工，不是临时用的副本。

当前状态：从 agent.py 原样复制而来，还没做任何修改。
改动清单见文件末尾的 TODO。

"""
import atexit
import argparse
import json
import sys
import os

from openai import OpenAI


from .config import API_KEY, BASE_URL, MAX_STEPS, MODEL, TOKEN_BUDGET
from .config import COMPRESS_ENABLED, COMPRESS_KEEP_RECENT_TOOLS, COMPRESS_TO_CHARS
from .config import set_session,current_dir,history_file
from .tools import SCHEMAS, dispatch

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """你是一个严谨的数据分析助手，服务于统计专业用户。

工作规则：
1. 先调用 load_data 看清列名、类型、样本，再动手计算；不要猜数据长什么样。
2. 所有数字都必须来自 run_python 的真实执行结果，禁止凭想象编造。
3. 代码报错时，读 traceback 定位原因，修改后重试，最多 3 次。
4. 涉及排名、占比必须实际算出数字，并说明依据哪一列。
5. 无法从数据得出结论时，明确说"无法确定"并说明缺什么。
6. 解释原因时，必须区分"数据支持的"和"你的推测"，推测要标注。
7. 最终回答用中文，先给结论，再给关键数字。
8. 需要对比多个数据源时，分别调用 load_data 并起有意义的名字，然后用 dfs['名字'] 引用。
9. 已加载的数据集清单见最近一次 load_data 的返回，不要重复加载同一个文件。
"""


def _short(text: str, limit: int = 160) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + " ..."

def load_session() ->list:
    """
    读档，没有则返回空列表
    """
    path = history_file()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))

def save_session(messages:list) -> None:
    """
    存档
    """
    history_file().write_text(json.dumps(messages,ensure_ascii=False,indent=2),encoding="utf-8")

def acquire_lock() -> None:
    lock = current_dir() / "lock"
    if lock.exists():
        info = lock.read_text(encoding="utf-8").strip()
        raise SystemExit(f"会话正被另一个进程使用（{info}）。"
                         f"确认没有的话，删掉{lock}再试。")
    lock.write_text(f"pid={os.getpid()}",encoding="utf-8")
    atexit.register(lambda: lock.unlink(missing_ok=True))

def _finalize(messages: list) -> str:
    """步数/预算用完时的收尾：不带 tools 再问一次，逼它输出"已有信息下的部分结论"。

    关键在「不传 tools」——模型没有工具可调，就只能用文本回答，
    比让它再试一次可靠得多。
    """
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages + [{
                "role": "user",
                "content": "步数或预算已用完。请只基于以上已经得到的结果作答；"
                           "如果信息不足，请明确说明：已经确认了什么、还缺什么、建议下一步怎么做。",
            }],
            temperature=0,
        )
        return resp.choices[0].message.content or "（收尾请求没有返回内容）"
    except Exception as exc:
        return f"步数或预算已用完，且收尾请求失败：{exc!r}"


def compress_history(messages: list,
                     keep_recent: int = COMPRESS_KEEP_RECENT_TOOLS,
                     to_chars: int = COMPRESS_TO_CHARS) -> int:
    """把历史里较早的 tool 结果压短。返回被压缩的条数。

    【绝对不要删消息】
        assistant(tool_calls) 和 tool 消息是成对的协议结构。
        删掉 tool 响应而留下 tool_calls，下一次调 API 会直接 400。
        所以这里只改 content，消息本身一条不动。

    【为什么保留最近 N 条原文】
        最近的工具结果是当前推理的直接依据；较早的只是背景。
        压错了会丢信息，压对了只省 token。

    幂等：已经压过的（长度已 ≤ to_chars）不会被再次处理。

    【默认关闭】见 config.COMPRESS_ENABLED 的注释：实测在「工具输出只有几百字符」
    的场景下，压缩省下的远小于「因丢信息而多走一步」的代价，且会破坏 prompt 前缀缓存。
    """
    if not COMPRESS_ENABLED:
        return 0

    idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    older = idx[:-keep_recent] if keep_recent > 0 else idx

    marker = "（早期工具结果已压缩"
    n = 0
    for i in older:
        content = messages[i].get("content") or ""
        # 已经压过的跳过：压缩后长度仍会大于 to_chars（因为加了说明文字），
        # 只靠长度判断会导致每次调用都再砍一刀。
        if marker in content:
            continue
        if len(content) > to_chars:
            messages[i]["content"] = (
                content[:to_chars]
                + f"\n……{marker}，原文 {len(content)} 字符）"
            )
            n += 1
    return n


def run(messages:list, max_steps: int = MAX_STEPS, stats: dict | None = None):
    """一轮任务。stats 传入一个 dict 时，会把 steps / tokens 写进去（给评测用）。"""
    total_tokens = 0
    last_sig = None          # 上一次工具调用的签名（工具名 + 参数）
    repeat = 0               # 这个签名连续出现了几次

    # 每轮开始前压一次历史（只压上一轮及更早的，本轮内不压，保护自修复用的报错）
    compressed = compress_history(messages)
    if stats is not None and compressed:
        stats["compressed"] = stats.get("compressed", 0) + compressed

    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            tools=SCHEMAS,
            messages=messages,
            temperature=0,
        )
        total_tokens += resp.usage.total_tokens
        if stats is not None:
            stats["steps"] = step
            stats["tokens"] = total_tokens
            # 缓存命中计量：命中部分价格比未命中低一个数量级，是成本关键指标。
            # 括号必须加：`a + b or 0` 等于 `(a + b) or 0`，
            # 而我们要的是 `a + (b or 0)`——这样才能挡住字段为 None 的情况。
            stats["cache_hit"] = stats.get("cache_hit", 0) + (
                getattr(resp.usage, "prompt_cache_hit_tokens", 0) or 0
            )
            stats["cache_miss"] = stats.get("cache_miss", 0) + (
                getattr(resp.usage, "prompt_cache_miss_tokens", 0) or 0
            )
        if total_tokens > TOKEN_BUDGET:
            return _finalize(messages)

        msg = resp.choices[0].message
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content":msg.content or ""})
            return msg.content

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [c.model_dump() for c in msg.tool_calls],
        })

        for call in msg.tool_calls:
            # 循环检测：工具名 + 参数完全相同，视为一次重复调用
            sig = (call.function.name, (call.function.arguments or "").strip())
            if sig == last_sig:
                repeat += 1
            else:
                last_sig, repeat = sig, 1

            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None
                obs = "参数不是合法 JSON，请重新生成。"

            if repeat >= 3:
                # 不执行，直接把"你在原地打转"这个事实喂回去
                obs = (f"你已经用完全相同的参数连续调用 {call.function.name} {repeat} 次，"
                       "结果不会改变。请换一个思路，或直接基于已有信息作答。")
                print(f"[{step}] {call.function.name} 重复调用被拦截（第 {repeat} 次）")
            elif args is not None:
                print(f"[{step}] {call.function.name}({_short(args, 120)})")
                obs = dispatch(call.function.name, args)

            print(f"    -> {_short(obs)}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": obs})

    return _finalize(messages)


def chat(max_steps:int) -> None:
    messages= load_session()
    if not messages:
        messages=[{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        q=input("\n你：").strip()
        if q in {"q","exit","quit"}:
            break
        if not q:
            continue

        messages.append({"role": "user", "content": q})
        answer=run(messages,max_steps)
        print(f"\nAI:{answer}")
        save_session(messages)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session",default="demo")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()
    if not API_KEY:
        sys.exit("缺少 AGENT_API_KEY，请检查 starter/.env")

    set_session(args.session)
    acquire_lock()
    print(f"模型: {MODEL}  |  会话: {args.session}\n")
    chat(args.max_steps)


if __name__ == "__main__":
    main()


# ===========================================================================
# TODO：把「多轮 + 持久化」合并进来。先回答下面 4 个问题，再动代码。
#
# 改动点（技术部分参照 scripts/minimal_chat_tools.py 和
# scripts/minimal_chat_persist.py，那两个已经跑通了）：
#
#   1. messages 从 run() 内部移到 chat() 里，跨轮存活
#   2. run() 不要销毁 messages（list 是可变对象，传进去 append 就行，不用 return）
#   3. 新增 chat()：while True -> input -> 每轮重置步数预算
#   4. 每轮结束追加写入 runs/sessions/<id>.jsonl（照抄 persist 示例的 append 做法）
#   5. 启动时 load_history() 恢复历史
#   6. tool 消息怎么持久化 + 怎么读回来（难点，见下面第 4 问）
#
# 4 个设计问题（先写答案，哪怕写错。这四问的答案就是面试答案）：
#
#   Q1 session_id 从哪来？命令行参数传、还是每次启动自动用时间戳新建一个？
#   Q2 current.pkl 跟会话绑定吗？用户第二轮说「换个文件」时 pkl 什么时候清？
#      会话恢复时如果 pkl 还在，但用户以为已经换了数据，会发生什么？
#   Q3 磁盘历史要不要跟着发送窗口一起裁？还是永久保留、只裁「发给模型的那部分」？
#   Q4 恢复历史时会撞上一个硬骨头：assistant 消息里有 tool_calls（嵌套结构），
#      tool 消息里有 tool_call_id。如果进程崩在「模型已经发起工具调用、
#      但 tool 结果还没写入」的位置，读回来的消息序列是不合法的——
#      下次调 API 会直接 400。你打算怎么检测和修复这种半截状态？
#
# 验收标准（三条全过才算做完）：
#   [ ] 连续追问，第二句用「它」指代，第三句答案正确
#   [ ] 中途换数据文件，「刚才那个数据有多少行」不会拿错数据答
#   [ ] 退出重开，还能接着上次聊
# ===========================================================================
