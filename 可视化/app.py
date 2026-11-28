"""数据分析 agent 的 Web 界面（第一档：会话 + 上传 + 提问 + 看图）。

启动（在项目根目录下运行）：
    python -m streamlit run 可视化/app.py

本文件只做 GUI 这一层，data_agent/ 里的函数一个都不改。
"""
import contextlib
import hashlib
import io
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# 让 `streamlit run` 在任意目录启动都能 import 到 data_agent。
# `streamlit run` 只会把「脚本所在目录」放进 sys.path，也就是 可视化/；
# 而 data_agent 在它的上一级。所以手动把项目根目录插到最前面。
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_agent.config import (DATA_DIR, output_dir,                      # noqa: E402
                               reset_session, set_session)
from data_agent.agent_chat import SYSTEM_PROMPT, load_session, save_session  # noqa: E402
from data_agent.agent_chat import run                                     # noqa: E402

UPLOAD_DIR = ROOT / "runs" / "uploads"      # runs/ 已被 .gitignore 忽略

st.set_page_config(page_title="数据分析 agent", layout="wide")

# ===========================================================================
# 0. 状态初始化
#    每次交互都会重跑整个文件，所以这几行每次都执行；
#    但只有「这个标签页第一次进来」时 ss 里还没有这些键，才会真正写入值。
# ===========================================================================
ss = st.session_state
if "session" not in ss:
    ss["session"] = None        # 已生效的会话名。调 set_session 必须用它
if "pending_name" not in ss:
    ss["pending_name"] = None   # 撞名时暂存，等用户回答「继续用 / 覆盖」
if "messages" not in ss:
    ss["messages"] = []         # 当前会话的对话历史
if "loaded_file_id" not in ss:
    ss["loaded_file_id"] = None # 已保存到磁盘的那次上传（防止重跑时重复保存）
if "pending_file" not in ss:
    ss["pending_file"] = None   # 上传了但还没告诉 agent 的文件（相对路径）
if "last_trace" not in ss:
    ss["last_trace"] = ""       # 上一轮工具调用记录（只存内存，不进历史）
if "new_pngs" not in ss:
    ss["new_pngs"] = []         # 本次新生成的图（文件名），只存内存

# set_session 改的是模块级全局变量，重跑不会重置它，但多标签页会互相覆盖。
if ss["session"]:
    set_session(ss["session"])


def switch_to(name: str) -> None:
    """切换 / 新建会话：磁盘和内存一起换，只换一边就一定出 bug。"""
    set_session(name)
    ss["session"] = name
    ss["pending_name"] = None
    ss["pending_file"] = None
    ss["loaded_file_id"] = None
    ss["last_trace"] = ""
    ss["new_pngs"] = []
    ss["messages"] = load_session() or [{"role": "system", "content": SYSTEM_PROMPT}]


def save_upload(up) -> str:
    """把上传的内存对象落盘到项目内，返回相对项目根的 posix 路径。

    为什么必须先落盘：load_data 只接受项目内的相对路径（tabular.py 有越界保护），
    而 file_uploader 给的是内存里的 BytesIO，没有路径。
    文件名带内容哈希：同一份文件重复上传落到同一个地方，不同内容永不互相覆盖。
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(up.getvalue()).hexdigest()[:8]
    stem, suffix = Path(up.name).stem, Path(up.name).suffix
    dest = UPLOAD_DIR / f"{stem}_{digest}{suffix}"
    dest.write_bytes(up.getvalue())
    return dest.relative_to(ROOT).as_posix()      # 如 runs/uploads/sales_3f2a91bc.csv


# ===========================================================================
# 1. 侧边栏：会话列表 + 新建 + 撞名确认 + 上传
# ===========================================================================
with st.sidebar:
    st.subheader("会话")

    # 每次重跑都从磁盘实时算 —— 磁盘会被别的进程改，不能靠记忆
    dirs = sorted(
        (p for p in DATA_DIR.iterdir()
         if p.is_dir() and not p.name.startswith("eval_")),   # eval_ 是评测会话
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not dirs:
        st.caption("还没有会话")

    for p in dirs:
        ts = datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M")
        if st.button(f"{p.name} · {ts}", key=f"pick_{p.name}", width="stretch",
                     type="primary" if p.name == ss["session"] else "secondary"):
            switch_to(p.name)
            st.rerun()

    st.divider()
    name = (st.text_input("新建会话", placeholder="例如 sales_2024") or "").strip()
    if st.button("新建", width="stretch"):
        if not name:
            st.warning("先输个名字")
        elif (DATA_DIR / name).is_dir():
            ss["pending_name"] = name       # 撞名：不替用户决定，先挂起来问他
        else:
            try:
                switch_to(name)
                st.rerun()
            except ValueError as exc:       # set_session 里那道正则校验没通过
                st.error(str(exc))

    # 只有 pending_name 有值时才渲染 —— 这就是「问用户一句」的写法
    if ss["pending_name"]:
        pn = ss["pending_name"]
        st.warning(f"会话「{pn}」已存在")
        col_a, col_b = st.columns(2)
        if col_a.button("继续用", width="stretch", type="primary"):
            switch_to(pn)                   # 一个文件都不删
            st.rerun()
        if col_b.button("覆盖", width="stretch"):
            # 界面层不碰会话目录结构：结构知识住在 config 里，以后再加子目录也不会崩
            reset_session(pn)               # 整棵子树重建（含 output/）
            switch_to(pn)                   # 内存状态统一交给 switch_to，别在这里手拼
            st.rerun()
        if st.button("取消", width="stretch"):
            ss["pending_name"] = None
            st.rerun()

    # ---- 上传：只负责落盘，不替 agent 加载 ----
    st.divider()
    up = st.file_uploader("上传数据（csv / xlsx / xls / txt）",
                          type=["csv", "xlsx", "xls", "txt"])
    if up is not None and up.file_id != ss["loaded_file_id"]:
        ss["pending_file"] = save_upload(up)    # 只在 file_id 变了时才真正写盘
        ss["loaded_file_id"] = up.file_id
    if ss["pending_file"]:
        st.caption(f"待处理：`{ss['pending_file']}`")


# ===========================================================================
# 2. 主区域：历史 + 对话框
# ===========================================================================
st.title("数据分析 agent")

if ss["session"] is None:
    st.info("先在左侧选一个会话，或新建一个。")
    st.stop()      # 停止「本次重跑」，后面的代码不执行。它不是 return

st.caption(f"当前会话：**{ss['session']}**")

# 只渲染人看得懂的两类消息；tool 消息和带 tool_calls 的 assistant 消息跳过
for m in ss["messages"]:
    if m["role"] == "user":
        st.chat_message("user").write(m["content"])
    elif m["role"] == "assistant" and m.get("content"):
        st.chat_message("assistant").write(m["content"])

if ss["last_trace"]:
    with st.expander("上一轮它做了什么", expanded=False):
        st.code(ss["last_trace"], language="text")

# 只显示「本次这一轮新生成的图」：跑之前记下已有文件名，跑完做差集。
# 旧图仍然躺在会话目录里（那是 agent 的工作现场），但界面不再重放它们。
if ss["new_pngs"]:
    st.subheader(f"本次生成的图（{len(ss['new_pngs'])} 张）")
    cols = st.columns(min(2, len(ss["new_pngs"])))
    for i, fn in enumerate(ss["new_pngs"]):
        p = output_dir() / fn               # 图在 <会话>/output/，不在会话根目录
        if p.exists():                      # 文件可能被下一次 run 覆盖或删掉
            with cols[i % len(cols)]:
                st.image(str(p), caption=fn)

prompt = st.chat_input("问点什么，例如：这个数据有多少行？")
if prompt:
    # 如果刚上传过文件，把路径并进这条消息 —— agent 自己会去 load_data
    content = prompt
    if ss["pending_file"]:
        content = f"【我刚上传了文件 {ss['pending_file']}】\n{prompt}"
    ss["messages"].append({"role": "user", "content": content})

    with st.chat_message("user"):
        st.write(content)

    # run() 是同步阻塞的，跑完之前这一轮什么都没法显示
    before = {p.name for p in output_dir().glob("*.png")}   # 跑之前的图
    buf = io.StringIO()
    with st.chat_message("assistant"):
        with st.status("正在分析…", expanded=True) as status:
            try:
                with contextlib.redirect_stdout(buf):     # 把 run() 里的 print 收进来
                    answer = run(ss["messages"])
            except Exception as exc:                      # 崩了也要把现场留下来
                status.update(label="出错了", state="error")
                st.exception(exc)
                st.stop()
            status.update(label=f"完成（{len(buf.getvalue().splitlines())} 行过程）",
                          state="complete")
            st.write(answer or "（本轮没有返回文字）")

    # 收尾：把过程和结果留在 session_state 里，再存盘
    ss["last_trace"] = buf.getvalue().strip()
    # 差集就是「本次新生成的图」；顺带解决"旧图累积重放"的问题
    ss["new_pngs"] = sorted({p.name for p in output_dir().glob("*.png")} - before)
    # run() 自己会往 messages 里写最终回答（agent_chat.py:170），
    # 但「步数/预算用完」那条出口走 _finalize()，是不写的。
    # 所以只在最后一条不是这条回答时才补写 —— 否则回答会存两遍、显示两遍。
    last = ss["messages"][-1] if ss["messages"] else {}
    if not (last.get("role") == "assistant" and last.get("content") == (answer or "")):
        ss["messages"].append({"role": "assistant", "content": answer or ""})
    ss["pending_file"] = None        # 文件已经交代过了，别在下一问里重复塞
    save_session(ss["messages"])
    st.rerun()                       # 重跑一次，让历史区按统一逻辑重画
