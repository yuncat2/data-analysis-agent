"""侧边栏 smoke test（不需要浏览器）。

用 Streamlit 自带的 AppTest 在无界面情况下模拟点击：
    python 可视化/smoke_test.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest          # noqa: E402

APP = str(ROOT / "可视化" / "app.py")
DATA_DIR = ROOT / "runs" / "sessions"
PROBE = "zz_ui_probe"

ok = True


def check(cond, msg):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    ok = ok and bool(cond)


def labels(at):
    return [b.label for b in at.sidebar.button]


# ---------------------------------------------------------------- 1. 首屏
at = AppTest.from_file(APP, default_timeout=60).run()
check(not at.exception, f"首屏无异常 {[e.value for e in at.exception]}")
print("  侧边栏按钮:", labels(at))
check(any("demo" in x for x in labels(at)), "列表里有 demo")
check(not any("eval_" in x for x in labels(at)), "eval_ 会话被过滤掉了")


# ------------------------------------------------------- 2. 切换已有会话
demo_btn = next(b for b in at.sidebar.button if b.label.startswith("demo"))
demo_btn.click().run()
check(not at.exception, f"点 demo 无异常 {[e.value for e in at.exception]}")
check(any("当前会话：**demo**" in c.value for c in at.caption), "已切到 demo")


# ------------------------------------------------------- 3. 新建撞名 → 确认区
at.sidebar.text_input[0].set_value("demo").run()
next(b for b in at.sidebar.button if b.label == "新建").click().run()
check(not at.exception, f"撞名无异常 {[e.value for e in at.exception]}")
check(any("已存在" in w.value for w in at.warning), "弹出「已存在」提示")
check({"继续用", "覆盖"} <= set(labels(at)), "出现 继续用 / 覆盖 两个按钮")


# ------------------------------- 4. 继续用：不该删任何文件（用探测会话验证）
probe_dir = DATA_DIR / PROBE
probe_dir.mkdir(parents=True, exist_ok=True)
(probe_dir / "dummy.txt").write_text("keep me", encoding="utf-8")
# 子目录也放一个：这正是「覆盖」以前 WinError 5 的场景
(probe_dir / "output").mkdir(exist_ok=True)
(probe_dir / "output" / "old_fig.png").write_bytes(b"\x89PNG\r\n\x1a\n")

at.sidebar.text_input[0].set_value(PROBE).run()
next(b for b in at.sidebar.button if b.label == "新建").click().run()
next(b for b in at.sidebar.button if b.label == "继续用").click().run()
check(not at.exception, f"继续用无异常 {[e.value for e in at.exception]}")
check((probe_dir / "dummy.txt").exists(), "继续用 保留了已有文件")
check(not any("已存在" in w.value for w in at.warning), "确认区用完就消失了")


# ------------------------------------------------------------ 5. 覆盖：清空
at.sidebar.text_input[0].set_value(PROBE).run()
next(b for b in at.sidebar.button if b.label == "新建").click().run()
next(b for b in at.sidebar.button if b.label == "覆盖").click().run()
check(not at.exception, f"覆盖无异常 {[e.value for e in at.exception]}")
check(not (probe_dir / "dummy.txt").exists(), "覆盖 清掉了旧文件")
check(not (probe_dir / "output" / "old_fig.png").exists(), "覆盖 清掉了 output/ 里的旧产物")
check((probe_dir / "output").is_dir(), "覆盖 之后 output/ 仍然存在（下游不用判存在）")


# --------------------------------------------------- 6. 新建不撞名 → 建目录
new_name = PROBE + "_new"
at.sidebar.text_input[0].set_value(new_name).run()
next(b for b in at.sidebar.button if b.label == "新建").click().run()
check(not at.exception, f"新建无异常 {[e.value for e in at.exception]}")
check((DATA_DIR / new_name).is_dir(), "新建 真的建了目录")

# --------------------------------------------------- 7. 非法名字被接口挡住
at.sidebar.text_input[0].set_value("../etc").run()
next(b for b in at.sidebar.button if b.label == "新建").click().run()
check(not at.exception, "非法名字没让页面崩（只是提示）")


# ------------------------------------- 8. 对话链路（用假 run()，不花 API 钱）
import json                                          # noqa: E402
import data_agent.agent_chat as ac                   # noqa: E402

calls = []


def fake_run(messages, max_steps=10, stats=None):
    """假装跑了一轮：模拟它调了一次工具 + 画了一张图 + 返回答案。"""
    calls.append([dict(m) for m in messages])
    print("[1] load_data({'path': 'runs/uploads/x.csv'})")
    print("    -> 已加载为数据集 'x'")
    import matplotlib                                 # noqa: E402
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                   # noqa: E402
    from data_agent.config import output_dir          # noqa: E402
    plt.plot([1, 2], [3, 4])
    plt.savefig(output_dir() / "fig_new.png")         # 模拟 agent 画图（cwd 已改成 output/）
    plt.close()
    messages.append({"role": "assistant", "content": "假答案：共 3 行"})
    return "假答案：共 3 行"


# app.py 里写的是 `from data_agent.agent_chat import run`，
# 每次重跑都会重新执行这行 import，所以换掉模块属性就够了。
ac.run = fake_run

chat_name = PROBE + "_chat"
at.sidebar.text_input[0].set_value(chat_name).run()
next(b for b in at.sidebar.button if b.label == "新建").click().run()
check(not at.exception, f"建 chat 探测会话无异常 {[e.value for e in at.exception]}")

# 会话目录的 output/ 里先放一张「上次留下的旧图」，它不该被显示
chat_dir = DATA_DIR / chat_name
(chat_dir / "output").mkdir(exist_ok=True)
old_png = chat_dir / "output" / "fig_old.png"
old_png.write_bytes(b"\x89PNG\r\n\x1a\n")            # 内容随便，我们只验文件名差集

at.chat_input[0].set_value("这个数据有多少行？").run()
check(not at.exception, f"提问无异常 {[e.value for e in at.exception]}")
check(len(calls) == 1, f"run() 被调了 {len(calls)} 次（应为 1）")
if calls:
    check(calls[0][-1]["content"] == "这个数据有多少行？", "用户那句话进了 messages")

codes = [c.value for c in at.code]
check(any("load_data" in c for c in codes), f"界面渲染出了工具调用痕迹（code 块 {len(codes)} 个）")

hist = DATA_DIR / chat_name / "history.json"
check(hist.exists(), "历史写进了 history.json")
if hist.exists():
    saved = json.loads(hist.read_text(encoding="utf-8"))
    roles = [m["role"] for m in saved]
    check(roles == ["system", "user", "assistant"],
          f"历史结构 {roles}（回答不能被存两遍）")

imgs = at.get("image")
check(len(imgs) == 1, f"只显示本次新图：渲染了 {len(imgs)} 张（应为 1）")


# --------------- 9. run() 没往 messages 写回答的那条出口（_finalize）---------
def fake_run_finalize(messages, max_steps=10, stats=None):
    calls.append([dict(m) for m in messages])
    print("[1] run_python(...)")
    return "假答案2：预算用完了"       # 故意不 append，模拟 _finalize()


ac.run = fake_run_finalize
at.chat_input[0].set_value("第二个问题").run()
check(not at.exception, f"第二问无异常 {[e.value for e in at.exception]}")
check(len(at.get("image")) == 0, "第二轮没画图，旧图也不该重放")
if hist.exists():
    roles = [m["role"] for m in json.loads(hist.read_text(encoding="utf-8"))]
    check(roles == ["system", "user", "assistant", "user", "assistant"],
          f"这条出口也要把回答补进历史 {roles}")

# ---------------------- 10. 覆盖之后，新会话必须重新带上系统提示词 -------------
ac.run = fake_run
rname = PROBE + "_reset"
at.sidebar.text_input[0].set_value(rname).run()
next(b for b in at.sidebar.button if b.label == "新建").click().run()
at.chat_input[0].set_value("第一问").run()
check((DATA_DIR / rname / "history.json").exists(), "重置前：会话有历史")

calls.clear()
at.sidebar.text_input[0].set_value(rname).run()
next(b for b in at.sidebar.button if b.label == "新建").click().run()
next(b for b in at.sidebar.button if b.label == "覆盖").click().run()
check(not (DATA_DIR / rname / "history.json").exists(), "覆盖 清掉了历史文件")

at.chat_input[0].set_value("第二问").run()
sent = calls[-1] if calls else []
check(bool(sent) and sent[0]["role"] == "system",
      f"覆盖后第一条必须是 system（实际 {sent[0]['role'] if sent else '无'}）")
check(len(sent) == 2, f"覆盖后只剩 system + 本条问题（实际 {len(sent)} 条）")

# ---------------------------------------------------------------- 清理
for d in (probe_dir, DATA_DIR / new_name, DATA_DIR / chat_name, DATA_DIR / rname):
    shutil.rmtree(d, ignore_errors=True)

print("\n结果:", "全部通过" if ok else "有失败")
sys.exit(0 if ok else 1)
