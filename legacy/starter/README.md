# starter —— 第 1 周骨架

这是路线图里「阶段 0」的可运行起点：一个 200 多行的 function-calling 循环。
**它的价值不在于能做什么，而在于你能把每一行都看懂。**

## 文件说明

| 文件 | 作用 |
|---|---|
| `agent_loop.py` | 主程序：工具定义 + 工具实现 + LLM 循环 |
| `sample.csv` | 示例数据（30 行销售流水，故意留了 2 个缺失值，用来测 agent 会不会自己发现） |
| `_selftest.py` | 不消耗 API 额度的自检：验证 `load_data` 和 `run_python` 真能跑 |
| `.env.example` | 配置模板，复制成 `.env` 后填 key |
| `_work/` | 运行时生成的中间产物：`current.pkl`（当前数据）、`step.py`（模型写的代码）、图片 |

## 跑起来

```powershell
# 1. 装依赖（如果还没建项目）
pip install pandas openpyxl openai python-dotenv matplotlib

# 2. 配置 key
copy .env.example .env
notepad .env        # 填 AGENT_API_KEY

# 3. 先自检工具链（不花 token，确认环境 OK）
python _selftest.py

# 4. 跑真正的 agent
python agent_loop.py --file sample.csv --ask "各区域销售额排名，给出前3名和占比"
```

> 提示：如果终端里中文显示成乱码，是 Windows 控制台编码问题，执行一次 `chcp 65001` 即可。
> 代码里已经强制子进程用 UTF-8 输出，所以这是纯显示层问题，不影响运行。

## 你该观察什么

跑的时候盯住打印出来的这几行：

```
[1] load_data({'path': 'sample.csv'})
    -> shape: 30 行 x 10 列 ...
[2] run_python({'code': "g = df.groupby('region')['amount']..."})
    -> stdout: region\n华东 13663.0 ...
===== 最终回答 =====
```

这就是 agent 的**全部**工作原理：模型不是"知道答案"，而是"决定调工具 → 看结果 → 再决定"。

## 三个可以立刻动手改的练习（比看教程有效十倍）

1. **把 `MAX_STEPS` 改成 2**，看模型会不会在中途被迫交出半成品结论 —— 理解"步数预算"如何影响输出质量。
2. **给 `run_python` 传一段必错的代码**（比如 `print(df['不存在的列'])`），观察 traceback 如何被回喂、模型如何自我修复 —— 这是 agent 最值钱的能力。
3. **删掉 `load_data` 的 `PICKLE` 保存逻辑**，看模型会怎么报错 —— 理解"工具之间的隐式状态传递"有多重要。

## 下一步（第 2 周）

按路线图的目录结构把它拆开：

```
sandbox.py   ← 抽出 run_python（后面要加 import 白名单、网络禁用、CPU/内存限制）
tools/       ← 每个工具一个文件 + 一个注册表
ingest/      ← 加 xlsx_adapter，然后是 image_adapter（第 3 周）
agent.py     ← 主循环 + 步数/成本上限 + 日志落盘
```
