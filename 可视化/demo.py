import streamlit as st
from datetime import datetime

st.title("第一个界面")
# 下面这行每重跑一次脚本就会变一次 —— 用来证明"整个文件从头又执行了一遍"
st.caption(f"本次脚本执行时间：{datetime.now().strftime('%H:%M:%S')}")

name = st.text_input("你叫什么")          # 返回值 = 输入框此刻的内容
if st.button("打招呼"):                   # 返回值 = 这次重跑是不是因为点了它
    st.write(f"你好，{name}")
    print("按钮被点了 -> 整个脚本重跑了一次")   # 看终端
