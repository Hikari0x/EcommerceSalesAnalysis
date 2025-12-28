import streamlit as st
import pandas as pd
from config import RAW_DATA_PATH
from data_loader import load_raw_data

# st.title('hello streamlit')
# st.write('第一个页面')
# df = pd.read_csv('../data/data_week2.csv')
# st.dataframe(df)
# st.dataframe(df.info())
# st.table(df.head())
#
#
# st.title("电商用户分析系统")      # 大标题
# st.header("数据概览")            # 一级标题
# st.subheader("缺失值分析")       # 二级标题
# st.write("这是说明文字")          # 普通文本
# st.markdown("**加粗 / Markdown**")

import streamlit as st

st.title("我的第一个 Streamlit App")
st.write("Hello, Streamlit! 🎈")

name = st.text_input("你的名字是？")
if name:
    st.success(f"欢迎你，{name}！")