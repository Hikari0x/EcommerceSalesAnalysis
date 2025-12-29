import numpy as np
import streamlit as st
import pandas as pd
from config import RAW_DATA_PATH
from data_loader import load_raw_data
import matplotlib.pyplot as plt
from data_visualize import plot_numeric_distribution
import seaborn as sns

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

df = load_raw_data()
st.title("我的第一个 Streamlit App")
st.write("Hello, Streamlit! 🎈")

name = st.text_input("你的名字是？")
if name:
    st.success(f"欢迎你，{name}！")

# 创建 Matplotlib 图形
fig, ax = plt.subplots(figsize=(8, 6))  # 可以设置大小

# 示例：绘制散点图
x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
ax.scatter(x, y, alpha=0.6, color='teal')
ax.set_title("随机散点图示例")
ax.set_xlabel("X轴")
ax.set_ylabel("Y轴")
ax.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig)

abc = numeric_cols = ['age']
ds = plot_numeric_distribution(df, numeric_cols)
st.pyplot(ds)
# 创建plt图形画布
plt.figure(figsize=(6, 4), dpi=300)
sns.histplot(df['age'], bins=50, kde=True)
plt.title(f"{'age'} 分布")
plt.tight_layout()
plt.show()
