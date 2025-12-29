import streamlit as st
import pandas as pd

import data_loader
from data_explore import (
    explore_missing_values,
    explore_numeric_features,
    explore_categorical_features,
    analyze_feature_by_group,
    explore_correlation,
    split_columns_clean
)

st.set_page_config(
    page_title="数据探索 EDA",
    layout="wide"
)

st.title("📊 电商用户数据探索（EDA）")

# =========================
# 1. 加载数据
# =========================
@st.cache_data
def load_data():
    return data_loader.load_raw_data()

df = load_data()
st.success(f"数据加载完成，共 {df.shape[0]} 行，{df.shape[1]} 列")

st.divider()

# =========================
# 2. 缺失值分析
# =========================
st.subheader("🔍 字段缺失率分析")

missing_df = explore_missing_values(df)
st.dataframe(missing_df, use_container_width=True)

st.divider()

# =========================
# 3. 数值特征描述统计
# =========================
st.subheader("📈 数值型特征描述性统计")

numeric_desc = explore_numeric_features(df)
st.dataframe(numeric_desc, use_container_width=True)

st.divider()

# =========================
# 4. 类别特征分布
# =========================
st.subheader("📊 类别型特征分布")

cat_result = explore_categorical_features(df)

for col, value_counts in cat_result.items():
    st.markdown(f"**{col}**")
    st.dataframe(value_counts.to_frame("count"))

st.divider()

# =========================
# 5. 分组特征分析
# =========================
st.subheader("🧩 分组特征分析")

group_col = st.selectbox(
    "选择分组字段",
    options=["lifecycle"]
)

feature_col = st.selectbox(
    "选择分析特征",
    options=df.columns
)

group_df = analyze_feature_by_group(
    df,
    group_col=group_col,
    feature_col=feature_col,
    normalize=True
)

st.dataframe(group_df, use_container_width=True)

st.divider()

# =========================
# 6. 相关性分析
# =========================
st.subheader("🔗 数值特征相关性分析")

threshold = st.slider(
    "强相关阈值",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05
)

corr_matrix, strong_corr = explore_correlation(
    df,
    method="pearson",
    threshold=threshold
)

with st.expander("📌 相关性矩阵"):
    st.dataframe(corr_matrix, use_container_width=True)

with st.expander("🔥 强相关特征对"):
    if strong_corr.empty:
        st.info("当前阈值下没有强相关特征对")
    else:
        st.dataframe(strong_corr, use_container_width=True)

st.divider()

# =========================
# 7. 自动列划分
# =========================
st.subheader("🧠 自动列类型划分")

numeric_cols, categorical_cols = split_columns_clean(df)

st.markdown("**数值列（用于建模）**")
st.code(numeric_cols)

st.markdown("**类别列**")
st.code(categorical_cols)
