"""
Streamlit 可视化大屏
电商用户数据探索（EDA Dashboard）

运行方式：
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 导入模块
import data_loader
from data_explore import (
    explore_missing_values,
    explore_numeric_features,
    explore_categorical_features,
    analyze_feature_by_group,
    explore_correlation,
    split_columns_clean
)

# 页面基础配置
st.set_page_config(
    page_title="电商用户数据分析大屏",
    layout="wide"
)

# 页面标题
st.title("📊 电商用户数据分析可视化大屏")
st.markdown("**课程项目：用户生命周期分析（lifecycle）**")

st.divider()


# 数据加载
@st.cache_data
def load_data():
    """
    加载原始数据（只读）
    """
    return data_loader.load_raw_data()


df = load_data()

st.success(
    f"数据加载完成：共 {df.shape[0]} 行，{df.shape[1]} 列"
)

# 5. 创建页面 Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 数据概览",
    "🔍 缺失值分析",
    "👤 用户画像",
    "🧩 分组分析",
    "🔗 相关性分析"
])

# 📌 Tab 1：数据概览
with tab1:
    st.subheader("📌 数据集概览")

    col1, col2, col3 = st.columns(3)
    col1.metric("样本数", df.shape[0])
    col2.metric("特征数", df.shape[1])
    col3.metric("目标变量", "lifecycle")

    st.divider()

    st.subheader("📄 数据预览（前 5 行）")
    st.dataframe(df.head(), use_container_width=True)

    st.info(
        "本数据集为电商用户行为数据，"
        "目标是分析不同生命周期（lifecycle）用户的特征差异。"
    )

# 🔍 Tab 2：缺失值分析
with tab2:
    st.subheader("🔍 字段缺失情况分析")

    missing_df = explore_missing_values(df)
    st.dataframe(missing_df, use_container_width=True)

    st.info(
        "缺失值分析用于指导后续数据清洗策略，"
        "如填充、删除或构建缺失值指示变量。"
    )

# 👤 Tab 3：用户画像分析
with tab3:
    st.subheader("👤 用户画像分析")

    numeric_cols, categorical_cols = split_columns_clean(df)

    # 生命周期分布（核心图）
    st.markdown("### 🎯 用户生命周期分布")
    lifecycle_count = df["lifecycle"].value_counts()

    st.bar_chart(lifecycle_count)

    st.divider()

    # 年龄分布
    if "age" in numeric_cols:
        st.markdown("### 🎂 用户年龄分布")
        age_series = df["age"].dropna()

        fig, ax = plt.subplots()
        ax.hist(age_series, bins=20)
        ax.set_xlabel("年龄")
        ax.set_ylabel("人数")

        st.pyplot(fig)

    st.info(
        "生命周期分布是建模和业务分析的核心，"
        "可以观察不同生命周期用户的规模差异。"
    )

# 🧩 Tab 4：分组特征分析
with tab4:
    st.subheader("🧩 不同生命周期下的特征分布")

    # 选择要分析的特征
    feature_col = st.selectbox(
        "选择一个特征进行分析",
        options=numeric_cols + categorical_cols
    )

    group_df = analyze_feature_by_group(
        df,
        group_col="lifecycle",
        feature_col=feature_col,
        normalize=True
    )

    st.dataframe(group_df, use_container_width=True)

    st.info(
        "该分析用于比较不同生命周期用户在某一特征上的分布差异，"
        "可为用户分层和精准运营提供依据。"
    )

# 🔗 Tab 5：相关性分析
with tab5:
    st.subheader("🔗 数值特征相关性分析")

    threshold = st.slider(
        "选择强相关阈值",
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

    with st.expander("📊 查看相关性矩阵"):
        st.dataframe(corr_matrix, use_container_width=True)

    with st.expander("🔥 查看强相关特征对"):
        if strong_corr.empty:
            st.warning("当前阈值下未发现强相关特征对")
        else:
            st.dataframe(strong_corr, use_container_width=True)

    st.info(
        "相关性分析可用于特征筛选，"
        "避免多重共线性对模型训练产生影响。"
    )

# 页面结束
st.divider()
st.success("✅ 可视化大屏加载完成")
