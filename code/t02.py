"""
Streamlit + Matplotlib
电商用户数据探索（EDA 可视化大屏）

运行方式：
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. 导入你已有的数据分析模块
# =========================
import data_loader
from data_explore import (
    explore_missing_values,
    explore_numeric_features,
    explore_categorical_features,
    analyze_feature_by_group,
    explore_correlation,
    split_columns_clean
)

# =========================
# 2. 页面配置（必须最前）
# =========================
st.set_page_config(
    page_title="电商用户数据分析大屏",
    layout="wide"
)

# =========================
# 3. 页面标题
# =========================
st.title("📊 电商用户生命周期分析可视化大屏")
st.markdown("**课程项目 | 使用 Streamlit + Matplotlib 构建 EDA Dashboard**")

st.divider()

# =========================
# 4. 数据加载（缓存）
# =========================
@st.cache_data
def load_data():
    return data_loader.load_raw_data()

df = load_data()
numeric_cols, categorical_cols = split_columns_clean(df)

st.success(f"数据加载完成：{df.shape[0]} 行，{df.shape[1]} 列")

# =========================
# 5. 页面结构（Tabs）
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 数据概览",
    "🔍 缺失值分析",
    "👤 用户画像",
    "🧩 分组分析",
    "🔗 相关性分析"
])

# ======================================================
# 📌 Tab 1：数据概览
# ======================================================
with tab1:
    st.subheader("📌 数据集基本信息")

    col1, col2, col3 = st.columns(3)
    col1.metric("样本数", df.shape[0])
    col2.metric("特征数", df.shape[1])
    col3.metric("目标变量", "lifecycle")

    st.divider()

    st.subheader("📄 数据预览")
    st.dataframe(df.head(), use_container_width=True)

    st.info(
        "本项目围绕用户生命周期（lifecycle）展开，"
        "目标是分析不同生命周期用户的行为与属性差异。"
    )

# ======================================================
# 🔍 Tab 2：缺失值分析（matplotlib）
# ======================================================
with tab2:
    st.subheader("🔍 字段缺失值分布")

    missing_df = explore_missing_values(df)

    # -------- matplotlib 缺失率柱状图 --------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        missing_df.index,
        missing_df["missing_rate"]
    )
    ax.set_ylabel("缺失率")
    ax.set_xlabel("字段名")
    ax.set_title("各字段缺失率分布")
    plt.xticks(rotation=45, ha="right")

    st.pyplot(fig)

    st.dataframe(missing_df, use_container_width=True)

    st.info(
        "缺失值分布图可以直观反映哪些字段需要重点清洗或构造缺失值指示变量。"
    )

# ======================================================
# 👤 Tab 3：用户画像（核心 matplotlib 图）
# ======================================================
with tab3:
    st.subheader("👤 用户画像分析")

    # ---------- 生命周期分布 ----------
    st.markdown("### 🎯 生命周期分布")

    lifecycle_counts = df["lifecycle"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(lifecycle_counts.index, lifecycle_counts.values)
    ax.set_xlabel("生命周期")
    ax.set_ylabel("用户数量")
    ax.set_title("不同生命周期用户数量分布")

    st.pyplot(fig)

    # ---------- 年龄分布 ----------
    if "age" in numeric_cols:
        st.markdown("### 🎂 年龄分布")

        age_data = df["age"].dropna()

        fig, ax = plt.subplots()
        ax.hist(age_data, bins=20)
        ax.set_xlabel("年龄")
        ax.set_ylabel("人数")
        ax.set_title("用户年龄分布直方图")

        st.pyplot(fig)

    st.info(
        "生命周期和年龄是用户画像中的关键维度，"
        "可以帮助理解不同阶段用户的构成特点。"
    )

# ======================================================
# 🧩 Tab 4：分组分析（matplotlib 版）
# ======================================================
with tab4:
    st.subheader("🧩 生命周期分组特征分析")

    feature_col = st.selectbox(
        "选择一个特征",
        options=numeric_cols + categorical_cols
    )

    group_df = analyze_feature_by_group(
        df,
        group_col="lifecycle",
        feature_col=feature_col,
        normalize=True
    )

    # -------- matplotlib 堆叠柱状图 --------
    fig, ax = plt.subplots(figsize=(8, 5))
    group_df.plot(kind="bar", stacked=True, ax=ax)

    ax.set_ylabel("比例")
    ax.set_title(f"{feature_col} 在不同生命周期下的分布")
    ax.legend(title=feature_col, bbox_to_anchor=(1.05, 1), loc="upper left")

    st.pyplot(fig)

    st.dataframe(group_df, use_container_width=True)

    st.info(
        "分组分析用于观察不同生命周期用户在特定特征上的结构差异，"
        "是用户分层分析的重要工具。"
    )

# ======================================================
# 🔗 Tab 5：相关性分析（matplotlib 热力图）
# ======================================================
with tab5:
    st.subheader("🔗 数值特征相关性分析")

    threshold = st.slider(
        "强相关阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )

    corr_matrix, strong_corr = explore_correlation(df, threshold=threshold)

    # -------- matplotlib 相关性热力图 --------
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr_matrix, cmap="coolwarm")
    fig.colorbar(cax)

    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90)
    ax.set_yticklabels(corr_matrix.columns)
    ax.set_title("数值特征相关性热力图")

    st.pyplot(fig)

    with st.expander("🔥 强相关特征对"):
        if strong_corr.empty:
            st.warning("当前阈值下未发现强相关特征对")
        else:
            st.dataframe(strong_corr, use_container_width=True)

    st.info(
        "相关性分析有助于发现冗余特征，"
        "为特征筛选和模型优化提供依据。"
    )

# =========================
# 页面结束
# =========================
st.divider()
st.success("✅ 可视化大屏构建完成")
