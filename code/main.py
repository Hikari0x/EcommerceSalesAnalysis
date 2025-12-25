import time
from config import START
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import seaborn as sns
from data_loader import load_raw_data, get_basic_info
from data_clean import clean_data
from data_explore import explore_missing_values, explore_numeric_features, explore_categorical_features, categorical_by_lifecycle
from data_visualize import plot_numeric_distribution, plot_categorical_distribution, plot_box_by_category, plot_correlation_heatmap, plot_categorical_by_lifecycle
from feature_engineer import build_features
from model_train import train_model
from model_evaluate import evaluate_model


def main():
    print(f'{'-'*30}电商销售数据分析项目{'-'*30}')

    # 1. 加载原始数据
    print("\n1. 加载原始数据...")
    df = load_raw_data()
    print(f"数据形状: {df.shape}")

    # 获取数据基本信息
    info = get_basic_info(df)
    print("数据基本信息:")
    for key, value in info.items():
        if key != 'dtypes':  # 避免输出过多信息
            print(f"  {key}: {value}")

    # 2. 数据探索
    print("\n2. 数据探索...")

    # 探索缺失值
    missing_info = explore_missing_values(df)
    print("缺失值统计:")
    print(missing_info[missing_info['missing_rate'] > 0])  # 只显示有缺失值的列

    # 探索数值型特征
    numeric_summary = explore_numeric_features(df)
    print("数值型特征描述统计:")
    print(numeric_summary)

    # 探索类别型特征
    categorical_summary = explore_categorical_features(df)
    print("类别型特征分布:")
    for col, value_counts in categorical_summary.items():
        print(f"  {col}:")
        print(f"    {value_counts.head()}")  # 只显示前几项避免输出过多

    # 3. 数据清洗
    print("\n3. 数据清洗...")

    # 假设已知的数值列和类别列（在实际项目中需要根据具体数据集调整）
    # 根据之前代码中的示例，我们使用这些列名
    numeric_cols = []
    categorical_cols = []

    # 动态识别数值列和类别列
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            numeric_cols.append(col)
        elif df[col].dtype in ['object', 'category']:
            categorical_cols.append(col)

    # 如果没有自动识别到数值列，使用常见数值列名
    if not numeric_cols:
        common_numeric_cols = ['age', 'revenue', 'days_since_last_order', 'quantity', 'price', 'amount']
        numeric_cols = [col for col in common_numeric_cols if col in df.columns]

    # 如果没有自动识别到类别列，使用常见类别列名
    if not categorical_cols:
        common_categorical_cols = ['gender', 'lifecycle', 'category', 'product_name', 'location']
        categorical_cols = [col for col in common_categorical_cols if col in df.columns]

    print(f"识别的数值列: {numeric_cols}")
    print(f"识别的类别列: {categorical_cols}")

    df_clean = clean_data(df, numeric_cols, categorical_cols)
    print(f"清洗后数据形状: {df_clean.shape}")

    # 4. 数据可视化
    print("\n4. 数据可视化...")

    # 如果有数值列，绘制分布图
    if numeric_cols:
        for col in numeric_cols[:2]:  # 只对前两个数值列进行可视化，避免输出过多图表
            if col in df_clean.columns:
                try:
                    plot_numeric_distribution(df_clean, col)
                except:
                    print(f"无法绘制 {col} 的分布图")

    # 如果有类别列，绘制分布图
    if categorical_cols:
        for col in categorical_cols[:2]:  # 只对前两个类别列进行可视化
            if col in df_clean.columns:
                try:
                    plot_categorical_distribution(df_clean, col)
                except:
                    print(f"无法绘制 {col} 的分布图")

    # 绘制相关性热力图
    try:
        plot_correlation_heatmap(df_clean)
    except:
        print("无法绘制相关性热力图")

    # 5. 特征工程
    print("\n5. 特征工程...")

    # 对类别变量进行独热编码
    df_features = build_features(df_clean, categorical_cols)
    print(f"特征工程后数据形状: {df_features.shape}")

    # 6. 模型训练
    print("\n6. 模型训练...")

    # 确定目标列（假设是lifecycle或revenue，需要根据实际数据集调整）
    target_col = None
    possible_target_cols = ['lifecycle', 'revenue', 'category']  # 常见的目标列
    for col in possible_target_cols:
        if col in df_features.columns:
            target_col = col
            break

    # 如果没有找到预设的目标列，选择一个类别型列作为目标
    if target_col is None:
        for col in df_features.columns:
            if df_features[col].dtype == 'object' or len(df_features[col].unique()) < 20:
                target_col = col
                break

    if target_col is not None:
        print(f"目标列: {target_col}")

        # 训练随机森林模型
        print("训练随机森林模型...")
        try:
            rf_model, x_test, y_test = train_model(df_features, target_col, model_type='rf')
            print("随机森林模型训练完成")

            # 模型评估
            print("\n7. 模型评估...")
            rf_results = evaluate_model(rf_model, x_test, y_test)
            print(f"随机森林模型准确率: {rf_results['accuracy']:.4f}")
            print("分类报告:")
            print(rf_results['report'])

            # 训练逻辑回归模型
            print("\n训练逻辑回归模型...")
            try:
                lr_model, x_test, y_test = train_model(df_features, target_col, model_type='lr')
                print("逻辑回归模型训练完成")

                # 模型评估
                lr_results = evaluate_model(lr_model, x_test, y_test)
                print(f"逻辑回归模型准确率: {lr_results['accuracy']:.4f}")
                print("分类报告:")
                print(lr_results['report'])

            except Exception as e:
                print(f"逻辑回归模型训练失败: {e}")

        except Exception as e:
            print(f"模型训练失败: {e}")
    else:
        print("未找到合适的目标列进行模型训练")

    print(f'{'-'*30}项目完成{'-'*30}')


if __name__ == '__main__':
    main()
    print(f'{time.time() - START:.2f}s')
