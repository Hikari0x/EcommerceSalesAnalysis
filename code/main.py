import time
from pprint import pprint

from config import START, TARGET_COL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import seaborn as sns
from data_loader import data_loader
from data_explore import data_explore, split_columns_clean
from data_clean import data_clean
from feature_engineer import build_features_for_dl
from ml_model import train_and_evaluate


def main():
    """
    程序主函数,一键启动
    :return: 
    """
    print(f'{'=' * 30}电商销售数据分析项目{'=' * 30}')
    # 数据加载
    print(f'{'-' * 30}数据加载{'-' * 30}')
    df = data_loader()
    # 数据探索
    # print(f'{'-' * 30}数据探索{'-' * 30}')
    # data_explore()
    # 自动划分原数据列
    numeric_cols, categorical_cols = split_columns_clean(df)
    # 数据清洗
    print(f'{'-' * 30}数据清洗{'-' * 30}')
    df_new = data_clean(df, numeric_cols, categorical_cols)
    # 特征工程
    print(f'{'-' * 30}特征工程{'-' * 30}')
    df_new, scaler, encoders = build_features_for_dl(df_new)
    # 简单模型训练
    print(f'{'-' * 30}简单模型训练{'-' * 30}')
    model, metrics, cv_metrics = train_and_evaluate(df_new, TARGET_COL, model_type='rf')
    print(f'模型信息:{model}')
    print('模型评估指标:')
    pprint(metrics)
    print(
        f"交叉验证 F1(macro)：均值 = {cv_metrics['cv_mean']:.4f}, "
        f"标准差 = {cv_metrics['cv_std']:.4f}"
    )

    print(f'{'=' * 30}项目完成{'=' * 30}')


if __name__ == '__main__':
    main()
    print(f'{time.time() - START:.2f}s')
