"""
代码逻辑和函数入口模块
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix,
                             classification_report
                             )
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import dump

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.family'] = 'Heiti TC'
plt.rcParams['axes.unicode_minus'] = False

# 全局配置
START = time.time()
RAW_DATA_PATH = '../data/data_week2.csv'
DF_NEW_PATH = '../data/df_new.csv'
TARGET_COL = 'lifecycle'


# 数据加载
def data_loader() -> pd.DataFrame:
    """加载原始用户数据,并规范列名"""
    # 加载csv数据
    df = pd.read_csv(RAW_DATA_PATH)
    # 去除列名中前后的空格,规范化列名
    df.columns = df.columns.str.strip()
    return df


# 数据探索
def data_explore(data: pd.DataFrame):
    """展示所有探索的数据"""
    # 统计各字段缺失率
    missing_count = data.isnull().sum()
    missing_rate = missing_count / len(data)
    result = pd.DataFrame({
        'missing_count': missing_count,
        'missing_rate': missing_rate
    }).sort_values(by='missing_rate', ascending=False)
    missing_pattern = []
    for col in data.columns:
        if missing_rate[col] > 0:
            if missing_rate[col] > 0.5:
                pattern = "大量缺失"
            elif missing_rate[col] > 0.2:
                pattern = "中等缺失"
            else:
                pattern = "少量缺失"
        else:
            pattern = "无缺失"
        missing_pattern.append(pattern)
    result['missing_pattern'] = missing_pattern
    print('统计各字段缺失率:')
    print(result)

    # 分析数值型特征的描述性统计
    numeric_df = data.select_dtypes(include=["int64", "float64"])
    numeric_desc = numeric_df.describe().T
    print('分析数值型特征的描述性统计:')
    print(numeric_desc)

    # 分析类别型特征的分布情况
    result_cat = {}
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        value_counts = data[col].value_counts(dropna=False)
        result_cat[col] = value_counts.head(10)  # 默认top_n=10
    print('分析类别型特征的分布情况:')
    print(result_cat)

    # 分析某个特征在不同分组中的分布
    if 'lifecycle' in data.columns and 'age' in data.columns:
        group_result = pd.crosstab(data['lifecycle'], data['age'], normalize='index')
        print('分析某个特征在不同分组中的分布:')
        print(group_result)

    # 分析数值特征之间的相关性
    numeric_df_corr = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_df_corr.corr(method='pearson')
    strong_corr_list = []
    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            feature_1 = columns[i]
            feature_2 = columns[j]
            corr_value = corr_matrix.loc[feature_1, feature_2]
            if abs(corr_value) >= 0.7:  # threshold = 0.7
                strong_corr_list.append([
                    feature_1,
                    feature_2,
                    corr_value,
                    abs(corr_value)
                ])
    strong_corr = pd.DataFrame(
        strong_corr_list,
        columns=['feature1', 'feature2', 'correlation', 'abs_correlation']
    )
    strong_corr = strong_corr.sort_values(
        by='abs_correlation',
        ascending=False
    )
    print('分析数值特征之间的相关性:')
    print("相关性矩阵:")
    print(corr_matrix)
    print("强相关特征对:")
    print(strong_corr)

    # 自动划分数值列和类别列
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
    print(f'建模列:{numeric_cols}')
    print(f'类别列:{categorical_cols}')


# 数据清洗
def data_clean(data: pd.DataFrame):
    """数据清洗主函数"""
    # 识别数值列和类别列
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()

    # 复制数据
    df_new = data.copy()

    # 处理数值型特征缺失值
    for col in numeric_cols:
        null_flag_col = f"{col}_is_null"
        df_new[null_flag_col] = df_new[col].isnull().astype(int)
        df_new[col] = df_new[col].fillna(-1)

    # 处理类别型特征缺失值
    for col in categorical_cols:
        df_new[col] = df_new[col].fillna('Unknown')

    # 标记异常值
    if "age" in df_new.columns:
        df_new["age_abnormal"] = (
                (df_new["age"] <= 0) | (df_new["age"] > 100)
        ).astype(int)

    # 处理重复值
    dup_count = df_new.duplicated().sum()
    if dup_count > 0:
        print(f"发现重复行数量：{dup_count}，已执行删除")
        df_new = df_new.drop_duplicates(keep="first")
    else:
        print("未发现重复行")

    # 保存清洗后的数据
    df_new.to_csv(DF_NEW_PATH, index=False)
    print(f"清洗后数据已保存至：{DF_NEW_PATH}")

    return df_new


# 特征工程
def feature_engineering(data: pd.DataFrame):
    """特征工程主函数"""
    df_new = data.copy()

    # 自动划分列类型
    numeric_cols = df_new.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df_new.select_dtypes(include=["object"]).columns.tolist()

    # 从类别特征中剔除标签列
    if TARGET_COL in categorical_cols:
        categorical_cols.remove(TARGET_COL)

    # 识别缺失值指示 & 异常标记列
    indicator_cols = [
        col for col in numeric_cols
        if col.endswith("_is_null") or col.endswith("_abnormal")
    ]
    # 真正用于数值建模的列（需要标准化）
    final_numeric_cols = [
        col for col in numeric_cols
        if col not in indicator_cols
    ]

    # 为数值型特征添加缺失值指示变量
    for col in final_numeric_cols:
        df_new[f"{col}_is_missing"] = (df_new[col] == -1).astype(int)

    # 数值特征标准化
    scaler = StandardScaler()
    df_new[final_numeric_cols] = scaler.fit_transform(df_new[final_numeric_cols])

    # 类别特征 One-Hot 编码
    df_new = pd.get_dummies(df_new, columns=categorical_cols, drop_first=False)

    return df_new, scaler


# 模型训练
def model_train(data: pd.DataFrame):
    """模型训练主函数"""
    # 拆分特征和标签
    x = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]

    # 划分训练集 / 测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=123,
        stratify=y
    )

    # 构建模型（使用随机森林）
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=123,
        n_jobs=-1
    )

    # 训练模型
    model.fit(x_train, y_train)

    # 保存模型
    dump(model, f'../model/rf.joblib')

    # 预测测试集
    y_pred = model.predict(x_test)

    # 评估模型
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        )
    }

    # 交叉验证
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(
        model,
        x,
        y,
        cv=5,
        scoring='f1_macro'
    )
    cv_metrics = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    print("模型训练完成！")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    print(f"交叉验证平均得分: {cv_metrics['cv_mean']:.4f}")

    return model, metrics


# 模型预测
def model_predict(data: pd.DataFrame, model=None):
    """模型预测函数"""
    if model is None:
        print("没有提供模型，无法进行预测")
        return None

    # 使用训练好的模型进行预测
    x = data.drop(columns=[TARGET_COL])  # 移除目标列
    predictions = model.predict(x)

    print(f"预测完成，前10个预测结果: {predictions[:10]}")
    return predictions


# 程序入口
def main():
    print("开始电商销售数据分析...")

    # 1. 加载数据
    print("\n1. 加载数据...")
    df = data_loader()
    print(f"数据加载完成，数据形状: {df.shape}")

    # 2. 探索数据
    print("\n2. 探索数据...")
    data_explore(df)

    # 3. 清洗数据
    print("\n3. 清洗数据...")
    df_cleaned = data_clean(df)
    print("数据清洗完成")

    # 4. 特征工程
    print("\n4. 特征工程...")
    df_processed, scaler = feature_engineering(df_cleaned)
    print("特征工程完成")

    # 5. 模型训练
    print("\n5. 模型训练...")
    model, metrics = model_train(df_processed)
    print("模型训练完成")

    # 6. 模型预测
    print("\n6. 模型预测...")
    predictions = model_predict(df_processed, model)
    print("模型预测完成")

    print(f"\n程序运行完成，总耗时: {time.time() - START:.2f}s")


if __name__ == '__main__':
    main()
