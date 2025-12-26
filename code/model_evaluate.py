import time
from config import START
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, x_test, y_test):
    """
        模型评估函数
        """
    # 1. 用模型对测试集做预测
    y_pred = model.predict(x_test)

    # 2. 准确率
    acc = accuracy_score(y_test, y_pred)

    # 3. 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 4. 分类报告
    report = classification_report(y_test, y_pred)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report
    }


if __name__ == '__main__':
   print(f'{time.time() - START:.2f}s')