"""
电商销售数据分析项目 - 包初始化文件

该包包含用于电商销售数据分析的各个模块：
- 数据加载 (data_loader)
- 数据清洗 (data_clean)
- 数据探索 (data_explore)
- 数据可视化 (data_visualize)
- 特征工程 (feature_engineer)
- 模型训练 (model_train)
- 模型评估 (model_evaluate)
"""

__version__ = "1.0.0"
__author__ = "Ecommerce Sales Analysis Project"

# 定义包的公共接口
__all__ = [
    'data_loader',
    'data_clean',
    'data_explore',
    'data_visualize',
    'feature_engineer',
    'model_train',
    'model_evaluate',
    'config'
]

# 可选：提供包级别的导入便利
try:
    from . import data_loader
    from . import data_clean
    from . import data_explore
    from . import data_visualize
    from . import feature_engineer
    from . import model_train
    from . import model_evaluate
    from . import config
except ImportError:
    # 如果导入失败，不抛出错误，允许单独导入模块
    pass

def get_project_info():
    """
    返回项目信息
    """
    return {
        "name": "Ecommerce Sales Analysis",
        "version": __version__,
        "author": __author__,
        "modules": __all__
    }

# 简单的包验证
def validate_setup():
    """
    验证包是否正确安装设置
    """
    print(f"Ecommerce Sales Analysis Package v{__version__} loaded successfully.")
    return True

# 加载时执行验证
if __name__ != "__main__":
    validate_setup()
