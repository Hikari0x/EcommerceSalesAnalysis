import time
from config import START
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import seaborn as sns
from data_loader import data_loader


def main():
    """
    程序主函数,一键启动
    :return: 
    """
    print(f'{'-' * 30}电商销售数据分析项目{'-' * 30}')
    print(f'{'-' * 30}项目完成{'-' * 30}')


if __name__ == '__main__':
    main()
    print(f'{time.time() - START:.2f}s')
