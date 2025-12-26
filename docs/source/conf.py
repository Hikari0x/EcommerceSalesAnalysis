# 让 Sphinx 找到你的项目代码（假设你的包在项目根目录下叫 mypackage）
import os
import sys

sys.path.insert(0, os.path.abspath('../../code/src'))  # 指向项目根目录

# 扩展列表
extensions = [
    'sphinx.ext.autodoc',  # 自动提取 docstring
    'sphinx.ext.napoleon',  # 支持 Google/Numpy 风格
    'sphinx.ext.viewcode',  # 显示源码链接
    'sphinx.ext.autosummary',  # 自动生成摘要
    'sphinx.ext.intersphinx'  # 链接到其他文档
]

# 默认选项：显示所有成员
autodoc_default_options = {
    'members': True,
    'undoc-members': True,  # 没写 docstring 的也显示
    'show-inheritance': True,
}

# 主题（可选，美观）
html_theme = 'sphinx_rtd_theme'

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'API参考'
copyright = '2025, lihetong'
author = 'lihetong'
release = '2025-12-30'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []

templates_path = ['_templates']
exclude_patterns = []

language = 'zh'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

latex_engine = 'xelatex'

# 自动摘要配置
autosummary_generate = True
