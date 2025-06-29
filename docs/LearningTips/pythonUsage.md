# Python

## Pipenv简介
pipenv 是 Python 官方推荐的虚拟环境和依赖管理工具，它结合了 pip 和 virtualenv 的功能，并通过 Pipfile 替代 requirements.txt。以下是核心用法指南：

## 1.安装
```
pip install pipenv  # 全局安装
```

## 2.核心工作流
### 2.1创建虚拟环境
进入项目目录后执行
```
pipenv install  # 自动创建虚拟环境（基于项目目录名）
```

- 若项目有 `Pipfile`，会按配置安装依赖
- 若无 `Pipfile`，生成 `Pipfile` 和 `Pipfile.lock`
  
### 安装包

```
pipenv install requests         # 安装生产依赖
pipenv install pytest --dev     # 安装开发依赖（写入 `[dev-packages]`）
pipenv install -r requirements.txt  # 导入旧依赖
pipenv install -r requirements.txt  # 导入旧依赖. 从 requirements.txt 迁移
pipenv requirements > requirements.txt          # 生产依赖导出 requirements.txt
```
### 2.3运行环境
```
pipenv shell    # 激活虚拟环境（启动子 shell）
exit            # 退出虚拟环境

# 或单命令运行（不进入 shell）
pipenv run python main.py
```
