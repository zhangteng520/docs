# Python 虚拟环境全解

## Python3 虚拟环境简介
Python 虚拟环境用于隔离项目依赖，避免不同项目间的包冲突。常见工具有：`venv`（官方推荐）、`virtualenv`、`conda`、`pipenv`。

### 1. venv（官方推荐，Python3.3+内置）
创建虚拟环境：
```
python -m venv venv_name
```
激活虚拟环境：
- Windows:
  ```
  .\venv_name\Scripts\activate
  ```
- macOS/Linux:
  ```
  source venv_name/bin/activate
  ```
退出虚拟环境：
```
deactivate
```
安装依赖
```
pip install -r requirements.txt
```

导出依赖
```
pip freeze > requirements.txt

```


### 2. virtualenv（兼容多版本 Python）
安装：
```
pip install virtualenv
```
创建虚拟环境：
```
virtualenv venv_name
```
激活/退出同 venv。

### 3. conda（Anaconda/Miniconda 环境管理器）
创建环境：
```
conda create -n env_name python=3.11
```
激活环境：
```
conda activate env_name
```
退出环境：
```
conda deactivate
```

### 4. pipenv（官方推荐的依赖和环境管理工具）
结合 pip 和 virtualenv，自动管理依赖和虚拟环境。

---

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
