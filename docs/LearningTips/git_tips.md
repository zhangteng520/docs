# Git 问题解决与命令指南

## 1. `.gitignore` 文件详解
**用户提问**：gitnore是什么  
**解答**：
`.gitignore` 是 Git 的配置文件，用于指定哪些文件/目录不应被版本控制

### 核心作用
- 过滤临时文件：`.DS_Store`, `Thumbs.db`
- 忽略编译产物：`*.class`, `/dist/`
- 保护敏感信息：`.env`, `*.key`
- 排除依赖目录：`node_modules/`, `venv/`

### 使用方法
```bash
# 创建文件
touch .gitignore

# 示例规则
*.log
/temp/
!important.log  # 例外规则
```
> **注意事项**
```
    # 已跟踪文件需要移除缓存
    git rm --cached config.ini
```


## 配置全局用户信息（独立于GitHub用户名）
```
git config --global user.name "您的姓名"
git config --global user.email "your-email@example.com"
```
## 仓库操作
```
git init  # 初始化仓库
git clone https://...  # 克隆远程仓库
git reomte add origin <https> #关联远端仓库
```

## 分支管理
```
git branch -a  # 查看所有分支
git branch -m <新分支名称> #修改当前分支名称
git checkout -b new-feature  # 创建并切换分支
git merge feature-branch  # 合并分支
```

## 提高工作流
```
git status  # 查看状态
git add .  # 添加所有修改
git commit -m "Message"  # 创建提交
git pull origin main #拉去最新的仓库
git push -u origin main  # 推送到远程
git push -f origin main #强制推送
```

## 历史查看
```
git log --oneline --graph  # 简洁图形化历史
git diff HEAD~1  # 与前一提交比较
```

## 撤销操作
```
git restore file.txt  # 丢弃工作区修改
git reset --soft HEAD~1  # 撤销提交但保留修改
git revert commit-id  # 创建撤销提交
git rm -r --cached path #删除远端文件而不删除本地文件
```