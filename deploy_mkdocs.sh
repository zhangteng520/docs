#!/bin/bash
# 自动构建 mkdocs 并部署到 GitHub Pages

set -e

# 1. 构建 mkdocs
pipenv run mkdocs build

echo "[INFO] MkDocs build 完成。"

# 2. 清空目标目录（保留 .git）

target_dir="../website/zhangteng520.github.io"
site_dir="site"
git_repo="https://github.com/zhangteng520/zhangteng520.github.io.git"


# 只保留 .git 文件夹，其余全部删除
if [ -d "$target_dir" ]; then
    for item in "$target_dir"/* "$target_dir"/.*; do
        name=$(basename "$item")
        if [ "$name" != "." ] && [ "$name" != ".." ] && [ "$name" != ".git" ]; then
            rm -rf "$item"
        fi
    done
else
    mkdir -p "$target_dir"
fi

echo "[INFO] 目标目录已清空。"

# 3. 拷贝 site 下所有内容到目标目录
cp -r $site_dir/* $target_dir/

echo "[INFO] 文件已复制到 $target_dir。"


# 4. Git 操作
cd $target_dir
git rev-parse --is-inside-work-tree
pwd

git add . -f
git commit -m "auto: update site $(date '+%Y-%m-%d %H:%M:%S')"
git push -f $git_repo main
if [ $? -ne 0 ]; then
    echo "[ERROR] Git push 失败，请检查网络连接或权限设置。"
    exit 1
fi

echo "[INFO] 已推送到远端仓库。"
