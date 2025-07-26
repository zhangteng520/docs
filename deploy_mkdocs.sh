#!/bin/bash
# 自动构建 mkdocs 并部署到 GitHub Pages

set -e

# 1. 构建 mkdocs
mkdocs build

echo "[INFO] MkDocs build 完成。"

# 2. 清空目标目录（保留 .git）
target_dir="zhangteng520.github.io/zhangteng520.github.io"
site_dir="site"

if [ -d "$target_dir" ]; then
    find "$target_dir" -mindepth 1 ! -name ".git" -exec rm -rf {} +
else
    mkdir -p "$target_dir"
fi

echo "[INFO] 目标目录已清空。"

# 3. 拷贝 site 下所有内容到目标目录
cp -r $site_dir/* $target_dir/

echo "[INFO] 文件已复制到 $target_dir。"

# 4. Git 操作
cd $target_dir
git add . -f
git commit -m "auto: update site $(date '+%Y-%m-%d %H:%M:%S')"
git push -f origin main
if [ $? -ne 0 ]; then
    echo "[ERROR] Git push 失败，请检查网络连接或权限设置。"
    exit 1
fi

echo "[INFO] 已推送到远端仓库。"
