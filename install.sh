#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

# 清理构建缓存，确保重新构建包
echo "cleaning build cache to ensure fresh build"
rm -rf build/ dist/ *.egg-info/

# 先卸载再安装，确保完全清理已移除的文件
echo "uninstalling existing msmodelslim package"
pip uninstall msmodelslim -y

# 重新安装包（不使用缓存）
echo "installing msmodelslim package without cache"
umask 027 && pip install . --no-cache-dir
