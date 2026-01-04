#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
"""
msmodelslim.utils.yaml_database 模块的单元测试
"""
from pathlib import Path
import yaml
import pytest

from msmodelslim.utils.exception import (
    SchemaValidateError,
    SecurityError,
    UnsupportedError,
)
from msmodelslim.utils.yaml_database import YamlDatabase


# 辅助函数：创建测试用YAML文件
def create_test_yaml(tmp_path: Path, filename: str, content: dict):
    file_path = tmp_path / f"{filename}.yaml"
    with open(file_path, "w") as f:
        yaml.safe_dump(content, f)


# ------------------------------ 测试迭代器方法 ------------------------------
def test_yaml_database_iter(tmp_path: Path):
    """测试迭代器功能"""
    create_test_yaml(tmp_path, "test1", {"key": "value1"})
    create_test_yaml(tmp_path, "test2", {"key": "value2"})
    (tmp_path / "test3.txt").write_text("not a yaml file")

    db = YamlDatabase(config_dir=tmp_path)

    items = list(db)

    assert len(items) == 2
    assert "test1" in items
    assert "test2" in items
    assert "test3" not in items


def test_yaml_database_iter_empty(tmp_path: Path):
    """测试空目录的迭代器"""
    db = YamlDatabase(config_dir=tmp_path)
    assert len(list(db)) == 0


def test_yaml_database_getitem_non_existing(tmp_path: Path):
    """测试获取不存在的键"""
    db = YamlDatabase(config_dir=tmp_path)

    with pytest.raises(UnsupportedError) as exc_info:
        _ = db["nonexistent"]

    assert "yaml database key nonexistent not found" in str(exc_info.value)
    assert exc_info.value.action == "Please check the yaml database"


def test_yaml_database_getitem_invalid_key_type(tmp_path: Path):
    """测试使用非字符串作为键"""
    db = YamlDatabase(config_dir=tmp_path)

    with pytest.raises(SchemaValidateError) as exc_info:
        _ = db[123]
    assert exc_info.value.action == "Please make sure the key is a string"


def test_yaml_database_setitem_read_only(tmp_path: Path):
    """测试在只读模式下尝试设置键值"""
    db = YamlDatabase(config_dir=tmp_path, read_only=True)

    with pytest.raises(SecurityError) as exc_info:
        db["new_key"] = {"new": "value"}

    assert f"yaml database {tmp_path} is read-only" in str(exc_info.value)
    assert exc_info.value.action == "Writing operation is forbidden"


def test_yaml_database_setitem_invalid_key_type(tmp_path: Path):
    """测试使用非字符串作为设置键"""
    db = YamlDatabase(config_dir=tmp_path, read_only=False)

    with pytest.raises(SchemaValidateError) as exc_info:
        db[123] = {"key": "value"}  # 使用整数作为键

    assert exc_info.value.action == "Please make sure the key is a string"


# ------------------------------ 测试__contains__方法 ------------------------------
def test_yaml_database_contains_existing(tmp_path: Path):
    """测试检查存在的键"""
    create_test_yaml(tmp_path, "existing", {"key": "value"})

    db = YamlDatabase(config_dir=tmp_path)

    assert "existing" in db
    assert "nonexistent" not in db


# ------------------------------ 测试values()方法 ------------------------------
def test_yaml_database_values(tmp_path: Path):
    """测试获取所有值"""
    create_test_yaml(tmp_path, "test1", {})
    create_test_yaml(tmp_path, "test2", {})

    db = YamlDatabase(config_dir=tmp_path)
    values = list(db.values())

    assert len(values) == 2
    assert {} in values
    assert {} in values
