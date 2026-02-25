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
from typing import Dict

from msmodelslim.core.context.base import BaseContext, BaseNamespace


class Namespace(BaseNamespace):
    def __init__(self) -> None:
        super().__init__()


class LocalDictContext(BaseContext):
    def __init__(self) -> None:
        self._namespaces: Dict[str, Namespace] = {}

    def create_namespace(self, key: str) -> Namespace:
        if key not in self._namespaces:
            self._namespaces[key] = Namespace()
        return self._namespaces[key]

