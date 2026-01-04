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

from abc import ABC, abstractmethod
from typing import Optional
import torch.nn as nn


class AscendV1SaveInterface(ABC):
    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        """
        导出件后处理
        @param model: 量化模型
        @param save_directory: 包含导出件（如config.json，quant_model_description.json等）的量化模型存储路径
        """
        pass

    def ascendv1_save_module_preprocess(self, prefix: str, module: nn.Module, model: nn.Module) -> Optional[nn.Module]:
        pass
        