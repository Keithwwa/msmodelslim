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

import torch
from collections import OrderedDict
from typing import Dict, List, Union, Type, Pattern, Tuple, Any, Callable


from msmodelslim.processor.flat_quant.flat_quant_utils.structure_pair import StructurePair
from msmodelslim.processor.flat_quant.flat_quant_utils.flat_fake_quant_linear import ForwardMode
from msmodelslim.processor.flat_quant.flat_quant_utils.utils import remove_after_substring


class FlatQuantManager:
    """管理 FlatQuant 算法的全流程操作：结构对注册、变换应用、模式切换与回退。"""

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any] = None) -> None:
        """初始化 FlatQuant 管理器，设置模型、配置和结构对注册表。"""
        self.model = model
        self.config = config or getattr(model, 'config', None)
        self._structure_pair_registry: Dict[str, List[StructurePair]] = {}
        self.layer_structure_pairs: Dict[str, List[StructurePair]] = {}

    def register_structure_pair(self, pair: StructurePair) -> None:
        """注册一个结构对实例（如 MLP 与 Linear 的配对），避免重复注册。"""
        if not isinstance(pair, StructurePair):
            raise TypeError(f"pair 必须是 StructurePair 类型，但得到的类型是：{type(pair)}")

        class_name = pair.__class__.__name__
        if class_name not in self._structure_pair_registry:
            self._structure_pair_registry[class_name] = []
        pair_list = self._structure_pair_registry[class_name]

        if not any(str(pair) == str(obj) for obj in pair_list):
            pair_list.append(pair)

    def register_pairs(self, structure_configs: List[Dict[str, Any]]) -> None:
        """根据结构配置列表分析模型结构，并注册所有结构对实例。"""
        for name, _ in self.model.named_modules():
            for config in structure_configs:
                if config["pattern"] in name:
                    linears = []
                    clean_name = remove_after_substring(name, config["pattern"])
                    for linear_name in config["linear_names"]:
                        linears.append(clean_name.replace(config["pattern"], linear_name))
                    prefix_name = '.'.join(clean_name.split('.')[:3])
                    self.register_structure_pair(
                        config["pair_class"](self.config, clean_name, linears, prefix_name, self.model)
                    )

        pairs_dict = self._structure_pair_registry
        pairs = []
        support_structure_pairs = StructurePair.support_structure_pairs
        num = max([len(pairs_dict[pair_type.__name__]) for pair_type in support_structure_pairs])
        for i in range(num):
            for pair_type in support_structure_pairs:
                if i < len(pairs_dict[pair_type.__name__]):
                    pairs.append(pairs_dict[pair_type.__name__][i])

        self.layer_structure_pairs = {}
        for pair in pairs:
            layer_key = pair.prefix_name
            if layer_key not in self.layer_structure_pairs:
                self.layer_structure_pairs[layer_key] = []
            self.layer_structure_pairs[layer_key].append(pair)

    def wrap_linear(self, prefix: str = "", device: Union[str, torch.device] = None) -> None:
        """替换指定前缀下的线性层，应用 FlatQuant 的变换逻辑。"""
        with torch.device(device=device):
            self._call_method_on_pairs(
                prefix=prefix, 
                method_name="wrap_linear"
            )

    def rollback_trans(self, prefix: str = "", pair_name: str = "") -> None:
        """回退已应用的变换矩阵（trans），恢复原始状态。"""
        self._call_method_on_pairs(
            prefix=prefix, 
            method_name="rollback_trans", 
            pair_name=pair_name
        )

    def change_mode(self, mod: ForwardMode, prefix: str = "") -> None:
        """切换所有结构对的前向传播模式（如训练/推理/量化模式）。"""
        self._call_method_on_pairs(
            prefix=prefix, 
            method_name="change_mode", 
            mod=mod
        )

    def _call_method_on_pairs(
        self,
        prefix: str,
        method_name: str,
        *args,
        **kwargs
    ) -> None:
        """遍历指定前缀下的所有结构对，统一调用指定方法。"""
        if prefix not in self.layer_structure_pairs:
            return
        for pair in self.layer_structure_pairs[prefix]:
            method = getattr(pair, method_name)
            method(*args, **kwargs)

    def match_pair(self, proj_name: str) -> None:
        """（预留）根据模块名称匹配对应的结构对。"""
        pass
