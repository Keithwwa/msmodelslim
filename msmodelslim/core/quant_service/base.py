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
from abc import abstractmethod
from pathlib import Path
from typing import Optional, List

from msmodelslim.core.quant_service.dataset_loader_infra import DatasetLoaderInfra
from msmodelslim.core.const import DeviceType
from .interface import IQuantService, BaseQuantConfig
from msmodelslim.model import IModel


class BaseQuantService(IQuantService):
    backend_name: str = "Unknown"

    def __init__(self, dataset_loader: DatasetLoaderInfra):
        self.dataset_loader = dataset_loader

    @abstractmethod
    def quantize(self,
                 quant_config: BaseQuantConfig,
                 model_adapter: IModel,
                 save_path: Optional[Path] = None,
                 device: DeviceType = DeviceType.NPU,
                 device_indices: Optional[List[int]] = None
                 ) -> None:
        ...
