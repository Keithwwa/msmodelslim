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
from typing import List, Dict, Any, Optional

from msmodelslim.core.const import DeviceType
from msmodelslim.core.quant_service.dataset_loader_infra import DatasetLoaderInfra
from msmodelslim.core.runner.pipeline_interface import PipelineInterface


class BaseAnalysisService(ABC):
    """Base class for model analysis services"""

    def __init__(self, dataset_loader: DatasetLoaderInfra):
        self.dataset_loader = dataset_loader

    @abstractmethod
    def analyze(self,
                model_adapter: PipelineInterface,
                patterns: List[str],
                analysis_config: Optional[Dict[str, Any]] = None,
                device: DeviceType = DeviceType.NPU):
        """
        Analyze model layers based on given patterns
        
        Args:
            model_adapter: The model to analyze
            patterns: List of layer name patterns to analyze (e.g., ['*linear*', 'attention.*'])
            analysis_config: Configuration for analysis method
            device: device

        """
        raise NotImplementedError

    @abstractmethod
    def export_results(self, result: Any, top_k: int = 10):
        """
        export analysis results in service-specific format
        
        Args:
            result: AnalysisResult containing layer scores and metadata
            top_k: Number of top layers to display
        """
        raise NotImplementedError
