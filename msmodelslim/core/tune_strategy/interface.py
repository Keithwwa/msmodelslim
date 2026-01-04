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
from typing import Generator, List, Optional

from pydantic import BaseModel, Field

from msmodelslim.core.practice import PracticeConfig
from msmodelslim.core.const import DeviceType
from msmodelslim.model import IModel
from msmodelslim.utils.plugin import TypedConfig

TUNING_STRATEGY_CONFIG_PLUGIN_PATH = "msmodelslim.strategy_config.plugins"


class EvaluateAccuracy(BaseModel):
    dataset: str
    accuracy: float


class AccuracyExpectation(BaseModel):
    dataset: str
    target: float
    tolerance: float


class EvaluateResult(BaseModel):
    accuracies: List[EvaluateAccuracy] = Field(default_factory=list)
    expectations: List[AccuracyExpectation] = Field(default_factory=list)
    is_satisfied: bool


@TypedConfig.plugin_entry(entry_point_group=TUNING_STRATEGY_CONFIG_PLUGIN_PATH)
class StrategyConfig(TypedConfig):
    type: TypedConfig.TypeField


class ITuningStrategy(ABC):
    @abstractmethod
    def generate_practice(self,
                          model: IModel,
                          device: DeviceType = DeviceType.NPU,
                          ) -> Generator[PracticeConfig, Optional[EvaluateResult], None]:
        ...


class ITuningStrategyFactory(ABC):
    @abstractmethod
    def create_strategy(self, strategy_config: StrategyConfig) -> ITuningStrategy:
        ...
