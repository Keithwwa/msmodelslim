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

from pydantic import BaseModel, Field
from typing_extensions import Self

from msmodelslim.core.const import RunnerType
from msmodelslim.processor.base import AutoProcessorConfigList
from .save.saver import AutoSaverConfigList
from ..interface import BaseQuantConfig


class ModelslimV1ServiceConfig(BaseModel):
    runner: RunnerType = RunnerType.AUTO
    process: AutoProcessorConfigList = Field(default_factory=list)
    save: AutoSaverConfigList = Field(default_factory=list)
    dataset: str = Field(default='mix_calib.jsonl')


class ModelslimV1QuantConfig(BaseQuantConfig):
    spec: ModelslimV1ServiceConfig  # quantization config specification

    @classmethod
    def from_base(cls, quant_config: BaseQuantConfig) -> Self:
        return cls(
            apiversion=quant_config.apiversion,
            spec=load_specific_config(quant_config.spec),
        )


def load_specific_config(yaml_spec: object) -> ModelslimV1ServiceConfig:
    """Load specific configuration from YAML spec"""
    if isinstance(yaml_spec, ModelslimV1ServiceConfig):
        return yaml_spec
    if not isinstance(yaml_spec, dict):
        raise ValueError("task spec must be dict")
    return ModelslimV1ServiceConfig.model_validate(yaml_spec)
