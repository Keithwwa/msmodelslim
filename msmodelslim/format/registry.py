#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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

from __future__ import annotations

from typing import Any, ClassVar, List, Tuple, Type, Union

from pydantic import Field, TypeAdapter
from typing_extensions import Annotated

from msmodelslim.format.ascendV1_format.ascendV1 import AscendV1QuantFormatConfig
from msmodelslim.format.compressed_tensors_format.compressed_tensors import (
    CompressedTensorsQuantFormat,
    CompressedTensorsQuantFormatConfig,
)
from msmodelslim.format.interface import IFormat
from msmodelslim.format.base import QuantFormatConfig
from msmodelslim.format.mindie_format.mindie import MindIEQuantFormatConfig
from msmodelslim.processor.save.registry import register_quant_format

QuantFormatConfigUnion = Annotated[
    Union[
        CompressedTensorsQuantFormatConfig,
        AscendV1QuantFormatConfig,
        MindIEQuantFormatConfig,
    ],
    Field(discriminator="type"),
]

QuantFormatConfigList = List[QuantFormatConfigUnion]

_format_config_adapter = TypeAdapter(QuantFormatConfigUnion)


def parse_format_config(data: dict[str, Any]) -> QuantFormatConfig:
    """Parse a format config dict via Pydantic tagged-union dispatch on ``type``."""
    return _format_config_adapter.validate_python(data)


class QuantFormatFactory:
    #: 仅注册走新 ``QuantSaveProcessor`` + ``IFormat`` 的格式；AscendV1/MindIE 由旧 Saver 处理。
    BUILTIN_BINDINGS: ClassVar[Tuple[Tuple[Type[QuantFormatConfig], Type[IFormat]], ...]] = (
        (CompressedTensorsQuantFormatConfig, CompressedTensorsQuantFormat),
    )

    @classmethod
    def install(cls) -> None:
        for config_cls, format_cls in cls.BUILTIN_BINDINGS:
            register_quant_format(config_cls, format_cls)


QuantFormatFactory.install()

__all__ = [
    "QuantFormatFactory",
    "QuantFormatConfigUnion",
    "QuantFormatConfigList",
    "parse_format_config",
]
