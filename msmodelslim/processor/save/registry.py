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

# pylint: disable=duplicate-code

from typing import Any, Literal, Type, get_args, get_origin

from msmodelslim.format.base import QuantFormatConfig
from msmodelslim.format.interface import IFormat
from msmodelslim.format.interface import ExportContext
from msmodelslim.infra.io.default_json_export_writer_creator import DefaultJsonExportWriterCreator
from msmodelslim.infra.io.default_safetensors_export_writer_creator import DefaultSafetensorsExportWriterCreator

_REGISTRY: dict[Type[QuantFormatConfig], Type[IFormat]] = {}


def _ensure_builtin_quant_formats_registered() -> None:
    import importlib

    importlib.import_module("msmodelslim.format.registry")


def register_quant_format(config_cls: Type[QuantFormatConfig], format_cls: Type[IFormat]) -> None:
    """注册：YAML ``type`` 对应的 Config 类 → :class:`IFormat` 实现类。"""
    _REGISTRY[config_cls] = format_cls


def create_quant_format(
    processor_config: Any,
    ctx: ExportContext,
    json_writer_infra: DefaultJsonExportWriterCreator | None = None,
    safetensors_writer_infra: DefaultSafetensorsExportWriterCreator | None = None,
) -> IFormat:
    """构造格式实例；writer 在 ``pre_run`` → ``prepare_export`` 中创建。"""
    _ensure_builtin_quant_formats_registered()
    format_config = processor_config.format
    format_cls = _REGISTRY.get(type(format_config))
    if format_cls is None:
        names = ", ".join(sorted(c.__name__ for c in _REGISTRY))
        raise TypeError(
            f"No quant save format registered for config type {type(format_config).__name__}. "
            f"Registered config types: {names or '(none)'}"
        )
    return format_cls(
        format_config,
        ctx,
        json_writer_infra or DefaultJsonExportWriterCreator(),
        safetensors_writer_infra or DefaultSafetensorsExportWriterCreator(),
    )


def _accepted_values(field: Any) -> set[Any]:
    annotation = getattr(field, "annotation", None)
    if get_origin(annotation) is Literal:
        return set(get_args(annotation))
    return {field.default}


def parse_save_config(data: dict[str, Any]) -> QuantFormatConfig:
    _ensure_builtin_quant_formats_registered()
    cfg_type = data.get("type")
    for config_cls in _REGISTRY:
        fields = getattr(config_cls, "model_fields", {})
        type_field = fields.get("type")
        if type_field is None:
            continue
        accepted_formats = _accepted_values(type_field)
        if cfg_type in accepted_formats:
            return config_cls.model_validate(data)
    names = ", ".join(sorted(c.__name__ for c in _REGISTRY))
    raise TypeError(
        f"No quant save config parser registered for type={cfg_type!r}. Registered config classes: {names or '(none)'}"
    )


__all__ = [
    "register_quant_format",
    "create_quant_format",
    "parse_save_config",
]
