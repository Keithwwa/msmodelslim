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

from typing import Any, Callable, Dict, Literal, Type

from abc import abstractmethod
from pydantic import BaseModel, Field
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.format.interface import ExportContext, IFormat

ModuleHandler = Callable[[str, nn.Module], None]


class QuantFormatConfig(BaseModel):
    """Base config for quantized export formats; subclasses are distinguished by ``type``."""

    type: Literal["_auto_save"] = "_auto_save"
    save_directory: str = Field(default=".", exclude=True)

    def set_save_directory(self, save_directory: str) -> None:
        """Set the output directory for export artifacts."""
        self.save_directory = str(save_directory)


class QuantFormatBase(IFormat):
    """Base class: traversal + handler map; subclasses decide how to open IO writers via ``_open_writers``."""

    def __init__(
        self,
        config: QuantFormatConfig,
        ctx: ExportContext,
        json_writer_infra: Any | None = None,
        safetensors_writer_infra: Any | None = None,
    ) -> None:
        self.config = config
        self.ctx = ctx
        self._json_writer_infra = json_writer_infra
        self._safetensors_writer_infra = safetensors_writer_infra
        self.json_writer: Any = None
        self.safetensors_writer: Any = None
        self.processed_modules: set[nn.Module] = set()
        self._module_handler_map = self.build_module_handler_map()

    def process_module_tensors(self, prefix: str, module: nn.Module) -> None:
        for name, sub_module in module.named_modules(memo=self.processed_modules, prefix=prefix):
            self._process_module_maybe_wrapper_ir(name, sub_module)

    def finalize_export(self, model: nn.Module) -> None:
        """Close open export writers after all modules have been processed.

        Safetensors and JSON writers opened during ``prepare_export`` are flushed
        and released here. Subclasses should call ``super().finalize_export(model)``
        before writing format-specific metadata (for example ``config.json``).

        :param model: The quantized model whose tensors were exported.
        """
        if self.safetensors_writer is not None:
            self.safetensors_writer.close()
            self.safetensors_writer = None
        if self.json_writer is not None:
            self.json_writer.close()
            self.json_writer = None

    def merge_ranks(self) -> None:
        pass

    @abstractmethod
    def build_module_handler_map(self) -> Dict[Type[nn.Module], ModuleHandler]:
        """子类实现：模块类型到落盘 handler 的映射表。"""
        pass

    def _process_module_maybe_wrapper_ir(self, prefix: str, module: nn.Module) -> None:
        if module in self.processed_modules:
            return
        if isinstance(module, qir.WrapperIR):
            wrapped = module.wrapped_module
            if not module.is_atomic():
                self._process_module_maybe_wrapper_ir(prefix, wrapped)
            self._write_quantized_leaf(prefix, module)
        else:
            self._write_quantized_leaf(prefix, module)
        self.processed_modules.add(module)

    def _write_quantized_leaf(self, prefix: str, module: nn.Module) -> None:
        if not self._module_handler_map:
            return
        handler = self._module_handler_map.get(type(module))
        if handler is not None:
            handler(prefix, module)
        elif nn.Module in self._module_handler_map:
            self._module_handler_map[nn.Module](prefix, module)


__all__ = ["QuantFormatConfig", "QuantFormatBase", "ModuleHandler"]
