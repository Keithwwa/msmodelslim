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

from typing import Any, Dict, Literal, Type, Callable

from torch import nn

import msmodelslim.ir as qir
from msmodelslim import logger
from msmodelslim.format.base import QuantFormatBase, QuantFormatConfig
from msmodelslim.format.interface import ExportContext

_HF_COPY_SUFFIXES = (".json", ".py", ".txt", ".jinja")
_HF_SKIP_SUFFIXES = ("index.json",)
_SAVE_PREFIX = "model"


class CompressedTensorsQuantFormatConfig(QuantFormatConfig):
    type: Literal["compressed_tensors"] = "compressed_tensors"
    part_file_size: int = 4


class CompressedTensorsQuantFormat(QuantFormatBase):
    """compressed-tensors 导出：层间写 safetensors，收尾从量化模型反向推导并写 ``config.json``。"""

    def __init__(
        self,
        config: QuantFormatConfig,
        ctx: ExportContext,
        json_writer_infra: Any | None = None,
        safetensors_writer_infra: Any | None = None,
    ) -> None:
        super().__init__(config, ctx, json_writer_infra, safetensors_writer_infra)
        self.json_writer = None
        self.safetensors_writer = None

    def prepare_export(self) -> None:
        if self._safetensors_writer_infra is None:
            logger.warning("Safetensors writer infra is not configured; compressed-tensors weight export is skipped.")
            return

        part_file_size = int(getattr(self.config, "part_file_size", 4))
        save_directory = str(self.ctx.save_directory)
        logger.info(
            "Preparing compressed-tensors export: save_directory=%s, part_file_size=%s, prefix=%s",
            save_directory,
            part_file_size,
            _SAVE_PREFIX,
        )
        self.safetensors_writer = self._safetensors_writer_infra.create_safetensors_writer(
            part_file_size,
            save_directory,
            _SAVE_PREFIX,
        )
        self.processed_modules = set()
        self._module_handler_map = self.build_module_handler_map()

    def support_distributed(self) -> bool:
        return False

    def build_module_handler_map(self) -> Dict[Type[nn.Module], Callable[[str, nn.Module], None]]:
        return {
            qir.W8A8StaticFakeQuantLinear: self.on_w8a8_static,
            qir.W8A8DynamicPerChannelFakeQuantLinear: self.on_w8a8_dynamic_per_channel,
            nn.Linear: self.on_float_linear,
            nn.Module: self.on_float_module,
        }

    def on_w8a8_static(self, name: str, module: nn.Module) -> None:
        pass

    def on_w8a8_dynamic_per_channel(self, name: str, module: nn.Module) -> None:
        pass

    def on_float_linear(self, name: str, module: nn.Module) -> None:
        pass

    def on_float_module(self, name: str, module: nn.Module) -> None:
        pass

    def finalize_export(self, model: nn.Module) -> None:
        super().finalize_export(model)
        if self.json_writer is not None:
            self.json_writer.close()
            self.json_writer = None
        if self.safetensors_writer is not None:
            self.safetensors_writer.close()
            self.safetensors_writer = None


__all__ = [
    "CompressedTensorsQuantFormatConfig",
    "CompressedTensorsQuantFormat",
]
