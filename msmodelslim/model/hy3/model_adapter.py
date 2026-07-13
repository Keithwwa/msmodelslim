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

import os
import time
from collections import defaultdict
from typing import Any, Generator, List, Optional

import torch
from torch import distributed as dist
from torch import nn
from safetensors import safe_open

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.security import json_safe_load, get_valid_read_path, MAX_READ_FILE_SIZE_32G
from ..common.layer_wise_forward import (
    transformers_generated_forward_func,
)
from ..default.model_adapter import DefaultModelAdapter
from ..interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
)
from transformers.models.hy_v3.modeling_hy_v3 import HYV3MoE
from .moe_utils import UnstackedHy3MoE, convert_hy3_moe_to_unstacked


def _promote_expert_bias_to_parameters(model: nn.Module) -> None:
    """Promote HF ``e_score_correction_bias`` buffer to Parameter before MoE unstack renames it."""
    for _, module in model.named_modules():
        if not hasattr(module, "e_score_correction_bias"):
            continue
        existing = getattr(module, "e_score_correction_bias")
        if existing is None or isinstance(existing, nn.Parameter):
            continue
        if not isinstance(existing, torch.Tensor):
            continue
        if "e_score_correction_bias" in module._buffers:
            del module._buffers["e_score_correction_bias"]
        module.register_parameter(
            "e_score_correction_bias",
            nn.Parameter(existing.detach().clone(), requires_grad=False),
        )


_MTP_MODULE_NAMES = ("enorm", "hnorm", "eh_proj", "final_layernorm")


def _attach_mtp_modules(model: nn.Module, config) -> None:
    mtp_layer_idx = config.num_hidden_layers - 1
    mtp_decoder = model.model.layers[mtp_layer_idx]
    hidden_size = config.hidden_size
    rms_norm_eps = config.rms_norm_eps

    mtp_decoder.enorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
    mtp_decoder.hnorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
    mtp_decoder.eh_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
    mtp_decoder.final_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    get_logger().info(
        "Attached MTP modules (enorm, hnorm, eh_proj, final_layernorm) to layer %d",
        mtp_layer_idx,
    )


def _load_mtp_weights(model: nn.Module, config, model_path: str) -> None:
    mtp_layer_idx = config.num_hidden_layers - 1
    mtp_decoder = model.model.layers[mtp_layer_idx]
    prefix = f"model.layers.{mtp_layer_idx}"

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    model_index = json_safe_load(index_path)
    weight_map = model_index.get("weight_map")

    files_to_keys = defaultdict(list)
    loaded_count = 0
    for module_name in _MTP_MODULE_NAMES:
        key = f"{prefix}.{module_name}.weight"
        if key in weight_map:
            files_to_keys[weight_map[key]].append((module_name, key))
            loaded_count += 1
        else:
            get_logger().debug("MTP weight key not found in weight map: %s", key)

    for file_name, key_pairs in files_to_keys.items():
        file_path = os.path.join(model_path, file_name)
        file_path = get_valid_read_path(
            file_path,
            extensions=".safetensors",
            size_max=MAX_READ_FILE_SIZE_32G,
        )

        with safe_open(file_path, framework="pt", device="cpu") as f:
            for module_name, key in key_pairs:
                tensor = f.get_tensor(key)
                module = getattr(mtp_decoder, module_name)
                module.weight = nn.Parameter(tensor, requires_grad=False)
                get_logger().debug("Loaded %s from %s", key, file_name)

    get_logger().info(
        "Loaded %d MTP modules (%s) for layer %d",
        loaded_count,
        ", ".join(_MTP_MODULE_NAMES),
        mtp_layer_idx,
    )


def _resolve_unstack_device() -> Optional[str]:
    """Prefer NPU for MoE weight copy; fall back to CPU when NPU is unavailable."""
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return f"npu:{torch.npu.current_device()}"
    except (ImportError, AttributeError):
        pass
    return None


@logger_setter()
class Hy3ModelAdapter(  # pylint: disable=too-many-ancestors
    DefaultModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
):
    """
    Hy3 (HF Transformers) 模型适配器。

    与 Qwen3 MoE 适配器对齐的调度能力（逐层 visit、标准 Transformers forward、KV 开关），
    支持 checkpoint 中 ``model.layers.{num_hidden_layers}`` 的 MTP 模块加载、
    FA3 激活量化与 AscendV1 保存。
    """

    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return "hy3"

    def get_hidden_dim(self):
        return self.config.hidden_size

    def _has_mtp(self) -> bool:
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            return False
        weight_map = json_safe_load(index_path).get("weight_map", {})
        for layer_idx in (self.config.num_hidden_layers - 1, self.config.num_hidden_layers):
            if f"model.layers.{layer_idx}.enorm.weight" in weight_map:
                return True
        return False

    def _prepare_mtp_config(self) -> None:
        if not self._has_mtp():
            return
        original_layers = self.config.num_hidden_layers
        self.config.num_hidden_layers += 1
        target_len = self.config.num_hidden_layers
        if getattr(self.config, "mlp_layer_types", None) is None:
            first_k = getattr(self.config, "first_k_dense_replace", 1)
            self.config.mlp_layer_types = ["dense"] * first_k + ["sparse"] * (target_len - first_k)
        elif len(self.config.mlp_layer_types) < target_len:
            extra = target_len - len(self.config.mlp_layer_types)
            self.config.mlp_layer_types.extend(["sparse"] * extra)

        get_logger().info(
            "MTP enabled: expanded num_hidden_layers from %d to %d (MTP layer at index %d)",
            original_layers,
            self.config.num_hidden_layers,
            original_layers,
        )

    def _ensure_layer_moe_unstacked(self, layer: nn.Module, layer_idx: int) -> None:
        """Unstack one MoE layer on NPU (if available) right before layer-wise quant."""
        if isinstance(layer.mlp, UnstackedHy3MoE):
            return
        if not isinstance(layer.mlp, HYV3MoE):
            return

        device = _resolve_unstack_device()
        t0 = time.time()
        get_logger().info(
            "MoE unstack layer %d on %s ...",
            layer_idx,
            device or "cpu",
        )
        mlp = layer.mlp.to(device) if device else layer.mlp
        layer.mlp = convert_hy3_moe_to_unstacked(mlp, self.config)
        get_logger().info(
            "MoE unstack layer %d done in %.1fs on %s",
            layer_idx,
            time.time() - t0,
            device or "cpu",
        )

    def load_model_with_mtp(self, device: DeviceType) -> nn.Module:
        self._prepare_mtp_config()
        model = self._load_model(device)

        _promote_expert_bias_to_parameters(model)
        if self._has_mtp():
            _attach_mtp_modules(model, self.config)
            _load_mtp_weights(model, self.config, str(self.model_path))

        unstack_device = _resolve_unstack_device()
        get_logger().info(
            "Model weights loaded; MoE unstack deferred to layer-wise visit on %s",
            unstack_device or "cpu",
        )
        return model

    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self.load_model_with_mtp(device)

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def handle_dataset_by_batch(
        self,
        dataset: Any,
        batch_size: int,
        device: DeviceType = DeviceType.NPU,
    ) -> List[Any]:
        return self._get_batch_tokenized_data(
            calib_list=dataset,
            batch_size=batch_size,
            device=device,
        )

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self.load_model_with_mtp(device)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        if dist.is_initialized():
            dist.barrier()

        for layer_idx, layer in enumerate(model.model.layers):
            name = f"model.layers.{layer_idx}"
            self._ensure_layer_moe_unstacked(layer, layer_idx)
            yield ProcessRequest(name, layer, tuple(), {})

    def generate_model_forward(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> Generator[ProcessRequest, Any, None]:
        return transformers_generated_forward_func(model, inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return self._enable_kv_cache(model, need_kv_cache)
