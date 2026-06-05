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

import unittest
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

import torch
from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.model.deepseek_v4.model_adapter import DeepSeekV4ModelAdapter


class DummyConfig:
    def __init__(self):
        self.num_hidden_layers = 2
        self.n_routed_experts = 2
        self.n_shared_experts = 1
        self.dim = 128
        self.q_lora_rank = 4


class DummyDecoderLayerV4(nn.Module):
    def __init__(self, layer_id=0, args=None):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.linear = nn.Linear(1, 1)
        # attributes used by tests / adapter checks
        self.enorm = None
        # provide a get_submodule helper so tests don't need to monkeypatch it

    def get_submodule(self, name):
        value = getattr(self, name)
        if value is None:
            raise AttributeError(f"Module '{name}' not found")
        return value

    def forward(self, x):
        return self.linear(x)


class DummyModelV4(nn.Module):
    def __init__(self, config=None, num_layers=None):
        super().__init__()
        if isinstance(config, DummyConfig):
            self.config = config
            num_layers = config.num_hidden_layers if num_layers is None else num_layers
        else:
            num_layers = config if isinstance(config, int) else 1
            self.config = None

        self.layers = nn.ModuleList([DummyDecoderLayerV4(layer_id=i, args=self.config) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestDeepSeekV4ModelAdapter(unittest.TestCase):
    def setUp(self):
        self.model_path = Path('.')
        self.model_type = 'DeepSeek-V4'
        self.dummy_config = DummyConfig()
        self.adapter_patcher = patch.object(DeepSeekV4ModelAdapter, '__init__', lambda x, model_path, model_type: None)

    def create_adapter(self, **kwargs):
        with self.adapter_patcher:
            adapter = DeepSeekV4ModelAdapter(model_path=self.model_path, model_type=self.model_type)
            adapter.config = self.dummy_config
            adapter.model_path = self.model_path
            for key, value in kwargs.items():
                setattr(adapter, key, value)
            return adapter

    # Verify the adapter reports the expected model pedigree string.
    def test_model_pedigree_returns_deepseek_v4_when_adapter_created(self):
        adapter = self.create_adapter()
        self.assertEqual(adapter.get_model_pedigree(), 'deepseek_v4')

    # Verify the adapter returns the provided model type.
    def test_model_type_returns_model_type_when_adapter_type_set(self):
        adapter = self.create_adapter(model_type=self.model_type)
        self.assertEqual(adapter.get_model_type(), self.model_type)

    # Verify dataset handling delegates to the tokenizer helper.
    def test_handle_dataset_calls_tokenizer_when_handling_data(self):
        adapter = self.create_adapter()
        adapter._get_tokenized_data = Mock(return_value=['tokenized'])
        result = adapter.handle_dataset('ds', device=DeviceType.CPU)
        adapter._get_tokenized_data.assert_called_once_with('ds', DeviceType.CPU)
        self.assertEqual(result, ['tokenized'])

    @patch('msmodelslim.model.deepseek_v4.model_adapter.json_safe_load')
    @patch('msmodelslim.model.deepseek_v4.model_adapter.os.path.join')
    # Verify weight map loading returns the parsed mapping and caches it.
    def test_get_weight_map_returns_weight_map_when_index_json_loaded(self, mock_join, mock_json_load):
        adapter = self.create_adapter()
        mock_json_load.return_value = {'weight_map': {'a': 'file1', 'b': 'file2'}}
        mock_join.return_value = self.model_path / 'model.safetensors.index.json'

        result = adapter.get_weight_map()

        mock_join.assert_called_once_with(self.model_path, 'model.safetensors.index.json')
        mock_json_load.assert_called_once()
        self.assertEqual(result, {'a': 'file1', 'b': 'file2'})

        adapter.get_weight_map()
        self.assertEqual(mock_json_load.call_count, 1)

    # Verify state dict loading works when all tensors are in a single safe tensors file.
    def test_get_state_dict_reads_single_file_when_single_file_contains_all_parameters(self):
        adapter = self.create_adapter()
        adapter.get_weight_map = Mock(
            return_value={'layer.weight': 'file1.safetensors', 'layer.bias': 'file1.safetensors'}
        )

        mock_module = Mock(spec=nn.Module)
        mock_module.named_parameters.return_value = [('layer.weight', Mock()), ('layer.bias', Mock())]

        mock_file = MagicMock()
        mock_file.get_tensor.side_effect = lambda name: f'tensor_{name}'

        with (
            patch('msmodelslim.model.deepseek_v4.model_adapter.get_valid_read_path', side_effect=lambda x, **kwargs: x),
            patch(
                'msmodelslim.model.deepseek_v4.model_adapter.safe_open',
                return_value=Mock(__enter__=Mock(return_value=mock_file), __exit__=Mock(return_value=False)),
            ),
        ):
            result = adapter.get_state_dict(mock_module)

        self.assertEqual(result['layer.weight'], 'tensor_layer.weight')
        self.assertEqual(result['layer.bias'], 'tensor_layer.bias')

    # Verify state dict loading works when tensors are spread across multiple files.
    def test_get_state_dict_reads_multiple_files_when_parameters_spread_across_files(self):
        adapter = self.create_adapter()
        adapter.get_weight_map = Mock(
            return_value={
                'layer1.weight': 'file1.safetensors',
                'layer1.bias': 'file1.safetensors',
                'layer2.weight': 'file2.safetensors',
                'layer2.bias': 'file2.safetensors',
            }
        )

        mock_module = Mock(spec=nn.Module)
        mock_module.named_parameters.return_value = [
            ('layer1.weight', Mock()),
            ('layer1.bias', Mock()),
            ('layer2.weight', Mock()),
            ('layer2.bias', Mock()),
        ]

        def safe_open_side_effect(path, framework, device):
            mock_file = MagicMock()
            mock_file.get_tensor.side_effect = lambda name: f'tensor_{name}'
            mock_context = MagicMock()
            mock_context.__enter__.return_value = mock_file
            mock_context.__exit__.return_value = False
            return mock_context

        with (
            patch('msmodelslim.model.deepseek_v4.model_adapter.get_valid_read_path', side_effect=lambda x, **kwargs: x),
            patch('msmodelslim.model.deepseek_v4.model_adapter.safe_open', side_effect=safe_open_side_effect),
        ):
            result = adapter.get_state_dict(mock_module)

        self.assertEqual(result['layer1.weight'], 'tensor_layer1.weight')
        self.assertEqual(result['layer2.bias'], 'tensor_layer2.bias')

    # Verify a missing weight file raises FileNotFoundError during state dict loading.
    def test_get_state_dict_raises_file_not_found_when_weight_file_missing(self):
        adapter = self.create_adapter()
        adapter.get_weight_map = Mock(return_value={'layer.weight': 'missing.safetensors'})

        mock_module = Mock(spec=nn.Module)
        mock_module.named_parameters.return_value = [('layer.weight', Mock())]

        with patch(
            'msmodelslim.model.deepseek_v4.model_adapter.get_valid_read_path',
            side_effect=FileNotFoundError('not found'),
        ):
            with self.assertRaises(FileNotFoundError):
                adapter.get_state_dict(mock_module)

    @patch('msmodelslim.model.deepseek_v4.model_adapter.get_mtp_layer')
    @patch('msmodelslim.model.deepseek_v4.model_adapter.wrap_mtp_decoder')
    @patch('msmodelslim.model.deepseek_v4.model_adapter.get_logger')
    # Verify missing MTP decoder triggers creation and wrapping of a new MTP layer.
    def test_load_mtp_if_not_load_creates_mtp_layer_when_decoder_missing_mtp(
        self, mock_get_logger, mock_wrap, mock_get_mtp
    ):
        adapter = self.create_adapter()
        dummy_decoder = DummyDecoderLayerV4()

        adapter.load_mtp_if_not_load(dummy_decoder, layer_prefix='layers.1')

        mock_get_mtp.assert_called_once_with(
            config=self.dummy_config, model_path=self.model_path, layer_prefix='layers.1', mtp_layer_prefix='layers.1.'
        )
        mock_wrap.assert_called_once_with(mtp_decoder=dummy_decoder, mtp_layer=mock_get_mtp.return_value)
        mock_get_logger.return_value.info.assert_any_call('Creating MTP layer')

    @patch('msmodelslim.model.deepseek_v4.model_adapter.get_mtp_layer')
    # Verify existing MTP decoder is not re-created when already loaded.
    def test_load_mtp_if_not_load_skips_when_mtp_already_exists(self, mock_get_mtp):
        adapter = self.create_adapter()
        dummy_decoder = DummyDecoderLayerV4()
        # ensure adapter checks via get_submodule; provide a get_submodule method
        dummy_decoder.enorm = nn.Linear(1, 1)

        adapter.load_mtp_if_not_load(dummy_decoder, layer_prefix='layers.1')
        mock_get_mtp.assert_not_called()

    @patch('msmodelslim.model.deepseek_v4.model_adapter.auto_dequant_state_dict')
    # Verify the existing decoder is returned and no auto-dequant is performed when the decoder exists.
    def test_load_decoder_if_not_exist_returns_existing_decoder_when_decoder_present(self, mock_auto_dequant):
        adapter = self.create_adapter()
        dummy_model = DummyModelV4(num_layers=1, config=self.dummy_config)

        result = adapter.load_decoder_if_not_exist(model=dummy_model, layer_prefix='layers.0', idx=0)

        self.assertEqual(result, dummy_model.get_submodule('layers.0'))
        mock_auto_dequant.assert_not_called()

    # Verify decoder layer generator yields the expected names and wraps the final layer.
    def test_generate_decoder_layer_returns_layer_names_when_layers_generated(self):
        adapter = self.create_adapter(config=Mock(num_hidden_layers=3))
        mock_decoders = [
            DummyDecoderLayerV4(layer_id=0),
            DummyDecoderLayerV4(layer_id=1),
            DummyDecoderLayerV4(layer_id=2),
        ]
        adapter.load_decoder_if_not_exist = Mock(side_effect=mock_decoders)
        adapter.load_mtp_if_not_load = Mock()

        layers = list(adapter.generate_decoder_layer(model=Mock()))

        self.assertEqual([name for name, _ in layers], ['layers.0', 'layers.1', 'layers.2'])
        adapter.load_mtp_if_not_load.assert_called_once_with(mock_decoders[2], layer_prefix='layers.2')

    # Verify adapter config generation for subgraph uses model configuration values.
    def test_get_adapter_config_for_subgraph_returns_configs_when_config_present(self):
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 3
        adapter.config.n_routed_experts = 2
        adapter.config.n_shared_experts = 1

        configs = adapter.get_adapter_config_for_subgraph()

        self.assertEqual(len(configs), 18)
        self.assertEqual(configs[0].mapping.source, 'layers.0.ffn.shared_experts.w3')
        self.assertEqual(configs[3].mapping.source, 'layers.0.attn.wo_a')
        # the 5th config corresponds to layer 0's norm-linear mapping targets
        self.assertIn('layers.0.attn.wq_a', configs[4].mapping.targets)

    # Verify layer norm fuse maps are returned for models with multiple layers.
    def test_get_ln_fuse_map_returns_maps_when_model_has_layers(self):
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 2
        adapter.config.n_routed_experts = 2
        adapter.config.n_shared_experts = 1

        pre_ln_map, ln_map = adapter.get_ln_fuse_map()

        self.assertEqual(pre_ln_map, {'norm': ['head']})
        self.assertIn('layers.0.ffn_norm', ln_map)
        self.assertIn('layers.1.enorm', ln_map)
        self.assertEqual(ln_map['layers.1.enorm'], ['layers.1.e_proj'])

    @patch('msmodelslim.model.deepseek_v4.model_adapter.QuaRotInterface.get_rotate_command')
    # Verify rotate map generation returns the pre-run list and rotation pairs.
    def test_get_rotate_map_returns_rotation_pairs_when_block_size_specified(self, mock_rotate_cmd):
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 2
        mock_rotate_cmd.side_effect = ['rot_a', 'rot_b']

        pre_run_list, rot_pairs = adapter.get_rotate_map(block_size=128)

        self.assertEqual(len(pre_run_list), 1)
        self.assertEqual(len(rot_pairs), 2)
        self.assertTrue(any('layers.0.attn.wq_a' in pair.right_rot for pair in rot_pairs))
        self.assertEqual(mock_rotate_cmd.call_count, 2)

    # Verify Ascend v1 save preprocessing rewrites layer prefixes for MTP modules.
    def test_ascendv1_save_module_preprocess_replaces_mtp_prefix_when_layer_prefix_given(self):
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 3
        prefix, module = adapter.ascendv1_save_module_preprocess('layers.2.some', nn.Linear(1, 1), Mock())
        self.assertEqual(prefix, 'mtp.0.some')

    def test_ascendv1_save_module_preprocess_unchanged_when_not_mtp_prefix(self):
        """验证非 MTP 层前缀不被修改"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 3
        prefix, module = adapter.ascendv1_save_module_preprocess('layers.0.some', nn.Linear(1, 1), Mock())
        self.assertEqual(prefix, 'layers.0.some')

    def test_enable_kv_cache_does_nothing_when_called(self):
        """验证 enable_kv_cache 是空操作"""
        adapter = self.create_adapter()
        result = adapter.enable_kv_cache(Mock(), True)
        self.assertIsNone(result)

    def test_get_bake_names_returns_empty_lists_when_called(self):
        """验证 get_bake_names 返回两个空列表"""
        adapter = self.create_adapter()
        left, right = adapter.get_bake_names()
        self.assertEqual(left, [])
        self.assertEqual(right, [])

    def test_generate_model_visit_delegates_to_decoder_visit_func_when_called(self):
        """验证 generate_model_visit 委托给 generated_decoder_layer_visit_func"""
        adapter = self.create_adapter()
        with patch('msmodelslim.model.deepseek_v4.model_adapter.generated_decoder_layer_visit_func') as mock_visit:
            adapter.generate_model_visit(Mock())
        mock_visit.assert_called_once()

    @patch('msmodelslim.model.deepseek_v4.model_adapter.get_mtp_layer')
    @patch('msmodelslim.model.deepseek_v4.model_adapter.wrap_mtp_decoder')
    @patch('msmodelslim.model.deepseek_v4.model_adapter.auto_dequant_state_dict')
    def test_load_decoder_if_not_exist_creates_new_decoder_when_missing(
        self, mock_auto_dequant, mock_wrap, mock_get_mtp
    ):
        """验证 decoder 不存在时创建新 decoder 并加载权重"""
        adapter = self.create_adapter()
        dummy_model = DummyModelV4(num_layers=1, config=self.dummy_config)
        adapter.get_weight_map = Mock(
            return_value={
                'mtp.0.linear.weight': 'fake.safetensors',
                'mtp.0.linear.bias': 'fake.safetensors',
                'linear.weight': 'fake.safetensors',
            }
        )

        with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
            with patch('msmodelslim.model.deepseek_v4.model_adapter.get_valid_read_path', return_value='/tmp/fake'):

                def _get_tensor(name):
                    return torch.ones(1, 1) if 'weight' in name else torch.ones(1)

                with patch(
                    'msmodelslim.model.deepseek_v4.model_adapter.safe_open',
                    return_value=MagicMock(
                        __enter__=MagicMock(return_value=MagicMock(get_tensor=MagicMock(side_effect=_get_tensor))),
                        __exit__=MagicMock(),
                    ),
                ):
                    decoder = adapter.load_decoder_if_not_exist(model=dummy_model, layer_prefix='layers.1', idx=1)

        self.assertIsNotNone(decoder)

    @patch('msmodelslim.model.deepseek_v4.model_adapter.auto_dequant_state_dict')
    def test_load_decoder_if_not_exist_creates_with_mtp_prefix_when_mtp_layer(self, mock_auto_dequant):
        """验证最后一层 decoder 使用 mtp.0 前缀加载权重"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 1
        dummy_model = DummyModelV4(num_layers=1, config=self.dummy_config)

        with patch('msmodelslim.model.deepseek_v4.model_adapter.get_valid_read_path', return_value='/tmp/fake'):
            with patch(
                'msmodelslim.model.deepseek_v4.model_adapter.safe_open',
                return_value=MagicMock(
                    __enter__=MagicMock(return_value=MagicMock(get_tensor=MagicMock(return_value=torch.ones(1)))),
                    __exit__=MagicMock(),
                ),
            ):
                with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
                    decoder = adapter.load_decoder_if_not_exist(model=dummy_model, layer_prefix='layers.0', idx=0)

                    self.assertIsNotNone(decoder)

    def test_get_adapter_config_for_subgraph_three_layers_when_multi_layer(self):
        """验证 3 层时每层生成正确数量的子图配置"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 3
        adapter.config.n_routed_experts = 2
        adapter.config.n_shared_experts = 1

        configs = adapter.get_adapter_config_for_subgraph()

        # 每层: 1 shared up-down + 2 routed up-down + 1 linear-linear + 2 norm-linear = 6
        # 3层: 18
        self.assertEqual(len(configs), 18)

    def test_get_adapter_config_for_subgraph_differs_by_layer_type(self):
        """验证不同 layer_idx 的 norm-linear 目标数量不同（layer 0 少, layer 1 多）"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 4
        adapter.config.n_routed_experts = 1
        adapter.config.n_shared_experts = 1

        configs = adapter.get_adapter_config_for_subgraph()

        # layer0: norm-linear targets 少（无 compressor/indexer）
        layer0_configs = [c for c in configs if 'layers.0.attn_norm' in (c.mapping.source or '')]
        self.assertEqual(len(layer0_configs), 1)
        # layer0 只有 wq_a + wkv
        self.assertEqual(len(layer0_configs[0].mapping.targets), 2)

    def test_ln_fuse_map_includes_mtp_mappings_when_mtp_present(self):
        """验证 ln_fuse_map 包含 MTP 层专用映射"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 1
        adapter.config.n_routed_experts = 1
        adapter.config.n_shared_experts = 1

        pre_ln, ln_map = adapter.get_ln_fuse_map()

        self.assertIn('layers.0.enorm', ln_map)
        self.assertIn('layers.0.hnorm', ln_map)
        self.assertIn('layers.0.norm', ln_map)

    def test_rotate_map_includes_mtp_rotations_when_mtp_layer(self):
        """验证 rotate_map 含 MTP 层专用旋转对"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 2
        adapter.config.n_routed_experts = 1
        adapter.config.n_shared_experts = 1

        with patch(
            'msmodelslim.model.deepseek_v4.model_adapter.QuaRotInterface.get_rotate_command', return_value='rot'
        ):
            pre_run, rot_pairs = adapter.get_rotate_map(block_size=128)

        self.assertEqual(len(rot_pairs), 2)
        # MTP 层 h_proj 在最后一层的右旋中
        last_layer_right = rot_pairs[0].right_rot
        self.assertIn('layers.1.h_proj', last_layer_right)

    def test_load_config_returns_model_args_when_called(self):
        """验证 _load_config 返回 ModelArgs 实例"""
        adapter = self.create_adapter()
        with patch('msmodelslim.model.deepseek_v4.model_adapter.ModelArgs') as mock_model_args:
            adapter._load_config()
        mock_model_args.assert_called_once()
