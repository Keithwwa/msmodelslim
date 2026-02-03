#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Unit tests for iter_smooth NonFusionSubgraph API (iter_smooth_impl_non_fusion_linear).
Covers code added in commit f2075e1 (non-fusion iter_smooth).
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from msmodelslim.ir.qal.qtypes import NonFusionSubgraph
from msmodelslim.processor.anti_outlier.iter_smooth.api import iter_smooth
from msmodelslim.processor.anti_outlier.common import (
    IterSmoothConfig,
    IterSmoothContext,
    SubgraphFusionFactory,
)
from msmodelslim.ir.non_fusion_smooth_quant_ir import NonFusionSmoothQuantHookIR


class TestIterSmoothImplNonFusionLinear(unittest.TestCase):
    """Tests for iter_smooth_impl_non_fusion_linear (NonFusionSubgraph)."""

    def setUp(self):
        self.config = IterSmoothConfig(alpha=0.5, scale_min=1e-5)
        self.a_scale = torch.tensor([[1.0, 2.0]])
        self.shift = torch.tensor([0.0])
        self.context = IterSmoothContext(
            version=1,
            a_smooth_scale=self.a_scale,
            shift=self.shift,
        )

    def test_empty_linears_raises_value_error(self):
        """Empty linears list raises ValueError."""
        subgraph = NonFusionSubgraph(linears=[])
        with self.assertRaises(ValueError) as ctx:
            iter_smooth(subgraph, self.config, self.context)
        self.assertIn("at least one linear layer", str(ctx.exception))

    def test_single_linear_applies_fusion_and_registers_hook(self):
        """Single linear: compute scales, apply fusion, register pre_hook."""
        linear = nn.Linear(2, 3)
        torch.nn.init.ones_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
        subgraph = NonFusionSubgraph(linears=[linear])

        with patch.object(
            SubgraphFusionFactory, "apply_fusion_to_subgraph"
        ) as mock_fusion:
            iter_smooth(subgraph, self.config, self.context)
            mock_fusion.assert_called_once()
            call_args = mock_fusion.call_args
            self.assertIs(call_args[0][0], subgraph)
            self.assertIn("scales", call_args[1])
            self.assertIn("scales", call_args[1]["scales"])

        # Check pre_hooks registered (one NonFusionSmoothQuantHookIR per linear)
        # PyTorch may store (priority, hook) or hook in _forward_pre_hooks.values()
        raw_hooks = list(linear._forward_pre_hooks.values())
        hook_objs = [
            h[1] if isinstance(h, tuple) and len(h) == 2 else h
            for h in raw_hooks
        ]
        ir_hooks = [h for h in hook_objs if isinstance(h, NonFusionSmoothQuantHookIR)]
        self.assertEqual(len(ir_hooks), 1)
        self.assertIsNotNone(ir_hooks[0].hook_handle)

    def test_multiple_linears_applies_fusion_and_registers_hooks(self):
        """Multiple linears: w_scale from concat of per-linear stats, fusion and hooks for each."""
        linear1 = nn.Linear(2, 3)
        linear2 = nn.Linear(2, 3)
        torch.nn.init.ones_(linear1.weight)
        torch.nn.init.ones_(linear2.weight)
        subgraph = NonFusionSubgraph(linears=[linear1, linear2])

        with patch.object(
            SubgraphFusionFactory, "apply_fusion_to_subgraph"
        ) as mock_fusion:
            iter_smooth(subgraph, self.config, self.context)
            mock_fusion.assert_called_once()

        raw_vals = lambda m: list(m._forward_pre_hooks.values())
        hook_objs = lambda m: [
            h[1] if isinstance(h, tuple) and len(h) == 2 else h
            for h in raw_vals(m)
        ]
        for linear in subgraph.linears:
            ir_hooks = [h for h in hook_objs(linear) if isinstance(h, NonFusionSmoothQuantHookIR)]
            self.assertEqual(len(ir_hooks), 1)

    def test_single_linear_forward_after_smooth(self):
        """After iter_smooth, single linear forward still runs (hook is no-op)."""
        linear = nn.Linear(2, 3)
        torch.nn.init.ones_(linear.weight)
        torch.nn.init.zeros_(linear.bias)
        subgraph = NonFusionSubgraph(linears=[linear])
        iter_smooth(subgraph, self.config, self.context)
        x = torch.randn(1, 2)
        out = linear(x)
        self.assertEqual(out.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
