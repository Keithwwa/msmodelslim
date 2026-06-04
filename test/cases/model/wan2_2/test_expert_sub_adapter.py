#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=consider-using-from-import

from unittest.mock import Mock

import torch.nn as nn

from msmodelslim.model.wan2_2.expert_sub_adapter import Wan2_2LowNoiseSubAdapter
from msmodelslim.processor.quant.fa3.interface import FA3QuantAdapterInterface
from msmodelslim.processor.quarot import OnlineQuaRotInterface


class TestExpertSubAdapterInterfaces:
    @staticmethod
    def test_low_noise_sub_adapter_implements_quarot_and_fa3():
        parent = Mock()
        parent.get_online_rotation_configs = Mock(return_value={})
        parent.inject_fa3_placeholders = Mock()

        sub = Wan2_2LowNoiseSubAdapter(parent, "low_noise_model")
        sub.bind_module(nn.Linear(4, 4))

        assert isinstance(sub, OnlineQuaRotInterface)
        assert isinstance(sub, FA3QuantAdapterInterface)

        sub.get_online_rotation_configs(sub._module)
        parent.get_online_rotation_configs.assert_called_once_with(sub._module)
