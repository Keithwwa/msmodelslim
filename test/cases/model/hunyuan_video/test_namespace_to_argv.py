#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=use-implicit-booleaness-not-comparison

from msmodelslim.model.hunyuan_video.model_adapter import HunyuanVideoModelAdapter


class TestNamespaceToArgv:
    @staticmethod
    def test_expands_video_size():
        argv = HunyuanVideoModelAdapter._namespace_to_argv(
            {
                "video_size": (720, 1280),
                "infer_steps": 50,
            }
        )
        assert argv == ["--video-size", "720", "1280", "--infer-steps", "50"]

    @staticmethod
    def test_skips_dict_values():
        argv = HunyuanVideoModelAdapter._namespace_to_argv(
            {
                "video_size": [480, 832],
                "extra": {"a": 1},
            }
        )
        assert "--extra" not in argv
        assert argv[:4] == ["--video-size", "480", "832"]

    @staticmethod
    def test_skips_list_values_not_registered_for_cli():
        argv = HunyuanVideoModelAdapter._namespace_to_argv(
            {
                "cfg_scale": [1.0, 2.0],
            }
        )
        assert argv == []
