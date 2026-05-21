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
from importlib import import_module
from pathlib import Path
from typing import Dict

from msmodelslim.model.plugin_factory.loader_interface import AdapterLoaderInterface
from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.dependency_check import (
    DependencyChecker,
    get_require_packages,
)
from msmodelslim.utils.exception import UnsupportedError, VersionError

MODEL_ADAPTER_ENTRY_POINTS = "msmodelslim.model_adapter.plugins"


class BaseModelAdapterLoader(AdapterLoaderInterface):
    ADAPTER_CLASS_PATH = ""

    def get_loader_requirements(self) -> Dict[str, str]:
        return get_require_packages(self)

    @staticmethod
    def check_requirements(
            plugin_name: str,
            requirements: Dict[str, str],
    ) -> None:
        DependencyChecker.set_plugin(plugin_name, requirements)
        try:
            DependencyChecker.check_plugin(plugin_name)
        except VersionError as exc:
            raise VersionError(
                f"Dependency check failed for plugin '{plugin_name}': {exc}"
            ) from exc

    def precheck(
            self,
            model_type: str,
            model_path: Path,
    ) -> None:
        del model_path
        plugin_name = f"{MODEL_ADAPTER_ENTRY_POINTS}:{model_type}"

        dependency_map = getattr(msmodelslim_config, "model_adapter_dependencies", {})
        if not isinstance(dependency_map, dict):
            dependency_map = {}
        config_requirements = dependency_map.get(plugin_name, {})
        if not isinstance(config_requirements, dict):
            config_requirements = {}

        loader_requirements = self.get_loader_requirements()

        # Priority: loader decorator < platform config.
        merged_requirements = dict(loader_requirements)
        merged_requirements.update(config_requirements)

        self.check_requirements(
            plugin_name=plugin_name,
            requirements=merged_requirements,
        )

    def load(
            self,
            model_type: str,
            model_path: Path,
            trust_remote_code: bool = False,
    ):
        if ":" not in self.ADAPTER_CLASS_PATH:
            raise UnsupportedError(
                f"Loader '{type(self).__name__}' must define ADAPTER_CLASS_PATH in "
                f"'module.path:ClassName' format."
            )
        module_path, class_name = self.ADAPTER_CLASS_PATH.split(":", 1)
        adapter_module = import_module(module_path)
        adapter_class = getattr(adapter_module, class_name)
        self.check_requirements(
            plugin_name=f"{MODEL_ADAPTER_ENTRY_POINTS}:{model_type}",
            requirements=get_require_packages(adapter_class),
        )
        return adapter_class(
            model_type=model_type,
            model_path=model_path,
            trust_remote_code=trust_remote_code,
        )

