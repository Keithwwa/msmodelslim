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
from abc import ABC, abstractmethod
from typing import Generator, Optional

from msmodelslim.core.practice import PracticeConfig


class PracticeManagerInfra(ABC):
    def get_config_url(self, model_pedigree: str, config_id: str) -> Optional[str]:
        """Return the URL/location of the config; in different scenarios url may be a file path, an HTTP URL, etc."""
        return None

    @abstractmethod
    def __contains__(self, model_pedigree) -> bool:
        """Check if model pedigree is supported"""
        raise NotImplementedError

    @abstractmethod
    def get_config_by_id(self, model_pedigree, config_id: str) -> PracticeConfig:
        """Get configuration by ID"""
        raise NotImplementedError

    @abstractmethod
    def iter_config(self, model_pedigree) -> Generator[PracticeConfig, None, None]:
        """Iterate configurations by priority"""
        raise NotImplementedError
