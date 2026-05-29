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

from msmodelslim.format.ascendV1_format.ascendV1_json_writer_infra import (
    AscendV1JsonWriterCreatorInfra,
    AscendV1JsonWriterInfra,
)
from msmodelslim.format.mindie_format.mindie_json_writer_infra import (
    MindIEJsonWriterCreatorInfra,
    MindIEJsonWriterInfra,
)
from msmodelslim.infra.io.json_writer import JsonWriter


class DefaultJsonExportWriterCreator(
    AscendV1JsonWriterCreatorInfra,
    MindIEJsonWriterCreatorInfra,
):
    """描述 JSON 落盘：构造 :class:`~msmodelslim.infra.io.json_writer.JsonWriter`。"""

    def create_json_writer(
        self, save_directory: str, file_name: str
    ) -> AscendV1JsonWriterInfra | MindIEJsonWriterInfra:
        return JsonWriter(save_directory, file_name)


__all__ = ["DefaultJsonExportWriterCreator"]
