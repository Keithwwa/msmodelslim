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
import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field, ConfigDict

from msmodelslim.app.auto_tuning import TuningHistoryManagerInfra, TuningHistory
from msmodelslim.core.tune_strategy import EvaluateResult
from msmodelslim.utils.security import yaml_safe_load, yaml_safe_dump
from msmodelslim.utils.yaml_database import YamlDatabase


class YamlDatabaseHistory(BaseModel):
    history_practice_database: YamlDatabase
    history_index_file_path: Path

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TuningHistoryIndexUnit(BaseModel):
    practice_id: str
    evaluation: EvaluateResult
    time: str = Field(default_factory=lambda: str(datetime.datetime.now()))


class TuningHistoryIndex(BaseModel):
    records: List[TuningHistoryIndexUnit] = Field(default_factory=list)


class YamlTuningHistoryManager(TuningHistoryManagerInfra):
    def __init__(self):
        self.database_histories: Dict[str, YamlDatabaseHistory] = {}

    def append_history(self, database: str, history: TuningHistory) -> None:
        if database not in self.database_histories:
            history_dir = Path(database)
            self.database_histories[database] = YamlDatabaseHistory(
                history_practice_database=YamlDatabase(
                    config_dir=history_dir,
                    read_only=False,
                ),
                history_index_file_path=history_dir / 'history.yaml',
            )

        database_history = self.database_histories[database]
        with _get_modifiable_history_index(database_history.history_index_file_path) as history_index:
            history_index.records.append(TuningHistoryIndexUnit(
                practice_id=history.practice.metadata.config_id,
                evaluation=history.evaluation,
            ))
        database_history.history_practice_database[history.practice.metadata.config_id] = history.practice


@contextmanager
def _get_modifiable_history_index(history_index_file_path: Path):
    """
    作为上下文管理器，负责：
    1. 从索引文件中加载 PracticeHistoryIndex
    2. 将修改后的 PracticeHistoryIndex 回写到索引文件
    """
    if history_index_file_path.exists():
        history_content = yaml_safe_load(str(history_index_file_path))
        history_index = TuningHistoryIndex.model_validate(history_content)
    else:
        history_index = TuningHistoryIndex()
    try:
        yield history_index
    finally:
        yaml_safe_dump(history_index.model_dump(), str(history_index_file_path))
