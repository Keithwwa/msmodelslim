#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""DTS 调度 backend：可替换的执行策略（默认 wave 在后续 PR 提供）。"""

from msmodelslim.utils.distributed.task_scheduler.backend.base import DTSBackend

__all__ = ["DTSBackend"]
