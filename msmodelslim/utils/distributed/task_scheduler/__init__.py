#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""DTS 对外公开接口（``DistributedTaskScheduler`` / wave backend 见后续 PR）。"""

from msmodelslim.utils.distributed.task_scheduler.backend import DTSBackend
from msmodelslim.utils.distributed.task_scheduler.constants import (
    clear_distributed_task_work_queue,
    get_distributed_task_work_queue,
    set_distributed_task_work_queue,
)
from msmodelslim.utils.distributed.task_scheduler.types import TaskExecutionRecord, TaskSyncContext
from msmodelslim.utils.distributed.task_scheduler.sync import DTSMixin

__all__ = [
    "DTSBackend",
    "DTSMixin",
    "TaskExecutionRecord",
    "TaskSyncContext",
    "set_distributed_task_work_queue",
    "get_distributed_task_work_queue",
    "clear_distributed_task_work_queue",
]
