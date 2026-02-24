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
from typing import List, Literal, Optional, Generator

from pydantic import Field, field_validator

from msmodelslim.core.const import DeviceType, QuantType
from msmodelslim.core.practice import PracticeConfig
from msmodelslim.core.tune_strategy import ITuningStrategy
from msmodelslim.core.tune_strategy.base import BaseTuningStrategy
from msmodelslim.core.tune_strategy.dataset_loader_infra import DatasetLoaderInfra
from msmodelslim.core.tune_strategy.common.config_builder.expert_experience import (
    ExpertExperienceConfigBuilder,
    ExpertExperienceLoader,
    StructureConfig,
)
from msmodelslim.core.tune_strategy.interface import StrategyConfig, EvaluateResult
from msmodelslim.core.tune_strategy.standing_high.strategy import (
    StandingHighStrategy,
    StandingHighStrategyConfig
)
from msmodelslim.core.tune_strategy.standing_high_with_experience.standing_high_with_experience_interface import (
    StandingHighWithExperienceInterface
)
from msmodelslim.model import IModel
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger


class StandingHighWithExperienceStrategyConfig(StrategyConfig):
    """基于专家经验的摸高算法策略配置"""
    type: Literal["standing_high_with_experience"] = "standing_high_with_experience"

    structure_configs: Optional[List[StructureConfig]] = Field(
        default=None,
        description="结构配置列表，每个配置包含结构类型和对应的 include/exclude，例如 [{'type': 'GQA', 'include': ['*self_attn*'], 'exclude': ['*kv_b_proj']}]"
    )
    quant_type: QuantType = Field(
        default=QuantType.W8A8,
        description="量化类型（QuantType 枚举），须在专家经验 supported_quant_types 范围内。"
    )

    @field_validator("quant_type")
    @classmethod
    def _quant_type_must_be_supported_by_expert_experience(cls, v: QuantType) -> QuantType:
        supported = ExpertExperienceLoader.get_supported_quant_types()
        if v.value not in supported:
            raise SchemaValidateError(
                f"quant_type '{v.value}' is not in expert experience supported_quant_types. "
                f"Supported: {supported}.",
                action="Please set quant_type to one of the values in expert_experience.yaml supported_quant_types.",
            )
        return v


@logger_setter("msmodelslim.core.tune_strategy.standing_high_with_experience")
class StandingHighWithExperienceStrategy(BaseTuningStrategy, ITuningStrategy):
    """
    基于专家经验的摸高算法策略
    
    使用组合关系，内部使用 StandingHighStrategy 来执行摸高算法。
    职责：
    1. 根据简化的配置利用专家经验生成完整的 StandingHighStrategyConfig
    2. 委托 StandingHighStrategy 执行实际的摸高算法
    """
    
    def __init__(self, config: StrategyConfig, dataset_loader: DatasetLoaderInfra):
        super().__init__(config, dataset_loader)

        if not isinstance(config, StandingHighWithExperienceStrategyConfig):
            raise SchemaValidateError(
                f"StandingHighWithExperienceStrategy requires StandingHighWithExperienceStrategyConfig, "
                f"got {type(config)}"
            )
        
        self.config: StandingHighWithExperienceStrategyConfig = config
    
    def generate_practice(
        self,
        model: IModel,
        device: DeviceType = DeviceType.NPU,
    ) -> Generator[PracticeConfig, Optional[EvaluateResult], None]:
        """
        生成实践配置
        
        在首次调用时，根据原始配置生成完整的 StandingHighStrategyConfig，
        然后创建 StandingHighStrategy 实例并委托其执行摸高算法。
        
        Note: 模型适配器需要同时实现 StandingHighWithExperienceInterface 和 
        StandingHighInterface，因为内部委托给 StandingHighStrategy 执行。
        """
        if not isinstance(model, StandingHighWithExperienceInterface):
            raise UnsupportedError(
                f"model must be StandingHighWithExperienceInterface, got {type(model)}"
            )
        # 处理 structure_configs：如果为 None，尝试自动检测
        structure_configs = self.config.structure_configs
        if structure_configs is None:
            get_logger().info("structure_configs not provided, attempting to auto-detect model structure from model")
            structure_configs = self._auto_detect_structure_configs(model)
            if structure_configs is None:
                get_logger().warning(
                    "Auto-detect not implemented yet (interface reserved), using default structure configs."
                )
                structure_configs = self._get_default_structure_configs()
            else:
                get_logger().info("Successfully auto-detected model structure")
                self.config.structure_configs = structure_configs
        
        # 生成完整的基础配置
        base_config = self._generate_base_config(self.config, structure_configs)
        
        # 创建 StandingHighStrategy 实例并委托其执行摸高算法
        standing_high_strategy = StandingHighStrategy(
            config=base_config,
            dataset_loader=self.dataset_loader
        )
        
        yield from standing_high_strategy.generate_practice(model, device)
    
    def _generate_base_config(
        self, 
        config: StandingHighWithExperienceStrategyConfig,
        structure_configs: List[StructureConfig]
    ) -> StandingHighStrategyConfig:
        """
        根据输入配置生成完整的 StandingHighStrategyConfig
        
        Args:
            config: StandingHighWithExperienceStrategyConfig 配置对象
            structure_configs: 结构配置列表（已处理，不会是 None）
            
        Returns:
            生成的 StandingHighStrategyConfig 配置对象
        """
        quant_type_str = config.quant_type.value

        get_logger().info(
            "Generating full config for standing_high_with_experience with "
            "structure_configs=%s, quant_type=%s",
            structure_configs, quant_type_str
        )

        builder = ExpertExperienceConfigBuilder()
        quant_config = builder.build(
            quant_type=quant_type_str,
            structure_configs=structure_configs,
        )
        search_space = builder.get_tuning_search_space(
            quant_type=quant_type_str,
            structure_configs=structure_configs,
        )
        if not search_space.anti_outlier_strategies:
            raise SchemaValidateError(
                "Expert experience builder must provide anti_outlier_strategies in tuning search space."
            )

        return StandingHighStrategyConfig(
            type="standing_high",
            anti_outlier_strategies=search_space.anti_outlier_strategies,
            template=quant_config.spec,
            metadata=quant_config.metadata
        )
    
    def _auto_detect_structure_configs(self, model: IModel) -> Optional[List[StructureConfig]]:
        """
        自动检测模型结构配置
        
        根据模型适配器自动识别结构类型并生成对应的配置。
        
        Args:
            model: 模型适配器对象，可以通过 model.load_model() 等方法访问模型结构
            
        Returns:
            如果成功检测到模型结构，返回结构配置列表；否则返回 None
        """
        # 暂未实现，保留接口
        return None
    
    def _get_default_structure_configs(self) -> List[StructureConfig]:
        """
        获取默认的结构配置
        
        默认配置不区分模型结构，对所有层统一应用量化配置。
        
        Returns:
            默认结构配置列表
        """
        return [
            StructureConfig(
                type="MHA",  # 使用MHA作为通用结构类型，对所有层应用w8a8量化
                include=["*"],  # 匹配所有层
                exclude=[]  # 不排除任何层
            )
        ]


def get_plugin():
    """
    获取 standing_high_with_experience 策略插件（返回配置类与组件类，由框架通过 config.ini 完成注册）。

    Returns:
        (StandingHighWithExperienceStrategyConfig, StandingHighWithExperienceStrategy) 元组
    """
    return StandingHighWithExperienceStrategyConfig, StandingHighWithExperienceStrategy