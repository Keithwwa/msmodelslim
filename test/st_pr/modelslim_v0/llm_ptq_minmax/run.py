# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

# -------------------------- 获取脚本自身所在目录（不受执行目录影响） --------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

model_resource_path = os.environ.get("MODEL_RESOURCE_PATH")
if not model_resource_path:
    raise Exception("获取不到模型路径，请先检查环境变量 MODEL_RESOURCE_PATH")

# for local path
LOAD_PATH = os.path.join(model_resource_path, "Qwen3-14B")
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=LOAD_PATH,
    torch_dtype='auto',
    device_map='auto',
    trust_remote_code=True,
    local_files_only=True).eval()

disable_names = ['lm_head']

quant_config = QuantConfig(
    a_bit=16, w_bit=8, disable_names=disable_names, dev_id=model.device.index, dev_type='npu',
    act_method=3, pr=1.0, w_sym=False, mm_tensor=False, w_method='MinMax')
calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')
calibrator.run()  # 执行PTQ量化校准

save_dir = os.path.join(script_dir, "output_llm_ptq_minmax")
calibrator.save(save_dir, save_type=["numpy", "safe_tensor"])
print('Save quant weight success!')
