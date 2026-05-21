# 开发者测试指南

## 简介

本文档旨在指导开发者在 msModelSlim 项目中运行单元测试，帮助您了解测试环境配置及执行流程，确保能够顺利执行测试。

## 环境要求

### 操作系统

msModelSlim 的单元测试目前仅支持 **Linux** 系统，不支持 Windows 或 macOS。请确保您在 Linux 环境下运行测试。

### Python版本

msModelSlim 推荐使用 Python 3.10 环境进行测试，建议使用 conda 创建独立的虚拟环境：

```bash
# 创建Python 3.10环境
conda create -n ut_py310 python=3.10
conda activate ut_py310

# 确认Python版本
python --version  # 应显示 Python 3.10.x
```

### 必须依赖

运行单元测试需要安装以下依赖包：

```bash
pip install pytest
pip install coverage
pip install torch==2.1.0
pip install easydict==1.13
pip install einops
pip install "pydantic>=2.10.1"
pip install wcmatch
pip install scipy
pip install pygtrie
pip install accelerate
pip install requests
pip install pyyaml
pip install numpy==1.26.4
pip install transformers==4.51.0
```

> [!NOTE]
>
> numpy 建议使用 1.26.4 版本以避免与 torch 2.1.0 产生兼容性问题。

## 单元测试执行

### 执行用例

使用项目根目录下的 `test/run_ut.sh` 脚本执行单元测试，测试 modelslim_v1 相关模块（app、core、ir、infra、processor、utils），使用 `--modelslim_v1` 参数：

```bash
cd test
bash run_ut.sh --modelslim_v1
```

该参数适用于仅涉及核心量化框架的测试场景，可以显著减少测试执行时间。

### 查看帮助信息

```bash
bash run_ut.sh --help
```

### 输出典型测试

运行 `bash run_ut.sh --modelslim_v1` 后，每个模块会显示测试进度和最终结果。输出格式示例如下：

```text
Running modelslim_v1 related test cases...
===== test session starts =====
collected 57 items

cases/app/analysis/test_analysis_app.py .......                       [ 26%]
cases/app/analysis/test_analysis_methods.py ....................      [ 73%]
cases/app/naive_quantization/test_naive_quantization_app.py ......    [100%]

----- generated xml file: /path/to/report/final_app.xml -----
===== 57 passed, 6 warnings in 6.17s =====

===== test session starts =====
collected 406 items

cases/core/context/test_base_context.py .................    [  4%]
...
----- generated xml file: /path/to/report/final_core.xml -----
===== 392 passed, 14 skipped, 4 warnings in 18.30s =====
```

> [!NOTE]
>
> 测试完成后会看到每个模块的统计信息，格式为 `X passed, Y skipped, Z warnings`。
