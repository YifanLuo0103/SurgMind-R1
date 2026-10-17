<div align="center">

<h1>SurgMind-R1</h1>
<h3>通过强化学习克服手术视觉语言模型中的推理坍塌</h3>

<p>
  <a href="README.md">
    <img src="https://img.shields.io/badge/Language-English-0078d4?style=flat-square" alt="Switch to English">
  </a>
</p>

<p>
  <a href="#论文与引用">
    <img src="https://img.shields.io/badge/Paper-Coming%20soon-b31b1b?style=for-the-badge" alt="论文：即将发布">
  </a>
  <a href="#训练">
    <img src="https://img.shields.io/badge/Code-Training-181717?style=for-the-badge&logo=github" alt="训练代码">
  </a>
  <a href="#surglite-bench">
    <img src="https://img.shields.io/badge/SurgLite--Bench-Coming%20soon-f9c74f?style=for-the-badge" alt="SurgLite-Bench：即将发布">
  </a>
  <a href="#模型">
    <img src="https://img.shields.io/badge/SurgMind--R1-Coming%20soon-2ea44f?style=for-the-badge&logo=huggingface" alt="SurgMind-R1 模型：即将发布">
  </a>
</p>

</div>

---

## 项目概述

手术视觉语言模型需要识别器械、理解手术阶段，并推理器械、动作与目标之间的交互关系。
然而，仅使用答案进行监督微调可能导致**推理坍塌**，削弱模型生成显式推理路径的能力。

**SurgMind-R1** 通过三阶段训练流程应对这一问题：

1. **领域自适应 SFT**：学习手术场景理解能力。
2. **推理增强 SFT**：使用 Cholec-Think 恢复显式推理能力。
3. **GRPO 强化学习**：结合任务感知的准确性奖励与回答格式奖励，巩固推理鲁棒性。

本仓库提供基于 LLaMA-Factory 的**两阶段 SFT 脚本**和**一阶段 GRPO 代码**。
**SurgLite-Bench 评测基准**与 **SurgMind-R1 模型权重**将在后续单独发布。

---

## 项目特点

- 基于 **Qwen2.5-VL-7B-Instruct**，面向手术视觉语言理解。
- 结合**领域适应**、**推理监督**与**强化学习**。
- 支持**器械识别**、**手术阶段识别**和**手术三元组识别**。
- 使用**任务感知的准确性奖励**与**结构化回答格式奖励**。
- 支持显式推理的 **think** 提示与直接回答的 **no-think** 提示。

---

## SurgLite-Bench

- **定位：** 轻量级、跨领域、跨任务的手术评测基准。
- **规模：** 6 个公开数据集、4 类手术场景、89,659 帧标注图像。
- **任务：** 器械识别、阶段识别、手势识别、手术三元组识别及安全关键视野（CVS）评估。
- **评估：** 域内性能与零样本域外泛化能力。

**Coming soon（即将发布）。** 发布后将补充下载链接与使用说明。

---

## 模型

**SurgMind-R1** 基于 **Qwen2.5-VL-7B-Instruct**，通过两阶段 SFT 和后续 GRPO 训练得到。

### 模型发布

**Coming soon（即将发布）。** 最终合并模型、tokenizer、processor 和模型使用说明将一并发布，下载链接将在此补充。

---

## 训练

请为 SFT 和 GRPO 准备独立环境，并将以下路径替换为实际绝对路径。

### 1. 两阶段 SFT

SFT 使用用户自行安装的 LLaMA-Factory。

```bash
python -m pip install -r sft/requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

请在 LLaMA-Factory 的数据集配置中注册数据集。

```bash
export DATASET_DIR="/absolute/path/to/sft_data"
export MEDIA_DIR="/absolute/path/to/images"

# Stage 1：领域自适应 SFT。
MODEL_PATH="/absolute/path/to/Qwen2.5-VL-7B-Instruct" bash sft/train_stage1.sh

# Stage 2：推理增强 SFT。
MODEL_PATH="/absolute/path/to/merged_stage1_model" bash sft/train_stage2.sh
```

训练参数位于 `sft/configs/`。

### 2. GRPO

```bash
python -m pip install -r grpo/requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

准备 Hugging Face 数据集。

```bash
MODEL_PATH="/absolute/path/to/merged_stage2_model" \
DATASET_PATH="/absolute/path/to/dataset_hf" \
bash grpo/train.sh
```

启动脚本默认使用 2 张 GPU，可通过 `GPU_NUM` 和 `OUTPUT_DIR` 设置 GPU 数量与输出目录。

---

## 论文与引用

**论文：Coming soon（即将公开）。** 当前稿件尚未公开，公开后将在此补充论文链接和 BibTeX 引用信息。

如果你在研究中使用 SurgMind-R1、SurgLite-Bench 或本仓库的训练代码，请在论文公开后引用我们的文章。

---

## 致谢

感谢 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)、
[Open-R1](https://github.com/huggingface/open-r1) 以及多模态 Open-R1 社区的开源贡献。

---

## 许可证

上游 Apache-2.0 许可证与版权声明保留在 [grpo/LICENSE](grpo/LICENSE) 和相应源文件中。
项目自有新增代码及模型产物的许可证将另行说明。
源数据集和基座模型仍遵循各自的使用条款。
