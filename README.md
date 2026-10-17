<div align="center">

<h1>SurgMind-R1</h1>
<h3>Overcoming Reasoning Collapse in Surgical Vision-Language Models<br>via Reinforcement Learning</h3>

<p>
  <a href="README_zh.md">
    <img src="https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-0078d4?style=flat-square" alt="切换到简体中文">
  </a>
</p>


<p>
  <a href="#paper-and-citation">
    <img src="https://img.shields.io/badge/Paper-Coming%20soon-b31b1b?style=for-the-badge" alt="Paper: Coming soon">
  </a>
  <a href="#training">
    <img src="https://img.shields.io/badge/Code-Training-181717?style=for-the-badge&logo=github" alt="Training Code">
  </a>
  <a href="#surglite-bench">
    <img src="https://img.shields.io/badge/SurgLite--Bench-Coming%20soon-f9c74f?style=for-the-badge" alt="SurgLite-Bench: Coming soon">
  </a>
  <a href="#models">
    <img src="https://img.shields.io/badge/SurgMind--R1-Coming%20soon-2ea44f?style=for-the-badge&logo=huggingface" alt="SurgMind-R1 Model: Coming soon">
  </a>
</p>
</div>

---

## Overview

Surgical vision-language models need to recognize instruments, understand procedural
phases, and reason about instrument–action–target interactions. However,
answer-only supervised fine-tuning can lead to **reasoning collapse**, weakening
the model's ability to produce explicit reasoning paths.

**SurgMind-R1** addresses this problem through a three-stage training pipeline:

1. **Domain-adaptive SFT** acquires surgical scene understanding.
2. **Reasoning-enhanced SFT** restores explicit reasoning using Cholec-Think.
3. **GRPO reinforcement learning** consolidates reasoning robustness with
   task-aware accuracy and response-format rewards.

This repository provides **two-stage SFT scripts** based on LLaMA-Factory and
**one-stage GRPO code**. The **SurgLite-Bench benchmark** and **SurgMind-R1 model
weights** will be released separately.

---

## Highlights

- Built on **Qwen2.5-VL-7B-Instruct** for surgical vision-language understanding.
- Combines **domain adaptation**, **reasoning supervision**, and **reinforcement learning**.
- Supports **tool recognition**, **phase recognition**, and **surgical triplet recognition**.
- Uses **task-aware accuracy rewards** together with **structured response-format rewards**.
- Supports explicit **think** and direct-answer **no-think** prompting.

---

## SurgLite-Bench

- **Scope:** A lightweight, cross-domain and cross-task surgical benchmark.
- **Scale:** 6 public datasets, 4 surgical scenarios, and 89,659 annotated frames.
- **Tasks:** Tool recognition, phase recognition, gesture recognition, surgical triplet recognition, and Critical View of Safety (CVS) assessment.
- **Evaluation:** In-domain performance and zero-shot out-of-distribution generalization.

**Coming soon.** Download links and usage instructions will be added upon release.

---

## Models

**SurgMind-R1** is built on **Qwen2.5-VL-7B-Instruct** and trained through
two-stage SFT followed by GRPO.

### Model Release

**Coming soon.** The final merged model, tokenizer, processor, and model usage
instructions will be released together. A model download link will be added here.

---

## Training

Use separate environments for SFT and GRPO. Replace the paths
below with your own absolute paths. 

### 1. Two-Stage SFT

SFT uses an externally installed LLaMA-Factory.

```bash
python -m pip install -r sft/requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

Register dataset in your LLaMA-Factory dataset configuration. 

```bash
export DATASET_DIR="/absolute/path/to/sft_data"
export MEDIA_DIR="/absolute/path/to/images"

# Stage 1: domain-adaptive SFT.
MODEL_PATH="/absolute/path/to/Qwen2.5-VL-7B-Instruct" bash sft/train_stage1.sh

# Stage 2: reasoning-enhanced SFT.
MODEL_PATH="/absolute/path/to/merged_stage1_model" bash sft/train_stage2.sh
```

Training settings are in `sft/configs/`.

### 2. GRPO

```bash
python -m pip install -r grpo/requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

Prepare a Hugging Face dataset. 

```bash
MODEL_PATH="/absolute/path/to/merged_stage2_model" \
DATASET_PATH="/absolute/path/to/dataset_hf" \
bash grpo/train.sh
```

All launchers default to 2 GPUs. Set `GPU_NUM` and `OUTPUT_DIR` to change the
GPU count and output location. 

---

## Paper and Citation

**Paper: Coming soon.** Our manuscript is not yet publicly available.
The paper link and BibTeX entry will be added once it is public.

If you use SurgMind-R1, SurgLite-Bench, or this training code in your research,
please cite our paper once it becomes publicly available.

---

## Acknowledgements

We thank [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),
[Open-R1](https://github.com/huggingface/open-r1), and the multimodal Open-R1
community for their open-source contributions.

---

## License

The upstream Apache-2.0 license and copyright headers are retained in
[grpo/LICENSE](grpo/LICENSE) and the corresponding source files.
Licensing for project-owned additions and model artifacts will be specified
separately. Source datasets and base models retain their respective terms.
