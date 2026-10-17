#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
require_path() {
    if [[ "$2" == path/to/your/* || -z "$2" ]]; then
        echo "Set $1 to your own model/data path before running this script." >&2
        exit 2
    fi
}
MODEL_PATH="${MODEL_PATH:-path/to/your/models/stage2_model}"
DATASET_PATH="${DATASET_PATH:-path/to/your/data/surg_grpo_hf}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/grpo}"
require_path MODEL_PATH "$MODEL_PATH"
require_path DATASET_PATH "$DATASET_PATH"
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export DATASET=CUSTOM_OPENQA
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_DISABLED=true
export DEBUG_MODE="${DEBUG_MODE:-false}"
export LOG_PATH="$OUTPUT_DIR/reward_debug.log"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
torchrun --nproc_per_node="${GPU_NUM:-2}" --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29500}" \
    "$SCRIPT_DIR/src/open_r1/grpo.py" \
    --output_dir "$OUTPUT_DIR" --model_name_or_path "$MODEL_PATH" \
    --dtype bfloat16 --dataset_name "$DATASET_PATH" --dataset_selection "" \
    --reward_funcs surg_accuracy format --deepspeed "$SCRIPT_DIR/configs/zero2.json" \
    --max_prompt_length 1024 --max_completion_length 1024 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 --max_steps 300 --num_train_epochs 2 \
    --logging_steps 1 --bf16 True --report_to tensorboard \
    --gradient_checkpointing True --attn_implementation flash_attention_2 \
    --max_pixels 401408 --min_pixels 3136 --save_steps 50 --save_only_model true \
    --num_generations 4 --use_peft true --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    "$@" 2>&1 | tee "$OUTPUT_DIR/training.log"
