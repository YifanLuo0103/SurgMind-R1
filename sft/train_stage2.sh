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
MODEL_PATH="${MODEL_PATH:-path/to/your/models/stage1_model}"
DATASET_DIR="${DATASET_DIR:-$SCRIPT_DIR/data}"
MEDIA_DIR="${MEDIA_DIR:-path/to/your/data/images}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/stage2}"
require_path MODEL_PATH "$MODEL_PATH"
require_path MEDIA_DIR "$MEDIA_DIR"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE="${GPU_NUM:-2}"
python -m llamafactory.cli train configs/stage2.yaml \
    "model_name_or_path=$MODEL_PATH" "dataset_dir=$DATASET_DIR" \
    "media_dir=$MEDIA_DIR" "output_dir=$OUTPUT_DIR" "$@"
