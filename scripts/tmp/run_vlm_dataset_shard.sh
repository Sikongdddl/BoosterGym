#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <seed> <save_dir> <log_path>" >&2
  exit 2
fi

SEED="$1"
SAVE_DIR="$2"
LOG_PATH="$3"

source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
set +u
conda activate booster
set -u

mkdir -p "$(dirname "$LOG_PATH")"

env OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is required}" \
    OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://models.sjtu.edu.cn/api/v1}" \
    python3 /home/ubuntu/jrWork/booster_gym/scripts/tmp/hypergym_vlm_bc_rl/generate_vlm_dataset.py \
      --episodes 100 \
      --max-steps 80 \
      --seed "$SEED" \
      --num-home 2 \
      --num-away 2 \
      --query-interval 5 \
      --save-dir "$SAVE_DIR" \
      --frame-max-width 320 \
      --frame-jpeg-quality 80 \
      --save-video \
      --fps 10 \
      > "$LOG_PATH" 2>&1
