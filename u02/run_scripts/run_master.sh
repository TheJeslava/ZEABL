#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_PATH="${OUTPUT_PATH:-results/reproduced-master.jsonl}"
METRICS_PATH="${METRICS_PATH:-results/reproduced-master-metrics.json}"
MODEL_PATH="${MODEL_PATH:-../checkpoints/ZoomEarth-3B}"
DATA_PATH="${DATA_PATH:-../data/XLRS-Bench-lite}"
SEGMENT_ATTENTION_FLAG=()
if [[ "${SEGMENT_VISION_ATTENTION:-1}" == "1" ]]; then
  SEGMENT_ATTENTION_FLAG=(--segment-vision-attention)
fi
RESUME_FLAG=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG=(--resume)
elif [[ -e "$OUTPUT_PATH" ]]; then
  echo "Refusing to overwrite $OUTPUT_PATH; set RESUME=1 or choose OUTPUT_PATH." >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON_BIN" src/eval/infer_xlrs_lite.py \
  --model-path "$MODEL_PATH" \
  --data-path "$DATA_PATH" \
  --output "$OUTPUT_PATH" \
  --metrics-output "$METRICS_PATH" \
  --seed 20260721 \
  "${SEGMENT_ATTENTION_FLAG[@]}" \
  "${RESUME_FLAG[@]}"
