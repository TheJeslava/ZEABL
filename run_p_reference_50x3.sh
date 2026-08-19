#!/usr/bin/env bash
set -euo pipefail

POSITIONS="$(sed -n 's/^POSITIONS="\(.*\)"/\1/p' p09/run_scripts/run_reference_50x3.sh)"
[ -n "$POSITIONS" ]

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_one() {
  local dir="$1"
  local stem="$2"
  echo "=== START $dir $stem $(date -u +%FT%TZ)"
  mkdir -p "$dir/results"
  (
    cd "$dir"
    python src/eval/infer_xlrs_lite.py \
      --model-path ../checkpoints/ZoomEarth-3B \
      --data-path ../data/XLRS-Bench-lite \
      --output "results/$stem.jsonl" \
      --metrics-output "results/$stem.metrics.json" \
      --positions "$POSITIONS" \
      --seed 20260721 \
      --segment-vision-attention
  )
  echo "=== DONE $dir $stem $(date -u +%FT%TZ)"
}

run_one p02 zoomearth-p02-reference-150
run_one p03 zoomearth-p03-reference-150
run_one p05 zoomearth-p05-reference-150
run_one p09 zoomearth-p09-reference-150
run_one p10 zoomearth-p10-reference-150
