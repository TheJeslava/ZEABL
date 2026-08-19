#!/usr/bin/env bash
set -euo pipefail

POSITIONS="$(sed -n 's/^POSITIONS="\(.*\)"/\1/p' u09/run_scripts/run_reference_50x3.sh)"
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

run_one u02 zoomearth-xlrs-reference-model-bbox-150
run_one u03 zoomearth-xlrs-reference-model-bbox-150
run_one u05 zoomearth-xlrs-reference-model-bbox-150
run_one u09 zoomearth-xlrs-reference-model-bbox-150
run_one u10 zoomearth-xlrs-reference-model-bbox-150
