#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_PATH="${OUTPUT_PATH:-results/zoomearth-xlrs-balanced-50.jsonl}"
METRICS_PATH="${METRICS_PATH:-results/zoomearth-xlrs-balanced-50.tbv-metrics.json}"
EVALUATION_PATH="${EVALUATION_PATH:-results/zoomearth-xlrs-balanced-50.tbv-evaluation.jsonl}"

"${PYTHON_BIN:-python}" src/eval/eval_xlrs.py \
  --results-file "$OUTPUT_PATH" \
  --metrics-output "$METRICS_PATH" \
  --evaluation-output "$EVALUATION_PATH"
