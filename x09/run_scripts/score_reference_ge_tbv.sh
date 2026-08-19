#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$VERSION_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_PATH="${INPUT_PATH:-$VERSION_DIR/results/zoomearth-xlrs-reference-all-1660.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$VERSION_DIR/results/zoomearth-xlrs-reference-all-1660}"
JUDGE_MODEL="${JUDGE_MODEL:-checkpoints/ZoomEarth-3B}"
JUDGE_CACHE="${JUDGE_CACHE:-results/ge-tbv-judge-cache.json}"

"$PYTHON_BIN" "$PROJECT_ROOT/evaluate_ge_tbv_xlrs.py" \
  --primary "$INPUT_PATH" \
  --output-prefix "$OUTPUT_PREFIX" \
  --judge-model "$JUDGE_MODEL" \
  --judge-cache "$JUDGE_CACHE"
