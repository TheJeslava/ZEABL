#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_version() {
  local version="$1"
  local output metrics selection log exit_file resume status
  output="$version/results/zoomearth-${version}-balanced-650.jsonl"
  metrics="$version/results/zoomearth-${version}-balanced-650.metrics.json"
  selection="$version/results/zoomearth-${version}-balanced-650.selection.json"
  log="$version/results/zoomearth-${version}-balanced-650.run.log"
  exit_file="$version/results/zoomearth-${version}-balanced-650.exitcode"

  if [[ -e "$output" && -e "$metrics" && -e "$exit_file" ]] \
    && [[ "$(wc -l < "$output")" -eq 650 ]] \
    && [[ "$(tr -d '[:space:]' < "$exit_file")" == "0" ]] \
    && "$PYTHON_BIN" - "$output" <<'PYTHON'
import json
from pathlib import Path
import sys

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line]
positions = {row["dataset_position"] for row in rows}
categories = {}
for row in rows:
    categories[row["category"]] = categories.get(row["category"], 0) + 1
valid = (
    len(rows) == 650
    and len(positions) == 650
    and all(row.get("status") == "ok" for row in rows)
    and len(categories) == 13
    and all(count == 50 for count in categories.values())
)
raise SystemExit(0 if valid else 1)
PYTHON
  then
    echo "[$(date -Is)] skipping complete $version"
    return 0
  fi

  resume=0
  if [[ -e "$output" ]]; then
    resume=1
  fi

  echo "[$(date -Is)] starting $version (resume=$resume)"
  set +e
  RESUME="$resume" \
  OUTPUT_PATH="results/zoomearth-${version}-balanced-650.jsonl" \
  METRICS_PATH="results/zoomearth-${version}-balanced-650.metrics.json" \
  SELECTION_PATH="results/zoomearth-${version}-balanced-650.selection.json" \
    "$version/run_scripts/run_balanced_50.sh" >"$log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" >"$exit_file"
  echo "[$(date -Is)] finished $version (exit=$status)"
  if [[ "$status" -ne 0 ]]; then
    return "$status"
  fi
}

run_version x02
run_version x03
run_version x09
run_version x10
run_version x05
