#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

run_version() {
  local version="$1"
  local output metrics selection log exit_file resume status
  output="$version/results/zoomearth-${version}-balanced-650.jsonl"
  metrics="$version/results/zoomearth-${version}-balanced-650.metrics.json"
  selection="$version/results/zoomearth-${version}-balanced-650.selection.json"
  log="$version/results/zoomearth-${version}-balanced-650.run.log"
  exit_file="$version/results/zoomearth-${version}-balanced-650.exitcode"

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

run_pair() {
  local first="$1"
  local second="$2"
  local first_pid second_pid first_status second_status

  run_version "$first" &
  first_pid=$!
  run_version "$second" &
  second_pid=$!

  set +e
  wait "$first_pid"
  first_status=$?
  wait "$second_pid"
  second_status=$?
  set -e

  if [[ "$first_status" -ne 0 ]]; then
    return "$first_status"
  fi
  if [[ "$second_status" -ne 0 ]]; then
    return "$second_status"
  fi
}

run_pair x02 x03
run_pair x09 x10
run_version x05
