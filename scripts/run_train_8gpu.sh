#!/usr/bin/env bash
# Launch R-Q-Evolve-v0.2 with exactly eight explicitly selected GPUs.
# Usage:
#   bash scripts/run_train_8gpu.sh --gpus 0,1,2,3,4,5,6,7
#   bash scripts/run_train_8gpu.sh --gpus 0,1,2,3,4,5,6,7 --detach
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="configs/rq_evolve_v02_8gpu.yaml"
EXPECTED_GPUS=8
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
DETACH=false
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      [[ $# -ge 2 ]] || { echo "[v0.2-8gpu] --gpus requires a value" >&2; exit 2; }
      GPUS="$2"
      shift 2
      ;;
    --detach)
      DETACH=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[v0.2-8gpu] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"
[[ -f "$CONFIG" ]] || { echo "[v0.2-8gpu] missing config: $CONFIG" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "[v0.2-8gpu] Python not found: $PYTHON_BIN" >&2; exit 1; }

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
[[ "${#GPU_IDS[@]}" -eq "$EXPECTED_GPUS" ]] || {
  echo "[v0.2-8gpu] --gpus must contain exactly $EXPECTED_GPUS IDs: $GPUS" >&2
  exit 2
}
declare -A SEEN_GPU_IDS=()
for gpu_id in "${GPU_IDS[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ ]] || { echo "[v0.2-8gpu] invalid GPU ID: $gpu_id" >&2; exit 2; }
  [[ -z "${SEEN_GPU_IDS[$gpu_id]:-}" ]] || { echo "[v0.2-8gpu] duplicate GPU ID: $gpu_id" >&2; exit 2; }
  SEEN_GPU_IDS[$gpu_id]=1
done

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[v0.2-8gpu] config: $CONFIG"
echo "[v0.2-8gpu] GPUs  : $CUDA_VISIBLE_DEVICES"
PATCH_RESULT="$("$PYTHON_BIN" patches/verl_agent_loop_sampling.py)"
echo "[v0.2-8gpu] VERL patch: $PATCH_RESULT"
"$PYTHON_BIN" -m rq_evolve_v02 preflight --config "$CONFIG"

LOG_DIR="$ROOT/logs/rq_evolve_v02_8gpu"
PID_FILE="$LOG_DIR/train.pid"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  prior_pid="$(tr -dc '0-9' < "$PID_FILE")"
  if [[ -n "$prior_pid" ]] && kill -0 "$prior_pid" 2>/dev/null; then
    prior_command="$(tr '\0' ' ' < "/proc/$prior_pid/cmdline" 2>/dev/null || true)"
    echo "[v0.2-8gpu] refusing to start: PID $prior_pid is alive" >&2
    [[ -n "$prior_command" ]] && echo "[v0.2-8gpu] command: $prior_command" >&2
    exit 1
  fi
fi

if $DETACH; then
  timestamp="$(date +%Y%m%d_%H%M%S)"
  log_file="$LOG_DIR/train_${timestamp}.log"
  nohup "$PYTHON_BIN" -m rq_evolve_v02 run --config "$CONFIG" > "$log_file" 2>&1 &
  train_pid=$!
  echo "$train_pid" > "$PID_FILE"
  ln -sfn "$log_file" "$LOG_DIR/latest.log"
  echo "[v0.2-8gpu] started PID $train_pid"
  echo "[v0.2-8gpu] log: $LOG_DIR/latest.log"
else
  "$PYTHON_BIN" -m rq_evolve_v02 run --config "$CONFIG"
fi
