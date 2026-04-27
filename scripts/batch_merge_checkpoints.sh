#!/usr/bin/env bash
set -euo pipefail

# Batch merge verl checkpoints to HuggingFace format.
#
# Default behavior:
# - Merge all discovered steps for each experiment.
# - Auto-detect backend (fsdp/megatron) from checkpoint contents.
#
# Example:
#   bash scripts/batch_merge_checkpoints.sh \
#     --project frontend_focal \
#     --experiment focal_v3_Qwen3-1.7B-Base \
#     --experiment baseline_v3_Qwen3-1.7B-Base \
#     --input-root checkpoints \
#     --output-root merged_checkpoints \
#     --step-count all

PROJECT_NAME="frontend_focal"
INPUT_ROOT="checkpoints"
OUTPUT_ROOT="merged_checkpoints"
ROLE="actor"
STEP_COUNT="all"            # all | positive integer (latest N steps)
STEPS_CSV=""                # optional, e.g. "5,20,75"; overrides STEP_COUNT
BACKEND="auto"              # auto | fsdp | megatron
PYTHON_BIN="python"
TIE_WORD_EMBEDDING="false"  # only used for megatron
TRUST_REMOTE_CODE="false"
USE_CPU_INITIALIZATION="false"
OVERWRITE="false"
DRY_RUN="false"

# You can keep experiments here, or pass --experiment multiple times.
EXPERIMENTS=()

SKIPPED_STEPS=()

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/batch_merge_checkpoints.sh [options]

Options:
  --project NAME                Project name under input root
  --experiment NAME             Experiment name (repeatable)
  --input-root DIR              Input root dir (default: checkpoints)
  --output-root DIR             Output root dir (default: merged_checkpoints)
  --role NAME                   Model role dir (default: actor)
  --step-count N|all            Merge latest N steps, or all (default: all)
  --steps 5,20,75               Explicit step list (overrides --step-count)
  --backend auto|fsdp|megatron  Backend selection (default: auto)
  --python-bin CMD              Python command (default: python)
  --tie-word-embedding          Pass for megatron merge
  --trust-remote-code           Pass to model merger
  --use-cpu-initialization      Pass to model merger
  --overwrite                   Re-merge even if output exists
  --dry-run                     Print commands without executing
  -h, --help                    Show this help

Notes:
  1) Input layout should be like:
     <input-root>/<project>/<experiment>/global_step_xx/<role>/...
  2) If --steps is not set, steps are auto-discovered from global_step_*.
  3) For --step-count N, script merges latest N discovered steps.
USAGE
}

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
  echo "Error: $*" >&2
  exit 1
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

join_by_comma() {
  local first="true"
  for x in "$@"; do
    if [[ "$first" == "true" ]]; then
      printf '%s' "$x"
      first="false"
    else
      printf ',%s' "$x"
    fi
  done
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --project)
        PROJECT_NAME="$2"
        shift 2
        ;;
      --experiment)
        EXPERIMENTS+=("$2")
        shift 2
        ;;
      --input-root)
        INPUT_ROOT="$2"
        shift 2
        ;;
      --output-root)
        OUTPUT_ROOT="$2"
        shift 2
        ;;
      --role)
        ROLE="$2"
        shift 2
        ;;
      --step-count)
        STEP_COUNT="$2"
        shift 2
        ;;
      --steps)
        STEPS_CSV="$2"
        shift 2
        ;;
      --backend)
        BACKEND="$2"
        shift 2
        ;;
      --python-bin)
        PYTHON_BIN="$2"
        shift 2
        ;;
      --tie-word-embedding)
        TIE_WORD_EMBEDDING="true"
        shift
        ;;
      --trust-remote-code)
        TRUST_REMOTE_CODE="true"
        shift
        ;;
      --use-cpu-initialization)
        USE_CPU_INITIALIZATION="true"
        shift
        ;;
      --overwrite)
        OVERWRITE="true"
        shift
        ;;
      --dry-run)
        DRY_RUN="true"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

auto_detect_backend() {
  local local_dir="$1"
  if [[ -f "$local_dir/fsdp_config.json" ]]; then
    echo "fsdp"
  elif [[ -d "$local_dir/dist_ckpt" ]]; then
    echo "megatron"
  else
    die "Cannot auto-detect backend for $local_dir (missing fsdp_config.json or dist_ckpt/)"
  fi
}

list_discovered_steps() {
  local exp_dir="$1"
  find "$exp_dir" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' -printf '%f\n' \
    | sed -nE 's/^global_step_([0-9]+)$/\1/p' \
    | sort -n
}

resolve_steps_for_experiment() {
  local exp_dir="$1"

  if [[ -n "$STEPS_CSV" ]]; then
    tr ',' '\n' <<<"$STEPS_CSV" | sed '/^$/d' | sort -n
    return 0
  fi

  mapfile -t discovered < <(list_discovered_steps "$exp_dir")

  if [[ "${#discovered[@]}" -eq 0 ]]; then
    die "No global_step_* directories found in $exp_dir"
  fi

  if [[ "$STEP_COUNT" == "all" ]]; then
    printf '%s\n' "${discovered[@]}"
    return 0
  fi

  if ! is_positive_int "$STEP_COUNT"; then
    die "--step-count must be 'all' or a positive integer, got: $STEP_COUNT"
  fi

  local count="$STEP_COUNT"
  local total="${#discovered[@]}"
  if (( count >= total )); then
    printf '%s\n' "${discovered[@]}"
    return 0
  fi

  printf '%s\n' "${discovered[@]:total-count:count}"
}

ensure_valid_config() {
  [[ "$BACKEND" == "auto" || "$BACKEND" == "fsdp" || "$BACKEND" == "megatron" ]] \
    || die "--backend must be auto|fsdp|megatron"

  if [[ "${#EXPERIMENTS[@]}" -eq 0 ]]; then
    die "At least one --experiment is required"
  fi

  if [[ ! -d "$INPUT_ROOT/$PROJECT_NAME" ]]; then
    die "Input project dir not found: $INPUT_ROOT/$PROJECT_NAME"
  fi

  mkdir -p "$OUTPUT_ROOT/$PROJECT_NAME"
}

run_merge_one_step() {
  local exp_name="$1"
  local step="$2"

  local local_dir="$INPUT_ROOT/$PROJECT_NAME/$exp_name/global_step_${step}/$ROLE"
  if [[ ! -d "$local_dir" ]]; then
    log "Skip missing checkpoint: $local_dir"
    SKIPPED_STEPS+=("$exp_name:global_step_$step")
    return 0
  fi

  local backend="$BACKEND"
  if [[ "$backend" == "auto" ]]; then
    backend="$(auto_detect_backend "$local_dir")"
  fi

  local target_dir="$OUTPUT_ROOT/$PROJECT_NAME/$exp_name/global_step_${step}/${ROLE}_hf"

  if [[ -d "$target_dir" && -n "$(find "$target_dir" -mindepth 1 -maxdepth 1 -print -quit)" && "$OVERWRITE" != "true" ]]; then
    log "Skip existing output: $target_dir"
    return 0
  fi

  mkdir -p "$target_dir"

  local cmd=("$PYTHON_BIN" -m verl.model_merger merge --backend "$backend" --local_dir "$local_dir" --target_dir "$target_dir")

  if [[ "$backend" == "megatron" && "$TIE_WORD_EMBEDDING" == "true" ]]; then
    cmd+=(--tie-word-embedding)
  fi
  if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
    cmd+=(--trust-remote-code)
  fi
  if [[ "$USE_CPU_INITIALIZATION" == "true" ]]; then
    cmd+=(--use_cpu_initialization)
  fi

  log "Merging: exp=$exp_name step=$step backend=$backend"
  log "Command: ${cmd[*]}"

  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi

  "${cmd[@]}"
}

main() {
  parse_args "$@"
  ensure_valid_config

  local merged_count=0

  for exp_name in "${EXPERIMENTS[@]}"; do
    local exp_dir="$INPUT_ROOT/$PROJECT_NAME/$exp_name"
    [[ -d "$exp_dir" ]] || die "Experiment dir not found: $exp_dir"

    mapfile -t steps < <(resolve_steps_for_experiment "$exp_dir")
    [[ "${#steps[@]}" -gt 0 ]] || die "No step selected for $exp_name"

    log "Selected steps for $exp_name: $(join_by_comma "${steps[@]}")"

    for step in "${steps[@]}"; do
      run_merge_one_step "$exp_name" "$step"
      merged_count=$((merged_count + 1))
    done
  done

  log "Done. Processed $merged_count step(s)."

  if [[ ${#SKIPPED_STEPS[@]} -gt 0 ]]; then
    log "Skipped ${#SKIPPED_STEPS[@]} step(s) due to missing checkpoints:"
    for skipped in "${SKIPPED_STEPS[@]}"; do
      log "  - $skipped"
    done
  fi
}

main "$@"
