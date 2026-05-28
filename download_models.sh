#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# FP8 model downloader for the local compare-lab setup.
#
# Default downloads in this repo:
# - Qwen/Qwen3.5-27B-FP8
# - RedHatAI/gemma-4-31B-it-FP8-block
#
# The script reads HF_TOKEN from the environment or a local .env file.
# Re-running resumes partial downloads automatically.
# -----------------------------------------------------------------------------

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f ".env" ]] && grep -qE '^[[:space:]]*(export[[:space:]]+)?HF_TOKEN=' ".env"; then
    # Load HF_TOKEN from local project env file if present.
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
fi

if ! command -v hf >/dev/null 2>&1 && [[ -x ".venv/bin/hf" ]]; then
  export PATH="$PWD/.venv/bin:$PATH"
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing dependency: 'hf' command not found."
  echo "Install with uv:"
  echo "  uv init"
  echo "  uv venv .venv"
  echo "  source .venv/bin/activate"
  echo "  uv add hf-transfer huggingface-hub"
  echo "Then activate your venv and run this script again."
  exit 127
fi

# High-performance downloads for current huggingface_hub (Xet backend).
export HF_XET_HIGH_PERFORMANCE=1
# Avoid deprecation warning if this variable exists in the user shell.
unset HF_HUB_ENABLE_HF_TRANSFER || true
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN found (from env or .env). Using token for downloads."
else
  echo "HF_TOKEN not found in environment or .env. Continuing without login."
  echo "For private/gated repos, set HF_TOKEN in .env or run:"
  echo "export HF_TOKEN=\"your_hf_token_here\""
fi

download_or_exit() {
  local repo="$1"
  shift

  local rc=0
  hf download "$repo" --type model "$@" || rc=$?
  if [[ $rc -eq 0 ]]; then
    return 0
  fi

  if [[ $rc -eq 130 ]]; then
    echo "Interrupted by Ctrl+C."
    exit 130
  fi

  echo "Failed: $repo (exit $rc)"
  exit "$rc"
}

GGUF_MODELS=(
  "unsloth/Qwen3.6-27B-MTP-GGUF:Q8_0"
)

HF_MODELS=(
  "RedHatAI/gemma-4-31B-it-FP8-block"
)

for SPEC in "${GGUF_MODELS[@]}"; do
  REPO="${SPEC%%:*}"
  QUANT="${SPEC#*:}"
  if [[ "$REPO" == "$QUANT" ]]; then
    QUANT="Q4_K_M"
  fi

  echo "Downloading ${QUANT} for $REPO..."
  download_or_exit "$REPO" --include "*${QUANT}*.gguf"

  echo "Done: $REPO (${QUANT})"
  echo "----------------------------------------"
done

if [[ ${#HF_MODELS[@]} -eq 0 ]] && [[ ${#GGUF_MODELS[@]} -eq 0 ]]; then
  echo "No models selected. Edit download_qwen_models.sh and add entries to HF_MODELS."
  exit 0
fi

for REPO in "${HF_MODELS[@]}"; do
  echo "Downloading full repo for $REPO..."
  download_or_exit "$REPO"

  echo "Done: $REPO"
  echo "----------------------------------------"
done
