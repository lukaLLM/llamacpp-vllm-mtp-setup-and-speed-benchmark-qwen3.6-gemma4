# MTP Benchmark Runbook

Local benchmark setup for comparing `NO MTP` vs `MTP` token throughput using:
- `llama.cpp` + `Qwen3.6-27B-MTP-GGUF`
- `vLLM` + `Gemma 4 31B FP8`

> **Note:** `llama.cpp` does **not** support Gemma 4 as of the date this repo was created. Gemma 4 benchmarks use `vLLM` only.

## YouTube Videos
-  Over 3x Faster AI. MTP Explained, Deployed & Benchmarked on Gemma 4 & Qwen 3.6 - https://www.youtube.com/watch?v=vN3At9GuSnc

## What This Repo Does

- compares `NO MTP` vs `MTP` throughput on local OpenAI-compatible endpoints
- supports `llama.cpp` Qwen3.6 MTP GGUF flows and `vLLM` Gemma 4 FP8 flows
- includes a bonus `vLLM` Qwen3.6 27B FP8 NO MTP vs MTP comparison (vLLM was also used for Qwen, not just Gemma 4)
- includes side-by-side and single-endpoint benchmark scripts
- stores per-request and per-run benchmark history in CSV:
  - `visualization/comparison_runs.csv`
  - `visualization/leaderboard_runs.csv`

## Requirements

Tested on: `Linux 6.14.0-29-generic #29~24.04.1-Ubuntu x86_64 GNU/Linux` — [Ubuntu 24.04 Desktop](https://ubuntu.com/download/desktop)

| Requirement | Notes | Install guide |
|---|---|---|
| Linux (Ubuntu 24.04) | tested OS | [Ubuntu Desktop](https://ubuntu.com/download/desktop) |
| NVIDIA GPU + driver | CUDA-capable GPU with a supported driver | [NVIDIA Driver Install Guide (Ubuntu)](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#ubuntu) |
| Docker Engine with `docker compose` | used to run llama.cpp and vLLM containers | [Docker Engine Install (Ubuntu)](https://docs.docker.com/engine/install/ubuntu/) |
| NVIDIA Container Toolkit | allows Docker containers to access the GPU | [NVIDIA Container Toolkit Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| Python `3.12+` | for benchmark scripts | — |
| `uv` | Python package/env manager | — |
| VSCode (or similar editor) | optional, for editing configs and scripts | [VSCode Install (Linux)](https://code.visualstudio.com/docs/setup/linux) |
| Hugging Face token | optional, needed for gated models | set as `HF_TOKEN=...` in repo-root `.env` |

## Quick Setup

1. Setup Python environment:

```bash
cd /home/luke/Documents/Code/MTP
uv venv .venv
source .venv/bin/activate
uv sync
```

2. Configure Hugging Face token in repo-root `.env`:

```bash
HF_TOKEN=your_hf_token_here
```

3. Download models:

```bash
./download_models.sh
```

4. Start servers:

```bash
docker compose --env-file .env -f docker/docker-compose_mtp_qwen3.6_27B.yaml up -d
docker compose --env-file .env -f docker/docker-compose_no_mtp_qwen3.6_27B.yaml up -d
docker compose --env-file .env -f docker/docker-compose_mtp_gemma4_31B.yaml up -d gemma4_mtp
```

5. Run benchmarks:

```bash
python3 visualization/app.py --help
python3 visualization/leaderboard.py --help
```

Optional check:

```bash
python3 --version
```

## 1) Setup With `uv`

```bash
cd /home/luke/Documents/Code/MTP
uv venv .venv
source .venv/bin/activate
uv sync
```

Optional check:

```bash
python3 --version
python3 visualization/app.py --help
python3 visualization/leaderboard.py --help
```

## 2) Configure Hugging Face Token

Create/update `.env` in repo root:

```bash
HF_TOKEN=your_hf_token_here
```

## 3) Download Models (Script)

Use the bundled downloader:

```bash
./download_models.sh
```

This script downloads:
- `unsloth/Qwen3.6-27B-MTP-GGUF` (`Q8_0` GGUF)
- `RedHatAI/gemma-4-31B-it-FP8-block`

Notes:
- Script reads `HF_TOKEN` from shell env or local `.env`.
- Re-running resumes partial downloads.

## 4) Start Servers

### Qwen (llama.cpp) MTP on `:8000`

```bash
docker compose --env-file .env -f docker/docker-compose_mtp_qwen3.6_27B.yaml up -d
```

### Qwen (llama.cpp) NO MTP on `:8001`

```bash
docker compose --env-file .env -f docker/docker-compose_no_mtp_qwen3.6_27B.yaml up -d
```

### Gemma 4 (vLLM) MTP on `:8000`

```bash
docker compose --env-file .env -f docker/docker-compose_mtp_gemma4_31B.yaml up -d gemma4_mtp
```

### Gemma 4 (vLLM) NO MTP on `:8000`

Run one service at a time on the same port:

```bash
docker compose --env-file .env -f docker/docker-compose_mtp_gemma4_31B.yaml up -d gemma4_default
```

### Qwen3.6 27B FP8 (vLLM) — NO MTP vs MTP Comparison

A bonus compose file runs `Qwen/Qwen3.6-27B-FP8` through `vLLM`, allowing a direct NO MTP vs MTP comparison for Qwen in FP8 precision via vLLM.

Two services are defined — run **one at a time** on the same port:

**NO MTP** on `:8000`:
```bash
docker compose --env-file .env -f docker/docker-compose_qwen3.6_27B_FP8_compare.yaml up -d qwen36_default
```

**MTP** (4 speculative tokens) on `:8000`:
```bash
docker compose --env-file .env -f docker/docker-compose_qwen3.6_27B_FP8_compare.yaml up -d qwen36_mtp
```

Key settings used in this compose file:
- Image: `vllm/vllm-openai:nightly`
- Model: `Qwen/Qwen3.6-27B-FP8`
- `--max-model-len 3000`, `--gpu-memory-utilization 0.45`, `--max-num-seqs 1`
- `--reasoning-parser qwen3`, `--no-enable-prefix-caching`
- MTP service adds: `--speculative-config '{"method":"mtp","num_speculative_tokens":4}'`

Stop when done:
```bash
docker compose -f docker/docker-compose_qwen3.6_27B_FP8_compare.yaml down
```

### Health checks

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8001/health
curl -sS http://127.0.0.1:8000/v1/models
```

## 5) Visualization App (`visualization/app.py`)

Compares two endpoints side by side and writes per-request rows to:
- `visualization/comparison_runs.csv`

### Example

```bash
python3 visualization/app.py \
  --mtp-url http://127.0.0.1:8000 \
  --no-mtp-url http://127.0.0.1:8001 \
  --runs 10 \
  --session-name Qwen3.6_27B_LLAMA_n_max_5 \
  --sequential
```

Example result:

```text
NO MTP avg:  43.22 tok/s
MTP avg:    117.95 tok/s
MTP speedup:2.73x
Saved per-request rows to: /home/luke/Documents/Code/MTP/visualization/comparison_runs.csv
```

### All flags

- `--no-mtp-url`, `--default-url`: NO MTP server URL (default `http://localhost:8001`)
- `--mtp-url`, `--onebonsai-url`: MTP server URL (default `http://localhost:8000`)
- `--runs`: number of benchmark requests (default `20`)
- `--n-predict`: generated tokens per request (default `1500`)
- `--use-full-context`: derive generation budget from `context-size - prompt - reserve`
- `--context-size`: server context window from Docker config (default `3000`)
- `--context-reserve`: reserved tokens headroom (default `128`)
- `--seed`: generation seed (default `1234`)
- `--output-lines`: visible output lines in UI (default `16`)
- `--prompt-lines`: visible prompt lines in UI (default `4`)
- `--session-name`: label stored in CSV (default `compare-YYYYmmdd-HHMMSS`)
- `--csv-file`: output CSV path (default `visualization/comparison_runs.csv`)
- `--sequential`: benchmark endpoints one after another instead of in parallel
- `--prompt`: prompt text used for benchmark

## 6) Leaderboard App (`visualization/leaderboard.py`)

Runs repeated requests against one endpoint and stores history in:
- `visualization/leaderboard_runs.csv`

### Example

```bash
python3 visualization/leaderboard.py \
  --url http://127.0.0.1:8000 \
  --run-name gemma4-mtp-3 \
  --api openai \
  --runs 10
```

Example result:

```text
Saved run 'gemma4-mtp-3' to /home/luke/Documents/Code/MTP/visualization/leaderboard_runs.csv | avg 124.26 tok/s | total 4740 tokens in 38.14s
```

### All flags

- `--url`: server URL (default `http://127.0.0.1:8000`)
- `--run-name`: required run label for grouping/history
- `--api`: `auto`, `completion`, or `openai` (default `auto`)
- `--model`: OpenAI model id (auto-detected when omitted)
- `--runs`: number of requests (default `10`)
- `--n-predict`: generation tokens for `/completion` API (default `1500`)
- `--max-tokens`: generation tokens for OpenAI chat API (default `1500`)
- `--use-full-context`: derive generation budget from `context-size - prompt - reserve`
- `--context-size`: server context window (default `8192`)
- `--context-reserve`: reserved tokens headroom (default `128`)
- `--seed`: generation seed (default `1234`)
- `--timeout`: request timeout in seconds (default `900`)
- `--history-file`: CSV history path (default `visualization/leaderboard_runs.csv`)
- `--show-top`: leaderboard rows to display (default `20`)
- `--output-lines`: visible output lines in UI (default `16`)
- `--prompt-lines`: visible prompt lines in UI (default `5`)
- `--prompt`: prompt text used for benchmark

## 7) Latest leaderboard snapshot

```text
┏━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━┓
┃ rank ┃ run           ┃ avg tok/s ┃ median ┃ tokens ┃ seconds ┃ runs ┃
┡━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━┩
│    1 │ gemma4-mtp-5  │    132.52 │ 133.10 │   4750 │   35.85 │   10 │
│    2 │ gemma4-mtp-4  │    129.82 │ 134.05 │   4750 │   37.07 │   10 │
│    3 │ qwen36-mtp-5  │    127.31 │ 126.68 │  13210 │  103.77 │   10 │
│    4 │ gemma4-mtp-3  │    124.26 │ 124.25 │   4740 │   38.14 │   10 │
│    5 │ qwen36-mtp-4  │    121.96 │ 122.01 │  12840 │  105.28 │   10 │
│    6 │ gemma4-mtp-2  │     94.00 │  96.42 │   4730 │   50.70 │   10 │
│    7 │ qwen36-no-mtp │     49.23 │  49.25 │  14530 │  295.12 │   10 │
│    8 │ gemma4-no-mtp │     39.69 │  39.54 │   5120 │  129.05 │   10 │
└──────┴───────────────┴───────────┴────────┴────────┴─────────┴──────┘
```

## 8) Stop servers

```bash
docker compose -f docker/docker-compose_mtp_qwen3.6_27B.yaml down
docker compose -f docker/docker-compose_no_mtp_qwen3.6_27B.yaml down
docker compose -f docker/docker-compose_mtp_gemma4_31B.yaml down
```

## References

- https://x.com/googlegemma/status/2051694045869879749
- https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
- https://ai.google.dev/gemma/docs/mtp/mtp
