# Qwen3.8-27B Serving on NVIDIA Tesla V100-32GB

A production-ready, distributable serving stack for **Qwen3.8-27B** on NVIDIA Volta (**sm_70**, e.g., Tesla V100 32GB/16GB, Titan V) powered by optimized `llama.cpp`.

```bash
# 1. Verify your environment against stack requirements
./check-env.sh

# 2. Build stock & patched binaries
./build.sh

# 3. Lock GPU clocks for sustained maximum throughput (once per boot)
sudo ./lock-clocks.sh lock

# 4. Start serving
./serve.sh                  # MTP on  — fastest (42.2 tok/s @ 128K)
./serve.sh --no-mtp         # MTP off — deterministic non-speculative (T2-001 packed build)
```

Exposes an OpenAI-compatible API endpoint at `http://127.0.0.1:8080`.

---

## Benchmark Performance

Single-stream autoregressive token decode throughput (tokens/second) measured at temperature 0 on an agentic corpus (code, tool-call JSON, shell, markdown) on a single Tesla V100-SXM2-32GB:

| Context Depth | Stock llama.cpp | `--no-mtp` (T2-001 Packed) | Default (`--mtp`) | Speedup vs Stock |
|:---:|:---:|:---:|:---:|:---:|
| **1K**   | 35.95 tok/s | 35.22 tok/s | **65.07 tok/s** | **+81.0%** |
| **8K**   | 33.74 tok/s | 34.39 tok/s | **62.41 tok/s** | **+85.0%** |
| **32K**  | 29.73 tok/s | 32.04 tok/s | **61.24 tok/s** | **+106.0%** |
| **64K**  | 25.18 tok/s | 29.04 tok/s | **52.51 tok/s** | **+108.5%** |
| **128K** | 16.49 tok/s | 23.83 tok/s | **42.22 tok/s** | **+156.0%** |

### Multi-Turn Agentic Sessions (Prompt Caching Win)

In agentic workflows where consecutive turns append to conversation history, prompt caching avoids reprocessing shared prompt prefixes:

| Serving Mode | Wall Time per Turn | Generation Speed |
|---|---|---|
| MTP (cold prefill every turn) | 22.94 s | 64.25 tok/s |
| **MTP + Prompt Caching (`--cache-prompt`)** | **3.08 s** | **64.18 tok/s** |
| **Net Speedup** | **7.45× faster wall time** | — |

---

## Architectural Rationale: Why Two Builds?

Serving this model at peak efficiency requires two distinct binaries depending on whether speculative decoding is enabled:

```
                          ┌────────────────────────────┐
                          │    Serving Qwen3.8-27B     │
                          └─────────────┬──────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
       [ MTP Enabled (Default) ]                     [ MTP Disabled (--no-mtp) ]
       - Speculative draft-mtp                       - Non-speculative, deterministic
       - Drafts 3-7 tokens per pass                  - Autoregressive 1-token decode
       - Batch verify amortizes KV reads             - T2-001 GQA Packing (ncols2=3)
       - Runs STOCK llama.cpp build                  - Eliminates 3x KV DRAM traffic
       - 42.22 tok/s @ 128K                          - 23.83 tok/s @ 128K (+44.9% over stock)
```

### 1. Default Mode: MTP on → Stock Build
Qwen3.8-27B includes an embedded Multi-Token Prediction (MTP) head in the GGUF (`blk.64`). In this mode, the server drafts multiple candidate tokens per forward pass and verifies them in a single batch.
- **Why stock?** Profiling shows that MTP's 4-wide verification step already amortizes KV reads across 4 query positions (30.0 µs/token vs 102.9 µs/token for single-token decode). Layering the T2-001 packing patch on top of MTP yielded only ~1% improvement. Running unpatched stock code is therefore optimal.

### 2. `--no-mtp` Mode: MTP off → T2-001 Packed-Attention Build
For deterministic argmax decoding, bit-exact reproducibility, or debugging without speculation:
- **The Problem**: Qwen3.8-27B has 24 Query heads and 4 KV heads ($24 / 4 = 6$ GQA ratio). Stock `llama.cpp`'s kernel selector only packs powers of two, packing 2 heads and reading each KV head 3 times into GPU registers.
- **The Fix**: The T2-001 patch (`patches/0001-t2-001-gqa-packing-sm70.patch`) adds `ncols2 = 3` support to `flash_attn_ext_vec`, eliminating the 3× redundant KV reads.
- **Result**: At 128K context, KV DRAM traffic drops from 26.37 GB to 8.59 GB per token, boosting non-speculative decode by **+44.9%** (16.45 → 23.83 tok/s).

---

## Stack Requirements Specification

See [`stack-requirements.txt`](file:///home/guthix/Projects/qwen38-v100-serve/stack-requirements.txt) for machine-parseable requirements.

| Component | Requirement | Critical Notes |
|---|---|---|
| **GPU Architecture** | NVIDIA **sm_70** (Volta) | Tesla V100 32GB/16GB, Titan V. 32GB required for 128K context; 16GB supports up to 32K. |
| **NVIDIA Driver** | **≥ 525.60.13, ≤ 580.xx** | **CRITICAL**: Driver branch R580 is the final driver release with Volta support. Upgrades past 580xx remove sm_70 support. |
| **CUDA Toolkit** | **CUDA 12.0 – 12.9** | **CRITICAL**: CUDA 13.x dropped `sm_70` compilation entirely. Must use CUDA 12.x. |
| **Host C/C++ Compiler**| **GCC 11 – 13** | GCC 14+ is incompatible with CUDA 12 `nvcc` headers. |
| **Build System** | CMake ≥ 3.22, Git, Make/Ninja | `ccache` strongly recommended for fast rebuilds. |
| **Model Weights** | `Qwen3.8-27B-UD-Q4_K_M.gguf` | Must retain the `blk.64` MTP module (e.g. Unsloth Q4_K_M or larger). |
| **Base llama.cpp** | Pinned tag `b10793` (`d230ddd`) | Verified clean upstream release. |

Validate your local environment at any time by running:
```bash
./check-env.sh
```

---

## Quick Start Guide

### 1. Acquire Model Weights
Obtain `Qwen3.8-27B-UD-Q4_K_M.gguf` from Hugging Face:
```bash
# Using huggingface-cli / hf
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_M.gguf --local-dir ./models
```
*(Note: Quants below Q3 strip the `blk.64` MTP head to save space. Use `UD-Q4_K_M` or higher to retain MTP).*

### 2. Configure Environment
Copy the configuration template:
```bash
cp serve.env.example .env
```
Edit `.env` to set your model path and any custom toolchain paths:
```ini
MODEL="/path/to/models/Qwen3.8-27B-UD-Q4_K_M.gguf"
# CUDA_HOME="/usr/local/cuda-12.9"  # If CUDA 12 is not in standard system PATH
# GCC_HOME="/usr/local/gcc-13"      # If host compiler is newer than GCC 13
```

### 3. Build Binaries
Run `build.sh` to configure and build both stock and patched binaries:
```bash
./build.sh
```
*(Options: `./build.sh --stock` for stock only; `./build.sh --patched` for patched only).*

### 4. Lock GPU Clocks
Lock clocks to maximum frequency to prevent thermal/power frequency drift:
```bash
sudo ./lock-clocks.sh lock
```

### 5. Launch the Server
```bash
./serve.sh
```

---

## Configuration Reference

### Command-Line Arguments for `serve.sh`

| Argument | Description | Default |
|---|---|---|
| `--mtp` | Enable MTP speculative decoding (uses stock build) | Enabled |
| `--no-mtp` | Disable MTP (uses T2-001 packed-attention build) | Disabled |
| `--draft-max N` | Max speculative draft tokens for MTP | `3` (optimal 0–64K; use `7` @ 128K) |
| `--ctx N` | Context window size in tokens | `131072` (128K) |
| `--port N` | HTTP port to bind | `8080` |
| `--host H` | Network interface to bind | `127.0.0.1` |
| `--device N` | GPU index for `CUDA_VISIBLE_DEVICES` | `0` |
| `--model PATH` | Path to GGUF model file | Sourced from `.env` or defaults |
| `--builds PATH`| Path to directory containing builds | `./builds` |
| `--bin PATH` | Explicit path to `llama-server` binary | Auto-detected |

Any additional flags (e.g. `--api-key secret`, `--ssl-key-file cert.key`) are passed through directly to `llama-server`.

### Environment Variables

All settings can be specified via environment variables or inside `.env` / `serve.env`:

| Variable | Description |
|---|---|
| `MODEL` | Path to the Qwen3.8-27B GGUF file |
| `CUDA_HOME` | Path to CUDA 12.x toolkit (e.g. `/usr/local/cuda-12.9`) |
| `GCC_HOME` | Path to GCC 11–13 toolchain (e.g. `/usr/local/gcc-13`) |
| `CUDA_VISIBLE_DEVICES` | Specific GPU device index to expose |
| `BUILDS_DIR` | Location of built binaries |
| `CTX` | Context window length in tokens |
| `MTP` | `1` for MTP, `0` for non-speculative |
| `DRAFT_MAX` | Number of draft tokens |

---

## Client API Integration

The server provides an OpenAI-compatible REST API.

### cURL Example
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a fast CUDA reduction kernel in C++."}
    ],
    "temperature": 0.0
  }'
```

### Python OpenAI SDK Example
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Explain grouped-query attention in 2 sentences."}
    ],
    temperature=0.0
)
print(response.choices[0].message.content)
```

---

## Critical Operational Traps

1. **`--parallel 1` is mandatory**: `llama-server` defaults to 4 concurrent slots and allocates KV cache for each slot up to `n_ctx`. At 128K context, multiple slots quadruple memory commitment and trigger immediate out-of-memory (OOM) aborts. `serve.sh` pins this to 1.
2. **`CUDA_VISIBLE_DEVICES=0` isolation**: On systems with secondary non-sm_70 GPUs (e.g., Pascal or display adapters), sm_70-only builds abort with `"no kernel image is available for execution on the device"` if other devices are visible.
3. **Prompt caching (`--cache-prompt`) is critical**: Multi-turn wall clock time drops from 22.94s to 3.08s per turn because prior turns' KV states are retained rather than recomputed.
4. **Lock GPU clocks (`./lock-clocks.sh lock`)**: Consumer enclosures or adapter-mounted V100s drift between 135 MHz and 1530 MHz without clock locks, causing throughput to wander unpredictably.
5. **Driver upgrades**: Never upgrade the NVIDIA driver beyond the 580xx branch on Volta machines.

---

## License & Provenance

- **llama.cpp**: Licensed under MIT.
- **Optimization Provenance**: Developed and verified in `../qwen-3-8-27b-inference-optimization` across experiments T0-001 through T2-002.
