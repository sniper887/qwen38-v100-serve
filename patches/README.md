# Patches

This directory contains hardware- and architecture-specific optimizations for `llama.cpp` targeting Qwen3.8-27B on NVIDIA Volta (`sm_70`, e.g., Tesla V100 32GB/16GB, Titan V).

---

## `0001-t2-001-gqa-packing-sm70.patch`

### The Problem

Qwen3.8-27B has 24 Query heads and 4 Key-Value heads, yielding a Grouped-Query Attention (GQA) ratio of:
$$\text{GQA Ratio} = \frac{24}{4} = 6$$

In stock `llama.cpp`, the FlashAttention vector kernel (`flash_attn_ext_vec` in `ggml-cuda/fattn-vec.cuh`) packs query heads sharing a KV head into GPU thread blocks using a powers-of-two loop (`ncols2 = 1, 2, 4, 8`). Because 6 is not a power of two ($6 = 2 \times 3$), the stock loop stops at `ncols2 = 2`.

This leaves an unoptimized factor of 3: each KV head is read into registers across 3 separate thread blocks. During single-stream autoregressive token decode at deep context (e.g., 128K context), memory bandwidth dominates execution time. The redundant reads generate **26.37 GB of DRAM traffic per decode step**.

### The Fix

This patch adds specialized `ncols2 = 3` support to `flash_attn_ext_vec`:
1. **Packs 3 Query heads per block** whenever `gqa_ratio % 3 == 0` and `ne02 % 3 == 0`.
2. **Maintains register limits**: Volta `sm_70` allows at most 255 registers per thread. To prevent register spilling when accumulating 3 heads, the patch spreads each row across threads using `D / (2 * cpy_ne)` (head dimension $D = 256$).
3. **Restricted to unquantized KV (`F16`/`BF16`)**: Quantized KV (`q8_0`) already operates at 252–255 registers; packing 3 heads on quantized KV spills registers into local memory. The patch cleanly branches via `if constexpr (packing_supported)` to preserve stock quantized behavior.

### Performance & Quality Results

Measured on NVIDIA Tesla V100-32GB (single stream, temp 0, `n_gen=128`):

| Context Depth | Stock Decode (tok/s) | Patched Decode (tok/s) | Speedup | KV DRAM Traffic / Token |
|:-------------:|:--------------------:|:----------------------:|:-------:|:-----------------------:|
| 1K            | 35.95                | 35.22                  | -2.0%   | —                       |
| 8K            | 34.42                | 34.39                  | within noise | —                  |
| 32K           | 30.29                | 32.04                  | **+5.8%** | —                    |
| 64K           | 25.63                | 29.04                  | **+13.3%** | —                   |
| 128K          | 16.45                | **23.83**              | **+44.9%** | **26.37 GB -> 8.59 GB** |

- **Numerical equivalence**: Mean KL-Divergence = `0.000000`, Same Top-1 Token = `100.000%` across the entire evaluation corpus.
- **Trade-off**: -2.0% throughput below 8K context due to fewer output blocks at shallow depths, traded for a +44.9% gain at 128K context.

### Why MTP Uses the Stock Build

Multi-Token Prediction (`--spec-type draft-mtp`) drafts 3 to 7 tokens and verifies them in a single batch forward pass. The 4-wide verification step naturally amortizes KV head reads across query tokens. Profiling confirmed that applying T2-001 on top of MTP yielded only ~1% improvement. Therefore:
- **`./serve.sh` (default, MTP on)** runs the **stock build** (unpatched).
- **`./serve.sh --no-mtp`** runs the **T2-001 patched build** for deterministic, non-speculative decode.

---

### Applying the Patch Manually

To apply this patch to a clean `llama.cpp` tree at commit `d230ddd` (tag `b10793`):

```bash
cd /path/to/llama.cpp
git apply /path/to/qwen38-v100-serve/patches/0001-t2-001-gqa-packing-sm70.patch
```
