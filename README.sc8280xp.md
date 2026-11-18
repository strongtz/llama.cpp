# llama.cpp for Radxa Dragon Q8B — Hexagon V68 NPU prefill + Adreno 690 GPU decode

This is the `sc8280xp-v68` branch of [strongtz/llama.cpp](https://github.com/strongtz/llama.cpp/tree/sc8280xp-v68),
developed by [Radxa](https://radxa.com) for the [Dragon Q8B](https://radxa.com/products/accessories/dragon-q8b/)
single-board computer (Snapdragon 8cx Gen 3 / SC8280XP).

The main contribution is a production-quality hybrid inference path:
prompt prefill on the **Hexagon V68 NPU** (HMX + HVX), token decode on the **Adreno 690 GPU** (OpenCL).
GPU and NPU share a single copy of the model weights via dma-bufs, saving 4–10 GB of memory.
Multimodal (image) inputs are supported: the vision encoder also runs on the NPU.

Patches relative to upstream llama.cpp are in `patches/llama.cpp/` (0001–0014).
Build scripts and board deployment helpers are in `deploy/`.

## Hardware
- SoC: Qualcomm SC8280XP (Snapdragon 8cx Gen 3)
- NPU: Hexagon V68, 4 HVX lanes, 1 HMX unit, 2×4 MB VTCM
- GPU: Adreno 690 (~50 GB/s DDR BW with the DT patch)
- RAM: 15 GB LPDDR5

## How it works
- **Prefill** (prompt): Hexagon V68 NPU via `HTP0` backend (`LLAMA_NPU_PREFILL=HTP0`).
  Prompt batches ≥ 32 tokens go to the NPU; the sequence state is then copied back to the
  GPU context. For multimodal inputs the vision encoder runs on the NPU too.
- **Decode**: Adreno 690 GPU via OpenCL (`-dev GPUOpenCL -ngl 99`).
- **Weight sharing**: MoE expert weights (Ling, gpt-oss) and dense layer weights (Qwen3.5)
  are stored once in `OpenCL_Shared` dma-bufs that the NPU maps directly.

## Changes
1. ggml-hexagon V68 port: emulated v73 int<->float vector conversions (exact, integer only;
   v68 qfloat is not exact), VTCM split into 2x4MB pages, UDMA must not set cache-bypass on
   VTCM addresses, HMX bias/scale stage absent on v68 (FA folds scale into Q), HMX operands
   kept in one VTCM page.
2. 32-bit DSP SMMU window: buffers <=256MB, on-demand mapping with eviction within budget.
3. Kernel fastrpc: contexts left pending by aborted/crashed processes leaked every mapping
   (IOVA exhaustion until reboot) -> fixed in fastrpc_device_release.
4. llama-server: NPU prefill context.
5. ggml-cpu: single-query flash attention for shared KV heads (MLA): blocked grouped-head
   NEON fp16 kernel (decode @16K ctx 10.6 -> 21 t/s).

## Models tested
| Model | Size | Prefill (NPU) | Decode (GPU) | Memory |
|---|---|---|---|---|
| Ling-3.0-tiny-Q4_0 | 4.5 GB | 420–500 t/s | 26–31 t/s | 7.5 GB |
| Qwen3.5-9B-MTP-Q4_0 | 4.8 GB | 205–218 t/s | 10–14 t/s (MTP) | 8.4 GB |
| gpt-oss-20b-Q4_0 | 11.3 GB | 240–290 t/s | 20–22 t/s | 14.1 GB |

Qwen3.5 and Ling include MTP speculative decoding (`SPEC_MTP=2`). All figures from the
OpenAI-compatible `llama-server` on the Dragon Q8B.

## Quick start

**Prerequisites:** Docker, an arm64 cross-compiler, and a board running Arch or Debian on the SC8280XP.

```sh
# 1. Clone
git clone -b sc8280xp-v68 https://github.com/strongtz/llama.cpp

# 2. Build (uses the public Snapdragon docker toolchain)
deploy/build-all.sh   # outputs pkg-snapdragon/

# 3. Deploy and run on the board
scp -r pkg-snapdragon arch@<board>:~/ling/
ssh arch@<board>
MODEL=~/ling/Qwen3.5-9B-MTP-Q4_0.gguf CTX=16384 GPU_UB=8 SPEC_MTP=2 \
  LLAMA_NPU_PREFILL_UBATCH=512 LLAMA_NPU_PREFILL_MIN=32 \
  ~/ling/run-server.sh       # OpenAI-compatible API on port 8080
```

For multimodal (image inputs), add `MMPROJ=~/ling/mmproj-Qwen3.5-9B-F16.gguf` to the command above.

The build requires the Docker image `ghcr.io/snapdragon-toolchain/arm64-linux:v0.8`,
which bundles the Hexagon SDK — the same one used in [upstream CI](https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/build-and-test-snapdragon.yml).

## Rebuild
deploy/build-all.sh (docker image ghcr.io/snapdragon-toolchain/arm64-linux:v0.8), then
`sudo systemctl restart ling-server` on the board.
