# NeuroNPU

A **cycle-approximate software simulator for an NPU**, written in C++17.

It executes a coarse, CISC-like ISA on a model with a systolic Tensor Engine, a Vector
Engine, a DMA engine, a software-managed SRAM scratchpad and external DDR — then logs
traces and computes performance metrics (utilization, bandwidth, roofline).

This repo is the **simulator + logging + interface**. A separate project (your offline
compiler) translates ONNX models into NeuroNPU ISA and targets the interface defined here.

## Layout

```
docs/        architecture.md · isa_spec.md · binary_format.md   ← the contract
configs/     default.yaml                                       ← tunable HW knobs
include/     public headers (neuronpu/*.h)
src/         simulator implementation
tests/       isa_programs/*.npuasm  (hand-written validation)
```

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```sh
# assemble + simulate a text program, write logs, print perf report
./build/neuronpu run tests/isa_programs/matmul.npuasm \
    --config configs/default.yaml --out build/logs --dump Y

# compile text ISA to the binary contract, then run the binary
./build/neuronpu asm tests/isa_programs/matmul.npuasm build/matmul.npubin
./build/neuronpu run build/matmul.npubin --out build/logs

# timing-only: skip functional compute, keep all timing/traffic/metrics
# (lets very large models be profiled in milliseconds)
./build/neuronpu run prog.npubin --out build/logs --timing-only

# golden comparison: inject input tensors and dump outputs as fp32 .npy
./build/neuronpu run prog.npubin --out build/logs \
    --load input=in.npy --save logits=out.npy
```

The bundled `matmul.npuasm` computes `Y = A @ B` (A = iota, B = ones), so
`Y[m,n] = 4096·m + 2016`; `--dump Y` should print `2016 …` for row 0. Outputs land in
`--out`: `isa_trace.jsonl`, `ddr_trace.csv`, `perf.json`, and `trace.json` (drag into
<https://ui.perfetto.dev> for the per-engine time sequence).

## LLM demo (TTFT / TPS / roofline)

[`tools/llm_demo.py`](tools/llm_demo.py) generates a small multi-layer transformer as
NeuroNPU ISA, runs **prefill** (process the prompt) and **decode** (one token, weights
streamed from DDR), and reports LLM-level metrics:

```sh
python3 tools/llm_demo.py --neuronpu build/neuronpu --config configs/default.yaml \
    --layers 6 --d 64 --ffn 256 --prompt 128 --gen 64
```

It derives **TTFT** (= prefill time), **TPS** (= 1 / decode-step time), per-phase **MAC
utilization**, **DDR bandwidth utilization**, **arithmetic intensity**, a roofline verdict,
and **energy per token**. The characteristic result falls straight out of the model:
prefill is **compute-bound**, decode is **memory-bound** (weights dominate per token).
Absolute TPS scales with model size — point `--config` at your own DDR/clock numbers and
grow `--layers/--d/--ffn` toward real dimensions. (`.npuasm`/`.npubin`/logs land in
`--outdir`.)

## Offline compiler (ONNX/Llama → ISA → profile)

The `compiler/` package lowers a coarse Graph IR to NeuroNPU ISA (bufferization,
DMA weight staging, cross-engine event scheduling, codegen) and is validated
bit-exact against a numpy reference:

```sh
python3 -m compiler.validate          # tiny Llama layer: compile, run, diff vs numpy
```

It profiles a **real Llama** from its `config.json` (architecture-driven frontend:
real dims/layer count → real-scale IR → ISA → timing-only profile):

```sh
# download just the 0.5 MB graph/config (weights not needed for timing)
python3 - <<'PY'
from huggingface_hub import hf_hub_download; import shutil,os
os.makedirs("models/llama32_1b",exist_ok=True)
for f in ["onnx/model.onnx","config.json"]:
    shutil.copy(hf_hub_download("onnx-community/Llama-3.2-1B",f),
                "models/llama32_1b/"+os.path.basename(f))
PY
python3 -m compiler.profile_llama --config-json models/llama32_1b/config.json
```

reporting per-phase MAC/DDR utilization, roofline, energy, **TTFT**, and a
**TPS-vs-context** curve. Llama-3.2-1B FP32 on the default config comes out
prefill **compute-bound** (~93% MAC util) and decode **memory-bound** (~5.4 GB
weights streamed per token), exactly as expected.

## Documentation

- [docs/architecture.md](docs/architecture.md) — engines, execution model, timing model.
- [docs/isa_spec.md](docs/isa_spec.md) — the ISA your compiler emits.
- [docs/binary_format.md](docs/binary_format.md) — the `.npubin` container layout.

## Status

Full coarse ISA implemented: DMA · MATMUL · CONV · VADD/RELU/GELU/SiLU/SOFTMAX/
RMSNORM/LAYERNORM/REQUANT/ROPE · event sync · nested LOOPs · multi-core (private SRAM).
The discrete-event scheduler models DDR + per-core SRAM bandwidth contention; the perf
report covers utilization, roofline, and an energy/power estimate. A hand-written
single-layer transformer block
([tests/isa_programs/transformer_block.npuasm](tests/isa_programs/transformer_block.npuasm))
runs the whole stack end to end. See the roadmap in
[docs/architecture.md](docs/architecture.md); remaining work is on the compiler side.
