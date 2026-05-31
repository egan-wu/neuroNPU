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
```

The bundled `matmul.npuasm` computes `Y = A @ B` (A = iota, B = ones), so
`Y[m,n] = 4096·m + 2016`; `--dump Y` should print `2016 …` for row 0. Outputs land in
`--out`: `isa_trace.jsonl`, `ddr_trace.csv`, `perf.json`.

## Documentation

- [docs/architecture.md](docs/architecture.md) — engines, execution model, timing model.
- [docs/isa_spec.md](docs/isa_spec.md) — the ISA your compiler emits.
- [docs/binary_format.md](docs/binary_format.md) — the `.npubin` container layout.

## Status

MVP vertical slice (DMA · MATMUL · VADD/RELU · event sync · logging · perf). See the
roadmap in [docs/architecture.md](docs/architecture.md) toward Llama / YOLO support.
