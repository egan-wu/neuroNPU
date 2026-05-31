# NeuroNPU — Architecture

NeuroNPU is a **cycle-approximate software model of an NPU**. It executes a coarse
(CISC-like) ISA, models on-chip/off-chip memory and three asynchronous engines,
and emits logs from which performance metrics are derived.

> Scope of this repository: **Simulator + Logging + Interface (ISA/binary/config)**.
> The frontend/compiler/backend (ONNX → NeuroNPU ISA) is a separate project that
> targets the interface defined in [`isa_spec.md`](isa_spec.md) and
> [`binary_format.md`](binary_format.md).

## Block diagram

```
            ┌────────────────────── NeuroNPU Core ──────────────────────┐
  DDR  ⇄    │   DMA Engine  ⇄  SRAM Scratchpad (software-managed, banked) │
(external)  │                        ↓             ↓                      │
            │              Tensor Engine        Vector Engine             │
            │             (systolic MAC array)  (SIMD lanes)              │
            └────────────────────────────────────────────────────────────┘
```

## Components

| Unit | Role | Numerics |
|------|------|----------|
| **Tensor Engine (TE)** | `MATMUL` / convolution on a systolic MAC array (`mac_rows × mac_cols`, weight-stationary). | inputs f16/bf16/i8/f32, **fp32 accumulate** |
| **Vector Engine (VE)** | Elementwise, activations, softmax, norm, requant, RoPE (currently `VADD`, `RELU`). | fp32 math |
| **DMA Engine** | DDR ↔ SRAM transfers. Asynchronous; serialised within its own queue. | — |
| **SRAM scratchpad** | On-chip, **software-managed** (the compiler allocates addresses and issues DMA). Not a cache. | — |
| **DDR** | External memory; the source of all traffic metrics. | — |

## Execution model

Instructions are partitioned into **three in-order per-engine queues** (DMA / TE / VE)
that execute **asynchronously and may overlap** — this is what models double-buffering
(load tile *n+1* while TE computes tile *n*). Cross-engine dependencies are expressed
explicitly by the compiler using **events**: an instruction may *wait* on an event
before issuing and *signal* an event on completion.

Within a single engine, instructions are already serialised in program order, so an
intra-engine dependency needs no event.

## Timing model (cycle-approximate)

The global time unit is **core-clock cycles**. Each instruction's duration comes from
an analytical formula, not per-cycle simulation:

| Op | Cycles |
|----|--------|
| DMA | `bytes / ddr_peak_bytes_per_ns + ddr_latency_ns`, converted to cycles via `core_clock_ghz` |
| MATMUL `[M,K]×[K,N]` | `M·N·K / (mac_rows·mac_cols) + (mac_rows + mac_cols)` (compute + systolic fill) |
| Vector | `elems / vector_lanes + overhead` |

**Functional results are exact** (fp32 accumulate) and fully decoupled from timing.
The known approximation is that op-internal, per-cycle effects (bank conflicts, pipeline
bubbles) are not modeled. The one effect that *is* modeled deliberately is shared-resource
contention; today a single DMA engine serialises DDR access, so there is no
under-counting. Multi-channel DDR contention is a planned extension (see below).

## Logging & metrics

A run produces (see [`../src/logger.cpp`](../src/logger.cpp)):

- `isa_trace.jsonl` — every instruction with `start/end/cycles`, engine, wait/signal, macs/bytes.
- `ddr_trace.csv` — every DDR burst: direction, descriptor, address, bytes, cycles.
- `perf.json` — the analysis below.

The analyzer ([`../src/perf.cpp`](../src/perf.cpp)) derives: total cycles/time, **MAC-array
utilization**, DMA/VE utilization, **DDR achieved vs peak bandwidth**, **arithmetic
intensity**, and a roofline **compute- vs memory-bound** verdict.

## Configuration

All capacity/timing knobs live in [`../configs/default.yaml`](../configs/default.yaml):
core clock, MAC array dims, vector lanes, DDR (freq/width/channels/latency/size),
SRAM (size/banks), feature flags. Loaded at startup.

## Roadmap (post-MVP)

1. VE breadth: `SOFTMAX`, `RMSNORM`/`LAYERNORM`, `GELU`/`SiLU`, `REQUANT`, `ROPE`, `CONV`.
2. `LOOP`/control flow to keep multi-layer LLM binaries compact.
3. Multi-DMA-channel + SRAM port contention modeling (shared-resource arbitration).
4. Multi-core / multi-tile scaling.
5. Optional energy/power estimation.
6. End goal: run a compiler-emitted Llama / YOLO ONNX graph end-to-end.
