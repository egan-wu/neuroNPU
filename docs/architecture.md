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
| DMA | `ddr_latency_ns` (fixed) + `bytes / effective_bandwidth`, where bandwidth is **shared fairly** among concurrent transfers |
| MATMUL `[M,K]×[K,N]` | `M·N·K / (mac_rows·mac_cols) + (mac_rows + mac_cols)` (compute + systolic fill) |
| Vector | `elems · passes / vector_lanes + overhead` (`passes` ≈ op cost: softmax≈5, norm≈3-4) |

The scheduler is a **discrete-event simulator**. Compute ops occupy their (single) engine
for a fixed duration. DMA is different: up to `dma_channels` transfers run concurrently and
**share total DDR bandwidth** via progressive (fluid) filling — bandwidth is re-divided
whenever the set of active transfers changes. So `dma_channels: 2` overlaps two loads but
each runs at ~half bandwidth: faster than serial, but **not** 2×. This is the modeled
DDR-contention effect.

**SRAM port contention** is also modeled, **per core**: each core has a private SRAM
scratchpad with its own aggregate bandwidth (`sram_banks × sram_bank_width_bytes`
bytes/cycle). Every engine on that core touching SRAM (its DMA transfers and compute ops)
draws on it; when a core's demand in an interval exceeds capacity, that core's active ops
are throttled by the same factor (a first-order shared-port model). With the default
config SRAM is provisioned above demand, so it only bites under heavy overlap or a starved
config. DDR bandwidth, by contrast, is shared **globally** across all cores.

**Functional results are exact** (fp32 accumulate) and fully decoupled from timing.
The known approximation is that op-internal, per-cycle effects (individual bank conflicts,
pipeline bubbles) are not modeled.

## Logging & metrics

A run produces (see [`../src/logger.cpp`](../src/logger.cpp)):

- `isa_trace.jsonl` — every instruction with `start/end/cycles`, engine, wait/signal, macs/bytes.
- `ddr_trace.csv` — every DDR burst: direction, descriptor, address, bytes, cycles.
- `trace.json` — Chrome Trace Event Format; drag into <https://ui.perfetto.dev> to see the
  time sequence as per-engine tracks plus a cumulative-DDR-bytes counter.
- `perf.json` — the analysis below.

The analyzer ([`../src/perf.cpp`](../src/perf.cpp)) derives: total cycles/time, **MAC-array
utilization**, DMA/VE utilization, **DDR achieved vs peak bandwidth**, **arithmetic
intensity**, a roofline **compute- vs memory-bound** verdict, and an **energy/power
estimate** (per-MAC, per-DDR-byte, per-SRAM-byte coefficients + static power; broken down
by source, with average power = energy / time).

## Configuration

All capacity/timing knobs live in [`../configs/default.yaml`](../configs/default.yaml):
core clock, MAC array dims, vector lanes, DDR (freq/width/channels/latency/size),
SRAM (size/banks), feature flags. Loaded at startup.

## Roadmap (post-MVP)

1. ~~VE breadth: `SOFTMAX`, `RMSNORM`/`LAYERNORM`, `GELU`/`SiLU`, `REQUANT`, `ROPE`~~ ✅
2. ~~`CONV`~~ ✅
3. ~~Multi-DMA-channel + DDR contention · SRAM port contention~~ ✅
4. ~~`LOOP`/control flow (nested) to keep multi-layer LLM binaries compact~~ ✅
5. ~~Multi-core / multi-tile scaling · per-core private SRAM~~ ✅
6. ~~Energy/power estimation~~ ✅
7. End goal: run a compiler-emitted Llama / YOLO ONNX graph end-to-end. A hand-written
   single-layer transformer block ([`../tests/isa_programs/transformer_block.npuasm`](../tests/isa_programs/transformer_block.npuasm))
   already runs the full op stack end to end; the remaining work is on the compiler side.
