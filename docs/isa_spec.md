# NeuroNPU — ISA Specification (v1)

This is the **contract between your compiler/backend and the simulator**. The backend
emits either `.npuasm` (text, this document) or `.npubin` (binary, see
[`binary_format.md`](binary_format.md)); the simulator consumes either.

The ISA is **CISC-like**: each instruction operates on a whole tensor *tile* described
by a **descriptor**, not on scalars or single registers.

## Concepts

### Memory spaces
- `ddr` — external memory (large; all traffic metrics come from here).
- `sram` — on-chip scratchpad, **software-managed**. The compiler owns allocation.

### Tensor descriptor
A typed, strided view into a memory space:

```
name | space(ddr|sram) | dtype | base_addr(bytes) | dims[] | strides[]
```

- `dtype` ∈ `f32`, `f16`, `bf16`, `i8` (compute accumulates in fp32).
- `base_addr` is a **byte** offset within its space.
- `strides` are in **elements**; if omitted, row-major contiguous is assumed. In `.npuasm`
  a trailing `:s0,s1,...` sets them — e.g. a transposed view `kT` of `k[S,D]` is
  `.desc kT sram f32 <addr> DxS :1,S`.
- an optional trailing `+N` gives an **iteration stride** (elements): inside a `LOOP`
  the descriptor's `base_addr` advances by `N` elements each iteration.

### Events (synchronization)
Engines run asynchronously. A cross-engine dependency is expressed with events:
- `@wait eN` — the instruction does not issue until event `N` has been signaled.
- `@sig eN`  — event `N` is signaled when the instruction completes.

Intra-engine ordering is implicit (per-engine in-order), so same-engine dependencies
need no event.

## Engines

| Engine | Opcodes |
|--------|---------|
| `DMA`    | `DMA.LOAD`, `DMA.STORE` |
| `TENSOR` | `MATMUL`, `CONV` |
| `VECTOR` (vision) | `MAXPOOL`, `UPSAMPLE`, `CONCAT` |
| `VECTOR` | `VADD`, `VSUB`, `VMUL`, `VMAX`, `RELU`, `GELU`, `SILU`, `SIGMOID`, `SOFTMAX`, `RMSNORM`, `LAYERNORM`, `REQUANT`, `ROPE`, `GATHER` |
| control  | `LOOP`, `ENDLOOP` (expanded at load) |
| (any)    | `NOP`, `HALT` |

## Instruction reference

| Opcode | Form | Semantics |
|--------|------|-----------|
| `DMA.LOAD`  | `DMA.LOAD dst_sram src_ddr` | copy `src` bytes DDR→SRAM |
| `DMA.STORE` | `DMA.STORE dst_ddr src_sram` | copy `src` bytes SRAM→DDR |
| `MATMUL`    | `MATMUL out a b [accum]` | `out[M,N] = a[M,K] @ b[K,N]`; `accum` adds into `out` |
| `CONV`      | `CONV out in weight $stride $pad` | 2D conv, `in[Ci,H,W] * weight[Co,Ci,Kh,Kw] = out[Co,Ho,Wo]`, zero-pad; `stride` default 1, `pad` default 0 |
| `MAXPOOL`   | `MAXPOOL out in $kernel $stride $pad` | 2D max pool `[C,H,W] -> [C,Ho,Wo]` |
| `UPSAMPLE`  | `UPSAMPLE out in $factor` | nearest-neighbour upsample `[C,H,W] -> [C,H·f,W·f]` |
| `CONCAT`    | `CONCAT out a b` | concatenate `a`,`b` along channel axis 0 |
| `REPEAT_KV` | `REPEAT_KV out in $group $head_dim` | GQA: repeat each KV head block `group`× (`[R,kv_dim] -> [R,kv_dim·group]`) |
| `VADD`/`VSUB`/`VMUL`/`VMAX` | `OP out a b` | elementwise `a+b` / `a-b` / `a*b` / `max(a,b)` (`VMUL` is SwiGLU gating) |
| `RELU`      | `RELU out in` | elementwise `out = max(0, in)` |
| `SIGMOID`   | `SIGMOID out in` | elementwise `out = 1/(1+e^-in)` |
| `GELU`      | `GELU out in` | elementwise GELU (tanh approximation) |
| `SILU`      | `SILU out in` | elementwise `out = in · sigmoid(in)` |
| `SOFTMAX`   | `SOFTMAX out in [$causal] [$q_offset]` | softmax along the **last** dim; `$causal=1` masks future keys (query row `r` attends keys `0..q_offset+r`; `q_offset` defaults to `kv_len - q_rows`) |
| `RMSNORM`   | `RMSNORM out in weight $eps` | `out = in / rms(row) · weight`; `eps` default 1e-6 |
| `LAYERNORM` | `LAYERNORM out in weight bias $eps` | `out = (in−mean)/std · weight + bias`; `eps` default 1e-5 |
| `REQUANT`   | `REQUANT out in $scale $zp` | `out = clamp(round(in/scale) + zp)` (e.g. fp32 -> `i8`) |
| `SCALE`     | `SCALE out in $factor` | `out = in · factor` (e.g. int8-GEMM dequant by `scale_a·scale_w`) |
| `ROPE`      | `ROPE out in $base $pos_offset` | rotary embedding on pairs along the last dim; position = row + `pos_offset`; `base` default 10000 |
| `GATHER`    | `GATHER out table $id0 $id1 ...` | embedding lookup: `out[n,:] = table[id_n,:]`; ids are immediates (known at compile time); table read counts as DDR traffic if the table is in DDR |
| `LOOP`      | `LOOP $count` … `ENDLOOP` | repeat the body `count` times (see loops below) |
| `ENDLOOP`   | `ENDLOOP` | close the nearest `LOOP` |
| `NOP`       | `NOP` | nothing |
| `HALT`      | `HALT` | end marker (0 cycles) |

**Immediates.** Scalar operands are written with a `$` prefix (e.g. `$1e-5`, `$0.0078`).
They are positional and follow the op's descriptor operands. (`#` and `;` start comments,
so immediates must not use `#`.)

Every instruction accepts optional trailing `@wait`, `@sig eN`, and `@core N`
(which core/tile runs it; default 0 — see Multi-core below). `@wait` may name **several**
events — `@wait 1,2` or repeated `@wait 1 @wait 2` — and the instruction issues only once
**all** are signaled (needed when an op depends on producers on different engines, e.g. a
MATMUL waiting on both a DMA-staged weight and a vector-engine activation). Events are
global, so a `@wait` can depend on another core's `@sig`.

## Assembly (`.npuasm`) directives

| Directive | Form | Purpose |
|-----------|------|---------|
| `.desc`   | `.desc NAME ddr\|sram DTYPE BASE DIMS [+ITERSTRIDE] [:STRIDES]` | declare a descriptor (declare before use) |
| `.data`   | `.data NAME MODE [ARG]` | preload a descriptor's memory |

`DIMS` accepts `x`- or `,`-separated extents (e.g. `64x64`). `BASE` accepts hex (`0x...`)
or decimal. `.data` MODE ∈ `iota`, `zeros`, `const V`, `rand SEED`.
Comments start with `#` or `;`.

## Example

```
.desc A  ddr  f32 0x00000 64x64
.desc B  ddr  f32 0x10000 64x64
.desc Y  ddr  f32 0x20000 64x64
.desc sA sram f32 0x0000  64x64
.desc sB sram f32 0x4000  64x64
.desc sY sram f32 0x8000  64x64
.data A iota
.data B const 1

DMA.LOAD  sA A      @sig 0
DMA.LOAD  sB B      @sig 1
MATMUL    sY sA sB  @wait 1 @sig 2
DMA.STORE Y  sY     @wait 2
HALT
```

See [`../tests/isa_programs/matmul.npuasm`](../tests/isa_programs/matmul.npuasm) for the
runnable version (with the expected-output check).

## Notes for the compiler backend

- **You own SRAM allocation and double-buffering.** The simulator does not relocate
  tensors; emit explicit `DMA.LOAD`/`DMA.STORE` and place tiles at non-overlapping
  SRAM addresses. To overlap load/compute, ping-pong between two SRAM regions and use
  events to gate the consumer.
- An instruction has **one** `@wait`. To depend on several producers, either chain them
  onto one engine (in-order) or have the last producer signal the event the consumer waits on.
- All coarse compute opcodes are implemented; remaining roadmap is energy modeling.

## Multi-core

Set `num_cores > 1` (config) and tag instructions with `@core N`. Each core has its own
TENSOR/VECTOR unit, its own `dma_channels`, and its own **private SRAM scratchpad** (a SRAM
descriptor address refers to the executing core's bank). **DDR is shared globally** across
cores (so they contend for DDR bandwidth); SRAM bandwidth contention is per core. Use
global events to synchronize across cores.

## Loops

`LOOP $count` … `ENDLOOP` brackets a body that the **simulator expands at load time**
(`flatten_loops`). This keeps multi-layer binaries compact while running the full work.
Per iteration `k`:

- any descriptor with an iteration stride (`+N`) has its `base_addr` advanced by `k·N`
  elements (a fresh address variant is generated);
- events are renamed into a per-iteration namespace, so an iteration's internal
  `@wait`/`@sig` never alias another iteration's.

Cross-iteration data dependencies (e.g. a residual buffer reused every layer) are the
compiler's responsibility: keep producer/consumer on the same engine (implicit in-order)
or reuse a fixed buffer (stride 0).

**Nesting is supported.** A `+N` descriptor advances **additively**: it gains `k·N`
elements for iteration `k` of *every* enclosing loop it sits in (so for outer/inner loops
the offset is `(k_outer + k_inner)·N`). Each iteration at each level also gets a disjoint
event namespace. Because the stride is a single per-descriptor value, distinct strides per
loop level aren't expressible with one descriptor — use separate descriptors per level if
you need them.
