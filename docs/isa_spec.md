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
- `strides` are in **elements**; if omitted, row-major contiguous is assumed.

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
| `VECTOR` | `VADD`, `RELU`, `GELU`, `SILU`, `SOFTMAX`, `RMSNORM`, `LAYERNORM`, `REQUANT`, `ROPE` |
| (any)    | `NOP`, `HALT` |

## Instruction reference

| Opcode | Form | Semantics |
|--------|------|-----------|
| `DMA.LOAD`  | `DMA.LOAD dst_sram src_ddr` | copy `src` bytes DDR→SRAM |
| `DMA.STORE` | `DMA.STORE dst_ddr src_sram` | copy `src` bytes SRAM→DDR |
| `MATMUL`    | `MATMUL out a b [accum]` | `out[M,N] = a[M,K] @ b[K,N]`; `accum` adds into `out` |
| `CONV`      | `CONV out in weight $stride $pad` | 2D conv, `in[Ci,H,W] * weight[Co,Ci,Kh,Kw] = out[Co,Ho,Wo]`, zero-pad; `stride` default 1, `pad` default 0 |
| `VADD`      | `VADD out a b` | elementwise `out = a + b` |
| `RELU`      | `RELU out in` | elementwise `out = max(0, in)` |
| `GELU`      | `GELU out in` | elementwise GELU (tanh approximation) |
| `SILU`      | `SILU out in` | elementwise `out = in · sigmoid(in)` |
| `SOFTMAX`   | `SOFTMAX out in` | softmax along the **last** dimension |
| `RMSNORM`   | `RMSNORM out in weight $eps` | `out = in / rms(row) · weight`; `eps` default 1e-6 |
| `LAYERNORM` | `LAYERNORM out in weight bias $eps` | `out = (in−mean)/std · weight + bias`; `eps` default 1e-5 |
| `REQUANT`   | `REQUANT out in $scale $zp` | `out = clamp(round(in/scale) + zp)` (e.g. to `i8`) |
| `ROPE`      | `ROPE out in $base $pos_offset` | rotary embedding on pairs along the last dim; position = row + `pos_offset`; `base` default 10000 |
| `NOP`       | `NOP` | nothing |
| `HALT`      | `HALT` | end marker (0 cycles) |

**Immediates.** Scalar operands are written with a `$` prefix (e.g. `$1e-5`, `$0.0078`).
They are positional and follow the op's descriptor operands. (`#` and `;` start comments,
so immediates must not use `#`.)

Every instruction accepts optional trailing `@wait eN` and/or `@sig eN`.

## Assembly (`.npuasm`) directives

| Directive | Form | Purpose |
|-----------|------|---------|
| `.desc`   | `.desc NAME ddr\|sram DTYPE BASE DIMS` | declare a descriptor (declare before use) |
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
- Planned opcodes (not yet implemented): `LOOP`.
