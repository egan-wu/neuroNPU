# NeuroNPU — `.npubin` Binary Format (v1)

The stable, machine-emitted form of a program. All integers are **little-endian**.
Encoder/decoder: [`../src/isa.cpp`](../src/isa.cpp) (`write_binary` / `read_binary`).

```
Header
  char[4]   magic      = "NPUB"
  u32       version    = 1

Descriptor table
  u32       num_descriptors
  repeat:
    u16     name_len
    char[name_len] name
    u8      space        (0 = DDR, 1 = SRAM)
    u8      dtype         (0 = F32, 1 = F16, 2 = BF16, 3 = I8)
    u64     base_addr     (bytes)
    u16     rank
    i64[rank]    dims
    u16     num_strides   (0 => row-major contiguous)
    i64[num_strides] strides   (elements)

Instruction stream
  u32       num_instructions
  repeat:
    u8      opcode        (see table below)
    u8      accumulate    (0/1)
    i32     wait_event    (-1 = none)
    i32     signal_event  (-1 = none)
    u16     num_args
    i32[num_args] args     (descriptor indices into the table above)

Init-data section
  u32       num_entries
  repeat:
    i32     descriptor_id
    u64     num_bytes
    u8[num_bytes] data     (written to that descriptor's base_addr before the run)
```

## Opcode encoding

| Value | Opcode |
|-------|--------|
| 0 | `NOP` |
| 1 | `HALT` |
| 2 | `DMA.LOAD` |
| 3 | `DMA.STORE` |
| 4 | `MATMUL` |
| 5 | `VADD` |
| 6 | `RELU` |

## Argument conventions

| Opcode | args |
|--------|------|
| `DMA.LOAD`  | `[dst_sram, src_ddr]` |
| `DMA.STORE` | `[dst_ddr, src_sram]` |
| `MATMUL`    | `[out, a, b]` |
| `VADD`      | `[out, a, b]` |
| `RELU`      | `[out, in]` |

> Versioning: bump `version` on any layout change; the loader rejects unknown versions.
