"""NeuroNPU compiler backend: lower a Graph IR to NeuroNPU ISA (.npuasm).

Responsibilities (bufferization + scheduling + codegen):
  * place weights / inputs / outputs in DDR, activations in on-chip SRAM
  * DMA-stage DDR operands into SRAM before the compute op that needs them
  * insert event sync (@wait/@sig) for cross-engine dependencies
  * emit text ISA (assembled to .npubin by `neuronpu asm`)

This is a straightforward (no-tiling) lowering that assumes activations fit in
SRAM — correct for the small validation model. Tiling for big tensors and
SRAM liveness reuse are separate passes layered on top.
"""
from __future__ import annotations
import re
import numpy as np
from .ir import Graph

F32 = 4
ALIGN = 256

# op kind -> (ISA mnemonic, engine)
TENSOR_OPS = {"matmul", "matmul_t"}
_MNEMONIC = {
    "rmsnorm": "RMSNORM", "rope": "ROPE", "softmax": "SOFTMAX", "silu": "SILU",
    "gelu": "GELU", "sigmoid": "SIGMOID", "mul": "VMUL", "add": "VADD",
    "sub": "VSUB", "matmul": "MATMUL", "matmul_t": "MATMUL", "gather": "GATHER",
}


def _engine(kind):
    return "TENSOR" if kind in TENSOR_OPS else "VECTOR"


class _Bump:
    def __init__(self): self.off = 0
    def alloc(self, nbytes):
        a = self.off
        self.off += (nbytes + ALIGN - 1) // ALIGN * ALIGN
        return a


class Backend:
    def __init__(self, g: Graph):
        self.g = g
        self.ddr, self.sram = _Bump(), _Bump()
        self.san = {}            # ir name -> asm identifier
        self.decls = []          # .desc / .data lines
        self.prog = []           # instruction lines
        self.ev = 0              # next event id
        self.ready = {}          # asm name -> (event_id, engine)
        self.ddr_addr = {}       # ir name -> (asm, addr)  for weights/inputs/outputs
        self.sram_of = {}        # ir name -> asm sram descriptor (activation or staged)
        self.ddr_data = {}       # asm DDR name -> ndarray  (weights, for --load)
        self.input_descs = {}    # ir input -> asm DDR name
        self.output_descs = {}   # ir output -> asm DDR name

    # ---- helpers ----
    def _name(self, ir_name, suffix=""):
        base = re.sub(r"[^A-Za-z0-9_]", "_", ir_name) + suffix
        if base[0].isdigit():
            base = "t_" + base
        return base

    def _new_ev(self):
        e = self.ev; self.ev += 1; return e

    def _dims(self, shape):
        return "x".join(str(int(s)) for s in shape)

    def _decl_ddr(self, ir_name):
        """Declare a DDR descriptor for a weight/input/output tensor (once)."""
        if ir_name in self.ddr_addr:
            return self.ddr_addr[ir_name][0]
        t = self.g.tensors[ir_name]
        asm = self._name(ir_name)
        addr = self.ddr.alloc(t.numel * F32)
        self.decls.append(f".desc {asm} ddr f32 0x{addr:x} {self._dims(t.shape)}")
        self.ddr_addr[ir_name] = (asm, addr)
        if t.data is not None:                       # value present (functional)
            self.ddr_data[asm] = t.data.astype(np.float32)
        return asm

    def _stage(self, ir_name):
        """Ensure a DDR-resident tensor (weight/input) is in SRAM; return asm name."""
        if ir_name in self.sram_of:
            return self.sram_of[ir_name]
        t = self.g.tensors[ir_name]
        ddr_asm = self._decl_ddr(ir_name)
        s_asm = self._name(ir_name, "_s")
        addr = self.sram.alloc(t.numel * F32)
        self.decls.append(f".desc {s_asm} sram f32 0x{addr:x} {self._dims(t.shape)}")
        ev = self._new_ev()
        self.prog.append(f"DMA.LOAD {s_asm} {ddr_asm} @sig {ev}")
        self.ready[s_asm] = (ev, "DMA")
        self.sram_of[ir_name] = s_asm
        return s_asm

    def _activation(self, ir_name):
        """Declare/alloc an SRAM descriptor for an op's activation output."""
        if ir_name in self.sram_of:
            return self.sram_of[ir_name]
        t = self.g.tensors[ir_name]
        s_asm = self._name(ir_name)
        addr = self.sram.alloc(t.numel * F32)
        self.decls.append(f".desc {s_asm} sram f32 0x{addr:x} {self._dims(t.shape)}")
        self.sram_of[ir_name] = s_asm
        return s_asm

    def _transposed_view(self, ir_name):
        """A transposed SRAM view of an [N,K] tensor as [K,N] (strides 1,K)."""
        base = self._stage_or_act(ir_name)
        t = self.g.tensors[ir_name]
        N, K = int(t.shape[-2]), int(t.shape[-1])
        # find base address from the descriptor we already emitted
        addr = self._addr_of(base)
        v = self._name(ir_name, "_T")
        self.decls.append(f".desc {v} sram f32 0x{addr:x} {K}x{N} :1,{K}")
        # the view shares the producer/event of its base
        self.ready[v] = self.ready.get(base, (None, "VECTOR"))
        return v

    def _addr_of(self, asm_name):
        for d in self.decls:
            parts = d.split()
            if len(parts) >= 5 and parts[0] == ".desc" and parts[1] == asm_name:
                return int(parts[4], 16)
        raise KeyError(asm_name)

    def _stage_or_act(self, ir_name):
        t = self.g.tensors[ir_name]
        return self._stage(ir_name) if (t.is_weight or t.is_input) else self._activation(ir_name)

    # ---- main lowering ----
    def compile(self):
        for op in self.g.ops:
            self._lower(op)
        # store graph outputs back to DDR
        for o in self.g.outputs:
            s = self.sram_of[o]
            ddr_asm = self._name(o, "_out")
            addr = self.ddr.alloc(self.g.tensors[o].numel * F32)
            self.decls.append(f".desc {ddr_asm} ddr f32 0x{addr:x} {self._dims(self.g.tensors[o].shape)}")
            ev, eng = self.ready[s]
            self.prog.append(f"DMA.STORE {ddr_asm} {s} @wait {ev}")
            self.output_descs[o] = ddr_asm
        for i in self.g.inputs:
            self.input_descs[i] = self._decl_ddr(i)
        self.prog.append("HALT")
        text = "\n".join(self.decls) + "\n\n" + "\n".join(self.prog) + "\n"
        return text

    def _lower(self, op):
        # take_last: a zero-cost view of the input's last row (no instruction).
        if op.kind == "take_last":
            base = self._stage_or_act(op.inputs[0])
            addr = self._addr_of(base)
            t = self.g.tensors[op.inputs[0]]
            rows, D = int(t.shape[-2]), int(t.shape[-1])
            v = self._name(op.outputs[0])
            self.decls.append(f".desc {v} sram f32 0x{addr + (rows-1)*D*F32:x} 1x{D}")
            self.sram_of[op.outputs[0]] = v
            self.ready[v] = self.ready.get(base, (None, "VECTOR"))
            return

        eng = _engine(op.kind)
        out = self._activation(op.outputs[0])

        if op.kind == "gather":
            table = self._decl_ddr(op.inputs[0])           # table stays in DDR
            ids = " ".join(f"${int(i)}" for i in op.attrs["ids"])
            ev = self._new_ev()
            self.prog.append(f"GATHER {out} {table} {ids} @sig {ev}")
            self.ready[out] = (ev, "VECTOR")
            return

        # resolve SRAM operands (+ events to wait on)
        operands, waits = [], []
        for k, inp in enumerate(op.inputs):
            if op.kind == "matmul_t" and k == 1:
                name = self._transposed_view(inp)
            else:
                name = self._stage_or_act(inp)
            operands.append(name)
            r = self.ready.get(name)
            if r and r[0] is not None and r[1] != eng:
                waits.append(r[0])

        imm = self._imms(op)
        ev = self._new_ev()
        ann = (f" @wait {','.join(map(str, sorted(set(waits))))}" if waits else "") + f" @sig {ev}"
        self.prog.append(f"{_MNEMONIC[op.kind]} {out} {' '.join(operands)}{imm}{ann}")
        self.ready[out] = (ev, eng)

    def _imms(self, op):
        a = op.attrs
        if op.kind == "rmsnorm":
            return f" ${a.get('eps', 1e-6)}"
        if op.kind == "rope":
            return f" ${a.get('base', 10000.0)} ${a.get('pos_offset', 0)}"
        if op.kind == "softmax" and a.get("causal"):
            q = a.get("q_offset")
            return f" $1" + (f" ${q}" if q is not None else "")
        return ""


def compile_graph(g: Graph):
    """Return (npuasm_text, ddr_data, input_descs, output_descs)."""
    be = Backend(g)
    text = be.compile()
    return text, be.ddr_data, be.input_descs, be.output_descs
