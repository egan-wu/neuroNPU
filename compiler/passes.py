"""IR-to-IR optimization passes (run before the backend)."""
from __future__ import annotations
import numpy as np
from .ir import Graph, Op


def _uses(g: Graph):
    u = {}
    for op in g.ops:
        for i in op.inputs:
            u[i] = u.get(i, 0) + 1
    for o in g.outputs:
        u[o] = u.get(o, 0) + 1
    return u


def fuse_residuals(g: Graph) -> Graph:
    """Fuse `add(x, matmul(a,W))` -> `matmul_acc(x, a, W)` (in-place accumulate).

    Removes one VADD and one intermediate buffer per residual by folding the add
    into the matmul epilogue (the ISA MATMUL already supports `accum`). Only fires
    when the matmul output feeds exactly one consumer (the add).
    """
    uses = _uses(g)
    producer = {op.outputs[0]: op for op in g.ops if op.outputs}
    skip = set()                       # matmul outputs folded into an add
    fuse_at = {}                       # add op id -> (x, a, W)

    for op in g.ops:
        if op.kind != "add":
            continue
        a, b = op.inputs
        for m, x in ((a, b), (b, a)):
            p = producer.get(m)
            if (p is not None and p.kind == "matmul" and uses.get(m, 0) == 1
                    and m not in g.outputs):
                skip.add(m)
                fuse_at[id(op)] = (x, p.inputs[0], p.inputs[1])
                break

    out = Graph(g.name + "_fused")
    out.tensors = dict(g.tensors)
    out.inputs = list(g.inputs)
    out.outputs = list(g.outputs)
    for op in g.ops:
        if op.kind == "matmul" and op.outputs[0] in skip:
            continue                                   # folded away
        if op.kind == "add" and id(op) in fuse_at:
            x, a, W = fuse_at[id(op)]
            out.ops.append(Op("matmul_acc", [x, a, W], list(op.outputs), {}))
        else:
            out.ops.append(op)
    return out


def quantize(g: Graph, inputs: dict) -> Graph:
    """Static int8 quantization of weight matmuls. For each matmul(a, W) with a
    weight W, computes per-tensor scales (scale = max|.|/127) from W and from a
    calibration run of `inputs`, then rewrites it as:
        a_i8 = requant(a, scale_a);  acc = matmul_i8(a_i8, W_i8);
        out  = scale(acc, scale_a * scale_w)
    i.e. int8 GEMM with fp32 accumulate and a dequant. Other ops stay fp32.
    """
    from . import reference
    vals = reference.execute(g, inputs)
    qmax = 127.0

    def scale_of(arr):
        m = float(np.max(np.abs(arr)))
        return m / qmax if m > 0 else 1.0

    out = Graph(g.name + "_int8")
    out.tensors = dict(g.tensors)
    out.inputs, out.outputs = list(g.inputs), list(g.outputs)
    for op in g.ops:
        W = g.tensors[op.inputs[1]] if op.kind == "matmul" and len(op.inputs) > 1 else None
        if op.kind == "matmul" and W is not None and W.data is not None:
            a, wname = op.inputs
            sw, sa = scale_of(W.data), scale_of(vals[a])
            wq = wname + "_i8"
            if wq not in out.tensors:
                codes = np.clip(np.round(W.data / sw), -128, 127).astype(np.float32)
                out.tensor(wq, W.shape, data=codes, dtype="i8")
            aq = f"{a}_q_{op.outputs[0]}"
            out.tensor(aq, g.tensors[a].shape, dtype="i8")
            out.ops.append(Op("requant", [a], [aq], {"scale": sa}))
            acc = op.outputs[0] + "_acc"
            out.tensor(acc, g.tensors[op.outputs[0]].shape)
            out.ops.append(Op("matmul", [aq, wq], [acc], {}))
            out.ops.append(Op("scale", [acc], [op.outputs[0]], {"factor": sa * sw}))
        else:
            out.ops.append(op)
    return out


def apply(g: Graph, opt: dict) -> Graph:
    if opt.get("fuse", False):
        g = fuse_residuals(g)
    return g
