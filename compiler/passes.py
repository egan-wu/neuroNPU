"""IR-to-IR optimization passes (run before the backend)."""
from __future__ import annotations
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


def apply(g: Graph, opt: dict) -> Graph:
    if opt.get("fuse", False):
        g = fuse_residuals(g)
    return g
