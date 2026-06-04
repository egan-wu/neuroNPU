"""A focused ONNX -> Graph IR frontend for conv-style vision nets (YOLO).

Runs shape inference, walks the nodes, and maps the compute-relevant ops
(Conv/MaxPool/Resize/Concat/Split/Sigmoid/Mul/Add) to IR. The detect head's
post-processing ops (TopK/GatherElements/ReduceMax/...) are stopped at, since
they carry negligible compute. Unhandled non-head ops pass through. Weights are
declared as params (no data) -- this is for timing/traffic profiling.
"""
from __future__ import annotations
import onnx
from onnx import shape_inference
from .ir import Graph

# Detect-head-exclusive ops: stop here (post-processing, negligible compute).
HEAD_OPS = {"TopK", "GatherElements", "ReduceMax", "Mod"}


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == a.INT:    return a.i
            if a.type == a.INTS:   return list(a.ints)
            if a.type == a.FLOAT:  return a.f
            if a.type == a.FLOATS: return list(a.floats)
    return default


def _chw(shape):                          # [N,C,H,W] -> (C,H,W); strip batch
    s = [int(d) for d in shape]
    return tuple(s[1:]) if len(s) == 4 else tuple(s)


def import_yolo(path) -> Graph:
    m = shape_inference.infer_shapes(onnx.load(path, load_external_data=False))
    go = m.graph
    init = {i.name: [int(d) for d in i.dims] for i in go.initializer}
    shp = {}
    for coll in (go.value_info, go.input, go.output):
        for v in coll:
            shp[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]

    g = Graph("yolov10n")
    m_ir = {}                              # onnx tensor -> IR tensor name
    inp = go.input[0]
    m_ir[inp.name] = g.tensor(inp.name.replace("/", "_"), _chw(shp[inp.name]), is_input=True)

    def ir(name):                          # IR tensor for an activation, else None
        return m_ir.get(name)

    nconv = 0
    for node in go.node:
        if node.op_type in HEAD_OPS:
            break                          # reached the detect head
        ot = node.output[0]
        osh = _chw(shp.get(ot, []))
        nm = ot.replace("/", "_")
        t = node.op_type

        if t == "Conv":
            x = ir(node.input[0])
            wsh = init[node.input[1]]      # [Co, Ci/g, Kh, Kw]
            w = g.tensor(nm + "_w", wsh, is_param=True)
            g.add("conv", [x, w], nm, osh,
                  {"stride": (_attr(node, "strides", [1])[0]),
                   "pad": (_attr(node, "pads", [0])[0]),
                   "group": _attr(node, "group", 1)})
            nconv += 1
        elif t == "Sigmoid":
            g.add("sigmoid", [ir(node.input[0])], nm, osh)
        elif t in ("Mul", "Add"):
            a, b = ir(node.input[0]), ir(node.input[1])
            if a is None or b is None:     # const operand -> passthrough activation
                m_ir[ot] = a or b
                continue
            g.add("mul" if t == "Mul" else "add", [a, b], nm, osh)
        elif t == "MaxPool":
            k = _attr(node, "kernel_shape", [2])[0]
            g.add("maxpool", [ir(node.input[0])], nm, osh,
                  {"kernel": k, "stride": _attr(node, "strides", [k])[0],
                   "pad": _attr(node, "pads", [0])[0]})
        elif t == "Resize":
            insh = _chw(shp[node.input[0]])
            f = osh[-1] // insh[-1] if insh[-1] else 2
            g.add("upsample", [ir(node.input[0])], nm, osh, {"factor": f})
        elif t == "Concat":
            ins = [ir(i) for i in node.input if ir(i) is not None]
            cur = ins[0]
            cc = self_c = _chw(shp[node.input[0]])[0]
            for k, b in enumerate(ins[1:]):
                bc = g.tensors[b].shape[0]
                cc += bc
                outn = nm if k == len(ins) - 2 else f"{nm}_cc{k}"
                hw = osh[1:]
                cur = g.add("concat", [cur, b], outn, (cc,) + tuple(hw))
            m_ir[ot] = cur
            continue
        elif t == "Split":
            x = ir(node.input[0])
            C = g.tensors[x].shape[0]
            splits = _attr(node, "split")
            n = len(node.output)
            if splits is None:
                splits = [C // n] * n
            c0 = 0
            for oi, sz in zip(node.output, splits):
                on = oi.replace("/", "_")
                osh2 = _chw(shp.get(oi, [0, sz] + list(g.tensors[x].shape[1:])))
                m_ir[oi] = g.add("slice", [x], on, osh2, {"c0": c0, "c1": c0 + sz})
                c0 += sz
            continue
        else:                              # unknown non-head op -> passthrough
            m_ir[ot] = ir(node.input[0])
            continue
        m_ir[ot] = nm

    # mark the last produced tensor as output
    last = list(m_ir.values())[-1]
    g.mark_output(last)
    g.meta_nconv = nconv
    return g
