"""Numpy reference executor for the Graph IR.

Defines the exact semantics of every op kind, and serves as the golden model
for validating the NeuroNPU backend (compiler output is diffed against this).
"""
from __future__ import annotations
import numpy as np
from .ir import Graph


def _rms_norm(x, w, eps):
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * w


def _rope(x, base, pos_offset):
    # rotate interleaved pairs; position = row index + offset
    S, D = x.shape
    out = np.empty_like(x)
    pos = (np.arange(S) + pos_offset)[:, None]
    i = np.arange(D // 2)
    theta = pos * (base ** (-(2.0 * i) / D))          # [S, D/2]
    c, s = np.cos(theta), np.sin(theta)
    x0, x1 = x[:, 0::2], x[:, 1::2]
    out[:, 0::2] = x0 * c - x1 * s
    out[:, 1::2] = x0 * s + x1 * c
    return out


def _conv2d(x, w, stride, pad):                     # x[Ci,H,W], w[Co,Ci,Kh,Kw]
    Ci, H, W = x.shape
    Co, _, Kh, Kw = w.shape
    Ho, Wo = (H + 2 * pad - Kh) // stride + 1, (W + 2 * pad - Kw) // stride + 1
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((Co, Ho, Wo), np.float32)
    for oh in range(Ho):
        for ow in range(Wo):
            patch = xp[:, oh * stride:oh * stride + Kh, ow * stride:ow * stride + Kw]
            out[:, oh, ow] = np.tensordot(w, patch, axes=([1, 2, 3], [0, 1, 2]))
    return out


def _maxpool(x, k, stride, pad):                    # x[C,H,W]
    stride = stride or k
    C, H, W = x.shape
    Ho, Wo = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), constant_values=-np.inf)
    out = np.empty((C, Ho, Wo), np.float32)
    for oh in range(Ho):
        for ow in range(Wo):
            out[:, oh, ow] = xp[:, oh * stride:oh * stride + k,
                                ow * stride:ow * stride + k].max(axis=(1, 2))
    return out


def _softmax(x, causal, q_offset):
    x = x.astype(np.float64)
    R, C = x.shape
    if causal:
        if q_offset is None:
            q_offset = C - R
        col = np.arange(C)[None, :]
        limit = (q_offset + np.arange(R))[:, None]
        x = np.where(col <= limit, x, -np.inf)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def execute(g: Graph, inputs: dict) -> dict:
    """Run the graph; return {tensor_name: ndarray} for every tensor."""
    vals = {}
    for name, t in g.tensors.items():
        if t.is_weight:
            vals[name] = t.data.astype(np.float32)
    for name, arr in inputs.items():
        vals[name] = arr.astype(np.float32)

    for op in g.ops:
        ins = [vals[i] for i in op.inputs]
        a = ins[0] if ins else None
        k = op.kind
        if k == "rmsnorm":
            r = _rms_norm(a, ins[1], op.attrs.get("eps", 1e-6))
        elif k == "matmul":
            r = a @ ins[1]
        elif k == "matmul_t":                       # a @ b^T
            r = a @ ins[1].T
        elif k == "matmul_acc":                      # x + a @ W  (fused residual)
            r = ins[0] + ins[1] @ ins[2]
        elif k == "rope":
            r = _rope(a, op.attrs.get("base", 10000.0), op.attrs.get("pos_offset", 0))
        elif k == "softmax":
            r = _softmax(a, op.attrs.get("causal", False), op.attrs.get("q_offset"))
        elif k == "silu":
            r = a / (1.0 + np.exp(-a))
        elif k == "gelu":
            r = 0.5 * a * (1.0 + np.tanh(0.7978845608 * (a + 0.044715 * a ** 3)))
        elif k == "sigmoid":
            r = 1.0 / (1.0 + np.exp(-a))
        elif k == "mul":
            r = a * ins[1]
        elif k == "add":
            r = a + ins[1]
        elif k == "sub":
            r = a - ins[1]
        elif k == "gather":
            r = a[np.asarray(op.attrs["ids"], dtype=np.int64)]
        elif k == "scale":
            r = a * op.attrs["factor"]
        elif k == "take_last":                      # last row -> [1, D]
            r = a[-1:, :]
        elif k == "conv":
            r = _conv2d(a, ins[1], op.attrs.get("stride", 1), op.attrs.get("pad", 0))
        elif k == "maxpool":
            r = _maxpool(a, op.attrs.get("kernel", 2), op.attrs.get("stride"),
                         op.attrs.get("pad", 0))
        elif k == "upsample":
            f = op.attrs.get("factor", 2)
            r = np.repeat(np.repeat(a, f, axis=-2), f, axis=-1)
        elif k == "concat":
            r = np.concatenate([a, ins[1]], axis=0)
        else:
            raise ValueError(f"reference: unknown op kind {k}")
        vals[op.outputs[0]] = np.asarray(r, dtype=np.float32)
    return vals
