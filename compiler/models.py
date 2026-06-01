"""Hand-built Graph IR models (used to validate the compiler before ONNX)."""
from __future__ import annotations
import numpy as np
from .ir import Graph


def tiny_llama_layer(S=4, D=8, H=16, seed=0) -> Graph:
    """One Llama-style decoder layer (single head) as Graph IR.

    RMSNorm -> Q/K/V -> RoPE -> causal attention -> out proj -> residual
    -> RMSNorm -> SwiGLU FFN (silu(xW1) * xW3 -> W2) -> residual.
    Attention scaling is omitted (kept identical to the reference) so the only
    thing under test is the compiler's lowering, not numerics conventions.
    """
    rng = np.random.default_rng(seed)
    g = Graph("tiny_llama_layer")

    def w(name, shape):
        return g.tensor(name, shape, data=(rng.standard_normal(shape) * 0.1).astype(np.float32))

    x = g.tensor("x", (S, D), is_input=True)
    g1, g2 = w("g1", (D,)), w("g2", (D,))
    Wq, Wk, Wv, Wo = (w(n, (D, D)) for n in ("Wq", "Wk", "Wv", "Wo"))
    W1, W3, W2 = w("W1", (D, H)), w("W3", (D, H)), w("W2", (H, D))

    xn = g.add("rmsnorm", [x, g1], "xn", (S, D), {"eps": 1e-5})
    q = g.add("matmul", [xn, Wq], "q", (S, D))
    k = g.add("matmul", [xn, Wk], "k", (S, D))
    v = g.add("matmul", [xn, Wv], "v", (S, D))
    qr = g.add("rope", [q], "qr", (S, D), {"base": 10000.0})
    kr = g.add("rope", [k], "kr", (S, D), {"base": 10000.0})
    sc = g.add("matmul_t", [qr, kr], "scores", (S, S))
    aw = g.add("softmax", [sc], "attn_w", (S, S), {"causal": True})
    at = g.add("matmul", [aw, v], "attn", (S, D))
    ao = g.add("matmul", [at, Wo], "attn_o", (S, D))
    xr = g.add("add", [x, ao], "x_attn", (S, D))

    xn2 = g.add("rmsnorm", [xr, g2], "xn2", (S, D), {"eps": 1e-5})
    gate0 = g.add("matmul", [xn2, W1], "gate0", (S, H))
    gate = g.add("silu", [gate0], "gate", (S, H))
    up = g.add("matmul", [xn2, W3], "up", (S, H))
    hh = g.add("mul", [gate, up], "ffn_h", (S, H))
    down = g.add("matmul", [hh, W2], "down", (S, D))
    out = g.add("add", [xr, down], "y", (S, D))
    g.mark_output(out)
    return g
