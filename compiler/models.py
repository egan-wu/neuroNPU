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


def llama_from_config(cfg: dict, q_rows: int, kv_rows: int, n_layers=None) -> Graph:
    """Build a real-scale Llama decoder forward pass as Graph IR (no weight data;
    for timing-only profiling).  q_rows==kv_rows => prefill; q_rows==1 & kv_rows>1
    => one decode step attending to a kv_rows-long cache.

    NOTE (approximation): attention is modeled with full-width K/V (MHA). Real
    Llama-3.2 uses GQA (8 KV heads), which would shrink k/v-proj and the KV cache
    ~4x; MAC-dominant costs (q/o proj, FFN, attention, lm_head) are exact.
    """
    D = cfg["hidden_size"]
    FFN = cfg["intermediate_size"]
    V = cfg["vocab_size"]
    L = n_layers if n_layers is not None else cfg["num_hidden_layers"]
    eps = cfg.get("rms_norm_eps", 1e-5)
    base = cfg.get("rope_theta", 500000.0)
    decode = (q_rows == 1 and kv_rows > 1)
    g = Graph(f"llama_{'decode' if decode else 'prefill'}")

    def param(name, shape):
        return g.tensor(name, shape, is_param=True)

    # token embedding (ids are placeholders; values irrelevant to timing)
    embed = param("embed_tokens", (V, D))
    x = g.add("gather", [embed], "hidden", (q_rows, D), {"ids": list(range(q_rows))})

    for li in range(L):
        p = f"L{li}."
        g1, g2 = param(p + "ln1", (D,)), param(p + "ln2", (D,))
        Wq, Wk, Wv, Wo = (param(p + n, (D, D)) for n in ("Wq", "Wk", "Wv", "Wo"))
        W1, W3, W2 = param(p + "W1", (D, FFN)), param(p + "W3", (D, FFN)), param(p + "W2", (FFN, D))

        xn = g.add("rmsnorm", [x, g1], p + "xn", (q_rows, D), {"eps": eps})
        q = g.add("matmul", [xn, Wq], p + "q", (q_rows, D))
        qr = g.add("rope", [q], p + "qr", (q_rows, D), {"base": base})
        if decode:
            g.add("matmul", [xn, Wk], p + "k", (1, D))      # new token's k/v (cost only)
            g.add("matmul", [xn, Wv], p + "v", (1, D))
            kc = param(p + "kcache", (kv_rows, D))           # KV cache (read from DDR)
            vc = param(p + "vcache", (kv_rows, D))
            sc = g.add("matmul_t", [qr, kc], p + "scores", (q_rows, kv_rows))
            aw = g.add("softmax", [sc], p + "aw", (q_rows, kv_rows), {"causal": True})
            at = g.add("matmul", [aw, vc], p + "attn", (q_rows, D))
        else:
            k = g.add("matmul", [xn, Wk], p + "k", (q_rows, D))
            v = g.add("matmul", [xn, Wv], p + "v", (q_rows, D))
            kr = g.add("rope", [k], p + "kr", (q_rows, D), {"base": base})
            sc = g.add("matmul_t", [qr, kr], p + "scores", (q_rows, q_rows))
            aw = g.add("softmax", [sc], p + "aw", (q_rows, q_rows), {"causal": True})
            at = g.add("matmul", [aw, v], p + "attn", (q_rows, D))
        ao = g.add("matmul", [at, Wo], p + "ao", (q_rows, D))
        x = g.add("add", [x, ao], p + "x1", (q_rows, D))

        xn2 = g.add("rmsnorm", [x, g2], p + "xn2", (q_rows, D), {"eps": eps})
        g0 = g.add("matmul", [xn2, W1], p + "g0", (q_rows, FFN))
        ga = g.add("silu", [g0], p + "gate", (q_rows, FFN))
        up = g.add("matmul", [xn2, W3], p + "up", (q_rows, FFN))
        hh = g.add("mul", [ga, up], p + "h", (q_rows, FFN))
        dn = g.add("matmul", [hh, W2], p + "down", (q_rows, D))
        x = g.add("add", [x, dn], p + "x2", (q_rows, D))

    gn = param("norm", (D,))
    xn = g.add("rmsnorm", [x, gn], "xn_final", (q_rows, D), {"eps": eps})
    last = g.add("take_last", [xn], "last", (1, D))      # logits only for last token
    lm = param("lm_head", (D, V))                        # tied to embed in real model
    logits = g.add("matmul", [last, lm], "logits", (1, V))
    g.mark_output(logits)
    return g

