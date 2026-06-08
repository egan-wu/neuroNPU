"""Profile Llama decode at different weight precisions (f32 / f16 / int8) to show
the memory-bandwidth (and therefore TPS) impact of quantization.

    python3 -m compiler.profile_quant --config-json models/llama32_1b/config.json
"""
from __future__ import annotations
import argparse, json
from .models import llama_from_config
from . import api


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", default="models/llama32_1b/config.json")
    ap.add_argument("--neuronpu", default="build/neuronpu")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--workdir", default="build/quant")
    ap.add_argument("--kv", type=int, default=128)
    ap.add_argument("--layers", type=int, default=None)
    a = ap.parse_args()
    cfg = json.load(open(a.config_json))
    L = a.layers or cfg["num_hidden_layers"]
    opt = {"reuse": True, "fuse": True, "tile": True, "nbuf": 2}

    print(f"=== weight-quantization profile: Llama decode (kv={a.kv}, {L} layers) ===\n")
    print(f"  {'weights':<8}{'time/token':>13}{'DDR/token':>12}{'TPS':>10}{'bound':>8}")
    base = None
    for dt in ("f32", "f16", "i8"):
        g = llama_from_config(cfg, 1, a.kv, L, wdtype=dt)
        p = api.compile(g, opt=opt, neuronpu=a.neuronpu, config=a.config,
                        workdir=a.workdir).profile(name=dt)
        base = base or p.tps
        vb = "MEM" if p.memory_bound else "CMP"
        print(f"  {dt:<8}{p.time_us:>11.1f}us{p.ddr_bytes/1e6:>10.0f}MB"
              f"{p.tps:>10.1f}{vb:>8}   {p.tps/base:.2f}x")
    print("\n  int8 weights cut DDR traffic ~4x and fp16 ~2x; since decode is "
          "memory-bound,\n  TPS scales almost directly with the weight-bandwidth "
          "reduction.")


if __name__ == "__main__":
    main()
