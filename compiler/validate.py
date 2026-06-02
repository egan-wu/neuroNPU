"""End-to-end check: compile the tiny Llama layer, run it on the simulator, and
diff against the numpy reference. Run from the repo root:

    python3 -m compiler.validate
"""
from __future__ import annotations
import argparse
import numpy as np
from .models import tiny_llama_layer
from .driver import validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neuronpu", default="build/neuronpu")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--workdir", default="build/compiler")
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--fuse", action="store_true")
    ap.add_argument("--tile", action="store_true")
    ap.add_argument("--tile-budget", type=int, default=16)
    ap.add_argument("--nbuf", type=int, default=2)
    a = ap.parse_args()

    g = tiny_llama_layer(S=a.S, D=a.D, H=a.H)
    x = (np.random.default_rng(1).standard_normal((a.S, a.D)) * 0.5).astype(np.float32)
    opt = {"reuse": True, "fuse": a.fuse, "tile": a.tile,
           "tile_budget": a.tile_budget, "nbuf": a.nbuf}
    print(f"tiny Llama layer: S={a.S} D={a.D} H={a.H}, {len(g.ops)} ops, opt={opt}")
    ok, perf = validate(g, {"x": x}, a.neuronpu, a.config, a.workdir, opt=opt)
    print(f"  cycles={perf['total_cycles']:.0f}  GMACs={perf['total_gmacs']:.5f}  "
          f"DDR={perf['ddr_achieved_gbps']:.1f} GB/s  energy={perf['energy_total_nj']:.1f} nJ")
    print("VALIDATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
