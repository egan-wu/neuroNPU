"""Compile a real Llama config to NeuroNPU ISA and profile it (timing-only).

Reads the downloaded model's config.json, builds the real-scale forward pass
(prefill + a decode step at several context lengths) via the compiler, runs each
on the simulator in timing-only mode, and reports LLM-level metrics.

    python3 -m compiler.profile_llama --config-json models/llama32_1b/config.json
"""
from __future__ import annotations
import argparse, json
from .models import llama_from_config
from .driver import compile_and_run


def _bytes(perf):
    return perf["ddr_achieved_gbps"] * perf["total_time_ns"]   # GB/s * ns = bytes


def _row(tag, perf):
    vb = "MEM " if perf["memory_bound"] else "CMP "
    sram = perf.get("compile_stats", {}).get("peak_sram_bytes", 0) / 1e6
    return (f"  {tag:<13}{perf['total_time_ns']/1e3:>10.1f}us"
            f"{perf['total_gmacs']:>9.2f}{perf['te_util']*100:>8.1f}%"
            f"{perf['ddr_bw_util']*100:>8.1f}%{_bytes(perf)/1e6:>9.0f}MB"
            f"{perf['arithmetic_intensity']:>8.2f}{vb:>6}"
            f"{perf['energy_total_nj']/1e6:>8.2f}mJ{sram:>8.0f}MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", default="models/llama32_1b/config.json")
    ap.add_argument("--neuronpu", default="build/neuronpu")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--workdir", default="build/llama")
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--kv", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--layers", type=int, default=None, help="override num layers")
    ap.add_argument("--tile", action="store_true", help="tile big weights (realistic SRAM)")
    ap.add_argument("--tile-budget", type=int, default=512 * 512)
    a = ap.parse_args()

    cfg = json.load(open(a.config_json))
    L = a.layers or cfg["num_hidden_layers"]
    print(f"=== Llama profile : {L} layers, hidden={cfg['hidden_size']}, "
          f"ffn={cfg['intermediate_size']}, vocab={cfg['vocab_size']} (FP32) ===\n")
    print(f"  {'phase':<13}{'time':>12}{'GMACs':>9}{'MACu':>9}{'DDRbw':>9}"
          f"{'DDR':>11}{'AI':>8}{'bound':>6}{'energy':>10}{'peakSRAM':>10}")

    opt = {"reuse": True, "fuse": True, "tile": a.tile, "tile_budget": a.tile_budget}

    def run(name, g):
        _, perf = compile_and_run(g, {}, a.neuronpu, a.config, a.workdir,
                                  timing_only=True, out_dir=a.workdir + "/" + name, opt=opt)
        print(_row(name, perf))
        return perf

    pf = run("prefill", llama_from_config(cfg, a.prompt, a.prompt, L))
    decs = {}
    for kv in a.kv:
        decs[kv] = run(f"decode kv={kv}", llama_from_config(cfg, 1, kv, L))

    print("\n  --- LLM-level metrics (config: %s) ---" % a.config)
    ttft = pf["total_time_ns"] / 1e6
    print(f"  TTFT (prefill {a.prompt} tok) : {ttft:9.3f} ms   "
          f"({'compute' if not pf['memory_bound'] else 'memory'}-bound)")
    for kv, p in decs.items():
        tps = 1e9 / p["total_time_ns"]
        print(f"  TPS @ context {kv:<5}      : {tps:9.1f} tok/s   "
              f"({p['total_time_ns']/1e3:.1f} us/token, "
              f"{'memory' if p['memory_bound'] else 'compute'}-bound)")
    base = decs[a.kv[0]]
    tot = (pf["total_time_ns"] + a.gen * base["total_time_ns"]) / 1e6
    print(f"  end-to-end {a.prompt}+{a.gen} tok   : {tot:9.3f} ms")
    print(f"  weights streamed / token   : {_bytes(base)/1e6:9.0f} MB (decode kv={a.kv[0]})")


if __name__ == "__main__":
    main()
