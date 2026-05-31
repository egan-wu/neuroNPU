#!/usr/bin/env python3
"""Generate a small multi-layer LLM as NeuroNPU ISA, run prefill + decode, and
report LLM-level metrics (TTFT, TPS, compute/bandwidth efficiency, roofline).

Each transformer layer streams its weights from DDR (so decode is weight-bound,
the realistic regime). The model is run in two phases:

  * prefill  - process the whole prompt (seq = prompt_len) -> time to first token
  * decode   - one step (seq = 1, attending to kv_len cached keys) -> per-token time

Usage:
  python3 tools/llm_demo.py --neuronpu build/neuronpu --config configs/default.yaml \
      --layers 6 --d 64 --ffn 256 --prompt 128 --gen 64 --outdir build/llm
"""
import argparse, json, os, subprocess, sys

ORDER = ["g1", "g2", "Wq", "Wk", "Wv", "Wo", "W1", "W2"]


def wdims(n, d, ffn):
    if n in ("g1", "g2"):           return f"{d}", d
    if n in ("Wq", "Wk", "Wv", "Wo"): return f"{d}x{d}", d * d
    if n == "W1":                   return f"{d}x{ffn}", d * ffn
    if n == "W2":                   return f"{ffn}x{d}", ffn * d
    raise ValueError(n)


def emit(layers, q_rows, kv_rows, d, ffn):
    """Return .npuasm text for one model run with the given shapes."""
    elems = {n: wdims(n, d, ffn)[1] for n in ORDER}
    blk = sum(elems[n] for n in ORDER)                 # weight elems per layer
    off, acc = {}, 0
    for n in ORDER:
        off[n] = acc; acc += elems[n]

    # bump allocators (256B aligned)
    ddr = [0]
    def dalloc(nb):
        a = ddr[0]; ddr[0] += (nb + 255) // 256 * 256; return a
    sram = [0]
    def salloc(nb):
        a = sram[0]; sram[0] += (nb + 255) // 256 * 256; return a

    dx = dalloc(q_rows * d * 4)
    dout = dalloc(q_rows * d * 4)
    wbase = dalloc(blk * layers * 4)

    a_x = salloc(q_rows * d * 4);  a_xn = salloc(q_rows * d * 4)
    a_q = salloc(q_rows * d * 4)
    a_k = salloc(kv_rows * d * 4); a_v = salloc(kv_rows * d * 4)
    a_sc = salloc(q_rows * kv_rows * 4)
    a_at = salloc(q_rows * d * 4); a_ao = salloc(q_rows * d * 4)
    a_f1 = salloc(q_rows * ffn * 4); a_f2 = salloc(q_rows * d * 4)
    sw = {n: salloc(elems[n] * 4) for n in ORDER}

    L = []
    P = L.append
    # DDR
    P(f".desc dx   ddr f32 0x{dx:x} {q_rows}x{d}")
    P(f".desc dout ddr f32 0x{dout:x} {q_rows}x{d}")
    P(f".desc dWall ddr f32 0x{wbase:x} {blk * layers}")
    for n in ORDER:
        dims, _ = wdims(n, d, ffn)
        P(f".desc d{n} ddr f32 0x{wbase + off[n] * 4:x} {dims} +{blk}")
    P(".data dx rand 1")
    P(".data dWall const 0.05")
    # SRAM activations
    P(f".desc x      sram f32 0x{a_x:x} {q_rows}x{d}")
    P(f".desc xn     sram f32 0x{a_xn:x} {q_rows}x{d}")
    P(f".desc q      sram f32 0x{a_q:x} {q_rows}x{d}")
    P(f".desc kproj  sram f32 0x{a_k:x} {q_rows}x{d}")
    P(f".desc kT     sram f32 0x{a_k:x} {d}x{kv_rows} :1,{d}")
    P(f".desc vproj  sram f32 0x{a_v:x} {q_rows}x{d}")
    P(f".desc v      sram f32 0x{a_v:x} {kv_rows}x{d}")
    P(f".desc scores sram f32 0x{a_sc:x} {q_rows}x{kv_rows}")
    P(f".desc attn   sram f32 0x{a_at:x} {q_rows}x{d}")
    P(f".desc attno  sram f32 0x{a_ao:x} {q_rows}x{d}")
    P(f".desc ffn1   sram f32 0x{a_f1:x} {q_rows}x{ffn}")
    P(f".desc ffn2   sram f32 0x{a_f2:x} {q_rows}x{d}")
    # SRAM weights
    for n in ORDER:
        dims, _ = wdims(n, d, ffn)
        P(f".desc {n} sram f32 0x{sw[n]:x} {dims}")

    # program: stage x once, then loop the layers (weights stream per layer)
    P("DMA.LOAD x dx")
    P(f"LOOP ${layers}")
    for i, n in enumerate(ORDER):
        sig = "  @sig 0" if i == len(ORDER) - 1 else ""
        P(f"  DMA.LOAD {n} d{n}{sig}")
    P("  RMSNORM xn x g1 $0.00001 @wait 0 @sig 1")
    P("  MATMUL  q xn Wq @wait 1 @sig 2")
    P("  MATMUL  kproj xn Wk @sig 3")
    P("  MATMUL  vproj xn Wv @sig 4")
    P("  ROPE    q q $10000 $0 @wait 2 @sig 5")
    P("  ROPE    kproj kproj $10000 $0 @wait 3 @sig 6")
    P("  MATMUL  scores q kT @wait 6 @sig 7")
    P("  SOFTMAX scores scores @wait 7 @sig 8")
    P("  MATMUL  attn scores v @wait 8 @sig 9")
    P("  MATMUL  attno attn Wo @sig 10")
    P("  VADD    x x attno @wait 10 @sig 11")
    P("  RMSNORM xn x g2 $0.00001 @sig 12")
    P("  MATMUL  ffn1 xn W1 @wait 12 @sig 13")
    P("  SILU    ffn1 ffn1 @wait 13 @sig 14")
    P("  MATMUL  ffn2 ffn1 W2 @wait 14 @sig 15")
    P("  VADD    x x ffn2 @wait 15 @sig 16")
    P("ENDLOOP")
    P("DMA.STORE dout x")
    P("HALT")
    return "\n".join(L) + "\n"


def run_phase(neuronpu, config, outdir, name, asm):
    os.makedirs(outdir, exist_ok=True)
    asm_path = os.path.join(outdir, name + ".npuasm")
    bin_path = os.path.join(outdir, name + ".npubin")
    log_dir = os.path.join(outdir, name + "_logs")
    with open(asm_path, "w") as f:
        f.write(asm)
    subprocess.run([neuronpu, "asm", asm_path, bin_path], check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run([neuronpu, "run", bin_path, "--config", config, "--out", log_dir],
                   check=True, stdout=subprocess.DEVNULL)
    with open(os.path.join(log_dir, "perf.json")) as f:
        return json.load(f), bin_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neuronpu", default="build/neuronpu")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--outdir", default="build/llm")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--ffn", type=int, default=256)
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--gen", type=int, default=64)
    a = ap.parse_args()

    pre = emit(a.layers, a.prompt, a.prompt, a.d, a.ffn)
    dec = emit(a.layers, 1, a.prompt + 1, a.d, a.ffn)
    pf, pbin = run_phase(a.neuronpu, a.config, a.outdir, "prefill", pre)
    df, dbin = run_phase(a.neuronpu, a.config, a.outdir, "decode", dec)

    ttft_ms = pf["total_time_ns"] / 1e6
    dec_ns = df["total_time_ns"]
    tps = 1e9 / dec_ns if dec_ns > 0 else 0.0
    total_ms = (pf["total_time_ns"] + a.gen * dec_ns) / 1e6

    def pct(x): return f"{x * 100:5.1f}%"

    print(f"\n=== NeuroNPU LLM demo : {a.layers} layers, d={a.d}, ffn={a.ffn}, "
          f"prompt={a.prompt}, gen={a.gen} ===")
    print(f"  artifacts: {pbin}, {dbin}\n")
    print(f"  {'phase':<9}{'time':>11}{'GMACs':>9}{'MAC util':>10}"
          f"{'DDR BW':>9}{'AI(m/B)':>9}{'roofline':>14}{'energy':>11}")
    for nm, p, t_ns in (("prefill", pf, pf["total_time_ns"]),
                        ("decode", df, dec_ns)):
        verdict = "MEMORY-bound" if p["memory_bound"] else "COMPUTE-bound"
        print(f"  {nm:<9}{t_ns/1e3:>9.2f}us{p['total_gmacs']:>9.4f}"
              f"{pct(p['te_util']):>10}{pct(p['ddr_bw_util']):>9}"
              f"{p['arithmetic_intensity']:>9.2f}{verdict:>14}"
              f"{p['energy_total_nj']:>9.1f}nJ")

    print("\n  --- LLM-level metrics ---")
    print(f"  TTFT (time to first token)   : {ttft_ms:8.3f} ms   "
          f"(prefill of {a.prompt} tokens; {'compute' if not pf['memory_bound'] else 'memory'}-bound)")
    print(f"  per-token latency (decode)   : {dec_ns/1e3:8.3f} us")
    print(f"  TPS (tokens / second)        : {tps:8.1f} tok/s  "
          f"({'memory' if df['memory_bound'] else 'compute'}-bound: weights streamed/token)")
    print(f"  energy / generated token     : {df['energy_total_nj']:8.2f} nJ   "
          f"avg power {df['avg_power_mw']:.0f} mW")
    print(f"  end-to-end ({a.prompt} prompt + {a.gen} gen): {total_ms:8.3f} ms total, "
          f"effective {a.gen/ (total_ms/1e3):.1f} tok/s")
    print()


if __name__ == "__main__":
    sys.exit(main())
