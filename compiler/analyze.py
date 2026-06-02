"""Post-run analysis of a NeuroNPU trace: per-op-type / per-layer breakdown,
critical-path attribution, and a scope-enriched Perfetto trace.

    python3 -m compiler.analyze build/llama/prefill
"""
from __future__ import annotations
import argparse, json, os
from collections import defaultdict


def _load(d):
    instrs = [json.loads(l) for l in open(os.path.join(d, "isa_trace.jsonl"))]
    scopes = json.load(open(os.path.join(d, "scopes.json")))
    for e in instrs:
        e["scope"] = scopes[e["idx"]] if e["idx"] < len(scopes) else "?"
    return instrs


def _table(title, groups, total_cyc):
    print(f"\n  {title}")
    print(f"    {'scope':<18}{'count':>7}{'cycles':>14}{'%':>7}{'GMACs':>9}{'DDR MB':>9}")
    rows = sorted(groups.items(), key=lambda kv: -kv[1]["cyc"])
    for name, g in rows:
        print(f"    {name:<18}{g['n']:>7}{g['cyc']:>14.0f}{100*g['cyc']/total_cyc:>6.1f}%"
              f"{g['macs']/1e9:>9.3f}{g['bytes']/1e6:>9.1f}")


def _critical_path(instrs):
    """Longest dependency chain by time. Each instr is bound by the latest of its
    wait events' signal times and the prior instr on its engine; trace that back."""
    by_idx = {e["idx"]: e for e in instrs}
    sig_to = {}                      # event id -> instr that signals it
    for e in instrs:
        if e["sig"] >= 0:
            sig_to[e["sig"]] = e
    prev_on_engine = {}              # engine -> last instr (by program order)
    eng_prev = {}
    for e in sorted(instrs, key=lambda x: x["idx"]):
        eng_prev[e["idx"]] = prev_on_engine.get(e["engine"])
        prev_on_engine[e["engine"]] = e
    # binding predecessor = the one whose end == this start
    end = max(instrs, key=lambda e: e["end"])
    chain, cur = [], end
    seen = set()
    while cur is not None and cur["idx"] not in seen:
        seen.add(cur["idx"]); chain.append(cur)
        preds = [sig_to.get(w) for w in cur["wait"]]
        preds.append(eng_prev.get(cur["idx"]))
        nxt, best = None, -1
        for p in preds:
            if p is not None and p["end"] > best and p["end"] <= cur["start"] + 1e-6:
                nxt, best = p, p["end"]
        cur = nxt
    return list(reversed(chain))


def analyze(d):
    instrs = _load(d)
    total_cyc = sum(e["cycles"] for e in instrs)
    wall = max(e["end"] for e in instrs)

    by_op, by_layer = defaultdict(lambda: dict(n=0, cyc=0, macs=0, bytes=0)), \
                      defaultdict(lambda: dict(n=0, cyc=0, macs=0, bytes=0))
    for e in instrs:
        for grp, key in ((by_op, e["scope"].split("/")[-1]), (by_layer, e["scope"].split("/")[0])):
            g = grp[key]; g["n"] += 1; g["cyc"] += e["cycles"]
            g["macs"] += e["macs"]; g["bytes"] += e["bytes"]

    print(f"=== analysis of {d} ===")
    print(f"  wall time {wall:.0f} cyc, total engine-busy {total_cyc:.0f} cyc "
          f"({total_cyc/wall:.2f}x overlap), {len(instrs)} instrs")
    _table("by op type:", by_op, total_cyc)
    _table("by layer:", by_layer, total_cyc)

    cp = _critical_path(instrs)
    cp_cyc = sum(e["cycles"] for e in cp)
    cp_by = defaultdict(float)
    for e in cp:
        cp_by[e["scope"].split("/")[-1]] += e["cycles"]
    print(f"\n  critical path: {len(cp)} instrs, {cp_cyc:.0f} cyc "
          f"({100*cp_cyc/wall:.0f}% of wall) — dominated by:")
    for k, v in sorted(cp_by.items(), key=lambda kv: -kv[1])[:6]:
        print(f"    {k:<18}{v:>12.0f} cyc{100*v/wall:>7.1f}% of wall")

    # scope-enriched Perfetto trace
    tp = os.path.join(d, "trace.json")
    if os.path.exists(tp):
        tr = json.load(open(tp))
        scopes = json.load(open(os.path.join(d, "scopes.json")))
        for ev in tr:
            if ev.get("ph") == "X":
                i = ev.get("args", {}).get("idx")
                if i is not None and i < len(scopes):
                    ev["name"] = scopes[i] + " " + ev["name"]
        out = os.path.join(d, "trace_scoped.json")
        json.dump(tr, open(out, "w"))
        print(f"\n  scoped Perfetto trace -> {out}  (open in https://ui.perfetto.dev)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    analyze(ap.parse_args().logdir)


if __name__ == "__main__":
    main()
