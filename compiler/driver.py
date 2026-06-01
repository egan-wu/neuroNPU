"""Compile a Graph IR, run it on the NeuroNPU simulator, and (optionally)
validate the result against the numpy reference executor."""
from __future__ import annotations
import os, subprocess
import numpy as np
from .ir import Graph
from .backend import compile_graph
from . import reference, passes


def _save_npy(path, arr):
    np.save(path, np.asarray(arr, dtype=np.float32).ravel())


def compile_and_run(g: Graph, inputs: dict, neuronpu: str, config: str,
                    workdir: str, timing_only=False, out_dir=None, opt=None):
    os.makedirs(workdir, exist_ok=True)
    out_dir = out_dir or os.path.join(workdir, "logs")
    g = passes.apply(g, opt or {})                 # IR optimization passes (fusion, ...)
    text, ddr_data, in_descs, out_descs, stats = compile_graph(g, opt)
    asm_path = os.path.join(workdir, "model.npuasm")
    bin_path = os.path.join(workdir, "model.npubin")
    with open(asm_path, "w") as f:
        f.write(text)
    subprocess.run([neuronpu, "asm", asm_path, bin_path], check=True,
                   stdout=subprocess.DEVNULL)

    cmd = [neuronpu, "run", bin_path, "--config", config, "--out", out_dir]
    if timing_only:
        cmd.append("--timing-only")
    else:
        for asm_name, arr in ddr_data.items():               # weights
            p = os.path.join(workdir, asm_name + ".npy"); _save_npy(p, arr)
            cmd += ["--load", f"{asm_name}={p}"]
        for ir_in, asm_name in in_descs.items():             # runtime inputs
            p = os.path.join(workdir, asm_name + ".npy"); _save_npy(p, inputs[ir_in])
            cmd += ["--load", f"{asm_name}={p}"]
        for ir_out, asm_name in out_descs.items():            # outputs
            cmd += ["--save", f"{asm_name}={os.path.join(workdir, asm_name + '.out.npy')}"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    import json
    with open(os.path.join(out_dir, "perf.json")) as f:
        perf = json.load(f)

    perf["compile_stats"] = stats
    outputs = {}
    if not timing_only:
        for ir_out, asm_name in out_descs.items():
            flat = np.load(os.path.join(workdir, asm_name + ".out.npy"))
            outputs[ir_out] = flat.reshape(g.tensors[ir_out].shape)
    return outputs, perf


def validate(g: Graph, inputs: dict, neuronpu: str, config: str, workdir: str,
             rtol=2e-3, atol=2e-3, opt=None):
    ref = reference.execute(g, inputs)
    got, perf = compile_and_run(g, inputs, neuronpu, config, workdir, opt=opt)
    ok = True
    for o in g.outputs:
        r, x = ref[o], got[o]
        close = np.allclose(r, x, rtol=rtol, atol=atol)
        maxerr = float(np.max(np.abs(r - x)))
        print(f"  output {o}: shape {tuple(r.shape)}  max|err|={maxerr:.2e}  "
              f"{'MATCH' if close else 'MISMATCH'}")
        ok = ok and close
    return ok, perf
