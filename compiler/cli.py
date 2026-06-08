"""`neuronpu` — the unified SDK command-line tool.

A single entry point that wraps the high-level API (compiler.api) so a user can
take a model file through the whole flow without touching Python internals:

    neuronpu compile  model.onnx -o model.npubin --opt O2 [--int8 calib.npz]
    neuronpu profile  model.onnx|model.npubin [--report out/]
    neuronpu run      model.onnx --input x=x.npy --output y=y.npy
    neuronpu inspect  model.onnx [-o model.npuasm]

`compile`/`run`/`inspect` take a model source (an .onnx file); `profile` also
accepts a saved .npubin artifact and re-profiles it standalone. (Llama, which
needs sequence/kv parameters, is profiled via `python3 -m compiler.profile_llama`.)
"""
from __future__ import annotations
import argparse
import os
import numpy as np

from . import api, analyze


def _inputs(pairs) -> dict:
    """Parse --input name=path.npy ... into {name: ndarray}."""
    d = {}
    for p in pairs or []:
        name, path = p.split("=", 1)
        d[name] = np.load(path)
    return d


def _quant(args):
    if not args.int8:
        return None
    calib = {}
    if args.int8 != "-":                       # "-" = empty calibration
        z = np.load(args.int8)
        calib = {k: z[k] for k in z.files} if hasattr(z, "files") else {"x": z}
    return api.Int8(calib)


def _model(args):
    return api.compile(args.model, opt=args.opt, quant=_quant(args),
                       neuronpu=args.neuronpu, config=args.config,
                       workdir=args.workdir)


def cmd_compile(args):
    m = _model(args)
    out = args.output or os.path.splitext(os.path.basename(args.model))[0] + ".npubin"
    m.save(out)
    print(f"compiled {args.model} -> {out}  (opt={args.opt}, {m.n_ops} ops)")
    print(f"  metadata: {out}.meta.json")


def cmd_profile(args):
    if args.model.endswith(".npubin"):
        prof = api.load(args.model, neuronpu=args.neuronpu,
                        config=args.config).profile(out_dir=args.workdir + "/loaded")
        print(f"=== profile {args.model} (artifact) ===")
        print("  " + prof.summary())
        return
    m = _model(args)
    out = os.path.join(args.workdir, "run")
    prof = m.profile(name="run")
    print(f"=== profile {args.model} (opt={args.opt}, {m.n_ops} ops) ===")
    print("  " + prof.summary())
    if args.report:
        analyze.analyze(out)


def cmd_run(args):
    m = _model(args)
    outs = m.run(_inputs(args.input))
    for name, save_to in (p.split("=", 1) for p in (args.output or [])):
        np.save(save_to, outs[name])
        print(f"  saved {name} {tuple(outs[name].shape)} -> {save_to}")
    if not args.output:
        for name, arr in outs.items():
            print(f"  {name}: shape {tuple(arr.shape)}")


def cmd_inspect(args):
    text = _model(args).inspect()
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"wrote ISA -> {args.output} ({text.count(chr(10))+1} lines)")
    else:
        print(text)


def _add_common(p):
    p.add_argument("model", help="model source (.onnx) or .npubin artifact")
    p.add_argument("--opt", default="O2", help="optimization level O0|O1|O2")
    p.add_argument("--neuronpu", default=api.DEFAULT_NEURONPU, help="simulator binary")
    p.add_argument("--config", default=api.DEFAULT_CONFIG, help="NPU config yaml")
    p.add_argument("--workdir", default="build/sdk", help="scratch/output dir")
    p.add_argument("--int8", default=None,
                   help="static int8 quant; arg is a calibration .npz ('-' for none)")


def build_parser():
    ap = argparse.ArgumentParser(prog="neuronpu", description="NeuroNPU SDK tool")
    ap.add_argument("--version", action="version", version=f"neuronpu {api.SDK_VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile", help="compile a model to a .npubin artifact")
    _add_common(c)
    c.add_argument("-o", "--output", help="output .npubin path")
    c.set_defaults(func=cmd_compile)

    pr = sub.add_parser("profile", help="profile a model or artifact (timing)")
    _add_common(pr)
    pr.add_argument("--report", action="store_true", help="also print per-layer breakdown")
    pr.set_defaults(func=cmd_profile)

    r = sub.add_parser("run", help="functionally execute a model")
    _add_common(r)
    r.add_argument("--input", action="append", help="name=path.npy (repeatable)")
    r.add_argument("--output", action="append", help="name=path.npy (repeatable)")
    r.set_defaults(func=cmd_run)

    i = sub.add_parser("inspect", help="dump the lowered ISA assembly")
    _add_common(i)
    i.add_argument("-o", "--output", help="write .npuasm here (else stdout)")
    i.set_defaults(func=cmd_inspect)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
