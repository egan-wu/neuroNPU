"""Download a real YOLO ONNX, compile it through the NeuroNPU compiler, and
profile it (timing-only). Also runs the per-layer/critical-path analysis.

    python3 -m compiler.profile_yolo
"""
from __future__ import annotations
import argparse, os, shutil
from . import api, analyze


def _ensure_model(repo, path):
    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shutil.copy(hf_hub_download(repo, "onnx/model.onnx"), path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="onnx-community/yolov10n")
    ap.add_argument("--onnx", default="models/yolov10n/model.onnx")
    ap.add_argument("--neuronpu", default="build/neuronpu")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--workdir", default="build/yolo")
    a = ap.parse_args()

    model = api.compile(_ensure_model(a.repo, a.onnx), opt={"reuse": True, "tile": True,
                        "nbuf": 2}, neuronpu=a.neuronpu, config=a.config, workdir=a.workdir)
    g = model._graph
    print(f"=== {a.repo} : {len(g.ops)} IR ops, {g.meta_nconv} convs (640x640 FP32) ===")
    out = os.path.join(a.workdir, "run")
    p = model.profile(name="run")
    print(f"  inference   : {p.time_us:8.1f} us/frame  ({p.fps:.0f} FPS)")
    print(f"  compute     : {p.gmacs:.2f} GMACs ({p.gflops:.1f} GFLOPs)")
    print(f"  MAC util    : {p.te_util*100:.1f}%   DDR {p.ddr_gbps:.1f} GB/s   "
          f"{'MEMORY' if p.memory_bound else 'COMPUTE'}-bound")
    print(f"  peak SRAM   : {p.peak_sram_bytes/1e6:.1f} MB   energy {p.energy_nj/1e6:.2f} mJ")
    print()
    analyze.analyze(out)


if __name__ == "__main__":
    main()
