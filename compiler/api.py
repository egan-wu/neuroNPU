"""NeuroNPU SDK — the stable, user-facing API.

This is the layer a user of the NPU is meant to touch. It hides the internal IR,
backend allocator, event scheduler, and raw `opt` flag dicts behind a small,
named surface:

    import compiler.api as npu

    model = npu.compile("yolov10n.onnx", opt=npu.O2)      # ONNX or a Graph
    prof  = model.profile()                                # timing-only
    print(prof.fps, prof.gmacs, prof.memory_bound)
    model.save("yolov10n.npubin")                          # artifact + metadata

    out = model.run({"x": x})                              # functional execution

Optimization is exposed as *named levels* (O0/O1/O2) instead of internal flags,
and quantization as a marker (`npu.Int8(calib)`), mirroring how real NPU SDKs
(Vela, SNPE, TensorRT, OpenVINO) present `-O3` / `--int8` rather than scheduler
internals. `profile()` returns a typed `Profile`, not a raw dict.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import os
import shutil
from typing import Optional, Union

from .ir import Graph
from .driver import compile_and_run
from . import reference, passes

# ---------------------------------------------------------------------------
# Named optimization levels. Users pick a level; the internal flag dict stays
# an implementation detail. Pass a raw dict to `opt=` to override per-flag.
# ---------------------------------------------------------------------------
O0: dict = {}                                              # no optimization
O1: dict = {"reuse": True, "fuse": True}                   # SRAM reuse + residual fusion
O2: dict = {"reuse": True, "fuse": True, "tile": True, "nbuf": 2}  # + tiling + double-buffer

_LEVELS = {"O0": O0, "O1": O1, "O2": O2}

DEFAULT_NEURONPU = "build/neuronpu"
DEFAULT_CONFIG = "configs/default.yaml"
SDK_VERSION = "0.1.0"


class Int8:
    """Static int8 quantization request. `calib` is a dict of representative
    inputs used to compute activation scales (per-tensor). Weight matmuls are
    rewritten to int8 GEMM with fp32 accumulate + dequant (see passes.quantize).
    """

    def __init__(self, calib: Optional[dict] = None):
        self.calib = calib or {}

    def __repr__(self):
        return f"Int8(calib={list(self.calib)})"


@dataclass
class Profile:
    """Typed profiling result. Friendly fields up front; `.raw` keeps the full
    simulator perf dict for anything not surfaced here."""
    time_ns: float
    gmacs: float
    te_util: float                 # tensor-engine (MAC) utilization [0..1]
    ddr_gbps: float                # achieved DDR bandwidth
    ddr_bw_util: float             # fraction of peak DDR bandwidth
    arithmetic_intensity: float    # MACs / byte
    memory_bound: bool
    energy_nj: float
    peak_sram_bytes: int
    cycles: float
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def gflops(self) -> float:
        return self.gmacs * 2

    @property
    def fps(self) -> float:
        return 1e9 / self.time_ns if self.time_ns else 0.0

    @property
    def tps(self) -> float:        # tokens/s for a single decode step
        return self.fps

    @property
    def time_us(self) -> float:
        return self.time_ns / 1e3

    @property
    def ddr_bytes(self) -> float:
        return self.ddr_gbps * self.time_ns      # GB/s * ns = bytes

    @property
    def bound(self) -> str:
        return "memory" if self.memory_bound else "compute"

    @classmethod
    def from_perf(cls, perf: dict) -> "Profile":
        cs = perf.get("compile_stats", {})
        return cls(
            time_ns=perf["total_time_ns"],
            gmacs=perf["total_gmacs"],
            te_util=perf["te_util"],
            ddr_gbps=perf["ddr_achieved_gbps"],
            ddr_bw_util=perf["ddr_bw_util"],
            arithmetic_intensity=perf["arithmetic_intensity"],
            memory_bound=perf["memory_bound"],
            energy_nj=perf["energy_total_nj"],
            peak_sram_bytes=int(cs.get("peak_sram_bytes", 0)),
            cycles=perf.get("total_cycles", 0.0),
            raw=perf,
        )

    def summary(self) -> str:
        return (f"{self.time_us:.1f}us  {self.fps:.0f}/s  {self.gflops:.2f} GFLOPs  "
                f"MACu {self.te_util*100:.1f}%  DDR {self.ddr_gbps:.1f}GB/s "
                f"({self.ddr_bytes/1e6:.0f}MB)  {self.bound}-bound  "
                f"{self.energy_nj/1e6:.2f}mJ  peakSRAM {self.peak_sram_bytes/1e6:.1f}MB")


def _resolve_opt(opt) -> dict:
    """Accept a named level ('O2'), one of the O0/O1/O2 dicts, a raw flag dict,
    or None (-> O2 default)."""
    if opt is None:
        return dict(O2)
    if isinstance(opt, str):
        if opt not in _LEVELS:
            raise ValueError(f"unknown opt level {opt!r}; use one of {list(_LEVELS)}")
        return dict(_LEVELS[opt])
    return dict(opt)


def _load_source(source: Union[Graph, str], name: Optional[str]) -> Graph:
    """Turn a Graph or a model path into a Graph IR."""
    if isinstance(source, Graph):
        if name:
            source.name = name
        return source
    if isinstance(source, str) and source.endswith(".onnx"):
        from .onnx_frontend import import_yolo
        g = import_yolo(source)
        if name:
            g.name = name
        return g
    raise TypeError(
        "compile() source must be a Graph or a path to an .onnx file; got "
        f"{type(source).__name__}. (Llama configs build a Graph via "
        "models.llama_from_config, then pass that Graph here.)")


class Model:
    """A compiled NeuroNPU model: a Graph + target/opt settings. Lowers lazily
    on the first profile()/run()/save() call."""

    def __init__(self, graph: Graph, *, opt: dict, quant: Optional[Int8] = None,
                 neuronpu: str = DEFAULT_NEURONPU, config: str = DEFAULT_CONFIG,
                 workdir: str = "build/sdk"):
        self._graph = graph
        self.name = graph.name
        self.opt = opt
        self.quant = quant
        self.neuronpu = neuronpu
        self.config = config
        self.workdir = workdir
        self._last: Optional[Profile] = None

    # -- internal: the graph the backend actually compiles (post quant pass) --
    def _lowered_graph(self) -> Graph:
        if self.quant is not None:
            return passes.quantize(self._graph, self.quant.calib)
        return self._graph

    @property
    def npubin(self) -> str:
        return os.path.join(self.workdir, "model.npubin")

    @property
    def n_ops(self) -> int:
        return len(self._graph.ops)

    def profile(self, name: str = "run") -> Profile:
        """Run the simulator in timing-only mode and return metrics."""
        _, perf = compile_and_run(self._lowered_graph(), {}, self.neuronpu,
                                  self.config, self.workdir, timing_only=True,
                                  out_dir=os.path.join(self.workdir, name),
                                  opt=self.opt)
        self._last = Profile.from_perf(perf)
        return self._last

    def run(self, inputs: dict, name: str = "run") -> dict:
        """Functionally execute the model and return {output_name: ndarray}."""
        outputs, perf = compile_and_run(self._lowered_graph(), inputs, self.neuronpu,
                                        self.config, self.workdir, timing_only=False,
                                        out_dir=os.path.join(self.workdir, name),
                                        opt=self.opt)
        self._last = Profile.from_perf(perf)
        return outputs

    def validate(self, inputs: dict, rtol=2e-3, atol=2e-3) -> bool:
        """Compare functional output against the numpy reference executor."""
        g = self._lowered_graph()
        ref = reference.execute(g, inputs)
        got = self.run(inputs)
        import numpy as np
        ok = True
        for o in g.outputs:
            ok = ok and np.allclose(ref[o], got[o], rtol=rtol, atol=atol)
        return ok

    def inspect(self) -> str:
        """Return the lowered ISA assembly text (also written to workdir)."""
        self.profile(name="inspect")
        with open(os.path.join(self.workdir, "model.npuasm")) as f:
            return f.read()

    def _metadata(self) -> dict:
        m = {
            "sdk_version": SDK_VERSION,
            "name": self.name,
            "opt": self.opt,
            "quant": repr(self.quant) if self.quant else None,
            "npu_config": self.config,
            "n_ops": self.n_ops,
            "n_tensors": len(self._graph.tensors),
        }
        if self._last is not None:
            m["profile"] = {
                "time_ns": self._last.time_ns,
                "gmacs": self._last.gmacs,
                "fps": self._last.fps,
                "te_util": self._last.te_util,
                "memory_bound": self._last.memory_bound,
                "peak_sram_bytes": self._last.peak_sram_bytes,
            }
        return m

    def save(self, path: str) -> str:
        """Write the compiled artifact (.npubin) and a `<path>.meta.json`
        sidecar with provenance. Compiles first if not already done."""
        if self._last is None:
            self.profile(name="save")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        shutil.copy(self.npubin, path)
        meta_path = path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(self._metadata(), f, indent=2)
        return path


@dataclass
class Artifact:
    """A loaded, standalone .npubin + its metadata. Re-profilable on the
    simulator without the original Graph (timing-only)."""
    npubin: str
    meta: dict
    neuronpu: str = DEFAULT_NEURONPU
    config: str = DEFAULT_CONFIG

    def profile(self, out_dir: str = "build/sdk/loaded") -> Profile:
        import subprocess
        os.makedirs(out_dir, exist_ok=True)
        cmd = [self.neuronpu, "run", self.npubin, "--config",
               self.meta.get("npu_config", self.config), "--out", out_dir,
               "--timing-only"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        with open(os.path.join(out_dir, "perf.json")) as f:
            return Profile.from_perf(json.load(f))


def load(path: str, *, neuronpu: str = DEFAULT_NEURONPU,
         config: str = DEFAULT_CONFIG) -> Artifact:
    """Load a saved .npubin artifact (+ sidecar metadata if present)."""
    meta = {}
    meta_path = path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return Artifact(npubin=path, meta=meta, neuronpu=neuronpu, config=config)


def compile(source: Union[Graph, str], *, opt=O2, quant: Optional[Int8] = None,
            neuronpu: str = DEFAULT_NEURONPU, config: str = DEFAULT_CONFIG,
            workdir: str = "build/sdk", name: Optional[str] = None) -> Model:
    """Compile a model into a NeuroNPU `Model`.

    source : a Graph IR, or a path to an .onnx file (vision/YOLO frontend).
    opt    : 'O0'|'O1'|'O2', one of the O0/O1/O2 dicts, or a raw flag dict.
    quant  : an Int8(calib) request, or None.
    """
    g = _load_source(source, name)
    return Model(g, opt=_resolve_opt(opt), quant=quant, neuronpu=neuronpu,
                 config=config, workdir=workdir)
