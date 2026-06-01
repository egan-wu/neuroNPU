"""NeuroNPU compiler — Graph IR.

A tiny, explicit graph of coarse tensor ops (the same altitude as the NeuroNPU
ISA). Frontends (ONNX, or the hand-built tiny model) produce a Graph; the
backend lowers it to NeuroNPU ISA. The op `kind`s map closely onto ISA opcodes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Tensor:
    name: str
    shape: tuple
    data: Optional[np.ndarray] = None   # set => constant/weight value present
    is_input: bool = False              # runtime input (loaded via --load)
    is_output: bool = False
    is_param: bool = False              # DDR-resident weight with no data (timing-only)

    @property
    def is_weight(self) -> bool:
        return self.data is not None or self.is_param

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


@dataclass
class Op:
    kind: str            # rmsnorm | matmul | matmul_t | rope | softmax | silu |
                         # mul | add | gather | (see reference.py for semantics)
    inputs: list         # input tensor names
    outputs: list        # output tensor names
    attrs: dict = field(default_factory=dict)


@dataclass
class Graph:
    name: str = "graph"
    tensors: dict = field(default_factory=dict)   # name -> Tensor
    ops: list = field(default_factory=list)       # topologically ordered
    inputs: list = field(default_factory=list)    # input tensor names
    outputs: list = field(default_factory=list)   # output tensor names

    # ---- construction helpers ----
    def tensor(self, name, shape, data=None, is_input=False, is_output=False,
               is_param=False) -> str:
        t = Tensor(name, tuple(int(s) for s in shape), data, is_input, is_output, is_param)
        self.tensors[name] = t
        if is_input and name not in self.inputs:
            self.inputs.append(name)
        if is_output and name not in self.outputs:
            self.outputs.append(name)
        return name

    def add(self, kind, inputs, out_name, out_shape, attrs=None) -> str:
        self.tensor(out_name, out_shape)
        self.ops.append(Op(kind, list(inputs), [out_name], attrs or {}))
        return out_name

    def mark_output(self, name):
        self.tensors[name].is_output = True
        if name not in self.outputs:
            self.outputs.append(name)
        return name
