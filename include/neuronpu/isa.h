// NeuroNPU — ISA model: descriptors, instructions, program, assembler & binary I/O.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "neuronpu/types.h"

namespace neuronpu {

enum class MemSpace { DDR, SRAM };
const char* memspace_name(MemSpace m);

// Which execution engine an opcode runs on.
enum class Engine { DMA, TENSOR, VECTOR };
const char* engine_name(Engine e);

enum class Opcode {
  NOP,
  HALT,
  DMA_LOAD,   // SRAM <- DDR
  DMA_STORE,  // DDR  <- SRAM
  MATMUL,     // out[M,N] = a[M,K] @ b[K,N]   (+= if accumulate)
  VADD,       // out = a + b   (elementwise)
  RELU,       // out = max(0, in)
  GELU,       // out = gelu(in)   (tanh approximation)
  SILU,       // out = in * sigmoid(in)
  SOFTMAX,    // out = softmax(in) along the last dimension
  RMSNORM,    // out = in / rms(in_row) * weight   (imm0 = eps)
  LAYERNORM,  // out = (in-mean)/std * weight + bias   (imm0 = eps)
  REQUANT,    // out = clamp(round(in/scale) + zp)   (imm0 = scale, imm1 = zp)
};
const char* opcode_name(Opcode o);
Engine      opcode_engine(Opcode o);

// A tensor descriptor: a typed, strided view into a memory space.
struct Descriptor {
  std::string          name;
  MemSpace             space = MemSpace::DDR;
  Dtype                dtype = Dtype::F32;
  uint64_t             base_addr = 0;     // byte offset within its space
  std::vector<int64_t> dims;              // logical shape
  std::vector<int64_t> strides;           // element strides; empty => row-major

  int64_t numel() const;
  int64_t bytes() const;
  // Byte offset of logical index `idx` (size == dims.size()).
  uint64_t offset_bytes(const std::vector<int64_t>& idx) const;
};

// One coarse-grained (CISC) instruction. Sync is expressed via optional
// wait-before / signal-after event ids (cross-engine dependencies).
struct Instr {
  Opcode              op = Opcode::NOP;
  std::vector<int>    args;            // descriptor ids; meaning is per-opcode
  std::vector<double> imms;            // scalar immediates (eps, scale, ...)
  bool                accumulate = false;
  int                 wait_event = -1; // issue blocks until this event is signaled
  int                 signal_event = -1; // signaled on completion
  Engine engine() const { return opcode_engine(op); }
  double imm(size_t i, double dflt) const { return i < imms.size() ? imms[i] : dflt; }
};

struct Program {
  std::vector<Descriptor> descriptors;
  std::vector<Instr>      instrs;
  // Initial bytes preloaded into a descriptor's memory before the run.
  std::map<int, std::vector<uint8_t>> init_data;  // descriptor id -> bytes

  int descriptor_id(const std::string& name) const;  // -1 if absent
};

// Assemble human-readable .npuasm text into a Program. Throws std::runtime_error
// with a line number on syntax errors.
Program assemble(const std::string& text);
Program assemble_file(const std::string& path);

// Binary container (.npubin) — the stable compiler<->simulator contract.
void    write_binary(const Program& p, const std::string& path);
Program read_binary(const std::string& path);

}  // namespace neuronpu
