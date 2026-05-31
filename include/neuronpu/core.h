// NeuroNPU — the core: event-driven scheduler + functional execution of a Program.
#pragma once
#include "neuronpu/config.h"
#include "neuronpu/isa.h"
#include "neuronpu/logger.h"
#include "neuronpu/memory.h"

#include <string>
#include <vector>

namespace neuronpu {

// Runs a program under `cfg`. Functional results are exact (fp32 accumulate);
// timing is cycle-approximate. The two are decoupled but produced in one pass.
class Core {
 public:
  Core(const Program& prog, const Config& cfg);
  RunResult run();

  // Read up to `max_elems` contiguous elements of a named tensor (post-run
  // inspection / functional validation).
  std::vector<float> read_tensor(const std::string& name, int64_t max_elems) const;

 private:
  // Timing models (return duration in core-clock cycles). DMA timing is handled
  // by the scheduler (fluid bandwidth sharing), not a fixed per-op formula.
  double matmul_cycles(int64_t M, int64_t N, int64_t K) const;
  double vector_cycles(int64_t elems, double passes = 1.0) const;

  // Functional execution of one instruction.
  void exec(const Instr& in, InstrRecord& rec, RunResult& out);

  const Program& prog_;
  const Config&  cfg_;
  Memory         mem_;
};

}  // namespace neuronpu
