// NeuroNPU — structured run traces (ISA trace + DDR traffic) and the logger.
#pragma once
#include <array>
#include <string>
#include <vector>

#include "neuronpu/isa.h"

namespace neuronpu {

// One executed instruction, with cycle-approximate timing on the global clock.
struct InstrRecord {
  int    idx = 0;
  Opcode op = Opcode::NOP;
  Engine engine = Engine::DMA;
  double start = 0;   // cycles
  double end = 0;     // cycles
  double cycles = 0;  // end - start
  std::vector<int> wait_events;
  int    signal_event = -1;
  double macs = 0;     // tensor ops only
  uint64_t bytes = 0;  // dma ops only
};

// One DDR burst (produced by DMA load/store).
struct DdrRecord {
  double      start = 0;
  double      end = 0;
  bool        is_load = true;  // true: DDR->SRAM, false: SRAM->DDR
  std::string descriptor;
  uint64_t    addr = 0;
  uint64_t    bytes = 0;
  double      cycles = 0;
};

// Aggregate result of a run, consumed by the perf analyzer.
struct RunResult {
  std::vector<InstrRecord> instrs;
  std::vector<DdrRecord>   ddr;
  double                   total_cycles = 0;
  double                   clock_ghz = 1.0;       // for cycle -> time conversion
  std::array<double, 3>    engine_busy{0, 0, 0};  // indexed by Engine
  double                   total_macs = 0;
  uint64_t                 ddr_bytes = 0;
  uint64_t                 sram_bytes = 0;  // SRAM bytes touched (energy model)
};

// Writes the traces to `out_dir` (isa_trace.jsonl, ddr_trace.csv).
class Logger {
 public:
  explicit Logger(std::string out_dir) : dir_(std::move(out_dir)) {}
  void write(const RunResult& r) const;

 private:
  void write_perfetto(const RunResult& r) const;  // Chrome/Perfetto trace.json
  std::string dir_;
};

}  // namespace neuronpu
