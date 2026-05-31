#include "neuronpu/perf.h"

#include <algorithm>
#include <cstdio>
#include <fstream>

namespace neuronpu {

PerfReport analyze(const RunResult& r, const Config& cfg) {
  PerfReport rep;
  rep.total_cycles = r.total_cycles;
  rep.total_time_ns = r.total_cycles / cfg.core_clock_ghz;

  double busy_dma = r.engine_busy[int(Engine::DMA)];
  double busy_ve  = r.engine_busy[int(Engine::VECTOR)];
  int    cores    = std::max(1, cfg.num_cores);
  // engine_busy aggregates across cores; normalize so utilization is per-core average.
  double denom = (r.total_cycles > 0 ? r.total_cycles : 1.0) * cores;

  rep.dma_util = busy_dma / denom;
  rep.ve_util  = busy_ve / denom;

  // MAC-array utilization: useful MACs / (peak MACs/cycle * total cycles * cores).
  double peak_macs = cfg.te_peak_macs_per_cycle() * denom;
  rep.te_util = peak_macs > 0 ? r.total_macs / peak_macs : 0.0;

  rep.total_gmacs = r.total_macs / 1e9;
  rep.ddr_peak_gbps = cfg.ddr_peak_bytes_per_ns();  // bytes/ns == GB/s
  rep.ddr_achieved_gbps = rep.total_time_ns > 0 ? double(r.ddr_bytes) / rep.total_time_ns : 0.0;
  rep.ddr_bw_util = rep.ddr_peak_gbps > 0 ? rep.ddr_achieved_gbps / rep.ddr_peak_gbps : 0.0;

  rep.arithmetic_intensity = r.ddr_bytes > 0 ? r.total_macs / double(r.ddr_bytes) : 0.0;
  // Compare compute-bound vs memory-bound time lower bounds.
  double compute_ns = cfg.te_peak_macs_per_cycle() > 0
      ? (r.total_macs / (cfg.te_peak_macs_per_cycle() * cores)) / cfg.core_clock_ghz : 0.0;
  double memory_ns = rep.ddr_peak_gbps > 0 ? double(r.ddr_bytes) / rep.ddr_peak_gbps : 0.0;
  rep.memory_bound = memory_ns > compute_ns;
  return rep;
}

void print_report(const PerfReport& rep) {
  std::printf("\n===== NeuroNPU Performance Report =====\n");
  std::printf("  total cycles        : %.0f\n", rep.total_cycles);
  std::printf("  total time          : %.2f us\n", rep.total_time_ns / 1000.0);
  std::printf("  compute             : %.4f GMACs\n", rep.total_gmacs);
  std::printf("  TE  (MAC) util      : %.1f %%\n", rep.te_util * 100.0);
  std::printf("  DMA engine util     : %.1f %%\n", rep.dma_util * 100.0);
  std::printf("  VE  engine util     : %.1f %%\n", rep.ve_util * 100.0);
  std::printf("  DDR bandwidth       : %.1f / %.1f GB/s (%.1f %%)\n",
              rep.ddr_achieved_gbps, rep.ddr_peak_gbps, rep.ddr_bw_util * 100.0);
  std::printf("  arithmetic intensity: %.2f MACs/byte\n", rep.arithmetic_intensity);
  std::printf("  roofline verdict    : %s\n", rep.memory_bound ? "MEMORY-bound" : "COMPUTE-bound");
  std::printf("=======================================\n");
}

void write_report_json(const PerfReport& rep, const std::string& path) {
  std::ofstream o(path);
  o << "{\n"
    << "  \"total_cycles\": " << rep.total_cycles << ",\n"
    << "  \"total_time_ns\": " << rep.total_time_ns << ",\n"
    << "  \"total_gmacs\": " << rep.total_gmacs << ",\n"
    << "  \"te_util\": " << rep.te_util << ",\n"
    << "  \"dma_util\": " << rep.dma_util << ",\n"
    << "  \"ve_util\": " << rep.ve_util << ",\n"
    << "  \"ddr_achieved_gbps\": " << rep.ddr_achieved_gbps << ",\n"
    << "  \"ddr_peak_gbps\": " << rep.ddr_peak_gbps << ",\n"
    << "  \"ddr_bw_util\": " << rep.ddr_bw_util << ",\n"
    << "  \"arithmetic_intensity\": " << rep.arithmetic_intensity << ",\n"
    << "  \"memory_bound\": " << (rep.memory_bound ? "true" : "false") << "\n"
    << "}\n";
}

}  // namespace neuronpu
