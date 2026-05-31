// NeuroNPU — post-run performance analysis (utilization, bandwidth, roofline).
#pragma once
#include <string>

#include "neuronpu/config.h"
#include "neuronpu/logger.h"

namespace neuronpu {

struct PerfReport {
  double   total_cycles = 0;
  double   total_time_ns = 0;
  double   te_util = 0;        // MAC-array utilization [0,1]
  double   dma_util = 0;       // DMA engine busy fraction
  double   ve_util = 0;        // vector engine busy fraction
  double   ddr_achieved_gbps = 0;
  double   ddr_peak_gbps = 0;
  double   ddr_bw_util = 0;
  double   total_gmacs = 0;
  double   arithmetic_intensity = 0;  // MACs per DDR byte
  bool     memory_bound = false;
};

PerfReport analyze(const RunResult& r, const Config& cfg);
void       print_report(const PerfReport& rep);
void       write_report_json(const PerfReport& rep, const std::string& path);

}  // namespace neuronpu
