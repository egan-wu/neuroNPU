// NeuroNPU — hardware configuration loaded from a YAML(-subset) file.
#pragma once
#include <cstdint>
#include <string>

namespace neuronpu {

// All knobs that shape timing and capacity. See configs/default.yaml.
struct Config {
  // Core
  double   core_clock_ghz = 1.0;   // global sim time unit is core-clock cycles
  int      num_cores      = 1;     // independent cores/tiles (share DDR + SRAM BW)

  // Tensor Engine (systolic MAC array)
  int      mac_rows = 32;
  int      mac_cols = 32;

  // Vector Engine
  int      vector_lanes = 256;     // fp32-equivalent lanes per cycle

  // DDR (external memory)
  double   ddr_freq_mhz   = 3200.0;
  int      ddr_width_bits = 64;    // per channel
  int      ddr_channels   = 2;
  double   ddr_latency_ns = 100.0; // fixed access latency added per DMA burst
  uint64_t ddr_size_mb    = 4096;
  int      dma_channels   = 1;     // concurrent DMA transfers (share DDR bandwidth)

  // SRAM scratchpad (on-chip, software-managed)
  uint64_t sram_size_kb        = 4096;
  int      sram_banks          = 16;
  int      sram_bank_width_bytes = 16;  // bytes/cycle per bank (port width)

  // Features
  bool     double_buffer  = true;  // documentation flag; overlap is compiler-driven

  // Peak DDR bandwidth in bytes/ns (= GB/s). DDR transfers 2 words/clock.
  double ddr_peak_bytes_per_ns() const;
  // Peak MACs/cycle of the tensor engine.
  double te_peak_macs_per_cycle() const { return double(mac_rows) * mac_cols; }
  // Aggregate SRAM bandwidth in bytes/cycle (shared by all engines touching SRAM).
  double sram_bw_bytes_per_cycle() const { return double(sram_banks) * sram_bank_width_bytes; }

  static Config load(const std::string& path);
  void dump() const;
};

}  // namespace neuronpu
