#include "neuronpu/config.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace neuronpu {

double Config::ddr_peak_bytes_per_ns() const {
  // GB/s = MHz * 2 (DDR) * bytes_per_channel * channels / 1000; GB/s == bytes/ns.
  double bytes_per_channel = ddr_width_bits / 8.0;
  return ddr_freq_mhz * 2.0 * bytes_per_channel * ddr_channels / 1000.0;
}

static std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

Config Config::load(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open config: " + path);

  std::unordered_map<std::string, std::string> kv;
  std::string line;
  while (std::getline(f, line)) {
    auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string key = trim(line.substr(0, colon));
    std::string val = trim(line.substr(colon + 1));
    if (key.empty() || val.empty()) continue;  // skip section headers
    kv[key] = val;
  }

  Config c;
  auto getd = [&](const char* k, double& dst) {
    auto it = kv.find(k); if (it != kv.end()) dst = std::stod(it->second);
  };
  auto geti = [&](const char* k, int& dst) {
    auto it = kv.find(k); if (it != kv.end()) dst = std::stoi(it->second);
  };
  auto getu = [&](const char* k, uint64_t& dst) {
    auto it = kv.find(k); if (it != kv.end()) dst = std::stoull(it->second);
  };
  auto getb = [&](const char* k, bool& dst) {
    auto it = kv.find(k);
    if (it != kv.end()) dst = (it->second == "true" || it->second == "1");
  };

  getd("core_clock_ghz", c.core_clock_ghz);
  geti("mac_rows", c.mac_rows);
  geti("mac_cols", c.mac_cols);
  geti("vector_lanes", c.vector_lanes);
  getd("ddr_freq_mhz", c.ddr_freq_mhz);
  geti("ddr_width_bits", c.ddr_width_bits);
  geti("ddr_channels", c.ddr_channels);
  getd("ddr_latency_ns", c.ddr_latency_ns);
  getu("ddr_size_mb", c.ddr_size_mb);
  geti("dma_channels", c.dma_channels);
  getu("sram_size_kb", c.sram_size_kb);
  geti("sram_banks", c.sram_banks);
  getb("double_buffer", c.double_buffer);
  return c;
}

void Config::dump() const {
  std::printf("Config:\n");
  std::printf("  core_clock      : %.3f GHz\n", core_clock_ghz);
  std::printf("  mac array       : %d x %d (%.0f MACs/cyc)\n", mac_rows, mac_cols,
              te_peak_macs_per_cycle());
  std::printf("  vector_lanes    : %d\n", vector_lanes);
  std::printf("  ddr             : %.0f MHz, %d-bit x%d ch, lat %.0f ns => %.1f GB/s\n",
              ddr_freq_mhz, ddr_width_bits, ddr_channels, ddr_latency_ns,
              ddr_peak_bytes_per_ns());
  std::printf("  dma_channels    : %d\n", dma_channels);
  std::printf("  sram            : %llu KB, %d banks\n",
              (unsigned long long)sram_size_kb, sram_banks);
  std::printf("  double_buffer   : %s\n", double_buffer ? "on" : "off");
}

}  // namespace neuronpu
