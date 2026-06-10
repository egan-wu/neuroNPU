// NeuroNPU runtime — C API implementation (wraps Core / Config / perf analyze).
#include "neuronpu/runtime.h"

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "neuronpu/config.h"
#include "neuronpu/core.h"
#include "neuronpu/isa.h"
#include "neuronpu/perf.h"

namespace fs = std::filesystem;
using namespace neuronpu;

namespace {
thread_local std::string g_err;
void set_err(const std::string& m) { g_err = m; }
}  // namespace

struct npu_model {
  Program prog;
  Config cfg;
  std::unordered_map<std::string, std::string> meta;
  std::unordered_map<std::string, std::vector<float>> inputs;  // bound tensors
  std::unique_ptr<Core> core;                                   // last functional run
};

extern "C" {

npu_model* npu_open(const char* npubin_path, const char* config_path) {
  try {
    auto m = std::make_unique<npu_model>();
    if (!npubin_path) throw std::runtime_error("npu_open: null path");
    m->prog = read_binary(npubin_path);
    std::string cfg = config_path ? config_path : "";
    m->cfg = (!cfg.empty() && fs::exists(cfg)) ? Config::load(cfg) : Config{};
    for (const auto& kv : m->prog.meta) m->meta[kv.first] = kv.second;
    return m.release();
  } catch (const std::exception& e) {
    set_err(e.what());
    return nullptr;
  }
}

void npu_close(npu_model* m) { delete m; }

const char* npu_meta(const npu_model* m, const char* key) {
  if (!m || !key) return nullptr;
  auto it = m->meta.find(key);
  return it == m->meta.end() ? nullptr : it->second.c_str();
}

int npu_profile_run(npu_model* m, npu_profile* out) {
  try {
    if (!m || !out) throw std::runtime_error("npu_profile_run: null arg");
    Core core(m->prog, m->cfg, /*functional=*/false);
    RunResult res = core.run();
    PerfReport r = analyze(res, m->cfg);
    out->time_ns = r.total_time_ns;
    out->cycles = r.total_cycles;
    out->gmacs = r.total_gmacs;
    out->te_util = r.te_util;
    out->ddr_gbps = r.ddr_achieved_gbps;
    out->ddr_bw_util = r.ddr_bw_util;
    out->arithmetic_intensity = r.arithmetic_intensity;
    out->energy_nj = r.energy_total_nj;
    out->memory_bound = r.memory_bound ? 1 : 0;
    return 0;
  } catch (const std::exception& e) {
    set_err(e.what());
    return 1;
  }
}

int npu_set_input(npu_model* m, const char* name, const float* data, int64_t n) {
  if (!m || !name || (!data && n > 0)) { set_err("npu_set_input: null arg"); return 1; }
  m->inputs[name] = std::vector<float>(data, data + n);
  return 0;
}

int npu_run(npu_model* m) {
  try {
    if (!m) throw std::runtime_error("npu_run: null model");
    m->core = std::make_unique<Core>(m->prog, m->cfg, /*functional=*/true);
    for (const auto& kv : m->inputs) m->core->write_tensor(kv.first, kv.second);
    m->core->run();
    return 0;
  } catch (const std::exception& e) {
    set_err(e.what());
    m->core.reset();
    return 1;
  }
}

int64_t npu_output_numel(npu_model* m, const char* name) {
  try {
    if (!m || !name) throw std::runtime_error("npu_output_numel: null arg");
    if (m->core) return m->core->tensor_numel(name);
    int id = m->prog.descriptor_id(name);              // pre-run: from the table
    if (id < 0) throw std::runtime_error(std::string("unknown tensor: ") + name);
    int64_t n = 1;
    for (int64_t d : m->prog.descriptors[id].dims) n *= d;
    return n;
  } catch (const std::exception& e) {
    set_err(e.what());
    return -1;
  }
}

int npu_get_output(npu_model* m, const char* name, float* out, int64_t n) {
  try {
    if (!m || !name || (!out && n > 0)) throw std::runtime_error("npu_get_output: null arg");
    if (!m->core) throw std::runtime_error("npu_get_output: call npu_run() first");
    std::vector<float> v = m->core->read_tensor(name, n);
    for (int64_t i = 0; i < n && i < int64_t(v.size()); ++i) out[i] = v[i];
    return 0;
  } catch (const std::exception& e) {
    set_err(e.what());
    return 1;
  }
}

const char* npu_last_error(void) { return g_err.c_str(); }

}  // extern "C"
