// NeuroNPU simulator CLI.
//
//   neuronpu run <prog.npuasm|.npubin> [--config cfg.yaml] [--out dir]
//   neuronpu asm <prog.npuasm> <out.npubin>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include "neuronpu/config.h"
#include "neuronpu/core.h"
#include "neuronpu/isa.h"
#include "neuronpu/logger.h"
#include "neuronpu/perf.h"

#include <fstream>
#include <vector>

namespace fs = std::filesystem;
using namespace neuronpu;

static bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// ---- minimal fp32 .npy I/O (C-order, little-endian) for golden comparison ----
static std::vector<float> load_npy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open npy: " + path);
  char magic[6]; f.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("bad npy magic: " + path);
  uint8_t major = 0, minor = 0; f.read((char*)&major, 1); f.read((char*)&minor, 1);
  uint32_t hlen = 0;
  if (major == 1) { uint16_t h; f.read((char*)&h, 2); hlen = h; }
  else            { f.read((char*)&hlen, 4); }
  std::string hdr(hlen, '\0'); f.read(&hdr[0], hlen);
  if (hdr.find("'<f4'") == std::string::npos)
    throw std::runtime_error("npy must be little-endian float32 (<f4): " + path);
  size_t lp = hdr.find('('), rp = hdr.find(')', lp);
  std::string shp = hdr.substr(lp + 1, rp - lp - 1);
  int64_t count = 1; bool any = false; std::string num;
  auto flush = [&]{ if (!num.empty()) { count *= std::stoll(num); any = true; num.clear(); } };
  for (char c : shp) { if (isdigit(c)) num += c; else flush(); }
  flush();
  if (!any) count = 1;  // scalar
  std::vector<float> data(count);
  f.read(reinterpret_cast<char*>(data.data()), count * 4);
  return data;
}

static void save_npy(const std::string& path, int64_t n, const std::vector<float>& data) {
  std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                    std::to_string(n) + ",), }";
  size_t total = 10 + hdr.size() + 1;
  hdr += std::string((64 - total % 64) % 64, ' ');
  hdr += "\n";
  uint16_t hlen = uint16_t(hdr.size());
  std::ofstream o(path, std::ios::binary);
  if (!o) throw std::runtime_error("cannot write npy: " + path);
  o.write("\x93NUMPY", 6);
  char ver[2] = {1, 0}; o.write(ver, 2);
  o.write(reinterpret_cast<char*>(&hlen), 2);
  o.write(hdr.data(), hdr.size());
  o.write(reinterpret_cast<const char*>(data.data()), data.size() * 4);
}

// Split "NAME=path" into (NAME, path).
static std::pair<std::string, std::string> split_eq(const std::string& s) {
  auto e = s.find('=');
  if (e == std::string::npos) throw std::runtime_error("expected NAME=path, got: " + s);
  return {s.substr(0, e), s.substr(e + 1)};
}

static Program load_program(const std::string& path) {
  if (ends_with(path, ".npubin")) return read_binary(path);
  return assemble_file(path);
}

static int cmd_run(int argc, char** argv) {
  if (argc < 1) { std::fprintf(stderr, "usage: neuronpu run <prog> [--config f] [--out d]\n"); return 2; }
  std::string prog_path = argv[0];
  std::string cfg_path = "configs/default.yaml";
  std::string out_dir = "logs";
  std::string dump_name;
  bool timing_only = false;
  std::vector<std::pair<std::string, std::string>> loads, saves;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
    else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump_name = argv[++i];
    else if (!std::strcmp(argv[i], "--timing-only")) timing_only = true;
    else if (!std::strcmp(argv[i], "--load") && i + 1 < argc) loads.push_back(split_eq(argv[++i]));
    else if (!std::strcmp(argv[i], "--save") && i + 1 < argc) saves.push_back(split_eq(argv[++i]));
  }
  if (timing_only && (!loads.empty() || !saves.empty()))
    throw std::runtime_error("--load/--save require functional mode (not --timing-only)");

  Config cfg = fs::exists(cfg_path) ? Config::load(cfg_path) : Config{};
  cfg.dump();
  if (timing_only) std::printf("  mode            : timing-only (no functional compute)\n");

  Program prog = load_program(prog_path);
  std::printf("\nLoaded %zu descriptors, %zu instructions from %s\n",
              prog.descriptors.size(), prog.instrs.size(), prog_path.c_str());

  Core core(prog, cfg, /*functional=*/!timing_only);
  for (auto& [name, path] : loads) {                 // inject input tensors
    core.write_tensor(name, load_npy(path));
    std::printf("loaded %s <- %s\n", name.c_str(), path.c_str());
  }
  RunResult res = core.run();

  fs::create_directories(out_dir);
  Logger(out_dir).write(res);

  for (auto& [name, path] : saves) {                 // dump output tensors
    save_npy(path, core.tensor_numel(name), core.read_tensor(name, core.tensor_numel(name)));
    std::printf("saved  %s -> %s\n", name.c_str(), path.c_str());
  }

  if (!dump_name.empty() && timing_only) {
    std::printf("\n(--dump ignored in --timing-only mode: no functional data)\n");
  } else if (!dump_name.empty()) {
    auto vals = core.read_tensor(dump_name, 8);
    std::printf("\ndump %s[0..%zu]:", dump_name.c_str(), vals.size());
    for (float v : vals) std::printf(" %g", v);
    std::printf("\n");
  }

  PerfReport rep = analyze(res, cfg);
  print_report(rep);
  write_report_json(rep, out_dir + "/perf.json");
  std::printf("\nlogs written to %s/ (isa_trace.jsonl, ddr_trace.csv, perf.json)\n",
              out_dir.c_str());
  return 0;
}

static int cmd_asm(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: neuronpu asm <in.npuasm> <out.npubin>\n"); return 2; }
  Program prog = assemble_file(argv[0]);
  write_binary(prog, argv[1]);
  std::printf("assembled %s -> %s (%zu descriptors, %zu instrs)\n",
              argv[0], argv[1], prog.descriptors.size(), prog.instrs.size());
  return 0;
}

static int cmd_meta(int argc, char** argv) {
  if (argc < 1) { std::fprintf(stderr, "usage: neuronpu meta <prog.npubin>\n"); return 2; }
  Program prog = read_binary(argv[0]);
  for (const auto& kv : prog.meta)
    std::printf("%s\t%s\n", kv.first.c_str(), kv.second.c_str());
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
        "NeuroNPU simulator\n"
        "  neuronpu run  <prog.npuasm|.npubin> [--config cfg.yaml] [--out dir]\n"
        "  neuronpu asm  <prog.npuasm> <out.npubin>\n"
        "  neuronpu meta <prog.npubin>           # print provenance header\n");
    return 2;
  }
  try {
    std::string sub = argv[1];
    if (sub == "run") return cmd_run(argc - 2, argv + 2);
    if (sub == "asm") return cmd_asm(argc - 2, argv + 2);
    if (sub == "meta") return cmd_meta(argc - 2, argv + 2);
    std::fprintf(stderr, "unknown subcommand: %s\n", sub.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
