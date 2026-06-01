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

namespace fs = std::filesystem;
using namespace neuronpu;

static bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
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
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
    else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump_name = argv[++i];
    else if (!std::strcmp(argv[i], "--timing-only")) timing_only = true;
  }

  Config cfg = fs::exists(cfg_path) ? Config::load(cfg_path) : Config{};
  cfg.dump();
  if (timing_only) std::printf("  mode            : timing-only (no functional compute)\n");

  Program prog = load_program(prog_path);
  std::printf("\nLoaded %zu descriptors, %zu instructions from %s\n",
              prog.descriptors.size(), prog.instrs.size(), prog_path.c_str());

  Core core(prog, cfg, /*functional=*/!timing_only);
  RunResult res = core.run();

  fs::create_directories(out_dir);
  Logger(out_dir).write(res);

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

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
        "NeuroNPU simulator\n"
        "  neuronpu run <prog.npuasm|.npubin> [--config cfg.yaml] [--out dir]\n"
        "  neuronpu asm <prog.npuasm> <out.npubin>\n");
    return 2;
  }
  try {
    std::string sub = argv[1];
    if (sub == "run") return cmd_run(argc - 2, argv + 2);
    if (sub == "asm") return cmd_asm(argc - 2, argv + 2);
    std::fprintf(stderr, "unknown subcommand: %s\n", sub.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
