#include "neuronpu/logger.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace neuronpu {

void Logger::write(const RunResult& r) const {
  std::ofstream isa(dir_ + "/isa_trace.jsonl");
  if (!isa) throw std::runtime_error("cannot write log dir: " + dir_);
  for (const auto& e : r.instrs) {
    isa << "{\"idx\":" << e.idx
        << ",\"op\":\"" << opcode_name(e.op) << "\""
        << ",\"engine\":\"" << engine_name(e.engine) << "\""
        << ",\"start\":" << e.start
        << ",\"end\":" << e.end
        << ",\"cycles\":" << e.cycles
        << ",\"wait\":" << e.wait_event
        << ",\"sig\":" << e.signal_event
        << ",\"macs\":" << e.macs
        << ",\"bytes\":" << e.bytes
        << "}\n";
  }

  std::ofstream ddr(dir_ + "/ddr_trace.csv");
  ddr << "start,end,dir,descriptor,addr,bytes,cycles\n";
  for (const auto& d : r.ddr) {
    ddr << d.start << ',' << d.end << ',' << (d.is_load ? "load" : "store") << ','
        << d.descriptor << ',' << d.addr << ',' << d.bytes << ',' << d.cycles << '\n';
  }

  write_perfetto(r);
}

// Chrome Trace Event Format (loads directly in https://ui.perfetto.dev).
// ts/dur are microseconds; one track (tid) per engine + a cumulative DDR counter.
void Logger::write_perfetto(const RunResult& r) const {
  const double cyc_to_us = 1.0 / (r.clock_ghz * 1000.0);  // cycles -> microseconds
  std::ofstream o(dir_ + "/trace.json");
  if (!o) throw std::runtime_error("cannot write trace.json in " + dir_);

  o << "[\n";
  bool first = true;
  auto comma = [&] { if (!first) o << ",\n"; first = false; };

  // Process + thread (engine) names.
  comma();
  o << R"({"name":"process_name","ph":"M","pid":1,"args":{"name":"NeuroNPU"}})";
  const char* tnames[3] = {"DMA", "TENSOR", "VECTOR"};
  for (int e = 0; e < 3; ++e) {
    comma();
    o << R"({"name":"thread_name","ph":"M","pid":1,"tid":)" << e
      << R"(,"args":{"name":")" << tnames[e] << R"("}})";
  }

  // One complete (X) slice per instruction, on its engine's track.
  for (const auto& e : r.instrs) {
    comma();
    o << R"({"name":")" << opcode_name(e.op) << R"(","cat":")" << engine_name(e.engine)
      << R"(","ph":"X","pid":1,"tid":)" << int(e.engine)
      << R"(,"ts":)" << e.start * cyc_to_us
      << R"(,"dur":)" << e.cycles * cyc_to_us
      << R"(,"args":{"idx":)" << e.idx
      << R"(,"cycles":)" << e.cycles
      << R"(,"macs":)" << e.macs
      << R"(,"bytes":)" << e.bytes
      << R"(,"wait":)" << e.wait_event
      << R"(,"sig":)" << e.signal_event << "}}";
  }

  // Cumulative DDR bytes as a counter track (sampled at each burst end).
  std::vector<DdrRecord> sorted = r.ddr;
  std::sort(sorted.begin(), sorted.end(),
            [](const DdrRecord& a, const DdrRecord& b) { return a.end < b.end; });
  double cum = 0;
  for (const auto& d : sorted) {
    cum += double(d.bytes);
    comma();
    o << R"({"name":"DDR_bytes","ph":"C","pid":1,"ts":)" << d.end * cyc_to_us
      << R"(,"args":{"cumulative_bytes":)" << cum << "}}";
  }

  o << "\n]\n";
}

}  // namespace neuronpu
