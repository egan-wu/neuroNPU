#include "neuronpu/logger.h"

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
}

}  // namespace neuronpu
