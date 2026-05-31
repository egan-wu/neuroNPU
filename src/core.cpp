#include "neuronpu/core.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace neuronpu {

Core::Core(const Program& prog, const Config& cfg)
    : prog_(prog), cfg_(cfg),
      mem_(cfg.ddr_size_mb * 1024ull * 1024ull, cfg.sram_size_kb * 1024ull) {
  // Preload initial data into each descriptor's memory space.
  for (const auto& kv : prog_.init_data) {
    const Descriptor& d = prog_.descriptors[kv.first];
    mem_.space(d.space).write(d.base_addr, kv.second.data(), kv.second.size());
  }
}

double Core::dma_cycles(uint64_t bytes) const {
  double ns = double(bytes) / cfg_.ddr_peak_bytes_per_ns() + cfg_.ddr_latency_ns;
  return ns * cfg_.core_clock_ghz;  // cycles = ns * (cycles/ns)
}

double Core::matmul_cycles(int64_t M, int64_t N, int64_t K) const {
  double macs = double(M) * double(N) * double(K);
  double compute = macs / cfg_.te_peak_macs_per_cycle();
  double fill = double(cfg_.mac_rows + cfg_.mac_cols);  // systolic fill/drain
  return compute + fill;
}

double Core::vector_cycles(int64_t elems, double passes) const {
  return double(elems) * passes / std::max(1, cfg_.vector_lanes) + 8.0;  // + fixed overhead
}

// ---- element access helpers ----
static float read_at(Memory& mem, const Descriptor& d, const std::vector<int64_t>& idx) {
  return load_elem(mem.space(d.space).at(d.offset_bytes(idx)), d.dtype);
}
static void write_at(Memory& mem, const Descriptor& d, const std::vector<int64_t>& idx, float v) {
  store_elem(mem.space(d.space).at(d.offset_bytes(idx)), d.dtype, v);
}

// Contiguous (row-major) flat access, used by vector-engine ops.
static float read_flat(Memory& mem, const Descriptor& d, int64_t i) {
  return load_elem(mem.space(d.space).at(d.base_addr + uint64_t(i) * dtype_size(d.dtype)), d.dtype);
}
static void write_flat(Memory& mem, const Descriptor& d, int64_t i, float v) {
  store_elem(mem.space(d.space).at(d.base_addr + uint64_t(i) * dtype_size(d.dtype)), d.dtype, v);
}

void Core::exec(const Instr& in, InstrRecord& rec, RunResult& out) {
  switch (in.op) {
    case Opcode::NOP:
    case Opcode::HALT:
      return;

    case Opcode::DMA_LOAD:
    case Opcode::DMA_STORE: {
      const Descriptor& dst = prog_.descriptors[in.args.at(0)];
      const Descriptor& src = prog_.descriptors[in.args.at(1)];
      uint64_t n = uint64_t(src.bytes());
      std::vector<uint8_t> buf(n);
      mem_.space(src.space).read(src.base_addr, buf.data(), n);
      mem_.space(dst.space).write(dst.base_addr, buf.data(), n);
      rec.bytes = n;
      rec.cycles = dma_cycles(n);
      out.ddr_bytes += n;
      DdrRecord dr;
      dr.is_load = (in.op == Opcode::DMA_LOAD);
      dr.descriptor = dr.is_load ? src.name : dst.name;
      dr.addr = dr.is_load ? src.base_addr : dst.base_addr;
      dr.bytes = n;
      dr.cycles = rec.cycles;
      dr.start = rec.start;
      dr.end = rec.start + rec.cycles;
      out.ddr.push_back(dr);
      return;
    }

    case Opcode::MATMUL: {
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& A = prog_.descriptors[in.args.at(1)];
      const Descriptor& B = prog_.descriptors[in.args.at(2)];
      int64_t M = O.dims.at(0), N = O.dims.at(1), K = A.dims.at(1);
      if (A.dims.at(0) != M || B.dims.at(0) != K || B.dims.at(1) != N)
        throw std::runtime_error("MATMUL shape mismatch for output " + O.name);
      for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
          float acc = in.accumulate ? read_at(mem_, O, {m, n}) : 0.f;
          for (int64_t k = 0; k < K; ++k)
            acc += read_at(mem_, A, {m, k}) * read_at(mem_, B, {k, n});
          write_at(mem_, O, {m, n}, acc);
        }
      rec.macs = double(M) * double(N) * double(K);
      rec.cycles = matmul_cycles(M, N, K);
      out.total_macs += rec.macs;
      return;
    }

    case Opcode::VADD: {
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& A = prog_.descriptors[in.args.at(1)];
      const Descriptor& B = prog_.descriptors[in.args.at(2)];
      int64_t n = O.numel();
      for (int64_t i = 0; i < n; ++i)
        write_flat(mem_, O, i, read_flat(mem_, A, i) + read_flat(mem_, B, i));
      rec.cycles = vector_cycles(n);
      return;
    }

    case Opcode::RELU: {
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      int64_t n = O.numel();
      for (int64_t i = 0; i < n; ++i) {
        float v = read_flat(mem_, I, i);
        write_flat(mem_, O, i, v > 0.f ? v : 0.f);
      }
      rec.cycles = vector_cycles(n);
      return;
    }

    case Opcode::GELU: {
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      int64_t n = O.numel();
      const float k = 0.7978845608f;  // sqrt(2/pi)
      for (int64_t i = 0; i < n; ++i) {
        float x = read_flat(mem_, I, i);
        float t = std::tanh(k * (x + 0.044715f * x * x * x));
        write_flat(mem_, O, i, 0.5f * x * (1.f + t));
      }
      rec.cycles = vector_cycles(n, 4.0);
      return;
    }

    case Opcode::SILU: {
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      int64_t n = O.numel();
      for (int64_t i = 0; i < n; ++i) {
        float x = read_flat(mem_, I, i);
        write_flat(mem_, O, i, x / (1.f + std::exp(-x)));
      }
      rec.cycles = vector_cycles(n, 4.0);
      return;
    }

    case Opcode::SOFTMAX: {  // along last dim
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      int64_t C = I.dims.back(), R = I.numel() / C;
      for (int64_t r = 0; r < R; ++r) {
        float mx = -std::numeric_limits<float>::infinity();
        for (int64_t c = 0; c < C; ++c) mx = std::max(mx, read_flat(mem_, I, r * C + c));
        float sum = 0.f;
        for (int64_t c = 0; c < C; ++c) {
          float e = std::exp(read_flat(mem_, I, r * C + c) - mx);
          write_flat(mem_, O, r * C + c, e);
          sum += e;
        }
        for (int64_t c = 0; c < C; ++c)
          write_flat(mem_, O, r * C + c, read_flat(mem_, O, r * C + c) / sum);
      }
      rec.cycles = vector_cycles(I.numel(), 5.0);
      return;
    }

    case Opcode::RMSNORM: {  // out = in / rms(row) * weight ; imm0 = eps
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      const Descriptor& W = prog_.descriptors[in.args.at(2)];
      float eps = float(in.imm(0, 1e-6));
      int64_t C = I.dims.back(), R = I.numel() / C;
      for (int64_t r = 0; r < R; ++r) {
        float ss = 0.f;
        for (int64_t c = 0; c < C; ++c) { float x = read_flat(mem_, I, r * C + c); ss += x * x; }
        float inv = 1.f / std::sqrt(ss / float(C) + eps);
        for (int64_t c = 0; c < C; ++c)
          write_flat(mem_, O, r * C + c, read_flat(mem_, I, r * C + c) * inv * read_flat(mem_, W, c));
      }
      rec.cycles = vector_cycles(I.numel(), 3.0);
      return;
    }

    case Opcode::LAYERNORM: {  // out = (in-mean)/std * weight + bias ; imm0 = eps
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      const Descriptor& W = prog_.descriptors[in.args.at(2)];
      const Descriptor& Bd = prog_.descriptors[in.args.at(3)];
      float eps = float(in.imm(0, 1e-5));
      int64_t C = I.dims.back(), R = I.numel() / C;
      for (int64_t r = 0; r < R; ++r) {
        float mean = 0.f;
        for (int64_t c = 0; c < C; ++c) mean += read_flat(mem_, I, r * C + c);
        mean /= float(C);
        float var = 0.f;
        for (int64_t c = 0; c < C; ++c) {
          float d = read_flat(mem_, I, r * C + c) - mean; var += d * d;
        }
        float inv = 1.f / std::sqrt(var / float(C) + eps);
        for (int64_t c = 0; c < C; ++c) {
          float norm = (read_flat(mem_, I, r * C + c) - mean) * inv;
          write_flat(mem_, O, r * C + c, norm * read_flat(mem_, W, c) + read_flat(mem_, Bd, c));
        }
      }
      rec.cycles = vector_cycles(I.numel(), 4.0);
      return;
    }

    case Opcode::CONV: {  // out[Co,Ho,Wo] = in[Ci,H,W] * w[Co,Ci,Kh,Kw] ; imm0=stride imm1=pad
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      const Descriptor& W = prog_.descriptors[in.args.at(2)];
      int stride = int(in.imm(0, 1.0));
      int pad = int(in.imm(1, 0.0));
      int64_t Ci = I.dims.at(0), H = I.dims.at(1), Wd = I.dims.at(2);
      int64_t Co = W.dims.at(0), Kh = W.dims.at(2), Kw = W.dims.at(3);
      int64_t Ho = O.dims.at(1), Wo = O.dims.at(2);
      if (W.dims.at(1) != Ci || O.dims.at(0) != Co)
        throw std::runtime_error("CONV channel mismatch for output " + O.name);
      if (Ho != (H + 2 * pad - Kh) / stride + 1 || Wo != (Wd + 2 * pad - Kw) / stride + 1)
        throw std::runtime_error("CONV output shape mismatch for " + O.name);
      for (int64_t co = 0; co < Co; ++co)
        for (int64_t oh = 0; oh < Ho; ++oh)
          for (int64_t ow = 0; ow < Wo; ++ow) {
            float acc = 0.f;
            for (int64_t ci = 0; ci < Ci; ++ci)
              for (int64_t kh = 0; kh < Kh; ++kh)
                for (int64_t kw = 0; kw < Kw; ++kw) {
                  int64_t ih = oh * stride - pad + kh, iw = ow * stride - pad + kw;
                  if (ih < 0 || ih >= H || iw < 0 || iw >= Wd) continue;  // zero pad
                  float x = read_flat(mem_, I, (ci * H + ih) * Wd + iw);
                  float wv = read_flat(mem_, W, ((co * Ci + ci) * Kh + kh) * Kw + kw);
                  acc += x * wv;
                }
            write_flat(mem_, O, (co * Ho + oh) * Wo + ow, acc);
          }
      rec.macs = double(Co) * Ho * Wo * Ci * Kh * Kw;
      rec.cycles = rec.macs / cfg_.te_peak_macs_per_cycle() + double(cfg_.mac_rows + cfg_.mac_cols);
      out.total_macs += rec.macs;
      return;
    }

    case Opcode::REQUANT: {  // out = clamp(round(in/scale) + zp) ; imm0=scale, imm1=zp
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      float scale = float(in.imm(0, 1.0));
      float zp = float(in.imm(1, 0.0));
      int64_t n = O.numel();
      for (int64_t i = 0; i < n; ++i)
        write_flat(mem_, O, i, read_flat(mem_, I, i) / scale + zp);  // I8 store rounds+clamps
      rec.cycles = vector_cycles(n);
      return;
    }
  }
}

std::vector<float> Core::read_tensor(const std::string& name, int64_t max_elems) const {
  int id = prog_.descriptor_id(name);
  if (id < 0) throw std::runtime_error("read_tensor: unknown descriptor " + name);
  const Descriptor& d = prog_.descriptors[id];
  int64_t n = std::min(d.numel(), max_elems);
  size_t es = dtype_size(d.dtype);
  std::vector<float> v;
  v.reserve(n);
  for (int64_t i = 0; i < n; ++i)
    v.push_back(load_elem(mem_.space(d.space).at(d.base_addr + uint64_t(i) * es), d.dtype));
  return v;
}

RunResult Core::run() {
  RunResult out;
  out.instrs.reserve(prog_.instrs.size());

  // Partition instructions into per-engine in-order queues.
  std::vector<int> q[3];
  for (int i = 0; i < int(prog_.instrs.size()); ++i)
    q[int(prog_.instrs[i].engine())].push_back(i);

  size_t head[3] = {0, 0, 0};
  double engfree[3] = {0, 0, 0};
  std::unordered_map<int, double> ev;  // event id -> time it was signaled
  int remaining = int(prog_.instrs.size());

  while (remaining > 0) {
    int best = -1;
    double best_start = std::numeric_limits<double>::max();
    for (int e = 0; e < 3; ++e) {
      if (head[e] >= q[e].size()) continue;
      const Instr& in = prog_.instrs[q[e][head[e]]];
      double wt = 0;
      if (in.wait_event >= 0) {
        auto it = ev.find(in.wait_event);
        if (it == ev.end()) continue;  // not ready
        wt = it->second;
      }
      double start = std::max(engfree[e], wt);
      if (start < best_start) { best_start = start; best = e; }
    }
    if (best < 0)
      throw std::runtime_error("scheduler deadlock: instruction waiting on an "
                               "event that is never signaled");

    int ii = q[best][head[best]++];
    const Instr& in = prog_.instrs[ii];
    InstrRecord rec;
    rec.idx = ii;
    rec.op = in.op;
    rec.engine = in.engine();
    rec.wait_event = in.wait_event;
    rec.signal_event = in.signal_event;
    rec.start = best_start;
    exec(in, rec, out);           // sets rec.cycles / macs / bytes
    rec.end = rec.start + rec.cycles;

    engfree[best] = rec.end;
    out.engine_busy[best] += rec.cycles;
    if (in.signal_event >= 0) ev[in.signal_event] = rec.end;
    out.instrs.push_back(rec);
    --remaining;
  }

  out.total_cycles = std::max({engfree[0], engfree[1], engfree[2]});
  out.clock_ghz = cfg_.core_clock_ghz;
  return out;
}

}  // namespace neuronpu
