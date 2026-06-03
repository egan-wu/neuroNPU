#include "neuronpu/core.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace neuronpu {

Core::Core(const Program& prog, const Config& cfg, bool functional)
    : prog_(flatten_loops(prog)), cfg_(cfg), functional_(functional),
      mem_(functional ? cfg.ddr_size_mb * 1024ull * 1024ull : 0,
           functional ? cfg.sram_size_kb * 1024ull : 0,
           std::max(1, cfg.num_cores)) {
  if (!functional_) return;  // timing-only: no real memory, no preload
  // Preload initial data: DDR is shared; SRAM init is replicated to every core.
  for (const auto& kv : prog_.init_data) {
    const Descriptor& d = prog_.descriptors[kv.first];
    if (d.space == MemSpace::DDR) {
      mem_.space(MemSpace::DDR, 0).write(d.base_addr, kv.second.data(), kv.second.size());
    } else {
      for (int c = 0; c < mem_.num_cores(); ++c)
        mem_.space(MemSpace::SRAM, c).write(d.base_addr, kv.second.data(), kv.second.size());
    }
  }
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

// ---- element access helpers (SRAM is per-core; `core` selects the bank) ----
static float read_at(Memory& mem, const Descriptor& d, int core, const std::vector<int64_t>& idx) {
  return load_elem(mem.space(d.space, core).at(d.offset_bytes(idx)), d.dtype);
}
static void write_at(Memory& mem, const Descriptor& d, int core,
                     const std::vector<int64_t>& idx, float v) {
  store_elem(mem.space(d.space, core).at(d.offset_bytes(idx)), d.dtype, v);
}

// Contiguous (row-major) flat access, used by vector-engine ops.
static float read_flat(Memory& mem, const Descriptor& d, int core, int64_t i) {
  return load_elem(mem.space(d.space, core).at(d.base_addr + uint64_t(i) * dtype_size(d.dtype)),
                   d.dtype);
}
static void write_flat(Memory& mem, const Descriptor& d, int core, int64_t i, float v) {
  store_elem(mem.space(d.space, core).at(d.base_addr + uint64_t(i) * dtype_size(d.dtype)),
             d.dtype, v);
}

void Core::dma_strided(const Descriptor& src, const Descriptor& dst, int core) {
  size_t es = dtype_size(src.dtype);
  int64_t n = src.numel();
  std::vector<int64_t> idx(src.dims.size(), 0);
  MemSpaceStore& ss = mem_.space(src.space, core);
  MemSpaceStore& ds = mem_.space(dst.space, core);
  for (int64_t c = 0; c < n; ++c) {
    std::memcpy(ds.at(dst.offset_bytes(idx)), ss.at(src.offset_bytes(idx)), es);
    for (int d = int(src.dims.size()) - 1; d >= 0; --d) {   // odometer over dims
      if (++idx[d] < src.dims[d]) break;
      idx[d] = 0;
    }
  }
}

void Core::exec(const Instr& in, InstrRecord& rec, RunResult& out) {
  switch (in.op) {
    case Opcode::NOP:
    case Opcode::HALT:
    case Opcode::LOOP:      // removed by flatten_loops(); no-op if it ever reaches here
    case Opcode::ENDLOOP:
      return;

    case Opcode::DMA_LOAD:
    case Opcode::DMA_STORE: {
      // Functional copy only; timing (+ DdrRecord) is finalised by the scheduler,
      // which models DDR-bandwidth contention across concurrent channels.
      const Descriptor& dst = prog_.descriptors[in.args.at(0)];
      const Descriptor& src = prog_.descriptors[in.args.at(1)];
      uint64_t n = uint64_t(src.bytes());
      if (functional_) {
        if (src.strides.empty() && dst.strides.empty()) {   // fast contiguous path
          std::vector<uint8_t> buf(n);
          mem_.space(src.space, in.core).read(src.base_addr, buf.data(), n);
          mem_.space(dst.space, in.core).write(dst.base_addr, buf.data(), n);
        } else {
          dma_strided(src, dst, in.core);                    // gather/scatter strided tile
        }
      }
      rec.bytes = n;
      out.ddr_bytes += n;
      return;
    }

    case Opcode::MATMUL: {
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& A = prog_.descriptors[in.args.at(1)];
      const Descriptor& B = prog_.descriptors[in.args.at(2)];
      int64_t M = O.dims.at(0), N = O.dims.at(1), K = A.dims.at(1);
      if (A.dims.at(0) != M || B.dims.at(0) != K || B.dims.at(1) != N)
        throw std::runtime_error("MATMUL shape mismatch for output " + O.name);
      if (functional_)
        for (int64_t m = 0; m < M; ++m)
          for (int64_t n = 0; n < N; ++n) {
            float acc = in.accumulate ? read_at(mem_, O, in.core, {m, n}) : 0.f;
            for (int64_t k = 0; k < K; ++k)
              acc += read_at(mem_, A, in.core, {m, k}) * read_at(mem_, B, in.core, {k, n});
            write_at(mem_, O, in.core, {m, n}, acc);
          }
      rec.macs = double(M) * double(N) * double(K);
      rec.cycles = matmul_cycles(M, N, K);
      out.total_macs += rec.macs;
      return;
    }

    case Opcode::VADD:
    case Opcode::VSUB:
    case Opcode::VMUL:
    case Opcode::VMAX: {  // binary elementwise
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& A = prog_.descriptors[in.args.at(1)];
      const Descriptor& B = prog_.descriptors[in.args.at(2)];
      int64_t n = O.numel();
      if (functional_)
        for (int64_t i = 0; i < n; ++i) {
          float a = read_flat(mem_, A, in.core, i), b = read_flat(mem_, B, in.core, i);
          float r = in.op == Opcode::VADD ? a + b
                  : in.op == Opcode::VSUB ? a - b
                  : in.op == Opcode::VMUL ? a * b
                                          : std::max(a, b);
          write_flat(mem_, O, in.core, i, r);
        }
      rec.cycles = vector_cycles(n);
      return;
    }

    case Opcode::SIGMOID: {
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      int64_t n = O.numel();
      if (functional_)
        for (int64_t i = 0; i < n; ++i) {
          float x = read_flat(mem_, I, in.core, i);
          write_flat(mem_, O, in.core, i, 1.f / (1.f + std::exp(-x)));
        }
      rec.cycles = vector_cycles(n, 4.0);
      return;
    }

    case Opcode::GATHER: {  // out[n,:] = table[imm[n],:] ; embedding lookup
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& T = prog_.descriptors[in.args.at(1)];
      int64_t D = T.dims.back();
      int64_t N = int64_t(in.imms.size());
      if (functional_)
        for (int64_t n = 0; n < N; ++n) {
          int64_t id = int64_t(in.imms[n]);
          for (int64_t d = 0; d < D; ++d)
            write_flat(mem_, O, in.core, n * D + d, read_flat(mem_, T, in.core, id * D + d));
        }
      uint64_t moved = uint64_t(N * D) * dtype_size(T.dtype);
      if (T.space == MemSpace::DDR) out.ddr_bytes += moved;  // table read from DDR
      out.sram_bytes += moved;
      rec.bytes = moved;
      rec.cycles = vector_cycles(N * D);
      return;
    }

    case Opcode::RELU: {
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      int64_t n = O.numel();
      if (functional_)
        for (int64_t i = 0; i < n; ++i) {
          float v = read_flat(mem_, I, in.core, i);
          write_flat(mem_, O, in.core, i, v > 0.f ? v : 0.f);
        }
      rec.cycles = vector_cycles(n);
      return;
    }

    case Opcode::GELU: {
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      int64_t n = O.numel();
      const float k = 0.7978845608f;  // sqrt(2/pi)
      if (functional_)
        for (int64_t i = 0; i < n; ++i) {
          float x = read_flat(mem_, I, in.core, i);
          float t = std::tanh(k * (x + 0.044715f * x * x * x));
          write_flat(mem_, O, in.core, i, 0.5f * x * (1.f + t));
        }
      rec.cycles = vector_cycles(n, 4.0);
      return;
    }

    case Opcode::SILU: {
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      int64_t n = O.numel();
      if (functional_)
        for (int64_t i = 0; i < n; ++i) {
          float x = read_flat(mem_, I, in.core, i);
          write_flat(mem_, O, in.core, i, x / (1.f + std::exp(-x)));
        }
      rec.cycles = vector_cycles(n, 4.0);
      return;
    }

    case Opcode::SOFTMAX: {  // along last dim; optional causal mask
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      int64_t C = I.dims.back(), R = I.numel() / C;
      bool causal = in.imm(0, 0.0) != 0.0;
      // query row r attends keys 0..(q_offset + r); default aligns the last R
      // queries to the end of the K range (prefill: q_offset=0; decode: C-1).
      int64_t q_offset = int64_t(in.imm(1, double(C - R)));
      if (functional_)
       for (int64_t r = 0; r < R; ++r) {
        int64_t limit = causal ? std::min(C - 1, q_offset + r) : C - 1;
        float mx = -std::numeric_limits<float>::infinity();
        for (int64_t c = 0; c <= limit; ++c) mx = std::max(mx, read_flat(mem_, I, in.core, r * C + c));
        float sum = 0.f;
        for (int64_t c = 0; c < C; ++c) {
          float e = (c <= limit) ? std::exp(read_flat(mem_, I, in.core, r * C + c) - mx) : 0.f;
          write_flat(mem_, O, in.core, r * C + c, e);
          sum += e;
        }
        for (int64_t c = 0; c < C; ++c)
          write_flat(mem_, O, in.core, r * C + c, read_flat(mem_, O, in.core, r * C + c) / sum);
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
      if (functional_)
       for (int64_t r = 0; r < R; ++r) {
        float ss = 0.f;
        for (int64_t c = 0; c < C; ++c) { float x = read_flat(mem_, I, in.core, r * C + c); ss += x * x; }
        float inv = 1.f / std::sqrt(ss / float(C) + eps);
        for (int64_t c = 0; c < C; ++c)
          write_flat(mem_, O, in.core, r * C + c, read_flat(mem_, I, in.core, r * C + c) * inv * read_flat(mem_, W, in.core, c));
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
      if (functional_)
       for (int64_t r = 0; r < R; ++r) {
        float mean = 0.f;
        for (int64_t c = 0; c < C; ++c) mean += read_flat(mem_, I, in.core, r * C + c);
        mean /= float(C);
        float var = 0.f;
        for (int64_t c = 0; c < C; ++c) {
          float d = read_flat(mem_, I, in.core, r * C + c) - mean; var += d * d;
        }
        float inv = 1.f / std::sqrt(var / float(C) + eps);
        for (int64_t c = 0; c < C; ++c) {
          float norm = (read_flat(mem_, I, in.core, r * C + c) - mean) * inv;
          write_flat(mem_, O, in.core, r * C + c, norm * read_flat(mem_, W, in.core, c) + read_flat(mem_, Bd, in.core, c));
        }
      }
      rec.cycles = vector_cycles(I.numel(), 4.0);
      return;
    }

    case Opcode::MAXPOOL: {  // [C,H,W] -> [C,Ho,Wo] ; imm0=kernel imm1=stride imm2=pad
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      int Kp = int(in.imm(0, 2)), stride = int(in.imm(1, Kp)), pad = int(in.imm(2, 0));
      int64_t C = I.dims.at(0), H = I.dims.at(1), W = I.dims.at(2);
      int64_t Ho = O.dims.at(1), Wo = O.dims.at(2);
      if (functional_)
        for (int64_t c = 0; c < C; ++c)
          for (int64_t oh = 0; oh < Ho; ++oh)
            for (int64_t ow = 0; ow < Wo; ++ow) {
              float m = -std::numeric_limits<float>::infinity();
              for (int kh = 0; kh < Kp; ++kh)
                for (int kw = 0; kw < Kp; ++kw) {
                  int64_t ih = oh * stride - pad + kh, iw = ow * stride - pad + kw;
                  if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    m = std::max(m, read_flat(mem_, I, in.core, (c * H + ih) * W + iw));
                }
              write_flat(mem_, O, in.core, (c * Ho + oh) * Wo + ow, m);
            }
      rec.cycles = vector_cycles(O.numel(), double(Kp * Kp));
      return;
    }

    case Opcode::UPSAMPLE: {  // nearest [C,H,W] -> [C,H*f,W*f] ; imm0=factor
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      int f = int(in.imm(0, 2));
      int64_t C = I.dims.at(0), H = I.dims.at(1), W = I.dims.at(2);
      int64_t Ho = H * f, Wo = W * f;
      if (functional_)
        for (int64_t c = 0; c < C; ++c)
          for (int64_t oh = 0; oh < Ho; ++oh)
            for (int64_t ow = 0; ow < Wo; ++ow)
              write_flat(mem_, O, in.core, (c * Ho + oh) * Wo + ow,
                         read_flat(mem_, I, in.core, (c * H + oh / f) * W + ow / f));
      rec.cycles = vector_cycles(O.numel());
      return;
    }

    case Opcode::CONCAT: {  // concat a,b along channel axis 0
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& A = prog_.descriptors[in.args.at(1)];
      const Descriptor& B = prog_.descriptors[in.args.at(2)];
      int64_t na = A.numel(), nb = B.numel();
      if (functional_) {
        for (int64_t i = 0; i < na; ++i) write_flat(mem_, O, in.core, i, read_flat(mem_, A, in.core, i));
        for (int64_t i = 0; i < nb; ++i) write_flat(mem_, O, in.core, na + i, read_flat(mem_, B, in.core, i));
      }
      rec.cycles = vector_cycles(na + nb);
      return;
    }

    case Opcode::REPEAT_KV: {  // [R,kv_dim] -> [R,kv_dim*g] ; imm0=group, imm1=head_dim
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      int g = int(in.imm(0, 1)), hd = int(in.imm(1, 1));
      int64_t kv_dim = I.dims.back(), R = I.numel() / kv_dim, kvh = kv_dim / hd;
      if (functional_)
        for (int64_t r = 0; r < R; ++r)
          for (int64_t kh = 0; kh < kvh; ++kh)
            for (int rep = 0; rep < g; ++rep)
              for (int d = 0; d < hd; ++d)
                write_flat(mem_, O, in.core, r * (kv_dim * g) + (kh * g + rep) * hd + d,
                           read_flat(mem_, I, in.core, r * kv_dim + kh * hd + d));
      rec.cycles = vector_cycles(O.numel());
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
      if (functional_)
       for (int64_t co = 0; co < Co; ++co)
        for (int64_t oh = 0; oh < Ho; ++oh)
          for (int64_t ow = 0; ow < Wo; ++ow) {
            float acc = 0.f;
            for (int64_t ci = 0; ci < Ci; ++ci)
              for (int64_t kh = 0; kh < Kh; ++kh)
                for (int64_t kw = 0; kw < Kw; ++kw) {
                  int64_t ih = oh * stride - pad + kh, iw = ow * stride - pad + kw;
                  if (ih < 0 || ih >= H || iw < 0 || iw >= Wd) continue;  // zero pad
                  float x = read_flat(mem_, I, in.core, (ci * H + ih) * Wd + iw);
                  float wv = read_flat(mem_, W, in.core, ((co * Ci + ci) * Kh + kh) * Kw + kw);
                  acc += x * wv;
                }
            write_flat(mem_, O, in.core, (co * Ho + oh) * Wo + ow, acc);
          }
      rec.macs = double(Co) * Ho * Wo * Ci * Kh * Kw;
      rec.cycles = rec.macs / cfg_.te_peak_macs_per_cycle() + double(cfg_.mac_rows + cfg_.mac_cols);
      out.total_macs += rec.macs;
      return;
    }

    case Opcode::ROPE: {  // rotate pairs along last dim ; imm0=base, imm1=pos offset
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      double base = in.imm(0, 10000.0);
      int64_t offset = int64_t(in.imm(1, 0.0));
      int64_t D = I.dims.back(), S = I.numel() / D;
      if (functional_)
       for (int64_t s = 0; s < S; ++s) {
        int64_t pos = s + offset;
        for (int64_t i = 0; i < D / 2; ++i) {
          double theta = double(pos) * std::pow(base, -double(2 * i) / double(D));
          float c = float(std::cos(theta)), sn = float(std::sin(theta));
          float x0 = read_flat(mem_, I, in.core, s * D + 2 * i);
          float x1 = read_flat(mem_, I, in.core, s * D + 2 * i + 1);
          write_flat(mem_, O, in.core, s * D + 2 * i, x0 * c - x1 * sn);
          write_flat(mem_, O, in.core, s * D + 2 * i + 1, x0 * sn + x1 * c);
        }
      }
      rec.cycles = vector_cycles(I.numel(), 6.0);
      return;
    }

    case Opcode::REQUANT: {  // out = clamp(round(in/scale) + zp) ; imm0=scale, imm1=zp
      const Descriptor& O = prog_.descriptors[in.args.at(0)];
      const Descriptor& I = prog_.descriptors[in.args.at(1)];
      float scale = float(in.imm(0, 1.0));
      float zp = float(in.imm(1, 0.0));
      int64_t n = O.numel();
      if (functional_)
        for (int64_t i = 0; i < n; ++i)
          write_flat(mem_, O, in.core, i, read_flat(mem_, I, in.core, i) / scale + zp);  // I8 rounds+clamps
      rec.cycles = vector_cycles(n);
      return;
    }
  }
}

int64_t Core::tensor_numel(const std::string& name) const {
  int id = prog_.descriptor_id(name);
  if (id < 0) throw std::runtime_error("tensor_numel: unknown descriptor " + name);
  return prog_.descriptors[id].numel();
}

void Core::write_tensor(const std::string& name, const std::vector<float>& data, int core) {
  if (!functional_) throw std::runtime_error("write_tensor needs functional mode");
  int id = prog_.descriptor_id(name);
  if (id < 0) throw std::runtime_error("write_tensor: unknown descriptor " + name);
  const Descriptor& d = prog_.descriptors[id];
  int64_t n = std::min<int64_t>(d.numel(), int64_t(data.size()));
  size_t es = dtype_size(d.dtype);
  for (int64_t i = 0; i < n; ++i)
    store_elem(mem_.space(d.space, core).at(d.base_addr + uint64_t(i) * es), d.dtype, data[i]);
}

std::vector<float> Core::read_tensor(const std::string& name, int64_t max_elems, int core) const {
  int id = prog_.descriptor_id(name);
  if (id < 0) throw std::runtime_error("read_tensor: unknown descriptor " + name);
  const Descriptor& d = prog_.descriptors[id];
  int64_t n = std::min(d.numel(), max_elems);
  size_t es = dtype_size(d.dtype);
  std::vector<float> v;
  v.reserve(n);
  for (int64_t i = 0; i < n; ++i)
    v.push_back(load_elem(mem_.space(d.space, core).at(d.base_addr + uint64_t(i) * es), d.dtype));
  return v;
}

// An instruction currently executing on an engine.
namespace {
struct Running {
  int         instr_idx;
  Engine      engine;
  int         core = 0;
  double      start;
  // compute ops: remaining cycles, rate 1. DMA ops: remaining bytes at a
  // bandwidth-share rate, after an initial latency phase ending at data_start.
  double      cycles_remaining = 0;
  double      bytes_remaining = 0;
  double      data_start = 0;    // DMA: time the latency phase ends
  bool        is_dma = false;
  double      sram_demand = 0;   // compute ops: SRAM bytes/cycle at full rate
  InstrRecord rec;
};
}  // namespace

RunResult Core::run() {
  RunResult out;
  out.instrs.reserve(prog_.instrs.size());
  out.clock_ghz = cfg_.core_clock_ghz;

  // Per-(core, engine) in-order queues.
  const int ncores = std::max(1, cfg_.num_cores);
  std::vector<std::array<std::vector<int>, 3>> q(ncores);
  std::vector<std::array<size_t, 3>> head(ncores);
  for (auto& h : head) h = {0, 0, 0};
  for (int i = 0; i < int(prog_.instrs.size()); ++i) {
    const Instr& in = prog_.instrs[i];
    if (in.core < 0 || in.core >= ncores)
      throw std::runtime_error("instruction core id " + std::to_string(in.core) +
                               " out of range (num_cores=" + std::to_string(ncores) + ")");
    q[in.core][int(in.engine())].push_back(i);
  }

  std::unordered_map<int, double> ev;     // event id -> signaled time
  std::vector<Running> running;
  int remaining = int(prog_.instrs.size());
  double now = 0;

  const double bytes_per_cycle = cfg_.ddr_peak_bytes_per_ns() / cfg_.core_clock_ghz;
  const double sram_bw = cfg_.sram_bw_bytes_per_cycle();
  const double latency_cycles = cfg_.ddr_latency_ns * cfg_.core_clock_ghz;
  const int    dma_channels = std::max(1, cfg_.dma_channels);
  const double EPS = 1e-9;

  auto ready = [&](const Instr& in) {
    for (int w : in.wait_events) if (ev.find(w) == ev.end()) return false;
    return true;
  };
  auto dma_in_flight = [&](int c) {
    int n = 0; for (auto& r : running) if (r.is_dma && r.core == c) ++n; return n;
  };

  // Start an instruction on (core c, engine e) at time `now`.
  auto start_instr = [&](int c, int e) {
    int ii = q[c][e][head[c][e]++];
    const Instr& in = prog_.instrs[ii];
    Running r;
    r.instr_idx = ii;
    r.engine = Engine(e);
    r.core = c;
    r.start = now;
    r.rec.idx = ii;
    r.rec.op = in.op;
    r.rec.engine = Engine(e);
    r.rec.wait_events = in.wait_events;
    r.rec.signal_event = in.signal_event;
    r.rec.start = now;
    exec(in, r.rec, out);  // functional work; sets macs/bytes and (non-DMA) cycles
    if (in.op == Opcode::DMA_LOAD || in.op == Opcode::DMA_STORE) {
      r.is_dma = true;
      r.bytes_remaining = double(r.rec.bytes);
      r.data_start = now + latency_cycles;
      out.sram_bytes += r.rec.bytes;  // the SRAM side of the transfer
    } else {
      r.cycles_remaining = r.rec.cycles;  // NOP/HALT have 0 cycles -> retire at once
      // SRAM bandwidth this op wants at full rate: operand bytes touched / cycles.
      if (r.rec.cycles > 0) {
        double touched = 0;
        for (int a : in.args) touched += double(prog_.descriptors[a].bytes());
        r.sram_demand = touched / r.rec.cycles;
        out.sram_bytes += uint64_t(touched);
      }
    }
    running.push_back(std::move(r));
  };

  while (remaining > 0) {
    // 1) Greedily start ready ops on free engines, across all cores.
    bool started = true;
    while (started) {
      started = false;
      for (int c = 0; c < ncores; ++c) {
        // DMA: in-order, up to dma_channels concurrent per core.
        if (head[c][0] < q[c][0].size() && dma_in_flight(c) < dma_channels &&
            ready(prog_.instrs[q[c][0][head[c][0]]])) {
          start_instr(c, 0); started = true;
        }
        // TENSOR / VECTOR: single unit each, per core.
        for (int e = 1; e < 3; ++e) {
          bool busy = false;
          for (auto& r : running) if (r.core == c && int(r.engine) == e) { busy = true; break; }
          if (!busy && head[c][e] < q[c][e].size() && ready(prog_.instrs[q[c][e][head[c][e]]])) {
            start_instr(c, e); started = true;
          }
        }
      }
    }

    if (running.empty())
      throw std::runtime_error("scheduler deadlock: instruction waiting on an "
                               "event that is never signaled");

    // 2) Current DDR bandwidth share among transferring DMA ops.
    int transferring = 0;
    for (auto& r : running)
      if (r.is_dma && now >= r.data_start - EPS && r.bytes_remaining > EPS) ++transferring;
    double share = bytes_per_cycle / std::max(1, transferring);

    // 2b) SRAM-port contention is PER CORE (each core has private SRAM banks).
    //     A DMA op's SRAM demand equals its DDR transfer rate; a compute op's is its
    //     constant operand-bytes/cycle. If a core's demand exceeds its SRAM bandwidth,
    //     that core's active ops are throttled by the same factor (first-order model).
    std::vector<double> sram_demand(ncores, 0.0);
    for (auto& r : running) {
      if (r.is_dma) { if (now >= r.data_start - EPS && r.bytes_remaining > EPS) sram_demand[r.core] += share; }
      else sram_demand[r.core] += r.sram_demand;
    }
    std::vector<double> sram_factor(ncores, 1.0);
    for (int c = 0; c < ncores; ++c)
      if (sram_demand[c] > sram_bw) sram_factor[c] = sram_bw / sram_demand[c];

    // 3) Time to the next event (completion or a latency phase ending).
    double dt = std::numeric_limits<double>::max();
    for (auto& r : running) {
      double sf = sram_factor[r.core];
      double t;
      if (!r.is_dma)                         t = r.cycles_remaining / sf;            // compute
      else if (now < r.data_start - EPS)     t = r.data_start - now;                 // latency
      else                                   t = r.bytes_remaining / (share * sf);   // transfer
      dt = std::min(dt, t);
    }

    // 4) Progress all running ops over the interval [now, now+dt], using the
    //    pre-advance time to classify each DMA op (latency vs transfer); event
    //    boundaries guarantee no op crosses its data_start mid-interval.
    for (auto& r : running) {
      double sf = sram_factor[r.core];
      if (!r.is_dma) r.cycles_remaining -= sf * dt;
      else if (now >= r.data_start - EPS) r.bytes_remaining -= share * sf * dt;
    }
    now += dt;

    // 5) Retire completed ops.
    for (auto it = running.begin(); it != running.end();) {
      bool done = it->is_dma ? (it->bytes_remaining <= EPS && now >= it->data_start - EPS)
                             : (it->cycles_remaining <= EPS);
      if (!done) { ++it; continue; }
      const Instr& in = prog_.instrs[it->instr_idx];
      it->rec.end = now;
      it->rec.cycles = now - it->start;
      out.engine_busy[int(it->engine)] += it->rec.cycles;
      if (it->is_dma) {
        DdrRecord dr;
        dr.is_load = (in.op == Opcode::DMA_LOAD);
        const Descriptor& d = prog_.descriptors[in.args.at(dr.is_load ? 1 : 0)];
        dr.descriptor = d.name;
        dr.addr = d.base_addr;
        dr.bytes = it->rec.bytes;
        dr.start = it->start;
        dr.end = now;
        dr.cycles = it->rec.cycles;
        out.ddr.push_back(dr);
      }
      if (in.signal_event >= 0) ev[in.signal_event] = now;
      out.instrs.push_back(it->rec);
      --remaining;
      it = running.erase(it);
    }
  }

  // Instruction records were retired out of issue order; sort by start time.
  std::sort(out.instrs.begin(), out.instrs.end(),
            [](const InstrRecord& a, const InstrRecord& b) {
              return a.start < b.start || (a.start == b.start && a.idx < b.idx);
            });
  out.total_cycles = now;
  return out;
}

}  // namespace neuronpu
