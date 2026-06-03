#include "neuronpu/isa.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace neuronpu {

const char* memspace_name(MemSpace m) { return m == MemSpace::DDR ? "ddr" : "sram"; }
const char* engine_name(Engine e) {
  switch (e) { case Engine::DMA: return "DMA"; case Engine::TENSOR: return "TENSOR";
               case Engine::VECTOR: return "VECTOR"; }
  return "?";
}
const char* opcode_name(Opcode o) {
  switch (o) {
    case Opcode::NOP: return "NOP";          case Opcode::HALT: return "HALT";
    case Opcode::DMA_LOAD: return "DMA.LOAD"; case Opcode::DMA_STORE: return "DMA.STORE";
    case Opcode::MATMUL: return "MATMUL";     case Opcode::VADD: return "VADD";
    case Opcode::RELU: return "RELU";         case Opcode::GELU: return "GELU";
    case Opcode::SILU: return "SILU";         case Opcode::SOFTMAX: return "SOFTMAX";
    case Opcode::RMSNORM: return "RMSNORM";   case Opcode::LAYERNORM: return "LAYERNORM";
    case Opcode::REQUANT: return "REQUANT";   case Opcode::CONV: return "CONV";
    case Opcode::ROPE: return "ROPE";
    case Opcode::LOOP: return "LOOP";         case Opcode::ENDLOOP: return "ENDLOOP";
    case Opcode::VMUL: return "VMUL";         case Opcode::VSUB: return "VSUB";
    case Opcode::VMAX: return "VMAX";         case Opcode::SIGMOID: return "SIGMOID";
    case Opcode::GATHER: return "GATHER";
    case Opcode::MAXPOOL: return "MAXPOOL";   case Opcode::UPSAMPLE: return "UPSAMPLE";
    case Opcode::CONCAT: return "CONCAT";
  }
  return "?";
}
Engine opcode_engine(Opcode o) {
  switch (o) {
    case Opcode::DMA_LOAD: case Opcode::DMA_STORE: return Engine::DMA;
    case Opcode::MATMUL:  case Opcode::CONV:       return Engine::TENSOR;
    case Opcode::VADD:    case Opcode::RELU:    case Opcode::GELU:
    case Opcode::SILU:    case Opcode::SOFTMAX: case Opcode::RMSNORM:
    case Opcode::LAYERNORM: case Opcode::REQUANT: case Opcode::ROPE:
    case Opcode::VMUL:    case Opcode::VSUB:    case Opcode::VMAX:
    case Opcode::SIGMOID: case Opcode::GATHER:
    case Opcode::MAXPOOL: case Opcode::UPSAMPLE: case Opcode::CONCAT:
                                                   return Engine::VECTOR;
    default:                                       return Engine::DMA;
  }
}

int64_t Descriptor::numel() const {
  int64_t n = 1; for (auto d : dims) n *= d; return n;
}
int64_t Descriptor::bytes() const { return numel() * int64_t(dtype_size(dtype)); }

uint64_t Descriptor::offset_bytes(const std::vector<int64_t>& idx) const {
  int64_t off = 0;
  if (!strides.empty()) {
    for (size_t i = 0; i < idx.size(); ++i) off += idx[i] * strides[i];
  } else {  // row-major
    int64_t stride = 1;
    for (int i = int(dims.size()) - 1; i >= 0; --i) {
      off += idx[i] * stride; stride *= dims[i];
    }
  }
  return base_addr + uint64_t(off) * dtype_size(dtype);
}

int Program::descriptor_id(const std::string& name) const {
  for (size_t i = 0; i < descriptors.size(); ++i)
    if (descriptors[i].name == name) return int(i);
  return -1;
}

// ----------------------------- assembler -----------------------------------

static std::vector<std::string> tokenize(const std::string& line) {
  std::vector<std::string> t;
  std::istringstream is(line);
  std::string w;
  while (is >> w) t.push_back(w);
  return t;
}

static std::vector<int64_t> parse_dims(const std::string& s) {
  std::vector<int64_t> dims;
  std::string cur;
  for (char ch : s) {
    if (ch == 'x' || ch == 'X' || ch == ',') {
      if (!cur.empty()) { dims.push_back(std::stoll(cur)); cur.clear(); }
    } else cur.push_back(ch);
  }
  if (!cur.empty()) dims.push_back(std::stoll(cur));
  return dims;
}

static uint64_t parse_addr(const std::string& s) {
  if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
    return std::stoull(s.substr(2), nullptr, 16);
  return std::stoull(s);
}

static Opcode parse_opcode(const std::string& s, bool& ok) {
  ok = true;
  if (s == "NOP") return Opcode::NOP;
  if (s == "HALT") return Opcode::HALT;
  if (s == "DMA.LOAD") return Opcode::DMA_LOAD;
  if (s == "DMA.STORE") return Opcode::DMA_STORE;
  if (s == "MATMUL") return Opcode::MATMUL;
  if (s == "VADD") return Opcode::VADD;
  if (s == "RELU") return Opcode::RELU;
  if (s == "GELU") return Opcode::GELU;
  if (s == "SILU") return Opcode::SILU;
  if (s == "SOFTMAX") return Opcode::SOFTMAX;
  if (s == "RMSNORM") return Opcode::RMSNORM;
  if (s == "LAYERNORM") return Opcode::LAYERNORM;
  if (s == "REQUANT") return Opcode::REQUANT;
  if (s == "CONV") return Opcode::CONV;
  if (s == "ROPE") return Opcode::ROPE;
  if (s == "LOOP") return Opcode::LOOP;
  if (s == "ENDLOOP") return Opcode::ENDLOOP;
  if (s == "VMUL") return Opcode::VMUL;
  if (s == "VSUB") return Opcode::VSUB;
  if (s == "VMAX") return Opcode::VMAX;
  if (s == "SIGMOID") return Opcode::SIGMOID;
  if (s == "GATHER") return Opcode::GATHER;
  if (s == "MAXPOOL") return Opcode::MAXPOOL;
  if (s == "UPSAMPLE") return Opcode::UPSAMPLE;
  if (s == "CONCAT") return Opcode::CONCAT;
  ok = false; return Opcode::NOP;
}

static void fill_init(Program& p, int id, const std::string& mode, const std::string& arg) {
  const Descriptor& d = p.descriptors[id];
  int64_t n = d.numel();
  std::vector<uint8_t> buf(d.bytes());
  size_t es = dtype_size(d.dtype);
  uint32_t lcg = 0x12345678u;
  if (mode == "rand" && !arg.empty()) lcg = uint32_t(std::stoul(arg));
  for (int64_t i = 0; i < n; ++i) {
    float v = 0.f;
    if (mode == "iota")       v = float(i);
    else if (mode == "const") v = arg.empty() ? 0.f : std::stof(arg);
    else if (mode == "zeros") v = 0.f;
    else if (mode == "rand") {
      lcg = lcg * 1664525u + 1013904223u;
      v = (float(lcg >> 8) / float(1u << 24)) * 2.f - 1.f;  // [-1,1)
    }
    store_elem(buf.data() + size_t(i) * es, d.dtype, v);
  }
  p.init_data[id] = std::move(buf);
}

Program assemble(const std::string& text) {
  Program p;
  std::istringstream is(text);
  std::string raw;
  int lineno = 0;
  auto err = [&](const std::string& m) {
    throw std::runtime_error("npuasm line " + std::to_string(lineno) + ": " + m);
  };

  while (std::getline(is, raw)) {
    ++lineno;
    auto semi = raw.find_first_of("#;");
    if (semi != std::string::npos) raw = raw.substr(0, semi);
    auto tok = tokenize(raw);
    if (tok.empty()) continue;

    if (tok[0] == ".desc") {
      if (tok.size() < 6) err(".desc NAME ddr|sram DTYPE BASE DIMS");
      Descriptor d;
      d.name = tok[1];
      d.space = (tok[2] == "sram") ? MemSpace::SRAM : MemSpace::DDR;
      d.dtype = dtype_from_string(tok[3]);
      d.base_addr = parse_addr(tok[4]);
      d.dims = parse_dims(tok[5]);
      for (size_t t = 6; t < tok.size(); ++t) {  // optional trailing modifiers
        if (tok[t][0] == '+')      d.iter_stride = std::stoll(tok[t].substr(1));  // iter stride
        else if (tok[t][0] == ':') d.strides = parse_dims(tok[t].substr(1));      // element strides
        else err("unexpected .desc modifier '" + tok[t] + "'");
      }
      if (!d.strides.empty() && d.strides.size() != d.dims.size())
        err(".desc strides rank must match dims");
      if (p.descriptor_id(d.name) != -1) err("duplicate descriptor " + d.name);
      p.descriptors.push_back(std::move(d));
      continue;
    }
    if (tok[0] == ".data") {
      if (tok.size() < 3) err(".data NAME iota|zeros|const V|rand SEED");
      int id = p.descriptor_id(tok[1]);
      if (id < 0) err("unknown descriptor " + tok[1]);
      fill_init(p, id, tok[2], tok.size() > 3 ? tok[3] : "");
      continue;
    }

    // instruction
    bool ok = false;
    Opcode op = parse_opcode(tok[0], ok);
    if (!ok) err("unknown opcode '" + tok[0] + "'");
    Instr in; in.op = op;
    for (size_t i = 1; i < tok.size(); ++i) {
      const std::string& a = tok[i];
      if (a == "accum") { in.accumulate = true; continue; }
      if (a == "@wait") { if (++i >= tok.size()) err("@wait needs event");
                          // accepts one or a comma-separated list (e.g. @wait 1,2)
                          std::string ev = tok[i]; size_t pos = 0;
                          while (pos < ev.size()) {
                            size_t comma = ev.find(',', pos);
                            std::string one = ev.substr(pos, comma - pos);
                            if (!one.empty()) in.wait_events.push_back(std::stoi(one.substr(one[0]=='e'?1:0)));
                            if (comma == std::string::npos) break; pos = comma + 1;
                          }
                          continue; }
      if (a == "@sig")  { if (++i >= tok.size()) err("@sig needs event");
                          in.signal_event = std::stoi(tok[i].substr(tok[i][0]=='e'?1:0)); continue; }
      if (a == "@core") { if (++i >= tok.size()) err("@core needs id");
                          in.core = std::stoi(tok[i]); continue; }
      if (a[0] == '$')  { in.imms.push_back(std::stod(a.substr(1))); continue; }  // immediate
      int id = p.descriptor_id(a);
      if (id < 0) err("unknown descriptor '" + a + "'");
      in.args.push_back(id);
    }
    p.instrs.push_back(std::move(in));
  }
  return p;
}

Program assemble_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open program: " + path);
  std::stringstream ss; ss << f.rdbuf();
  return assemble(ss.str());
}

// ----------------------------- loop flattening -----------------------------
//
// Supports nested LOOPs. A `+N` descriptor's base advances additively: it gains
// k*N elements for iteration k of EVERY enclosing loop it sits in. Each loop
// iteration also gets a disjoint event namespace (sized to the body's needs so
// nesting stays correct).

namespace {

// Index just past the ENDLOOP matching the LOOP at `lo` (handles nesting).
size_t match_endloop(const std::vector<Instr>& s, size_t lo) {
  int depth = 1;
  for (size_t j = lo + 1; j < s.size(); ++j) {
    if (s[j].op == Opcode::LOOP) ++depth;
    else if (s[j].op == Opcode::ENDLOOP && --depth == 0) return j;
  }
  throw std::runtime_error("LOOP without matching ENDLOOP");
}

// Size of the event-id namespace used by instructions in [begin, end). A nested
// loop consumes count * span(body); sequential siblings share the namespace.
int event_span(const std::vector<Instr>& s, size_t begin, size_t end) {
  int sp = 0;
  for (size_t i = begin; i < end;) {
    if (s[i].op == Opcode::LOOP) {
      size_t j = match_endloop(s, i);
      int count = int(s[i].imm(0, 1.0));
      sp = std::max(sp, count * event_span(s, i + 1, j));
      i = j + 1;
    } else {
      if (s[i].op == Opcode::ENDLOOP) throw std::runtime_error("ENDLOOP without matching LOOP");
      sp = std::max(sp, s[i].signal_event + 1);
      for (int w : s[i].wait_events) sp = std::max(sp, w + 1);
      ++i;
    }
  }
  return sp;
}

}  // namespace

Program flatten_loops(const Program& p) {
  bool has_loop = false;
  for (const auto& in : p.instrs)
    if (in.op == Opcode::LOOP || in.op == Opcode::ENDLOOP) { has_loop = true; break; }
  if (!has_loop) return p;

  Program out;
  out.descriptors = p.descriptors;
  out.init_data = p.init_data;

  // Cache of (descriptor, total element offset) -> variant descriptor id.
  std::map<std::pair<int, int64_t>, int> variant;
  auto variant_id = [&](int id, int64_t off_elems) -> int {
    if (off_elems == 0) return id;
    auto key = std::make_pair(id, off_elems);
    auto it = variant.find(key);
    if (it != variant.end()) return it->second;
    Descriptor v = out.descriptors[id];
    v.base_addr += uint64_t(off_elems) * dtype_size(v.dtype);
    v.iter_stride = 0;
    v.name = out.descriptors[id].name + "@" + std::to_string(off_elems);
    int nid = int(out.descriptors.size());
    out.descriptors.push_back(std::move(v));
    variant[key] = nid;
    return nid;
  };

  // Recursively expand [begin, end): `off` is the accumulated per-descriptor
  // element offset from enclosing loops; `ev_base` shifts this scope's events.
  std::function<void(size_t, size_t, const std::map<int, int64_t>&, int)> expand =
      [&](size_t begin, size_t end, const std::map<int, int64_t>& off, int ev_base) {
        for (size_t i = begin; i < end;) {
          const Instr& cur = p.instrs[i];
          if (cur.op == Opcode::LOOP) {
            size_t j = match_endloop(p.instrs, i);
            int count = int(cur.imm(0, 1.0));
            int span = event_span(p.instrs, i + 1, j);
            for (int k = 0; k < count; ++k) {
              std::map<int, int64_t> child = off;
              // Advance every descriptor that has an iteration stride.
              for (int d = 0; d < int(out.descriptors.size()); ++d)
                if (out.descriptors[d].iter_stride != 0)
                  child[d] += int64_t(k) * out.descriptors[d].iter_stride;
              expand(i + 1, j, child, ev_base + k * span);
            }
            i = j + 1;
          } else {
            if (cur.op == Opcode::ENDLOOP)
              throw std::runtime_error("ENDLOOP without matching LOOP");
            Instr ni = cur;
            for (auto& a : ni.args) {
              auto it = off.find(a);
              a = variant_id(a, it == off.end() ? 0 : it->second);
            }
            for (int& w : ni.wait_events) w += ev_base;
            if (ni.signal_event >= 0) ni.signal_event += ev_base;
            out.instrs.push_back(std::move(ni));
            ++i;
          }
        }
      };

  expand(0, p.instrs.size(), {}, 0);
  return out;
}

// ----------------------------- binary I/O ----------------------------------

template <class T> static void put(std::ostream& o, T v) {
  o.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <class T> static T get(std::istream& i) {
  T v; i.read(reinterpret_cast<char*>(&v), sizeof(T)); return v;
}
static void put_str(std::ostream& o, const std::string& s) {
  put<uint16_t>(o, uint16_t(s.size())); o.write(s.data(), s.size());
}
static std::string get_str(std::istream& i) {
  uint16_t n = get<uint16_t>(i); std::string s(n, '\0'); i.read(&s[0], n); return s;
}

void write_binary(const Program& p, const std::string& path) {
  std::ofstream o(path, std::ios::binary);
  if (!o) throw std::runtime_error("cannot write binary: " + path);
  o.write("NPUB", 4); put<uint32_t>(o, 5u);

  put<uint32_t>(o, uint32_t(p.descriptors.size()));
  for (const auto& d : p.descriptors) {
    put_str(o, d.name);
    put<uint8_t>(o, uint8_t(d.space));
    put<uint8_t>(o, uint8_t(d.dtype));
    put<uint64_t>(o, d.base_addr);
    put<uint16_t>(o, uint16_t(d.dims.size()));
    for (auto x : d.dims) put<int64_t>(o, x);
    put<uint16_t>(o, uint16_t(d.strides.size()));
    for (auto x : d.strides) put<int64_t>(o, x);
    put<int64_t>(o, d.iter_stride);
  }

  put<uint32_t>(o, uint32_t(p.instrs.size()));
  for (const auto& in : p.instrs) {
    put<uint8_t>(o, uint8_t(in.op));
    put<uint8_t>(o, in.accumulate ? 1 : 0);
    put<uint16_t>(o, uint16_t(in.wait_events.size()));
    for (int w : in.wait_events) put<int32_t>(o, w);
    put<int32_t>(o, in.signal_event);
    put<int32_t>(o, in.core);
    put<uint16_t>(o, uint16_t(in.args.size()));
    for (auto a : in.args) put<int32_t>(o, a);
    put<uint16_t>(o, uint16_t(in.imms.size()));
    for (auto v : in.imms) put<double>(o, v);
  }

  put<uint32_t>(o, uint32_t(p.init_data.size()));
  for (const auto& kv : p.init_data) {
    put<int32_t>(o, kv.first);
    put<uint64_t>(o, uint64_t(kv.second.size()));
    o.write(reinterpret_cast<const char*>(kv.second.data()), kv.second.size());
  }
}

Program read_binary(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read binary: " + path);
  char magic[4]; in.read(magic, 4);
  if (std::memcmp(magic, "NPUB", 4) != 0) throw std::runtime_error("bad magic in " + path);
  uint32_t ver = get<uint32_t>(in);
  if (ver != 5) throw std::runtime_error("unsupported .npubin version (expected 5)");

  Program p;
  uint32_t nd = get<uint32_t>(in);
  for (uint32_t k = 0; k < nd; ++k) {
    Descriptor d;
    d.name = get_str(in);
    d.space = MemSpace(get<uint8_t>(in));
    d.dtype = Dtype(get<uint8_t>(in));
    d.base_addr = get<uint64_t>(in);
    uint16_t r = get<uint16_t>(in);
    for (uint16_t j = 0; j < r; ++j) d.dims.push_back(get<int64_t>(in));
    uint16_t ns = get<uint16_t>(in);
    for (uint16_t j = 0; j < ns; ++j) d.strides.push_back(get<int64_t>(in));
    d.iter_stride = get<int64_t>(in);
    p.descriptors.push_back(std::move(d));
  }
  uint32_t ni = get<uint32_t>(in);
  for (uint32_t k = 0; k < ni; ++k) {
    Instr in2;
    in2.op = Opcode(get<uint8_t>(in));
    in2.accumulate = get<uint8_t>(in) != 0;
    uint16_t nw = get<uint16_t>(in);
    for (uint16_t w = 0; w < nw; ++w) in2.wait_events.push_back(get<int32_t>(in));
    in2.signal_event = get<int32_t>(in);
    in2.core = get<int32_t>(in);
    uint16_t na = get<uint16_t>(in);
    for (uint16_t j = 0; j < na; ++j) in2.args.push_back(get<int32_t>(in));
    uint16_t nm = get<uint16_t>(in);
    for (uint16_t j = 0; j < nm; ++j) in2.imms.push_back(get<double>(in));
    p.instrs.push_back(std::move(in2));
  }
  uint32_t nin = get<uint32_t>(in);
  for (uint32_t k = 0; k < nin; ++k) {
    int32_t id = get<int32_t>(in);
    uint64_t n = get<uint64_t>(in);
    std::vector<uint8_t> buf(n);
    in.read(reinterpret_cast<char*>(buf.data()), n);
    p.init_data[id] = std::move(buf);
  }
  return p;
}

}  // namespace neuronpu
