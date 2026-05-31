#include "neuronpu/types.h"

#include <cstring>
#include <stdexcept>

namespace neuronpu {

const char* dtype_name(Dtype d) {
  switch (d) {
    case Dtype::F32:  return "f32";
    case Dtype::F16:  return "f16";
    case Dtype::BF16: return "bf16";
    case Dtype::I8:   return "i8";
  }
  return "?";
}

Dtype dtype_from_string(const std::string& s) {
  if (s == "f32" || s == "fp32") return Dtype::F32;
  if (s == "f16" || s == "fp16") return Dtype::F16;
  if (s == "bf16")               return Dtype::BF16;
  if (s == "i8"  || s == "int8") return Dtype::I8;
  throw std::runtime_error("unknown dtype: " + s);
}

std::size_t dtype_size(Dtype d) {
  switch (d) {
    case Dtype::F32:  return 4;
    case Dtype::F16:  return 2;
    case Dtype::BF16: return 2;
    case Dtype::I8:   return 1;
  }
  return 0;
}

float load_elem(const void* p, Dtype d) {
  switch (d) {
    case Dtype::F32: { float v; std::memcpy(&v, p, 4); return v; }
    case Dtype::F16: { uint16_t h; std::memcpy(&h, p, 2); return f16_to_f32(h); }
    case Dtype::BF16: {
      uint16_t b; std::memcpy(&b, p, 2);
      uint32_t bits = uint32_t(b) << 16; float v; std::memcpy(&v, &bits, 4); return v;
    }
    case Dtype::I8: { int8_t v; std::memcpy(&v, p, 1); return float(v); }
  }
  return 0.f;
}

void store_elem(void* p, Dtype d, float v) {
  switch (d) {
    case Dtype::F32: std::memcpy(p, &v, 4); return;
    case Dtype::F16: { uint16_t h = f32_to_f16(v); std::memcpy(p, &h, 2); return; }
    case Dtype::BF16: {
      uint32_t bits; std::memcpy(&bits, &v, 4);
      uint16_t b = uint16_t(bits >> 16); std::memcpy(p, &b, 2); return;
    }
    case Dtype::I8: {
      float r = v < -128.f ? -128.f : (v > 127.f ? 127.f : v);
      int8_t q = int8_t(r < 0 ? r - 0.5f : r + 0.5f); std::memcpy(p, &q, 1); return;
    }
  }
}

uint16_t f32_to_f16(float f) {
  uint32_t x; std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t  exp  = int32_t((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFF) == 0xFF)              // inf/nan
    return uint16_t(sign | 0x7C00u | (mant ? 0x200u : 0));
  if (exp >= 0x1F) return uint16_t(sign | 0x7C00u);   // overflow -> inf
  if (exp <= 0) {                                     // subnormal/zero
    if (exp < -10) return uint16_t(sign);
    mant |= 0x800000u;
    uint32_t shift = uint32_t(14 - exp);
    return uint16_t(sign | (mant >> shift));
  }
  return uint16_t(sign | (uint32_t(exp) << 10) | (mant >> 13));
}

float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t(h) & 0x8000u) << 16;
  uint32_t exp  = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) { bits = sign; }
    else {                                   // subnormal
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
      mant &= 0x3FFu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float v; std::memcpy(&v, &bits, 4); return v;
}

}  // namespace neuronpu
