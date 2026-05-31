// NeuroNPU — core numeric types and dtype helpers.
#pragma once
#include <cstdint>
#include <cstddef>
#include <string>

namespace neuronpu {

// Supported element data types. Compute always accumulates in fp32.
enum class Dtype { F32, F16, BF16, I8 };

const char* dtype_name(Dtype d);
Dtype       dtype_from_string(const std::string& s);
std::size_t dtype_size(Dtype d);  // bytes per element

// Read/write a single element at `p` interpreted as dtype `d`, in fp32 space.
float load_elem(const void* p, Dtype d);
void  store_elem(void* p, Dtype d, float v);

// IEEE-754 half <-> float (used for F16; BF16 is the high 16 bits of fp32).
uint16_t f32_to_f16(float f);
float    f16_to_f32(uint16_t h);

}  // namespace neuronpu
