// NeuroNPU — flat byte-addressable memory spaces (DDR + SRAM scratchpad).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "neuronpu/isa.h"

namespace neuronpu {

// A contiguous byte array standing in for one address space.
class MemSpaceStore {
 public:
  explicit MemSpaceStore(uint64_t size_bytes) : bytes_(size_bytes, 0) {}

  uint8_t*       at(uint64_t off)       { check(off); return bytes_.data() + off; }
  const uint8_t* at(uint64_t off) const { check(off); return bytes_.data() + off; }
  uint64_t       size() const { return bytes_.size(); }

  void write(uint64_t off, const uint8_t* src, uint64_t n);
  void read(uint64_t off, uint8_t* dst, uint64_t n) const;

 private:
  void check(uint64_t off) const;
  std::vector<uint8_t> bytes_;
};

// The two address spaces of the core.
class Memory {
 public:
  Memory(uint64_t ddr_bytes, uint64_t sram_bytes) : ddr_(ddr_bytes), sram_(sram_bytes) {}

  MemSpaceStore&       space(MemSpace s)       { return s == MemSpace::DDR ? ddr_ : sram_; }
  const MemSpaceStore& space(MemSpace s) const { return s == MemSpace::DDR ? ddr_ : sram_; }

 private:
  MemSpaceStore ddr_;
  MemSpaceStore sram_;
};

}  // namespace neuronpu
