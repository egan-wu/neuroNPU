// NeuroNPU — flat byte-addressable memory spaces (DDR + SRAM scratchpad).
#pragma once
#include <algorithm>
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

// Address spaces: one shared DDR plus a private SRAM scratchpad per core.
class Memory {
 public:
  Memory(uint64_t ddr_bytes, uint64_t sram_bytes, int num_cores) : ddr_(ddr_bytes) {
    for (int i = 0; i < std::max(1, num_cores); ++i) sram_.emplace_back(sram_bytes);
  }

  MemSpaceStore& space(MemSpace s, int core) {
    return s == MemSpace::DDR ? ddr_ : sram_[core];
  }
  const MemSpaceStore& space(MemSpace s, int core) const {
    return s == MemSpace::DDR ? ddr_ : sram_[core];
  }
  int num_cores() const { return int(sram_.size()); }

 private:
  MemSpaceStore              ddr_;
  std::vector<MemSpaceStore> sram_;  // one per core
};

}  // namespace neuronpu
