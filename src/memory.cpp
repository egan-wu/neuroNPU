#include "neuronpu/memory.h"

#include <cstring>
#include <stdexcept>

namespace neuronpu {

void MemSpaceStore::check(uint64_t off) const {
  if (off >= bytes_.size())
    throw std::runtime_error("memory access out of range: offset " + std::to_string(off) +
                             " >= size " + std::to_string(bytes_.size()));
}

void MemSpaceStore::write(uint64_t off, const uint8_t* src, uint64_t n) {
  if (n == 0) return;
  check(off + n - 1);
  std::memcpy(bytes_.data() + off, src, n);
}

void MemSpaceStore::read(uint64_t off, uint8_t* dst, uint64_t n) const {
  if (n == 0) return;
  check(off + n - 1);
  std::memcpy(dst, bytes_.data() + off, n);
}

}  // namespace neuronpu
