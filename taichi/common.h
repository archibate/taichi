#pragma once
#include "constants.h"

#define FUNC_DECL

#define TLANG_NAMESPACE_BEGIN \
  namespace taichi {          \
  namespace Tlang {
#define TLANG_NAMESPACE_END \
  }                         \
  }

#include <atomic>
#include <numeric>
#include <mutex>
#include <unordered_map>
#include <iostream>

#if !defined(TI_INCLUDED)

#ifdef _WIN64
#define TI_FORCE_INLINE __forceinline
#else
#define TI_FORCE_INLINE inline __attribute__((always_inline))
#endif
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <array>
#include <vector>

using float32 = float;
using float64 = double;
using uint8 = std::uint8_t;
using uint16 = std::uint16_t;
using uint32 = std::uint32_t;
using uint64 = std::uint64_t;
using int8 = std::int8_t;
using int16 = std::int16_t;
using int32 = std::int32_t;
using int64 = std::int64_t;

namespace taichi {
TI_FORCE_INLINE uint32 rand_int() noexcept {
  static unsigned int x = 123456789, y = 362436069, z = 521288629, w = 88675123;
  unsigned int t = x ^ (x << 11);
  x = y;
  y = z;
  z = w;
  return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
}

TI_FORCE_INLINE uint64 rand_int64() noexcept {
  return ((uint64)rand_int() << 32) + rand_int();
}

template <typename T>
TI_FORCE_INLINE T rand() noexcept;

template <>
TI_FORCE_INLINE float rand<float>() noexcept {
  return rand_int() * (1.0f / 4294967296.0f);
}

template <>
TI_FORCE_INLINE double rand<double>() noexcept {
  return rand_int() * (1.0 / 4294967296.0);
}

template <>
TI_FORCE_INLINE int rand<int>() noexcept {
  return rand_int();
}

template <typename T>
TI_FORCE_INLINE T rand() noexcept;
}  // namespace taichi

#endif

TLANG_NAMESPACE_BEGIN

using size_t = std::size_t;

constexpr int max_num_indices = taichi_max_num_indices;
constexpr int max_num_args = taichi_max_num_args;
constexpr int max_num_snodes = taichi_max_num_snodes;
constexpr int max_gpu_block_dim = 1024;

struct SNodeMeta {
  int indices[max_num_indices];
  int active;
  int start_loop;
  int end_loop;
  int _;
  void **snode_ptr;
  void *ptr;
};

struct AllocatorStat {
  int snode_id;
  size_t pool_size;
  size_t num_resident_blocks;
  size_t num_recycled_blocks;
  SNodeMeta *resident_metas;
};

template <typename T, typename G>
T union_cast(G g) {
  static_assert(sizeof(T) == sizeof(G), "");
  union {
    T t;
    G g;
  } u;
  u.g = g;
  return u.t;
}

template <typename T, typename G>
T union_cast_different_size(G g) {
  union {
    T t;
    G g;
  } u;
  u.g = g;
  return u.t;
}

TLANG_NAMESPACE_END
