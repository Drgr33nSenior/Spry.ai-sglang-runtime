// SPDX-License-Identifier: MIT
// Experimental UltraQuant-derived runtime C ABI.  All functions return zero
// on success.  On failure, call spry_uq_last_error() on the calling thread.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Increment this only for an incompatible ABI change.
int spry_uq_hip_abi_version(void);
const char* spry_uq_last_error(void);

// CPU reference entry points.  Inputs and outputs are binary32 arrays, but the
// Q/K/V operands use the HIP binary16 boundaries. Output exposes the FP32
// accumulator result before HIP's final binary16 rounding, for numerical checks.
int spry_uq_cpu_write(uint8_t* compressed_k, uint8_t* compressed_v,
                      uint8_t* validity, const float* input_k,
                      const float* input_v, const int64_t* locations,
                      int tokens, int slots, int kv_heads, int head_dim);
int spry_uq_cpu_attention(const uint8_t* compressed_k,
                          const uint8_t* compressed_v,
                          const uint8_t* validity, const float* query,
                          const int64_t* indices, const int64_t* lengths,
                          float* output, int slots, int queries,
                          int max_length, int query_heads, int kv_heads,
                          int head_dim, float attention_scale);

// HIP entry points.  All pointer arguments except stream are device pointers.
// K/V/query/output are contiguous IEEE binary16 tensors.  locations, indices,
// and lengths are int64 device tensors.  stream is a hipStream_t passed as void.
int spry_uq_hip_write(uint8_t* compressed_k, uint8_t* compressed_v,
                      uint8_t* validity, const void* input_k,
                      const void* input_v, const int64_t* locations,
                      int tokens, int slots, int kv_heads, int head_dim,
                      void* stream, int device);
int spry_uq_hip_attention(const uint8_t* compressed_k,
                          const uint8_t* compressed_v,
                          const uint8_t* validity, const void* query,
                          const int64_t* indices, const int64_t* lengths,
                          void* output, int slots, int queries,
                          int max_length, int query_heads, int kv_heads,
                          int head_dim, float attention_scale, void* stream,
                          int device);

#ifdef __cplusplus
}
#endif
