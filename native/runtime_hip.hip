// SPDX-License-Identifier: MIT
// Correctness-first gfx1201 implementation.  It deliberately uses scalar HIP
// arithmetic: compressed cache rows remain uint8 in device memory and are
// unpacked into per-thread local arrays while attention visits each token.
#include "spry/runtime.h"

#include "spry/codec.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "The experimental Spry HIP kernel must be compiled for gfx1201"
#endif
#include <limits>

extern "C" int spry_uq_runtime_set_error(int status, const char* message);

namespace {

constexpr int kThreads = 64;

__device__ void set_device_error(int* status, int value) {
  atomicCAS(status, SPRY_UQ_OK, value);
}

__device__ bool finite_in_range(float value) {
  return isfinite(value) && fabsf(value) <= 256.0f;
}

__device__ bool gpu_safe_exponent(std::uint8_t exponent) {
  return spry::valid_exponent_byte(exponent) && exponent <= 140u;
}

__device__ std::size_t row_bytes(int head_dim) {
  return static_cast<std::size_t>(head_dim / spry::kGroupSize) *
         spry::kEncodedGroupBytes;
}

__global__ void validate_write_kernel(const __half* input_k, const __half* input_v,
                                      const std::int64_t* locations, int tokens,
                                      int slots, int kv_heads, int head_dim,
                                      int* status) {
  const int token = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (token >= tokens || *status != SPRY_UQ_OK) return;
  const std::int64_t location = locations[token];
  if (location < 0 || location >= slots) {
    set_device_error(status, SPRY_UQ_INVALID_ARGUMENT);
    return;
  }
  // This preflight rejects duplicate destinations before the write kernel runs.
  for (int previous = 0; previous < token; ++previous) {
    if (locations[previous] == location) {
      set_device_error(status, SPRY_UQ_INVALID_ARGUMENT);
      return;
    }
  }
  const std::size_t base = static_cast<std::size_t>(token) * kv_heads * head_dim;
  const std::size_t count = static_cast<std::size_t>(kv_heads) * head_dim;
  for (std::size_t index = 0; index < count; ++index) {
    if (!finite_in_range(__half2float(input_k[base + index])) ||
        !finite_in_range(__half2float(input_v[base + index]))) {
      set_device_error(status, SPRY_UQ_NONFINITE_INPUT);
      return;
    }
  }
}

__global__ void write_kernel(std::uint8_t* compressed_k, std::uint8_t* compressed_v,
                             std::uint8_t* validity, const __half* input_k,
                             const __half* input_v, const std::int64_t* locations,
                             int tokens, int kv_heads, int head_dim) {
  const int item = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = tokens * kv_heads;
  if (item >= total) return;
  const int token = item / kv_heads;
  const int head = item % kv_heads;
  const std::size_t encoded_row = row_bytes(head_dim);
  const std::size_t input_base = static_cast<std::size_t>(item) * head_dim;
  const std::size_t cache_base =
      (static_cast<std::size_t>(locations[token]) * kv_heads + head) * encoded_row;
  float source[256];
  float rotated[256];
  for (int channel = 0; channel < head_dim; ++channel) {
    source[channel] = __half2float(input_k[input_base + channel]);
  }
  spry::rotate(source, rotated, static_cast<std::size_t>(head_dim));
  for (int channel = 0; channel < head_dim; ++channel) source[channel] = rotated[channel];
  for (int group = 0; group < head_dim / static_cast<int>(spry::kGroupSize); ++group) {
    spry::encode_group(source + group * spry::kGroupSize,
                       compressed_k + cache_base + group * spry::kEncodedGroupBytes);
  }
  for (int channel = 0; channel < head_dim; ++channel) {
    source[channel] = __half2float(input_v[input_base + channel]);
  }
  for (int group = 0; group < head_dim / static_cast<int>(spry::kGroupSize); ++group) {
    spry::encode_group(source + group * spry::kGroupSize,
                       compressed_v + cache_base + group * spry::kEncodedGroupBytes);
  }
  if (head == 0) validity[locations[token]] = 1u;
}

__global__ void validate_attention_kernel(
    const std::uint8_t* compressed_k, const std::uint8_t* compressed_v,
    const std::uint8_t* validity, const __half* query, const std::int64_t* indices,
    const std::int64_t* lengths, int slots, int queries, int max_length,
    int query_heads, int kv_heads, int head_dim, int* status) {
  const int query_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (query_index >= queries || *status != SPRY_UQ_OK) return;
  const std::size_t query_base = static_cast<std::size_t>(query_index) * query_heads * head_dim;
  for (std::size_t channel = 0; channel < static_cast<std::size_t>(query_heads) * head_dim; ++channel) {
    if (!finite_in_range(__half2float(query[query_base + channel]))) {
      set_device_error(status, SPRY_UQ_NONFINITE_INPUT);
      return;
    }
  }
  const std::int64_t length = lengths[query_index];
  if (length < 0 || length > max_length) {
    set_device_error(status, SPRY_UQ_INVALID_ARGUMENT);
    return;
  }
  const std::size_t encoded_row = row_bytes(head_dim);
  for (std::int64_t token = 0; token < length; ++token) {
    const std::int64_t slot = indices[static_cast<std::size_t>(query_index) * max_length + token];
    if (slot < 0 || slot >= slots || validity[slot] == 0u) {
      set_device_error(status, SPRY_UQ_INVALID_ARGUMENT);
      return;
    }
    for (int head = 0; head < kv_heads; ++head) {
      const std::size_t base = (static_cast<std::size_t>(slot) * kv_heads + head) * encoded_row;
      for (std::size_t group = 0; group < encoded_row; group += spry::kEncodedGroupBytes) {
        if (!gpu_safe_exponent(compressed_k[base + group + 16]) ||
            !gpu_safe_exponent(compressed_v[base + group + 16])) {
          set_device_error(status, SPRY_UQ_MALFORMED_ENCODING);
          return;
        }
      }
    }
  }
}

__device__ void decode_row_half(const std::uint8_t* encoded, int head_dim, float* output) {
  for (int group = 0; group < head_dim / static_cast<int>(spry::kGroupSize); ++group) {
    const std::uint8_t* block = encoded + group * spry::kEncodedGroupBytes;
    const std::uint8_t exponent = block[16];
    for (int lane = 0; lane < static_cast<int>(spry::kGroupSize); ++lane) {
      const std::uint8_t code = static_cast<std::uint8_t>(block[lane / 2] >> ((lane & 1) * 4));
      output[group * spry::kGroupSize + lane] =
          __half2float(__float2half_rn(spry::decode_value(code, exponent)));
    }
  }
}

__global__ void attention_kernel(
    const std::uint8_t* compressed_k, const std::uint8_t* compressed_v,
    const __half* query, const std::int64_t* indices, const std::int64_t* lengths,
    __half* output, int queries, int max_length, int query_heads,
    int kv_heads, int head_dim, float attention_scale) {
  const int item = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = queries * query_heads;
  if (item >= total) return;
  const int query_index = item / query_heads;
  const int query_head = item % query_heads;
  const int kv_head = query_head / (query_heads / kv_heads);
  const std::size_t query_base = static_cast<std::size_t>(item) * head_dim;
  const std::size_t encoded_row = row_bytes(head_dim);
  float source[256];
  float rotated_query[256];
  float decoded_k[256];
  float decoded_v[256];
  float accumulator[256];
  for (int channel = 0; channel < head_dim; ++channel) {
    source[channel] = __half2float(query[query_base + channel]);
    accumulator[channel] = 0.0f;
  }
  spry::rotate(source, rotated_query, static_cast<std::size_t>(head_dim));
  for (int channel = 0; channel < head_dim; ++channel) {
    rotated_query[channel] = __half2float(__float2half_rn(rotated_query[channel]));
  }
  float maximum = -INFINITY;
  float denominator = 0.0f;
  const std::int64_t length = lengths[query_index];
  for (std::int64_t token = 0; token < length; ++token) {
    const std::int64_t slot = indices[static_cast<std::size_t>(query_index) * max_length + token];
    const std::size_t cache_base = (static_cast<std::size_t>(slot) * kv_heads + kv_head) * encoded_row;
    decode_row_half(compressed_k + cache_base, head_dim, decoded_k);
    decode_row_half(compressed_v + cache_base, head_dim, decoded_v);
    float score = 0.0f;
    for (int channel = 0; channel < head_dim; ++channel) score += rotated_query[channel] * decoded_k[channel];
    score *= attention_scale;
    const float new_maximum = score > maximum ? score : maximum;
    const float old_weight = expf(maximum - new_maximum);
    const float token_weight = expf(score - new_maximum);
    for (int channel = 0; channel < head_dim; ++channel) {
      accumulator[channel] = accumulator[channel] * old_weight + token_weight * decoded_v[channel];
    }
    denominator = denominator * old_weight + token_weight;
    maximum = new_maximum;
  }
  for (int channel = 0; channel < head_dim; ++channel) {
    output[query_base + channel] = __float2half_rn(length == 0 ? 0.0f : accumulator[channel] / denominator);
  }
}

bool supported_dimension(int dimension) { return dimension == 64 || dimension == 128 || dimension == 256; }

bool exceeds_narrow_index_range(int first, int second, int third = 1) {
  constexpr std::int64_t limit = std::numeric_limits<int>::max();
  if (first < 0 || second < 0 || third < 0) return true;
  if (first == 0 || second == 0 || third == 0) return false;
  if (first != 0 && first > limit / second) return true;
  const std::int64_t product = static_cast<std::int64_t>(first) * second;
  return product != 0 && product > limit / third;
}

int host_validate_common(int slots, int kv_heads, int head_dim) {
  if (slots <= 0 || kv_heads <= 0) return spry_uq_runtime_set_error(SPRY_UQ_INVALID_ARGUMENT, "slots and kv_heads must be positive");
  if (!supported_dimension(head_dim)) return spry_uq_runtime_set_error(SPRY_UQ_UNSUPPORTED_DIMENSION, "head_dim must be 64, 128, or 256");
  if (exceeds_narrow_index_range(slots, kv_heads, head_dim)) {
    return spry_uq_runtime_set_error(SPRY_UQ_INVALID_ARGUMENT, "cache shape exceeds the narrow experimental index range");
  }
  return SPRY_UQ_OK;
}

const char* hip_message(hipError_t error) {
  return hipGetErrorString(error);
}

int check_hip(hipError_t error) {
  return error == hipSuccess ? SPRY_UQ_OK : spry_uq_runtime_set_error(SPRY_UQ_INVALID_ARGUMENT, hip_message(error));
}

int status_message(int status) {
  switch (status) {
    case SPRY_UQ_NONFINITE_INPUT:
      return spry_uq_runtime_set_error(status, "input contains a nonfinite or out-of-range value");
    case SPRY_UQ_MALFORMED_ENCODING:
      return spry_uq_runtime_set_error(status, "cache scale is outside the FP16 decode range");
    default:
      return spry_uq_runtime_set_error(status, "device-side cache validation failed");
  }
}

int require_device_pointer(const void* pointer, int device) {
  hipPointerAttribute_t attributes{};
  const hipError_t result = hipPointerGetAttributes(&attributes, pointer);
  if (result != hipSuccess || attributes.type != hipMemoryTypeDevice || attributes.device != device) {
    return spry_uq_runtime_set_error(SPRY_UQ_INVALID_ARGUMENT, "all HIP ABI buffers must be device allocations");
  }
  return SPRY_UQ_OK;
}

class DeviceScope {
 public:
  explicit DeviceScope(int requested) : previous_(-1), active_(false) {
    if (requested < 0 || hipGetDevice(&previous_) != hipSuccess || hipSetDevice(requested) != hipSuccess) return;
    active_ = true;
  }
  ~DeviceScope() { if (active_) hipSetDevice(previous_); }
  bool active() const { return active_; }
 private:
  int previous_;
  bool active_;
};

int prepare_call(void* opaque_stream, int device, const void* const* pointers, int pointer_count) {
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, device) != hipSuccess ||
      std::strncmp(properties.gcnArchName, "gfx1201", 7) != 0 ||
      (properties.gcnArchName[7] != '\0' && properties.gcnArchName[7] != ':')) {
    return spry_uq_runtime_set_error(SPRY_UQ_INVALID_ARGUMENT, "HIP device must be exactly gfx1201");
  }
  hipStream_t stream = reinterpret_cast<hipStream_t>(opaque_stream);
  hipStreamCaptureStatus capture_status;
  if (hipStreamIsCapturing(stream, &capture_status) != hipSuccess || capture_status != hipStreamCaptureStatusNone) {
    return spry_uq_runtime_set_error(SPRY_UQ_INVALID_ARGUMENT, "HIP graph capture is unsupported by the experimental backend");
  }
  int stream_device = -1;
  if (hipStreamGetDevice(stream, &stream_device) != hipSuccess || stream_device != device) {
    return spry_uq_runtime_set_error(SPRY_UQ_INVALID_ARGUMENT, "stream does not belong to the requested device");
  }
  for (int i = 0; i < pointer_count; ++i) {
    const int status = require_device_pointer(pointers[i], device);
    if (status != SPRY_UQ_OK) return status;
  }
  return SPRY_UQ_OK;
}

template <typename Launch>
int validate_then_launch(hipStream_t stream, Launch launch) {
  int* device_status = nullptr;
  hipError_t error = hipMalloc(&device_status, sizeof(int));
  if (error != hipSuccess) return check_hip(error);
  error = hipMemsetAsync(device_status, 0, sizeof(int), stream);
  if (error == hipSuccess) launch(device_status);
  if (error == hipSuccess) error = hipGetLastError();
  int host_status = SPRY_UQ_OK;
  if (error == hipSuccess) error = hipMemcpyAsync(&host_status, device_status, sizeof(int), hipMemcpyDeviceToHost, stream);
  if (error == hipSuccess) error = hipStreamSynchronize(stream);
  hipError_t release_error = hipFree(device_status);
  if (error != hipSuccess) return check_hip(error);
  if (release_error != hipSuccess) return check_hip(release_error);
  return host_status == SPRY_UQ_OK ? SPRY_UQ_OK : status_message(host_status);
}

}  // namespace

extern "C" int spry_uq_hip_write(
    std::uint8_t* compressed_k, std::uint8_t* compressed_v, std::uint8_t* validity,
    const void* input_k, const void* input_v, const std::int64_t* locations,
    int tokens, int slots, int kv_heads, int head_dim, void* opaque_stream, int device) {
  if (compressed_k == nullptr || compressed_v == nullptr || validity == nullptr || input_k == nullptr ||
      input_v == nullptr || (tokens != 0 && locations == nullptr) || tokens < 0) {
    return spry_uq_runtime_set_error(SPRY_UQ_INVALID_ARGUMENT, "write received a null pointer or negative token count");
  }
  int status = host_validate_common(slots, kv_heads, head_dim);
  if (status != SPRY_UQ_OK) return status;
  if (exceeds_narrow_index_range(tokens, kv_heads, head_dim)) {
    return spry_uq_runtime_set_error(SPRY_UQ_INVALID_ARGUMENT, "write shape exceeds the narrow experimental index range");
  }
  if (tokens == 0) return spry_uq_runtime_set_error(SPRY_UQ_OK, "ok");
  DeviceScope scope(device);
  if (!scope.active()) return spry_uq_runtime_set_error(SPRY_UQ_INVALID_ARGUMENT, "cannot select requested HIP device");
  const void* pointers[] = {compressed_k, compressed_v, validity, input_k, input_v, locations};
  status = prepare_call(opaque_stream, device, pointers, 6);
  if (status != SPRY_UQ_OK) return status;
  hipStream_t stream = reinterpret_cast<hipStream_t>(opaque_stream);
  status = validate_then_launch(stream, [&](int* device_status) {
    hipLaunchKernelGGL(validate_write_kernel, dim3((tokens + kThreads - 1) / kThreads), dim3(kThreads), 0,
                       stream, static_cast<const __half*>(input_k), static_cast<const __half*>(input_v), locations,
                       tokens, slots, kv_heads, head_dim, device_status);
  });
  if (status != SPRY_UQ_OK) return status;
  hipLaunchKernelGGL(write_kernel, dim3((tokens * kv_heads + kThreads - 1) / kThreads), dim3(kThreads), 0,
                     stream, compressed_k, compressed_v, validity, static_cast<const __half*>(input_k),
                     static_cast<const __half*>(input_v), locations, tokens, kv_heads, head_dim);
  hipError_t error = hipGetLastError();
  if (error == hipSuccess) error = hipStreamSynchronize(stream);
  return check_hip(error);
}

extern "C" int spry_uq_hip_attention(
    const std::uint8_t* compressed_k, const std::uint8_t* compressed_v,
    const std::uint8_t* validity, const void* query, const std::int64_t* indices,
    const std::int64_t* lengths, void* output, int slots, int queries,
    int max_length, int query_heads, int kv_heads, int head_dim, float attention_scale,
    void* opaque_stream, int device) {
  if (compressed_k == nullptr || compressed_v == nullptr || validity == nullptr || query == nullptr ||
      output == nullptr || (queries != 0 && (indices == nullptr || lengths == nullptr)) || queries < 0 ||
      max_length < 0 || query_heads <= 0 || kv_heads <= 0 || query_heads % kv_heads != 0 ||
      !std::isfinite(attention_scale) || attention_scale <= 0.0f) {
    return spry_uq_runtime_set_error(SPRY_UQ_INVALID_ARGUMENT, "invalid attention shape, scale, or pointer");
  }
  int status = host_validate_common(slots, kv_heads, head_dim);
  if (status != SPRY_UQ_OK) return status;
  if (queries == 0) return spry_uq_runtime_set_error(SPRY_UQ_OK, "ok");
  if (exceeds_narrow_index_range(queries, query_heads, head_dim) ||
      exceeds_narrow_index_range(queries, max_length)) {
    return spry_uq_runtime_set_error(SPRY_UQ_INVALID_ARGUMENT, "attention shape exceeds the narrow experimental index range");
  }
  const float max_dot = 4096.0f * 65504.0f * static_cast<float>(head_dim);
  if (attention_scale > std::numeric_limits<float>::max() / max_dot) {
    return spry_uq_runtime_set_error(SPRY_UQ_INVALID_ARGUMENT, "attention scale can overflow the FP32 score");
  }
  DeviceScope scope(device);
  if (!scope.active()) return spry_uq_runtime_set_error(SPRY_UQ_INVALID_ARGUMENT, "cannot select requested HIP device");
  const void* pointers[] = {compressed_k, compressed_v, validity, query, indices, lengths, output};
  status = prepare_call(opaque_stream, device, pointers, 7);
  if (status != SPRY_UQ_OK) return status;
  hipStream_t stream = reinterpret_cast<hipStream_t>(opaque_stream);
  status = validate_then_launch(stream, [&](int* device_status) {
    hipLaunchKernelGGL(validate_attention_kernel, dim3((queries + kThreads - 1) / kThreads), dim3(kThreads), 0,
                       stream, compressed_k, compressed_v, validity, static_cast<const __half*>(query), indices,
                       lengths, slots, queries, max_length, query_heads, kv_heads, head_dim, device_status);
  });
  if (status != SPRY_UQ_OK) return status;
  hipLaunchKernelGGL(attention_kernel, dim3((queries * query_heads + kThreads - 1) / kThreads), dim3(kThreads), 0,
                     stream, compressed_k, compressed_v, static_cast<const __half*>(query), indices, lengths,
                     static_cast<__half*>(output), queries, max_length, query_heads, kv_heads, head_dim,
                     attention_scale);
  hipError_t error = hipGetLastError();
  if (error == hipSuccess) error = hipStreamSynchronize(stream);
  return check_hip(error);
}
