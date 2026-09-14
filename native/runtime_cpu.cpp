// SPDX-License-Identifier: MIT
#include "spry/runtime.h"

#include "spry/codec.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace {

thread_local const char* g_error = "ok";

extern "C" int spry_uq_runtime_set_error(int status, const char* message) {
  g_error = message;
  return status;
}

int fail(int status, const char* message) {
  return spry_uq_runtime_set_error(status, message);
}

int ok() {
  g_error = "ok";
  return SPRY_UQ_OK;
}

bool supported_dimension(int dimension) {
  return dimension == 64 || dimension == 128 || dimension == 256;
}

bool exceeds_narrow_index_range(int first, int second, int third = 1) {
  constexpr std::int64_t limit = std::numeric_limits<int>::max();
  if (first < 0 || second < 0 || third < 0) return true;
  if (first == 0 || second == 0 || third == 0) return false;
  if (first != 0 && first > limit / second) return true;
  const std::int64_t product = static_cast<std::int64_t>(first) * second;
  return product != 0 && product > limit / third;
}

bool finite_in_range(float value) {
  return std::isfinite(value) && std::fabs(value) <= 256.0f;
}

bool gpu_safe_exponent(std::uint8_t exponent) {
  // `6 * 2^(exponent - 127)` must survive the specified FP16 decode boundary.
  return spry::valid_exponent_byte(exponent) && exponent <= 140u;
}

// Explicit IEEE-754 binary32/binary16 conversion keeps the CPU oracle
// independent of compiler-specific half types.
std::uint16_t f32_to_f16_bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16u) & 0x8000u;
  const std::uint32_t exponent = (bits >> 23u) & 0xffu;
  std::uint32_t mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu) {
    return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
  }
  int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    mantissa |= 0x800000u;
    const unsigned shift = static_cast<unsigned>(14 - half_exponent);
    std::uint32_t rounded = mantissa >> shift;
    const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0u)) {
      ++rounded;
    }
    return static_cast<std::uint16_t>(sign | rounded);
  }
  std::uint32_t rounded = mantissa >> 13u;
  const std::uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u) != 0u)) {
    ++rounded;
    if (rounded == 0x400u) {
      rounded = 0;
      ++half_exponent;
      if (half_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
      }
    }
  }
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(half_exponent) << 10u) |
                                    rounded);
}

float f16_bits_to_f32(std::uint16_t half) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(half) & 0x8000u) << 16u;
  std::uint32_t exponent = (half >> 10u) & 0x1fu;
  std::uint32_t mantissa = half & 0x3ffu;
  std::uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400u) == 0u) {
        mantissa <<= 1u;
        --exponent;
      }
      mantissa &= 0x3ffu;
      bits = sign | (exponent << 23u) | (mantissa << 13u);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13u);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23u) | (mantissa << 13u);
  }
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float half_round(float value) {
  return f16_bits_to_f32(f32_to_f16_bits(value));
}

std::size_t row_bytes(int head_dim) {
  return static_cast<std::size_t>(head_dim / spry::kGroupSize) *
         spry::kEncodedGroupBytes;
}

int validate_common(int slots, int kv_heads, int head_dim) {
  if (slots <= 0 || kv_heads <= 0) {
    return fail(SPRY_UQ_INVALID_ARGUMENT, "slots and kv_heads must be positive");
  }
  if (!supported_dimension(head_dim)) {
    return fail(SPRY_UQ_UNSUPPORTED_DIMENSION, "head_dim must be 64, 128, or 256");
  }
  if (exceeds_narrow_index_range(slots, kv_heads, head_dim)) {
    return fail(SPRY_UQ_INVALID_ARGUMENT, "cache shape exceeds the narrow experimental index range");
  }
  return SPRY_UQ_OK;
}

int validate_write_inputs(const float* input_k, const float* input_v,
                          const std::int64_t* locations, int tokens, int slots,
                          int kv_heads, int head_dim) {
  if (tokens < 0 || input_k == nullptr || input_v == nullptr ||
      (tokens != 0 && locations == nullptr)) {
    return fail(SPRY_UQ_INVALID_ARGUMENT, "write received a null pointer or negative token count");
  }
  if (exceeds_narrow_index_range(tokens, kv_heads, head_dim)) {
    return fail(SPRY_UQ_INVALID_ARGUMENT, "write shape exceeds the narrow experimental index range");
  }
  std::unordered_set<std::int64_t> used;
  const std::size_t values = static_cast<std::size_t>(tokens) * kv_heads * head_dim;
  for (int token = 0; token < tokens; ++token) {
    if (locations[token] < 0 || locations[token] >= slots || !used.insert(locations[token]).second) {
      return fail(SPRY_UQ_INVALID_ARGUMENT, "write locations must be unique in range");
    }
  }
  for (std::size_t index = 0; index < values; ++index) {
    if (!finite_in_range(input_k[index]) || !finite_in_range(input_v[index])) {
      return fail(SPRY_UQ_NONFINITE_INPUT, "K/V values must be finite with absolute value at most 256");
    }
  }
  return SPRY_UQ_OK;
}

int validate_attention_inputs(const std::uint8_t* compressed_k,
                              const std::uint8_t* compressed_v,
                              const std::uint8_t* validity, const float* query,
                              const std::int64_t* indices,
                              const std::int64_t* lengths, float* output,
                              int slots, int queries, int max_length,
                              int query_heads, int kv_heads, int head_dim,
                              float attention_scale) {
  if (queries < 0 || max_length < 0 || query_heads <= 0 || kv_heads <= 0 ||
      query_heads % kv_heads != 0 || !std::isfinite(attention_scale) ||
      attention_scale <= 0.0f || compressed_k == nullptr || compressed_v == nullptr ||
      validity == nullptr || query == nullptr || output == nullptr ||
      (queries != 0 && (indices == nullptr || lengths == nullptr))) {
    return fail(SPRY_UQ_INVALID_ARGUMENT, "invalid attention shape, scale, or pointer");
  }
  if (exceeds_narrow_index_range(queries, query_heads, head_dim) ||
      exceeds_narrow_index_range(queries, max_length)) {
    return fail(SPRY_UQ_INVALID_ARGUMENT, "attention shape exceeds the narrow experimental index range");
  }
  const float max_dot = 4096.0f * 65504.0f * static_cast<float>(head_dim);
  if (attention_scale > std::numeric_limits<float>::max() / max_dot) {
    return fail(SPRY_UQ_INVALID_ARGUMENT, "attention scale can overflow the FP32 score");
  }
  const std::size_t query_values = static_cast<std::size_t>(queries) * query_heads * head_dim;
  for (std::size_t i = 0; i < query_values; ++i) {
    if (!finite_in_range(query[i])) {
      return fail(SPRY_UQ_NONFINITE_INPUT, "query values must be finite with absolute value at most 256");
    }
  }
  const std::size_t encoded_row = row_bytes(head_dim);
  for (int query_index = 0; query_index < queries; ++query_index) {
    const std::int64_t length = lengths[query_index];
    if (length < 0 || length > max_length) {
      return fail(SPRY_UQ_INVALID_ARGUMENT, "attention length is outside max_length");
    }
    for (std::int64_t token = 0; token < length; ++token) {
      const std::int64_t slot = indices[static_cast<std::size_t>(query_index) * max_length + token];
      if (slot < 0 || slot >= slots || validity[slot] == 0u) {
        return fail(SPRY_UQ_INVALID_ARGUMENT, "attention read references an invalid cache slot");
      }
      for (int head = 0; head < kv_heads; ++head) {
        const std::size_t base = (static_cast<std::size_t>(slot) * kv_heads + head) * encoded_row;
        for (std::size_t group = 0; group < encoded_row; group += spry::kEncodedGroupBytes) {
          if (!gpu_safe_exponent(compressed_k[base + group + 16]) ||
              !gpu_safe_exponent(compressed_v[base + group + 16])) {
            return fail(SPRY_UQ_MALFORMED_ENCODING, "attention read found a scale outside the FP16 decode range");
          }
        }
      }
    }
  }
  return SPRY_UQ_OK;
}

void decode_row_half(const std::uint8_t* encoded, int head_dim, float* output) {
  for (int group = 0; group < head_dim / static_cast<int>(spry::kGroupSize); ++group) {
    const std::uint8_t* block = encoded + group * spry::kEncodedGroupBytes;
    const std::uint8_t exponent = block[16];
    for (int lane = 0; lane < static_cast<int>(spry::kGroupSize); ++lane) {
      const std::uint8_t code = static_cast<std::uint8_t>(block[lane / 2] >> ((lane & 1) * 4));
      output[group * spry::kGroupSize + lane] = half_round(spry::decode_value(code, exponent));
    }
  }
}

}  // namespace

extern "C" int spry_uq_hip_abi_version(void) { return 1; }

extern "C" const char* spry_uq_last_error(void) { return g_error; }

extern "C" int spry_uq_cpu_write(
    std::uint8_t* compressed_k, std::uint8_t* compressed_v, std::uint8_t* validity,
    const float* input_k, const float* input_v, const std::int64_t* locations,
    int tokens, int slots, int kv_heads, int head_dim) {
  if (compressed_k == nullptr || compressed_v == nullptr || validity == nullptr) {
    return fail(SPRY_UQ_INVALID_ARGUMENT, "write received a null cache pointer");
  }
  int status = validate_common(slots, kv_heads, head_dim);
  if (status != SPRY_UQ_OK) return status;
  status = validate_write_inputs(input_k, input_v, locations, tokens, slots, kv_heads, head_dim);
  if (status != SPRY_UQ_OK) return status;

  float source[256];
  float rotated[256];
  const std::size_t encoded_row = row_bytes(head_dim);
  for (int token = 0; token < tokens; ++token) {
    const std::size_t destination = static_cast<std::size_t>(locations[token]);
    for (int head = 0; head < kv_heads; ++head) {
      const std::size_t input_base = (static_cast<std::size_t>(token) * kv_heads + head) * head_dim;
      const std::size_t cache_base = (destination * kv_heads + head) * encoded_row;
      for (int channel = 0; channel < head_dim; ++channel) source[channel] = half_round(input_k[input_base + channel]);
      spry::rotate(source, rotated, static_cast<std::size_t>(head_dim));
      for (int channel = 0; channel < head_dim; ++channel) source[channel] = rotated[channel];
      for (int group = 0; group < head_dim / static_cast<int>(spry::kGroupSize); ++group) {
        spry::encode_group(source + group * spry::kGroupSize,
                           compressed_k + cache_base + group * spry::kEncodedGroupBytes);
      }
      for (int channel = 0; channel < head_dim; ++channel) source[channel] = half_round(input_v[input_base + channel]);
      for (int group = 0; group < head_dim / static_cast<int>(spry::kGroupSize); ++group) {
        spry::encode_group(source + group * spry::kGroupSize,
                           compressed_v + cache_base + group * spry::kEncodedGroupBytes);
      }
    }
    validity[destination] = 1u;
  }
  return ok();
}

extern "C" int spry_uq_cpu_attention(
    const std::uint8_t* compressed_k, const std::uint8_t* compressed_v,
    const std::uint8_t* validity, const float* query, const std::int64_t* indices,
    const std::int64_t* lengths, float* output, int slots, int queries,
    int max_length, int query_heads, int kv_heads, int head_dim,
    float attention_scale) {
  int status = validate_common(slots, kv_heads, head_dim);
  if (status != SPRY_UQ_OK) return status;
  status = validate_attention_inputs(compressed_k, compressed_v, validity, query, indices,
                                     lengths, output, slots, queries, max_length, query_heads,
                                     kv_heads, head_dim, attention_scale);
  if (status != SPRY_UQ_OK) return status;

  const std::size_t encoded_row = row_bytes(head_dim);
  float source[256];
  float rotated_query[256];
  float decoded_k[256];
  float decoded_v[256];
  float accumulator[256];
  for (int query_index = 0; query_index < queries; ++query_index) {
    const std::int64_t length = lengths[query_index];
    for (int query_head = 0; query_head < query_heads; ++query_head) {
      const std::size_t query_base = (static_cast<std::size_t>(query_index) * query_heads + query_head) * head_dim;
      const std::size_t output_base = query_base;
      for (int channel = 0; channel < head_dim; ++channel) source[channel] = half_round(query[query_base + channel]);
      spry::rotate(source, rotated_query, static_cast<std::size_t>(head_dim));
      for (int channel = 0; channel < head_dim; ++channel) rotated_query[channel] = half_round(rotated_query[channel]);
      for (int channel = 0; channel < head_dim; ++channel) accumulator[channel] = 0.0f;
      float maximum = -std::numeric_limits<float>::infinity();
      float denominator = 0.0f;
      const int kv_head = query_head / (query_heads / kv_heads);
      for (std::int64_t token = 0; token < length; ++token) {
        const std::int64_t slot = indices[static_cast<std::size_t>(query_index) * max_length + token];
        const std::size_t cache_base = (static_cast<std::size_t>(slot) * kv_heads + kv_head) * encoded_row;
        decode_row_half(compressed_k + cache_base, head_dim, decoded_k);
        decode_row_half(compressed_v + cache_base, head_dim, decoded_v);
        float score = 0.0f;
        for (int channel = 0; channel < head_dim; ++channel) score += rotated_query[channel] * decoded_k[channel];
        score *= attention_scale;
        const float new_maximum = score > maximum ? score : maximum;
        const float previous_weight = std::exp(maximum - new_maximum);
        const float token_weight = std::exp(score - new_maximum);
        for (int channel = 0; channel < head_dim; ++channel) {
          accumulator[channel] = accumulator[channel] * previous_weight + token_weight * decoded_v[channel];
        }
        denominator = denominator * previous_weight + token_weight;
        maximum = new_maximum;
      }
      for (int channel = 0; channel < head_dim; ++channel) {
        // Expose the FP32 accumulator result for tight implementation-error
        // checks, before the HIP entry point's final binary16 output rounding.
        output[output_base + channel] = length == 0 ? 0.0f : accumulator[channel] / denominator;
      }
    }
  }
  return ok();
}
