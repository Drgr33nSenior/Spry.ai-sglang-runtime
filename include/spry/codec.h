#ifndef SPRY_CODEC_H_
#define SPRY_CODEC_H_

// Scalar codec primitives shared by the CPU reference library and HIP kernels.
// The storage layout is intentionally fixed: each 32-value group has sixteen
// low-nibble-first FP4 bytes followed by one biased exponent byte.

#include <cstddef>
#include <cstdint>
#include <cmath>

#if defined(__HIPCC__)
#define SPRY_HD __host__ __device__
#else
#define SPRY_HD
#endif

namespace spry {

constexpr std::size_t kGroupSize = 32;
constexpr std::size_t kEncodedGroupBytes = 17;
constexpr float kExponentScale = 0.156f;

// Scalar arithmetic avoids a dynamically indexed host constexpr array in HIP
// device code. All operations here are exact for the eight E2M1 magnitudes.
SPRY_HD inline float fp4_magnitude(std::uint8_t code) {
  const unsigned magnitude = code & 7u;
  return magnitude < 4u ? 0.5f * static_cast<float>(magnitude)
                       : ::ldexpf(1.0f + 0.5f * static_cast<float>(magnitude & 1u),
                                  static_cast<int>(magnitude / 2u) - 1);
}

SPRY_HD inline bool valid_exponent_byte(std::uint8_t exponent) {
  // 255 is reserved so malformed cache pages can be detected.  It is never
  // emitted by the encoder (the largest representable E is +127, or 254).
  return exponent != 255u;
}

SPRY_HD inline float round_ties_to_even(float value) {
  const float lower = ::floorf(value);
  const float fraction = value - lower;
  if (fraction < 0.5f) {
    return lower;
  }
  if (fraction > 0.5f) {
    return lower + 1.0f;
  }
  const int lower_integer = static_cast<int>(lower);
  return (lower_integer & 1) == 0 ? lower : lower + 1.0f;
}

SPRY_HD inline std::uint8_t nearest_magnitude_code(float magnitude) {
  if (magnitude >= 6.0f) {
    return 7u;
  }

  std::uint8_t selected = 0u;
  float selected_distance = ::fabsf(magnitude);
  for (std::uint8_t candidate = 1u; candidate < 8u; ++candidate) {
    const float distance = ::fabsf(magnitude - fp4_magnitude(candidate));
    if (distance < selected_distance ||
        (distance == selected_distance && (candidate & 1u) == 0u &&
         (selected & 1u) != 0u)) {
      selected = candidate;
      selected_distance = distance;
    }
  }
  return selected;
}

SPRY_HD inline float decode_value(std::uint8_t code, std::uint8_t exponent) {
  const float magnitude = fp4_magnitude(code);
  const float scaled = ::ldexpf(magnitude, static_cast<int>(exponent) - 127);
  // Preserve an encoded negative zero for diagnostic decoding.  The encoder
  // emits only positive zero, so this does not create non-canonical storage.
  return (code & 0x8u) == 0u ? scaled : -scaled;
}

SPRY_HD inline void encode_group(const float* values, std::uint8_t* encoded) {
  float maximum = 0.0f;
  for (std::size_t index = 0; index < kGroupSize; ++index) {
    const float magnitude = ::fabsf(values[index]);
    if (magnitude > maximum) {
      maximum = magnitude;
    }
  }

  int exponent = 0;
  if (maximum != 0.0f) {
    // Avoid log2(0) after a subnormal product underflows.  The documented
    // policy is to retain the smallest finite scale and quantize such values
    // to zero when they cannot reach the FP4 range.
    const float scaled_maximum = kExponentScale * maximum;
    if (scaled_maximum == 0.0f) {
      exponent = -127;
    } else {
      exponent = static_cast<int>(round_ties_to_even(::log2f(scaled_maximum)));
      if (exponent < -127) {
        exponent = -127;
      } else if (exponent > 127) {
        exponent = 127;
      }
    }
  }
  const std::uint8_t exponent_byte =
      static_cast<std::uint8_t>(exponent + 127);
  encoded[16] = exponent_byte;
  const float scale = ::ldexpf(1.0f, exponent);

  for (std::size_t index = 0; index < kGroupSize; index += 2) {
    std::uint8_t packed = 0u;
    for (std::size_t lane = 0; lane < 2; ++lane) {
      const float value = values[index + lane];
      std::uint8_t code = nearest_magnitude_code(::fabsf(value) / scale);
      // Canonicalize both signed zero forms to the positive zero code.
      if (code != 0u && value < 0.0f) {
        code |= 0x8u;
      }
      packed |= static_cast<std::uint8_t>(code << (lane * 4u));
    }
    encoded[index / 2u] = packed;
  }
}

// A deterministic normalized Sylvester Walsh-Hadamard transform.  Callers
// provide distinct or identical input/output buffers; partially overlapping
// buffers are not supported.
SPRY_HD inline void rotate(const float* input, float* output,
                           std::size_t dimension) {
  for (std::size_t index = 0; index < dimension; ++index) {
    output[index] = input[index];
  }
  for (std::size_t width = 1; width < dimension; width *= 2u) {
    for (std::size_t base = 0; base < dimension; base += width * 2u) {
      for (std::size_t offset = 0; offset < width; ++offset) {
        const float left = output[base + offset];
        const float right = output[base + width + offset];
        output[base + offset] = left + right;
        output[base + width + offset] = left - right;
      }
    }
  }
  const float normalization = 1.0f / ::sqrtf(static_cast<float>(dimension));
  for (std::size_t index = 0; index < dimension; ++index) {
    output[index] *= normalization;
  }
}

}  // namespace spry

#ifdef __cplusplus
extern "C" {
#endif

enum spry_uq_status {
  SPRY_UQ_OK = 0,
  SPRY_UQ_INVALID_ARGUMENT = 1,
  SPRY_UQ_UNSUPPORTED_DIMENSION = 2,
  SPRY_UQ_BUFFER_SIZE = 3,
  SPRY_UQ_NONFINITE_INPUT = 4,
  SPRY_UQ_MALFORMED_ENCODING = 5,
  SPRY_UQ_VALUE_OUT_OF_RANGE = 6,
};

int spry_uq_encoded_bytes(std::size_t dimension, std::size_t* encoded_bytes);
int spry_uq_encode_f32(const float* values, std::size_t dimension,
                       std::uint8_t* encoded, std::size_t encoded_bytes);
int spry_uq_decode_f32(const std::uint8_t* encoded, std::size_t encoded_bytes,
                       std::size_t dimension, float* values);
int spry_uq_rotate_wht_f32(const float* input, std::size_t dimension,
                           float* output);
const char* spry_uq_status_string(int status);

#ifdef __cplusplus
}
#endif

#endif  // SPRY_CODEC_H_
