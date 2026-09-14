#include "spry/codec.h"

#include <cmath>
#include <limits>

namespace {

bool supported_dimension(std::size_t dimension) {
  return dimension == 64u || dimension == 128u || dimension == 256u;
}

bool finite_values(const float* values, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  return true;
}

bool rotation_range_is_safe(const float* values, std::size_t dimension) {
  float maximum = 0.0f;
  for (std::size_t index = 0; index < dimension; ++index) {
    maximum = std::fmax(maximum, std::fabs(values[index]));
  }
  // Every unnormalized butterfly output is bounded by dimension * maximum.
  return maximum <= std::numeric_limits<float>::max() /
                        static_cast<float>(dimension);
}

int validate_dimension_and_size(std::size_t dimension, std::size_t bytes) {
  if (!supported_dimension(dimension)) {
    return SPRY_UQ_UNSUPPORTED_DIMENSION;
  }
  const std::size_t expected = (dimension / spry::kGroupSize) *
                               spry::kEncodedGroupBytes;
  return bytes == expected ? SPRY_UQ_OK : SPRY_UQ_BUFFER_SIZE;
}

}  // namespace

extern "C" int spry_uq_encoded_bytes(std::size_t dimension,
                                      std::size_t* encoded_bytes) {
  if (encoded_bytes == nullptr) {
    return SPRY_UQ_INVALID_ARGUMENT;
  }
  if (!supported_dimension(dimension)) {
    return SPRY_UQ_UNSUPPORTED_DIMENSION;
  }
  *encoded_bytes = (dimension / spry::kGroupSize) * spry::kEncodedGroupBytes;
  return SPRY_UQ_OK;
}

extern "C" int spry_uq_encode_f32(const float* values, std::size_t dimension,
                                   std::uint8_t* encoded,
                                   std::size_t encoded_bytes) {
  if (values == nullptr || encoded == nullptr) {
    return SPRY_UQ_INVALID_ARGUMENT;
  }
  const int status = validate_dimension_and_size(dimension, encoded_bytes);
  if (status != SPRY_UQ_OK) {
    return status;
  }
  // Validate the entire input before writing any cache byte.
  if (!finite_values(values, dimension)) {
    return SPRY_UQ_NONFINITE_INPUT;
  }
  for (std::size_t group = 0; group < dimension / spry::kGroupSize; ++group) {
    spry::encode_group(values + group * spry::kGroupSize,
                       encoded + group * spry::kEncodedGroupBytes);
  }
  return SPRY_UQ_OK;
}

extern "C" int spry_uq_decode_f32(const std::uint8_t* encoded,
                                   std::size_t encoded_bytes,
                                   std::size_t dimension, float* values) {
  if (encoded == nullptr || values == nullptr) {
    return SPRY_UQ_INVALID_ARGUMENT;
  }
  const int status = validate_dimension_and_size(dimension, encoded_bytes);
  if (status != SPRY_UQ_OK) {
    return status;
  }
  for (std::size_t group = 0; group < dimension / spry::kGroupSize; ++group) {
    const std::uint8_t* encoded_group =
        encoded + group * spry::kEncodedGroupBytes;
    if (!spry::valid_exponent_byte(encoded_group[16])) {
      return SPRY_UQ_MALFORMED_ENCODING;
    }
    for (std::size_t index = 0; index < spry::kGroupSize; ++index) {
      const std::uint8_t code = static_cast<std::uint8_t>(
          (encoded_group[index / 2u] >> ((index % 2u) * 4u)) & 0x0fu);
      if (!std::isfinite(spry::decode_value(code, encoded_group[16]))) {
        return SPRY_UQ_MALFORMED_ENCODING;
      }
    }
  }
  for (std::size_t group = 0; group < dimension / spry::kGroupSize; ++group) {
    const std::uint8_t* encoded_group =
        encoded + group * spry::kEncodedGroupBytes;
    float* output_group = values + group * spry::kGroupSize;
    for (std::size_t index = 0; index < spry::kGroupSize; ++index) {
      const std::uint8_t code = static_cast<std::uint8_t>(
          (encoded_group[index / 2u] >> ((index % 2u) * 4u)) & 0x0fu);
      output_group[index] = spry::decode_value(code, encoded_group[16]);
    }
  }
  return SPRY_UQ_OK;
}

extern "C" int spry_uq_rotate_wht_f32(const float* input,
                                       std::size_t dimension, float* output) {
  if (input == nullptr || output == nullptr) {
    return SPRY_UQ_INVALID_ARGUMENT;
  }
  if (!supported_dimension(dimension)) {
    return SPRY_UQ_UNSUPPORTED_DIMENSION;
  }
  if (!finite_values(input, dimension)) {
    return SPRY_UQ_NONFINITE_INPUT;
  }
  if (!rotation_range_is_safe(input, dimension)) {
    return SPRY_UQ_VALUE_OUT_OF_RANGE;
  }
  spry::rotate(input, output, dimension);
  return SPRY_UQ_OK;
}

extern "C" const char* spry_uq_status_string(int status) {
  switch (status) {
    case SPRY_UQ_OK:
      return "ok";
    case SPRY_UQ_INVALID_ARGUMENT:
      return "invalid argument";
    case SPRY_UQ_UNSUPPORTED_DIMENSION:
      return "unsupported dimension";
    case SPRY_UQ_BUFFER_SIZE:
      return "incorrect buffer size";
    case SPRY_UQ_NONFINITE_INPUT:
      return "non-finite input";
    case SPRY_UQ_MALFORMED_ENCODING:
      return "malformed encoding";
    case SPRY_UQ_VALUE_OUT_OF_RANGE:
      return "value out of range";
    default:
      return "unknown status";
  }
}
