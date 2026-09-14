"""Regression tests for the production C ABI, without third-party modules."""

from __future__ import annotations

import ctypes
import math
import os
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from spry_uq.native import CodecError, CpuCodec  # noqa: E402


LIBRARY = os.environ.get("SPRY_UQ_CPU_LIBRARY")
F32_EXPONENT_SCALE = struct.unpack("<f", struct.pack("<f", 0.156))[0]


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def next_f32_positive(value: float, upward: bool) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 1 if upward else -1
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def expected_exponent_byte(values: list[float]) -> int:
    maximum = max(abs(f32(value)) for value in values)
    if maximum == 0.0:
        return 127
    scaled = f32(F32_EXPONENT_SCALE * maximum)
    if scaled == 0.0:
        return 0
    exponent = max(-127, min(127, round(math.log2(scaled))))
    return exponent + 127


def constant_code_page(code: int, exponent_byte: int) -> bytearray:
    encoded = bytearray(34)
    encoded[:16] = bytes([code | (code << 4)] * 16)
    encoded[16] = exponent_byte
    encoded[33] = 127
    return encoded


@unittest.skipUnless(LIBRARY, "set SPRY_UQ_CPU_LIBRARY to the explicitly built CPU library")
class CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.codec = CpuCodec(LIBRARY)
        cls.lib = cls.codec._library

    def test_encoded_size_and_supported_dimensions(self) -> None:
        self.assertEqual(self.codec.encoded_bytes(64), 34)
        self.assertEqual(self.codec.encoded_bytes(128), 68)
        self.assertEqual(self.codec.encoded_bytes(256), 136)
        with self.assertRaisesRegex(CodecError, "unsupported dimension"):
            self.codec.encoded_bytes(32)

    def test_exhaustive_small_fp4_decode_and_nibble_order(self) -> None:
        encoded = bytearray(34)
        for index in range(16):
            encoded[index // 2] |= index << ((index % 2) * 4)
        encoded[16] = 127
        encoded[33] = 127
        decoded = self.codec.decode(encoded, 64)
        expected = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        for index, value in enumerate(expected):
            self.assertEqual(decoded[index], value)
        self.assertEqual(math.copysign(1.0, decoded[8]), -1.0)

    def test_golden_encoding_is_bit_exact(self) -> None:
        values = [0.0, -0.5, 1.0, -1.5, 2.0, -3.0, 4.0, -6.0, 6.4]
        values.extend([0.0] * (64 - len(values)))
        encoded = self.codec.encode(values)
        expected_group = bytes([0x90, 0xB2, 0xD4, 0xF6, 0x07] + [0] * 11 + [127])
        self.assertEqual(encoded, expected_group + bytes(16) + bytes([127]))
        decoded = self.codec.decode(encoded, 64)
        self.assertEqual(decoded[:9], [0.0, -0.5, 1.0, -1.5, 2.0, -3.0, 4.0, -6.0, 6.0])

    def test_magnitude_ties_clip_and_zero_canonicalization(self) -> None:
        # The first seven values lie exactly between adjacent FP4 magnitudes.
        values = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 8.0, -0.0, 6.4]
        values.extend([0.0] * (64 - len(values)))
        encoded = self.codec.encode(values)
        # Ties select an even magnitude-code index: 0,2,2,4,4,6,6.  8 clips to 7.
        self.assertEqual(list(encoded[:5]), [0x20, 0x42, 0x64, 0x76, 0x70])
        self.assertEqual(encoded[16], 127)

    def test_nextafter_fp4_ties_are_stable_at_float32_precision(self) -> None:
        quarter = f32(0.25)
        five = f32(5.0)
        values = [
            next_f32_positive(quarter, False), quarter,
            next_f32_positive(quarter, True), next_f32_positive(five, False),
            five, next_f32_positive(five, True), 6.4,
        ]
        values.extend([0.0] * (64 - len(values)))
        encoded = self.codec.encode(values)
        # Below/tie/above .25 maps 0/0/1.  At 5 the even code is 6, then the
        # next representable float above it maps to 7.
        self.assertEqual(list(encoded[:4]), [0x00, 0x61, 0x76, 0x07])

    def test_group_boundaries_keep_independent_scales(self) -> None:
        values = [6.4] + [0.0] * 31 + [0.4] + [0.0] * 31
        encoded = self.codec.encode(values)
        self.assertEqual(encoded[16], expected_exponent_byte(values[:32]))
        self.assertEqual(encoded[33], expected_exponent_byte(values[32:]))
        self.assertEqual((encoded[0] & 0x0F), 7)
        self.assertEqual((encoded[17] & 0x0F), 7)

    def test_exponent_boundaries_use_float32_scale_math(self) -> None:
        for requested_exponent in (-127, -126, -1, 0, 1, 125):
            maximum = f32(math.ldexp(1.0, requested_exponent) / F32_EXPONENT_SCALE)
            values = [maximum] + [0.0] * 63
            encoded = self.codec.encode(values)
            self.assertEqual(encoded[16], expected_exponent_byte(values))

    def test_underflow_and_float32_extreme_are_defined(self) -> None:
        smallest_subnormal = float.fromhex("0x0.000002p-126")
        encoded = self.codec.encode([smallest_subnormal] * 64)
        self.assertEqual(encoded[16], 0)
        self.assertEqual(encoded[:16], bytes(16))

        float32_max = ctypes.c_float.from_buffer_copy(b"\xff\xff\x7f\x7f").value
        encoded = self.codec.encode([float32_max] * 64)
        self.assertEqual(encoded[16], 252)
        self.assertEqual(encoded[:16], bytes([0x77] * 16))

    def test_nonfinite_input_does_not_partially_write(self) -> None:
        values = (ctypes.c_float * 64)(*[0.0] * 63, math.inf)
        output = (ctypes.c_uint8 * 34)(*[0xA5] * 34)
        status = self.lib.spry_uq_encode_f32(values, 64, output, 34)
        self.assertEqual(status, 4)
        self.assertEqual(bytes(output), bytes([0xA5] * 34))
        for bad in (math.nan, -math.inf):
            with self.assertRaisesRegex(CodecError, "non-finite"):
                self.codec.encode([bad] + [0.0] * 63)

    def test_malformed_encoding_and_exact_buffer_length_are_rejected(self) -> None:
        malformed = bytearray(34)
        malformed[16] = 255
        with self.assertRaisesRegex(CodecError, "malformed"):
            self.codec.decode(malformed, 64)

        source = (ctypes.c_float * 64)()
        output = (ctypes.c_uint8 * 35)()
        self.assertEqual(self.lib.spry_uq_encode_f32(source, 64, output, 35), 3)
        decoded = (ctypes.c_float * 64)()
        self.assertEqual(self.lib.spry_uq_decode_f32(output, 33, 64, decoded), 3)

    def test_decode_scale_boundaries_and_nonrepresentable_values(self) -> None:
        cases = [(0, 7), (1, 7), (126, 7), (127, 7), (128, 7),
                 (253, 1), (254, 2)]
        for exponent_byte, code in cases:
            decoded = self.codec.decode(constant_code_page(code, exponent_byte), 64)
            expected = f32(math.ldexp([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][code],
                                      exponent_byte - 127))
            self.assertEqual(decoded[0], expected)

        for exponent_byte, code in ((253, 7), (254, 4)):
            with self.assertRaisesRegex(CodecError, "malformed"):
                self.codec.decode(constant_code_page(code, exponent_byte), 64)

    def test_normalized_rotation_matches_hand_computable_basis(self) -> None:
        rotated = self.codec.rotate([1.0] + [0.0] * 63)
        expected = 1.0 / 8.0
        for value in rotated:
            self.assertAlmostEqual(value, expected, places=7)

        values = [float((index % 7) - 3) for index in range(64)]
        rotated = self.codec.rotate(values)
        self.assertAlmostEqual(sum(value * value for value in values),
                               sum(value * value for value in rotated), places=4)

    def test_dense_wht_sign_matrix_and_dot_products_for_larger_heads(self) -> None:
        for dimension in (128, 256):
            query = [f32(((index * 17) % 31 - 15) / 7.0) for index in range(dimension)]
            key = [f32(((index * 11) % 29 - 14) / 5.0) for index in range(dimension)]
            rotated_query = self.codec.rotate(query)
            rotated_key = self.codec.rotate(key)
            normalization = math.sqrt(dimension)
            expected_query = [
                sum((1.0 if ((row & column).bit_count() % 2) == 0 else -1.0) * query[column]
                    for column in range(dimension)) / normalization
                for row in range(dimension)
            ]
            for actual, expected in zip(rotated_query, expected_query):
                self.assertAlmostEqual(actual, expected, places=5)
            self.assertAlmostEqual(sum(query[index] * key[index] for index in range(dimension)),
                                   sum(rotated_query[index] * rotated_key[index]
                                       for index in range(dimension)), places=3)

    def test_rotation_rejects_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(CodecError, "non-finite"):
            self.codec.rotate([math.nan] + [0.0] * 63)

    def test_rotation_refuses_values_that_can_overflow_a_butterfly(self) -> None:
        float32_max = ctypes.c_float.from_buffer_copy(b"\xff\xff\x7f\x7f").value
        source = (ctypes.c_float * 64)(*[float32_max] + [0.0] * 63)
        output = (ctypes.c_float * 64)(*[123.0] * 64)
        self.assertEqual(self.lib.spry_uq_rotate_wht_f32(source, 64, output), 6)
        self.assertEqual(list(output), [123.0] * 64)


if __name__ == "__main__":
    unittest.main()
