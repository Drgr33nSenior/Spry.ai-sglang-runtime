"""CPU regression coverage for the production compressed-cache C ABI."""

from __future__ import annotations

import ctypes
import math
import os
import random
import struct
import unittest


LIBRARY = os.environ.get("SPRY_UQ_CPU_LIBRARY")


def half(value: float) -> float:
    """Round exactly at the IEEE binary16 boundary used by the HIP path."""
    return struct.unpack("<e", struct.pack("<e", value))[0]


def rotate(values: list[float]) -> list[float]:
    result = list(values)
    width = 1
    while width < len(result):
        for base in range(0, len(result), 2 * width):
            for offset in range(width):
                left, right = result[base + offset], result[base + width + offset]
                result[base + offset] = left + right
                result[base + width + offset] = left - right
        width *= 2
    inverse_norm = 1.0 / math.sqrt(len(result))
    return [value * inverse_norm for value in result]


def decode_row(payload: bytes, head_dim: int) -> list[float]:
    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    result: list[float] = []
    for group in range(head_dim // 32):
        block = payload[group * 17 : (group + 1) * 17]
        assert len(block) == 17
        exponent = block[16]
        if exponent > 140:
            raise ValueError("malformed scale")
        scale = math.ldexp(1.0, exponent - 127)
        for lane in range(32):
            code = block[lane // 2] >> (4 * (lane & 1))
            value = magnitudes[code & 7] * scale
            result.append(half(-value if code & 8 else value))
    return result


def dense_attention_from_encoded(k: bytes, v: bytes, q: list[float], indices: list[list[int]],
                                 lengths: list[int], slots: int, kv_heads: int,
                                 query_heads: int, head_dim: int, scale: float) -> list[float]:
    row_bytes = 17 * (head_dim // 32)
    output = [0.0] * (len(lengths) * query_heads * head_dim)
    for query_index, length in enumerate(lengths):
        for query_head in range(query_heads):
            q_base = (query_index * query_heads + query_head) * head_dim
            rotated_q = [half(value) for value in rotate([half(value) for value in q[q_base : q_base + head_dim]])]
            kv_head = query_head // (query_heads // kv_heads)
            scores, rows = [], []
            for token in range(length):
                slot = indices[query_index][token]
                row_base = (slot * kv_heads + kv_head) * row_bytes
                decoded_k = decode_row(k[row_base : row_base + row_bytes], head_dim)
                decoded_v = decode_row(v[row_base : row_base + row_bytes], head_dim)
                scores.append(sum(a * b for a, b in zip(rotated_q, decoded_k)) * scale)
                rows.append(decoded_v)
            if not rows:
                continue
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            denominator = sum(weights)
            out_base = q_base
            for channel in range(head_dim):
                output[out_base + channel] = (
                    sum(weight * row[channel] for weight, row in zip(weights, rows)) / denominator
                )
    return output


def dense_full_precision(k: list[float], v: list[float], q: list[float], locations: list[int],
                         indices: list[list[int]], lengths: list[int], slots: int, kv_heads: int,
                         query_heads: int, head_dim: int, scale: float) -> list[float]:
    rows: dict[tuple[int, int], tuple[list[float], list[float]]] = {}
    for token, slot in enumerate(locations):
        for head in range(kv_heads):
            base = (token * kv_heads + head) * head_dim
            rows[slot, head] = ([half(value) for value in k[base : base + head_dim]],
                                [half(value) for value in v[base : base + head_dim]])
    output = [0.0] * (len(lengths) * query_heads * head_dim)
    for query_index, length in enumerate(lengths):
        for query_head in range(query_heads):
            q_base = (query_index * query_heads + query_head) * head_dim
            rotated_q = [half(value) for value in q[q_base : q_base + head_dim]]
            kv_head = query_head // (query_heads // kv_heads)
            selected = [rows[indices[query_index][token], kv_head] for token in range(length)]
            if not selected:
                continue
            scores = [sum(a * b for a, b in zip(rotated_q, row[0])) * scale for row in selected]
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            denominator = sum(weights)
            for channel in range(head_dim):
                output[q_base + channel] = (
                    sum(weight * row[1][channel] for weight, row in zip(weights, selected)) / denominator
                )
    return output


class Runtime:
    def __init__(self, path: str) -> None:
        self.lib = ctypes.CDLL(path)
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        self.lib.spry_uq_hip_abi_version.restype = integer
        self.lib.spry_uq_last_error.restype = ctypes.c_char_p
        self.lib.spry_uq_cpu_write.argtypes = [pointer] * 6 + [integer] * 4
        self.lib.spry_uq_cpu_write.restype = integer
        self.lib.spry_uq_cpu_attention.argtypes = [pointer] * 7 + [integer] * 6 + [ctypes.c_float]
        self.lib.spry_uq_cpu_attention.restype = integer

    def error(self) -> str:
        return self.lib.spry_uq_last_error().decode("utf-8")


@unittest.skipUnless(LIBRARY, "set SPRY_UQ_CPU_LIBRARY to the explicitly built CPU library")
class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = Runtime(LIBRARY)
        if cls.runtime.lib.spry_uq_hip_abi_version() != 1:
            raise RuntimeError("unexpected native runtime ABI")

    @staticmethod
    def fixture() -> tuple[int, int, int, int, int, list[int], list[float], list[float], list[float], list[list[int]], list[int]]:
        slots, kv_heads, query_heads, head_dim, tokens = 6, 2, 4, 64, 4
        locations = [4, 1, 3, 0]  # Paged physical-slot order, not token order.
        k = [((index * 37) % 29 - 14) / 37.0 for index in range(tokens * kv_heads * head_dim)]
        v = [((index * 19) % 31 - 15) / 41.0 for index in range(tokens * kv_heads * head_dim)]
        q = [((index * 11) % 23 - 11) / 29.0 for index in range(2 * query_heads * head_dim)]
        indices = [[4, 1, 3, 0], [1, 3, 0, 5]]
        lengths = [4, 3]
        return slots, kv_heads, query_heads, head_dim, tokens, locations, k, v, q, indices, lengths

    def run_write(self, locations: list[int], k: list[float], v: list[float], slots: int,
                  kv_heads: int, head_dim: int) -> tuple[ctypes.Array, ctypes.Array, ctypes.Array]:
        row_bytes = 17 * (head_dim // 32)
        compressed_k = (ctypes.c_uint8 * (slots * kv_heads * row_bytes))()
        compressed_v = (ctypes.c_uint8 * (slots * kv_heads * row_bytes))()
        valid = (ctypes.c_uint8 * slots)()
        k_array = (ctypes.c_float * len(k))(*k)
        v_array = (ctypes.c_float * len(v))(*v)
        location_array = (ctypes.c_int64 * len(locations))(*locations)
        status = self.runtime.lib.spry_uq_cpu_write(compressed_k, compressed_v, valid, k_array, v_array,
                                                    location_array, len(locations), slots, kv_heads, head_dim)
        self.assertEqual(status, 0, self.runtime.error())
        return compressed_k, compressed_v, valid

    def test_real_producer_consumer_matches_independent_decoded_attention(self) -> None:
        (slots, kv_heads, query_heads, head_dim, tokens, locations, k, v, q, indices, lengths) = self.fixture()
        compressed_k, compressed_v, valid = self.run_write(locations, k, v, slots, kv_heads, head_dim)
        self.assertEqual(list(valid), [1, 1, 0, 1, 1, 0])
        self.assertEqual(len(compressed_k), slots * kv_heads * 34)
        index_array = (ctypes.c_int64 * 8)(*(value for row in indices for value in row))
        length_array = (ctypes.c_int64 * len(lengths))(*lengths)
        query_array = (ctypes.c_float * len(q))(*q)
        output = (ctypes.c_float * len(q))()
        status = self.runtime.lib.spry_uq_cpu_attention(
            compressed_k, compressed_v, valid, query_array, index_array, length_array, output,
            slots, len(lengths), 4, query_heads, kv_heads, head_dim, ctypes.c_float(1.0 / 8.0),
        )
        self.assertEqual(status, 0, self.runtime.error())
        expected = dense_attention_from_encoded(bytes(compressed_k), bytes(compressed_v), q, indices, lengths,
                                                slots, kv_heads, query_heads, head_dim, 1.0 / 8.0)
        self.assertEqual(len(output), len(expected))
        for actual, reference in zip(output, expected):
            self.assertLessEqual(abs(actual - reference), 2e-5 + 2e-5 * abs(reference))

        full_precision = dense_full_precision(k, v, q, locations, indices, lengths, slots, kv_heads,
                                             query_heads, head_dim, 1.0 / 8.0)
        rms = math.sqrt(sum((actual - reference) ** 2 for actual, reference in zip(output, full_precision)) /
                        len(output))
        self.assertLessEqual(rms, 0.15)

    def test_write_is_atomic_for_duplicate_and_nonfinite_inputs(self) -> None:
        slots, kv_heads, head_dim = 3, 1, 64
        row_bytes = 34
        compressed_k = (ctypes.c_uint8 * (slots * row_bytes))(*([0xA5] * (slots * row_bytes)))
        compressed_v = (ctypes.c_uint8 * (slots * row_bytes))(*([0x5A] * (slots * row_bytes)))
        valid = (ctypes.c_uint8 * slots)(0, 0, 0)
        values = [0.25] * (2 * kv_heads * head_dim)
        locations = (ctypes.c_int64 * 2)(1, 1)
        status = self.runtime.lib.spry_uq_cpu_write(compressed_k, compressed_v, valid,
                                                    (ctypes.c_float * len(values))(*values),
                                                    (ctypes.c_float * len(values))(*values), locations,
                                                    2, slots, kv_heads, head_dim)
        self.assertEqual(status, 1)
        self.assertEqual(bytes(compressed_k), bytes([0xA5] * len(compressed_k)))
        self.assertEqual(bytes(compressed_v), bytes([0x5A] * len(compressed_v)))
        self.assertEqual(list(valid), [0, 0, 0])
        values[-1] = math.inf
        locations = (ctypes.c_int64 * 2)(1, 2)
        status = self.runtime.lib.spry_uq_cpu_write(compressed_k, compressed_v, valid,
                                                    (ctypes.c_float * len(values))(*values),
                                                    (ctypes.c_float * len(values))(*values), locations,
                                                    2, slots, kv_heads, head_dim)
        self.assertEqual(status, 4)
        self.assertEqual(bytes(compressed_k), bytes([0xA5] * len(compressed_k)))
        self.assertEqual(list(valid), [0, 0, 0])

    def check_attention(self, buffers, q, indices, lengths, slots, kvh, qh, d):
        width = len(indices[0])
        out = (ctypes.c_float * len(q))()
        status = self.runtime.lib.spry_uq_cpu_attention(
            *buffers, (ctypes.c_float * len(q))(*q),
            (ctypes.c_int64 * (len(indices) * width))(*(i for row in indices for i in row)),
            (ctypes.c_int64 * len(lengths))(*lengths), out,
            slots, len(lengths), width, qh, kvh, d, 1 / math.sqrt(d),
        )
        self.assertEqual(status, 0, self.runtime.error())
        expected = dense_attention_from_encoded(
            bytes(buffers[0]), bytes(buffers[1]), q, indices, lengths,
            slots, kvh, qh, d, 1 / math.sqrt(d),
        )
        for actual, reference in zip(out, expected):
            self.assertLessEqual(abs(actual - reference), 2e-5 + 2e-5 * abs(reference))
        return list(out)

    def test_target_dimensions_gqa_causal_prefill_append_reuse_and_eviction(self):
        for d, kvh, qh in ((64, 1, 4), (128, 2, 4), (256, 4, 16), (256, 4, 24)):
            with self.subTest(head_dim=d, kv_heads=kvh, query_heads=qh):
                rng = random.Random(2904 + d + qh)
                slots, locations = 8, [5, 1, 6]
                k = [rng.uniform(-1, 1) for _ in range(3 * kvh * d)]
                v = [rng.uniform(-1, 1) for _ in range(3 * kvh * d)]
                q = [rng.uniform(-1, 1) for _ in range(3 * qh * d)]
                buffers = self.run_write(locations[:2], k[:2 * kvh * d], v[:2 * kvh * d], slots, kvh, d)
                first = self.check_attention(buffers, q[:2 * qh * d],
                                             [[5, -99], [5, 1]], [1, 2], slots, kvh, qh, d)
                old_bytes = bytes(buffers[0]), bytes(buffers[1])
                # Append into a partial physical page; existing prefix stays encoded.
                status = self.runtime.lib.spry_uq_cpu_write(
                    *buffers, (ctypes.c_float * (kvh * d))(*k[2 * kvh * d:]),
                    (ctypes.c_float * (kvh * d))(*v[2 * kvh * d:]),
                    (ctypes.c_int64 * 1)(6), 1, slots, kvh, d,
                )
                self.assertEqual(status, 0, self.runtime.error())
                for old, new in zip(old_bytes, buffers[:2]):
                    self.assertEqual(old[:6 * kvh * 17 * (d // 32)], bytes(new)[:6 * kvh * 17 * (d // 32)])
                indices, lengths = [[5, 1, 6]] * 3, [1, 2, 3]
                full = self.check_attention(buffers, q, indices, lengths, slots, kvh, qh, d)
                self.assertEqual(first, full[:2 * qh * d])
                baseline = dense_full_precision(k, v, q, locations, indices, lengths, slots, kvh, qh, d, 1 / math.sqrt(d))
                rms = math.sqrt(sum((a - b)**2 for a, b in zip(full, baseline)) / len(full))
                self.assertLessEqual(rms, 0.15)
                self.assertEqual(full, self.check_attention(buffers, q, indices, lengths, slots, kvh, qh, d))
                # Eviction marks a slot unreadable; overwrite/reuse restores it.
                buffers[2][1] = 0
                out = (ctypes.c_float * (qh * d))()
                status = self.runtime.lib.spry_uq_cpu_attention(
                    *buffers, (ctypes.c_float * (qh * d))(*q[:qh * d]),
                    (ctypes.c_int64 * 1)(1), (ctypes.c_int64 * 1)(1), out,
                    slots, 1, 1, qh, kvh, d, 1 / math.sqrt(d),
                )
                self.assertEqual(status, 1)
                status = self.runtime.lib.spry_uq_cpu_write(
                    *buffers, (ctypes.c_float * (kvh * d))(),
                    (ctypes.c_float * (kvh * d))(*([0.5] * (kvh * d))),
                    (ctypes.c_int64 * 1)(1), 1, slots, kvh, d,
                )
                self.assertEqual(status, 0)
                reused = self.check_attention(buffers, q[:qh*d], [[1]], [1], slots, kvh, qh, d)
                # .156*.5 -> E=-4, .5/2^-4 clips to code6: .375.
                self.assertTrue(all(value == 0.375 for value in reused))

    def test_empty_mask_and_large_logits_use_stable_softmax(self):
        d = 64
        buffers = self.run_write([0, 1], [256.0] * (2*d), [1.0] * d + [2.0] * d, 2, 1, d)
        q = [256.0] * d
        result = self.check_attention(buffers, q, [[0, 1]], [2], 2, 1, 1, d)
        # Constant V=1 and V=2 encode to .75 and 1.5, respectively.
        self.assertTrue(all(value == 1.125 for value in result))
        result = self.check_attention(buffers, q, [[-999, -999]], [0], 2, 1, 1, d)
        self.assertEqual(result, [0.0] * d)

    def test_bad_lengths_indices_gqa_scale_and_half_overflow_are_rejected(self):
        buffers = self.run_write([0], [0.0] * 64, [0.5] * 64, 2, 1, 64)
        q, out = (ctypes.c_float * 128)(), (ctypes.c_float * 128)(*([123.0] * 128))
        cases = [(-1, 0, 1, 1, 0.125), (2, 0, 1, 1, 0.125),
                 (1, -1, 1, 1, 0.125), (1, 2, 1, 1, 0.125),
                 (1, 0, 1, 0, 0.125), (1, 0, 1, 2, 0.125),
                 (1, 0, 1, 1, math.nan), (1, 0, 1, 1, 0.0)]
        for length, index, qh, kvh, scale in cases:
            status = self.runtime.lib.spry_uq_cpu_attention(
                *buffers, q, (ctypes.c_int64 * 1)(index), (ctypes.c_int64 * 1)(length), out,
                2, 1, 1, qh, kvh, 64, scale,
            )
            self.assertNotEqual(status, 0)
            self.assertEqual(list(out), [123.0] * 128)
        buffers[0][16] = 141
        status = self.runtime.lib.spry_uq_cpu_attention(
            *buffers, q, (ctypes.c_int64 * 1)(0), (ctypes.c_int64 * 1)(1), out, 2, 1, 1, 1, 1, 64, 0.125,
        )
        self.assertEqual(status, 5)

    def test_raw_abi_rejects_extreme_shape_before_dereference(self) -> None:
        byte = (ctypes.c_uint8 * 1)(0)
        value = (ctypes.c_float * 1)(0.0)
        location = (ctypes.c_int64 * 1)(0)
        status = self.runtime.lib.spry_uq_cpu_write(
            byte, byte, byte, value, value, location, 1, 2**31 - 1, 2**31 - 1, 64
        )
        self.assertEqual(status, 1)

    def test_attention_rejects_uninitialized_and_malformed_cache_rows(self) -> None:
        slots, kv_heads, query_heads, head_dim, _, locations, k, v, q, _, _ = self.fixture()
        compressed_k, compressed_v, valid = self.run_write(locations, k, v, slots, kv_heads, head_dim)
        query = (ctypes.c_float * (query_heads * head_dim))(*q[: query_heads * head_dim])
        output = (ctypes.c_float * len(query))()
        indices = (ctypes.c_int64 * 1)(2)
        lengths = (ctypes.c_int64 * 1)(1)
        status = self.runtime.lib.spry_uq_cpu_attention(compressed_k, compressed_v, valid, query, indices, lengths,
                                                        output, slots, 1, 1, query_heads, kv_heads, head_dim, 0.125)
        self.assertEqual(status, 1)
        valid[2] = 1
        compressed_k[(2 * kv_heads) * 34 + 16] = 255
        status = self.runtime.lib.spry_uq_cpu_attention(compressed_k, compressed_v, valid, query, indices, lengths,
                                                        output, slots, 1, 1, query_heads, kv_heads, head_dim, 0.125)
        self.assertEqual(status, 5)


if __name__ == "__main__":
    unittest.main()
