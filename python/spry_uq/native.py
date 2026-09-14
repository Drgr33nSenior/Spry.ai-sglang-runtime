"""Explicit ctypes binding for the GPU-free UltraQuant CPU reference library."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable


class CodecError(RuntimeError):
    """A rejected codec request, including its stable native status code."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"spry_uq status {status}: {message}")
        self.status = status


class CpuCodec:
    """CPU-only binding; constructing it never loads a HIP or GPU library."""

    def __init__(self, library_path: str | os.PathLike[str] | None = None) -> None:
        candidate = library_path or os.environ.get("SPRY_UQ_CPU_LIBRARY")
        if not candidate:
            raise ValueError(
                "pass library_path or set SPRY_UQ_CPU_LIBRARY to the CPU library"
            )
        self.path = Path(candidate)
        self._library = ctypes.CDLL(str(self.path))
        self._configure()

    def _configure(self) -> None:
        lib = self._library
        lib.spry_uq_encoded_bytes.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        lib.spry_uq_encoded_bytes.restype = ctypes.c_int
        lib.spry_uq_encode_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ]
        lib.spry_uq_encode_f32.restype = ctypes.c_int
        lib.spry_uq_decode_f32.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
        ]
        lib.spry_uq_decode_f32.restype = ctypes.c_int
        lib.spry_uq_rotate_wht_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.spry_uq_rotate_wht_f32.restype = ctypes.c_int
        lib.spry_uq_status_string.argtypes = [ctypes.c_int]
        lib.spry_uq_status_string.restype = ctypes.c_char_p

    def _check(self, status: int) -> None:
        if status != 0:
            message = self._library.spry_uq_status_string(status).decode("utf-8")
            raise CodecError(status, message)

    @staticmethod
    def _validate_dimension(dimension: int) -> None:
        if dimension not in (64, 128, 256):
            raise CodecError(2, "unsupported dimension")

    def encoded_bytes(self, dimension: int) -> int:
        self._validate_dimension(dimension)
        result = ctypes.c_size_t()
        self._check(self._library.spry_uq_encoded_bytes(dimension, ctypes.byref(result)))
        return result.value

    def encode(self, values: Iterable[float]) -> bytes:
        source = tuple(values)
        output_bytes = self.encoded_bytes(len(source))
        input_array = (ctypes.c_float * len(source))(*source)
        output = (ctypes.c_uint8 * output_bytes)()
        self._check(self._library.spry_uq_encode_f32(
            input_array, len(source), output, len(output)
        ))
        return bytes(output)

    def decode(self, encoded: bytes | bytearray, dimension: int) -> list[float]:
        self._validate_dimension(dimension)
        if len(encoded) != self.encoded_bytes(dimension):
            raise CodecError(3, "incorrect buffer size")
        input_array = (ctypes.c_uint8 * len(encoded)).from_buffer_copy(encoded)
        output = (ctypes.c_float * dimension)()
        self._check(self._library.spry_uq_decode_f32(
            input_array, len(input_array), dimension, output
        ))
        return list(output)

    def rotate(self, values: Iterable[float]) -> list[float]:
        source = tuple(values)
        self._validate_dimension(len(source))
        input_array = (ctypes.c_float * len(source))(*source)
        output = (ctypes.c_float * len(source))()
        self._check(self._library.spry_uq_rotate_wht_f32(
            input_array, len(source), output
        ))
        return list(output)
