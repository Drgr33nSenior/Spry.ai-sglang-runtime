"""Torch allocation and validated producer/consumer dispatch to the HIP library.

There is deliberately no JIT build, device fallback, or import-time GPU probe.
"""

import ctypes
import hashlib
import math
import os
from pathlib import Path

from . import BACKEND, CODEC_VERSION, IMPLEMENTATION


class CompressedCache:
    """One attention layer, one device, and one immutable cache representation."""

    def __init__(self, slots, kv_heads, head_dim, device, dtype=None, library_path=None):
        import torch

        self._torch = torch
        for name, value in (("slots", slots), ("kv_heads", kv_heads)):
            if type(value) is not int or not 0 < value <= 2**30:
                raise ValueError(f"{name} must be a positive bounded integer")
        if head_dim not in (64, 128, 256):
            raise ValueError("supported head dimensions: 64, 128, 256")
        if dtype is not None and dtype != torch.float16:
            raise ValueError("spry_uq_rdna4 requires float16 Q/K/V")
        self.device = torch.device(device)
        if self.device.type != "cuda" or torch.version.hip is None:
            raise ValueError("spry_uq_rdna4 requires a ROCm PyTorch CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        props = torch.cuda.get_device_properties(self.device)
        arch = getattr(props, "gcnArchName", "").split(":")[0]
        if arch != "gfx1201":
            raise ValueError(f"spry_uq_rdna4 requires gfx1201; observed {arch!r}")
        self._check_capture()
        path = library_path or os.environ.get("SPRY_UQ_HIP_LIBRARY")
        if not path:
            raise ValueError("set SPRY_UQ_HIP_LIBRARY to the prebuilt HIP library")
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("HIP library must be a regular file")
        self.library_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self._lib = ctypes.CDLL(str(path))
        self._lib.spry_uq_hip_abi_version.restype = ctypes.c_int
        if self._lib.spry_uq_hip_abi_version() != 1:
            raise RuntimeError("unsupported spry HIP library ABI")
        self._lib.spry_uq_build_target.restype = ctypes.c_char_p
        self._lib.spry_uq_build_identity.restype = ctypes.c_char_p
        if self._lib.spry_uq_build_target() != b"gfx1201":
            raise RuntimeError("loaded library was not built for gfx1201")
        self.build_identity = self._lib.spry_uq_build_identity().decode("ascii")
        self._lib.spry_uq_last_error.restype = ctypes.c_char_p
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self._lib.spry_uq_hip_write.argtypes = [ptr] * 6 + [integer] * 4 + [ptr, integer]
        self._lib.spry_uq_hip_write.restype = integer
        self._lib.spry_uq_hip_attention.argtypes = (
            [ptr] * 7 + [integer] * 6 + [ctypes.c_float, ptr, integer]
        )
        self._lib.spry_uq_hip_attention.restype = integer
        self.slots, self.kv_heads, self.head_dim = slots, kv_heads, head_dim
        self._layout = (slots, kv_heads, head_dim)
        self.row_bytes = 17 * (head_dim // 32)
        if slots * kv_heads * self.row_bytes > 2**40:
            raise ValueError("cache extent exceeds the supported address bound")
        if slots * kv_heads * head_dim > 2**31 - 1:
            raise ValueError("cache extent exceeds the native experimental index range")
        with torch.cuda.device(self.device):
            self.k = torch.zeros((slots, kv_heads, self.row_bytes), dtype=torch.uint8, device=self.device)
            self.v = torch.zeros_like(self.k)
            self.valid = torch.zeros(slots, dtype=torch.uint8, device=self.device)
            torch.cuda.current_stream(self.device).synchronize()
        self._poisoned = False
        self.write_calls = 0
        self.attention_calls = 0
        self.max_tracked_call_tensor_bytes = 0

    @property
    def allocated_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.k, self.v, self.valid))

    def _check_capture(self):
        with self._torch.cuda.device(self.device):
            if self._torch.cuda.is_current_stream_capturing():
                raise RuntimeError("spry_uq_rdna4 does not support graph capture")

    def _ready(self):
        if self._poisoned:
            raise RuntimeError("cache is unusable after a native failure; start a fresh process/cache")
        self._check_capture()
        if (self.slots, self.kv_heads, self.head_dim) != self._layout:
            raise RuntimeError("cache layout changed; start a fresh process/cache")
        shape = (self.slots, self.kv_heads, 17 * (self.head_dim // 32))
        self._tensor(self.k, self._torch.uint8, shape, "compressed K")
        self._tensor(self.v, self._torch.uint8, shape, "compressed V")
        self._tensor(self.valid, self._torch.uint8, (self.slots,), "validity")
        if self.k.data_ptr() == self.v.data_ptr():
            raise ValueError("K and V allocations must be distinct")

    def _tensor(self, value, dtype, shape, name):
        torch = self._torch
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch tensor")
        if value.device != self.device or value.dtype != dtype:
            raise ValueError(f"{name} has the wrong device or dtype")
        if tuple(value.shape) != tuple(shape) or not value.is_contiguous():
            raise ValueError(f"{name} has unsupported shape or strides")
        if value.requires_grad:
            raise ValueError("compressed cache is inference-only")

    def _locations(self, loc, name, unique=False):
        if not hasattr(loc, "ndim") or loc.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional tensor")
        self._tensor(loc, self._torch.int64, (loc.numel(),), name)
        if loc.numel() > 2**30:
            raise ValueError("too many locations")
        if loc.numel() and ((loc < 0).any().item() or (loc >= self.slots).any().item()):
            raise ValueError(f"{name} contains an out-of-range slot")
        if unique and self._torch.unique(loc).numel() != loc.numel():
            raise ValueError(f"{name} contains duplicate write slots")

    def _call(self, function, *args):
        with self._torch.cuda.device(self.device):
            stream = self._torch.cuda.current_stream(self.device)
            status = function(*args, stream.cuda_stream, self.device.index)
        if status:
            self._poisoned = True
            detail = self._lib.spry_uq_last_error()
            message = detail.decode("utf-8", errors="replace") if detail else "unknown native failure"
            raise RuntimeError(f"spry_uq_rdna4: {message}")

    def write(self, loc, k, v):
        self._ready()
        self._locations(loc, "locations", unique=True)
        shape = (loc.numel(), self.kv_heads, self.head_dim)
        self._tensor(k, self._torch.float16, shape, "K")
        self._tensor(v, self._torch.float16, shape, "V")
        if not loc.numel():
            return
        self._call(self._lib.spry_uq_hip_write, self.k.data_ptr(), self.v.data_ptr(),
                   self.valid.data_ptr(), k.data_ptr(), v.data_ptr(), loc.data_ptr(),
                   loc.numel(), self.slots, self.kv_heads, self.head_dim)
        self.write_calls += 1

    def attention(self, q, slot_indices, lengths, scale):
        self._ready()
        if not hasattr(q, "ndim") or q.ndim != 3:
            raise ValueError("Q must have shape [queries, query_heads, head_dim]")
        nq, qh, d = q.shape
        if nq > 2**30 or not 0 < qh <= 256 or qh % self.kv_heads or d != self.head_dim:
            raise ValueError("unsupported query shape or GQA relationship")
        if not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("attention scale must be finite and in (0, 1]")
        self._tensor(q, self._torch.float16, (nq, qh, d), "Q")
        if not hasattr(slot_indices, "ndim") or slot_indices.ndim != 2:
            raise ValueError("indices must have shape [queries, maximum_prefix]")
        width = slot_indices.shape[1]
        if not 0 < width <= self.slots:
            raise ValueError("invalid attention prefix extent")
        self._tensor(slot_indices, self._torch.int64, (nq, width), "indices")
        self._tensor(lengths, self._torch.int64, (nq,), "lengths")
        out = self._torch.empty_like(q)
        if nq:
            self._call(self._lib.spry_uq_hip_attention, self.k.data_ptr(), self.v.data_ptr(),
                       self.valid.data_ptr(), q.data_ptr(), slot_indices.data_ptr(),
                       lengths.data_ptr(), out.data_ptr(), self.slots, nq, width,
                       qh, self.kv_heads, d, scale)
            self.attention_calls += 1
        self.max_tracked_call_tensor_bytes = max(
            self.max_tracked_call_tensor_bytes,
            out.numel() * out.element_size() + slot_indices.numel() * 8 + lengths.numel() * 8,
        )
        return out

    def invalidate(self, loc):
        self._ready()
        self._locations(loc, "locations")
        self.valid[loc] = 0
        self._torch.cuda.current_stream(self.device).synchronize()

    def move(self, dst, src):
        self._ready()
        self._locations(dst, "destinations", unique=True)
        self._locations(src, "sources")
        if src.numel() != dst.numel():
            raise ValueError("move source/destination sizes differ")
        if src.numel() and not (self.valid[src] != 0).all().item():
            raise ValueError("move reads an unwritten slot")
        # Snapshot all sources before writes, including overlapping permutations.
        k, v, valid = self.k[src], self.v[src], self.valid[src]
        self.k[dst], self.v[dst], self.valid[dst] = k, v, valid
        self._torch.cuda.current_stream(self.device).synchronize()
        self.max_tracked_call_tensor_bytes = max(
            self.max_tracked_call_tensor_bytes, k.numel() + v.numel() + valid.numel()
        )

    def evidence(self):
        return {
            "schema_version": 1,
            "requested_backend": BACKEND,
            "effective_implementation": IMPLEMENTATION,
            "codec": CODEC_VERSION,
            "qualification": "unqualified",
            "architecture": "gfx1201",
            "device_index": self.device.index,
            "library_sha256": self.library_sha256,
            "build_identity": self.build_identity,
            "layout": {"slots": self.slots, "kv_heads": self.kv_heads, "head_dim": self.head_dim},
            "allocated_tensor_bytes": self.allocated_bytes,
            "max_tracked_call_tensor_bytes": self.max_tracked_call_tensor_bytes,
            "total_peak_temporary_bytes": None,
            "native_validation_status_bytes": 4,
            "kernel_scratch_bytes": None,
            "successful_write_calls": self.write_calls,
            "successful_attention_calls": self.attention_calls,
            "cache_usable": not self._poisoned,
            "allocator_reserved_bytes": None,
            "end_to_end_qualification": "NOT RUN",
        }
