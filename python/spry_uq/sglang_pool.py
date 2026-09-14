"""SGLang KV-cache adapter for the experimental Spry UltraQuant codec."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from sglang.srt.mem_cache.memory_pool import KVCache
from spry_uq import CODEC_VERSION
from spry_uq.runtime import CompressedCache


SUPPORTED_HEAD_DIMS = frozenset((64, 128, 256))


class SpryUQTokenToKVPool(KVCache):
    """Compressed K/V byte tensors for ordinary causal MHA layers only."""

    spry_uq_codec_version = CODEC_VERSION

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        library_path: Optional[str] = None,
        **_: Any,
    ) -> None:
        if any(type(value) is not int or value <= 0 for value in (size, page_size, layer_num)):
            raise ValueError("Spry-UQ requires positive integer size, page size and layer count.")
        if dtype is not torch.float16:
            raise ValueError(f"Spry-UQ accepts FP16 Q/K/V only; received dtype={dtype}.")
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError(
                "Spry-UQ supports head dimensions 64, 128, and 256; "
                f"received {head_dim}."
            )
        if v_head_dim is not None and v_head_dim != head_dim:
            raise ValueError(
                "Spry-UQ requires equal key and value head dimensions; "
                f"received key={head_dim}, value={v_head_dim}."
            )
        if not str(device).startswith("cuda"):
            raise ValueError(f"Spry-UQ requires a HIP CUDA-alike device, not {device!r}.")
        if not library_path:
            raise ValueError(
                "Spry-UQ requires an explicit --spry-uq-library-path; "
                "the overlay never searches or changes site-packages."
            )
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        # ``dtype`` remains the FP16 compute contract. SGLang metadata must
        # describe the actual persistent bytes rather than an FP16 cache.
        self.store_dtype = torch.uint8
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = head_dim
        self.library_path = library_path
        # Preserve SGLang's allocator range, including its padded write slots.
        self.slot_count = size + page_size
        self.caches = [
            CompressedCache(
                slots=self.slot_count,
                kv_heads=head_num,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
                library_path=library_path,
            )
            for _ in range(layer_num)
        ]
        # These are persistent encoded bytes, never expanded FP16 cache views.
        self.k_buffer = [cache.k for cache in self.caches]
        self.v_buffer = [cache.v for cache in self.caches]
        self.valid_buffer = [cache.valid for cache in self.caches]
        self._finalize_allocation_log(size)

    def _cache(self, layer_id: int) -> CompressedCache:
        index = layer_id - self.start_layer
        if index < 0 or index >= len(self.caches):
            raise IndexError(
                f"Spry-UQ layer {layer_id} is outside "
                f"[{self.start_layer}, {self.start_layer + len(self.caches)})."
            )
        return self.caches[index]

    @staticmethod
    def _indices(loc: torch.Tensor) -> torch.Tensor:
        if not isinstance(loc, torch.Tensor) or loc.ndim != 1 or loc.dtype not in (
            torch.int32, torch.int64
        ):
            raise ValueError("Spry-UQ locations require a rank-1 int32 or int64 tensor.")
        return loc.to(torch.int64).contiguous()

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise RuntimeError(
            "Spry-UQ encoded K storage has no writable dense tensor view; "
            "use the Spry-UQ attention backend."
        )

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise RuntimeError(
            "Spry-UQ encoded V storage has no writable dense tensor view; "
            "use the Spry-UQ attention backend."
        )

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "Spry-UQ refuses fused dense KV writers because its persistent "
            "storage is encoded bytes."
        )

    def get_kv_buffer_shape(self):
        cache = self._cache(self.start_layer)
        return cache.k.shape, cache.v.shape

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        # Validity belongs to the K/V allocation as a whole; charge it once.
        key_bytes = sum(
            int(cache.k.nbytes) + int(cache.valid.nbytes) for cache in self.caches
        )
        value_bytes = sum(int(cache.v.nbytes) for cache in self.caches)
        return key_bytes, value_bytes

    def set_kv_buffer(
        self,
        layer: Any,
        loc_info: Any,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        **_: Any,
    ) -> None:
        if k_scale is not None or v_scale is not None:
            raise ValueError("Spry-UQ does not accept a second KV-cache scale.")
        loc = self._indices(getattr(loc_info, "loc", loc_info))
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        expected = (loc.numel(), self.head_num, self.head_dim)
        if tuple(cache_k.shape) != expected or tuple(cache_v.shape) != expected:
            raise ValueError(
                f"Spry-UQ requires K/V shape {expected}; got "
                f"K={tuple(cache_k.shape)}, V={tuple(cache_v.shape)}."
            )
        if cache_k.dtype is not torch.float16 or cache_v.dtype is not torch.float16:
            raise ValueError("Spry-UQ cache writes require FP16 K and V tensors.")
        # SGLang QKV split views can be non-contiguous. These bounded write
        # intermediates are transient and are not reported as KV payload bytes.
        self._cache(layer_id).write(loc, cache_k.contiguous(), cache_v.contiguous())

    def set_kv_buffer_prefix_valid(
        self,
        layer: Any,
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        layer_id_override: Optional[int] = None,
        **_: Any,
    ) -> None:
        if loc_2d.ndim != 2 or commit_lens.ndim != 1:
            raise ValueError("Spry-UQ prefix writes require rank-2 locations and rank-1 lengths.")
        if loc_2d.dtype not in (torch.int32, torch.int64) or commit_lens.dtype not in (
            torch.int32, torch.int64
        ):
            raise ValueError("Spry-UQ prefix locations and lengths require integer tensors.")
        if loc_2d.device != commit_lens.device:
            raise ValueError("Spry-UQ prefix locations and lengths require the same device.")
        if loc_2d.shape[0] != commit_lens.shape[0]:
            raise ValueError("Spry-UQ prefix lengths do not match location rows.")
        if ((commit_lens < 0) | (commit_lens > loc_2d.shape[1])).any().item():
            raise ValueError("Spry-UQ prefix lengths are outside the supplied location rows.")
        if cache_k.shape[0] != loc_2d.numel() or cache_v.shape[0] != loc_2d.numel():
            raise ValueError("Spry-UQ prefix K/V rows do not match locations.")
        columns = torch.arange(loc_2d.shape[1], device=loc_2d.device)
        mask = columns.unsqueeze(0) < commit_lens.to(torch.int64).unsqueeze(1)
        selected = mask.reshape(-1).nonzero(as_tuple=False).flatten()
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        self._cache(layer_id).write(
            loc_2d.reshape(-1).index_select(0, selected).to(torch.int64).contiguous(),
            cache_k.index_select(0, selected).contiguous(),
            cache_v.index_select(0, selected).contiguous(),
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        target, source = self._indices(tgt_loc), self._indices(src_loc)
        for cache in self.caches:
            cache.move(target, source)

    def invalidate(self, loc: torch.Tensor) -> None:
        """Clear validity when SGLang releases physical slots for reuse."""
        indices = self._indices(loc)
        for cache in self.caches:
            cache.invalidate(indices)

    def get_cpu_copy(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Spry-UQ refuses CPU KV offload and cache export.")

    def load_cpu_copy(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Spry-UQ refuses CPU KV offload and cache import.")

    def get_contiguous_buf_infos(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Spry-UQ refuses KV transfer/disaggregation buffers.")
