"""Strict opt-in SGLang attention backend for ``spry-uq-e2m1-v1``.

The implementation is intentionally narrow.  It uses SGLang's existing
request-to-physical-slot table and the native compressed-cache ABI; it never
materializes an FP16 persistent KV cache or delegates unsupported work to an
ordinary attention backend.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from spry_uq import BACKEND as BACKEND_NAME
from spry_uq import CODEC_VERSION, IMPLEMENTATION
from spry_uq.sglang_pool import SUPPORTED_HEAD_DIMS, SpryUQTokenToKVPool


logger = logging.getLogger(__name__)


def _backend_is_disabled(value: Any) -> bool:
    return str(getattr(value, "value", value)).lower() == "disabled"


def _phase_is_disabled(config: Any, name: str) -> bool:
    phase = getattr(config, name, None)
    return phase is not None and _backend_is_disabled(getattr(phase, "backend", None))


def _blocked_settings(server_args: Any) -> Iterable[str]:
    checks = (
        ("tensor parallelism", getattr(server_args, "tp_size", 1) != 1),
        ("pipeline parallelism", getattr(server_args, "pp_size", 1) != 1),
        ("data parallelism", getattr(server_args, "dp_size", 1) != 1),
        ("data-parallel attention", getattr(server_args, "enable_dp_attention", False)),
        ("decode context parallelism", getattr(server_args, "dcp_size", 1) != 1),
        ("speculative decoding", getattr(server_args, "speculative_algorithm", None) is not None),
        ("PD disaggregation", getattr(server_args, "disaggregation_mode", "null") != "null"),
        ("CPU KV offload", getattr(server_args, "cpu_offload_gb", 0) != 0),
        ("hierarchical cache", getattr(server_args, "enable_hierarchical_cache", False)),
        ("LMCache", getattr(server_args, "enable_lmcache", False)),
        ("PDMux", getattr(server_args, "enable_pdmux", False)),
        ("two-batch overlap", getattr(server_args, "enable_two_batch_overlap", False)),
        ("mixed chunk scheduling", getattr(server_args, "enable_mixed_chunk", False)),
        ("embedding mode", getattr(server_args, "is_embedding", False)),
        ("prefill-only cache disable", getattr(server_args, "prefill_only_disable_kv_cache", False)),
        ("memory saver", getattr(server_args, "enable_memory_saver", False)),
        ("page-major KV layout", getattr(server_args, "enable_page_major_kv_layout", False)),
        ("unified memory pool", getattr(server_args, "enable_unified_memory", False)),
        ("non-auto KV cache dtype", getattr(server_args, "kv_cache_dtype", "auto") != "auto"),
        (
            "conflicting attention backend selection",
            any(
                value not in (None, "triton")
                for value in (
                    getattr(server_args, "attention_backend", None),
                    getattr(server_args, "prefill_attention_backend", None),
                    getattr(server_args, "decode_attention_backend", None),
                )
            ),
        ),
    )
    for name, blocked in checks:
        if blocked:
            yield name


def validate_spry_uq_server_args(server_args: Any) -> None:
    """Reject unsupported modes before model execution or cache allocation."""

    requested = getattr(server_args, "spry_uq_backend", "disabled")
    if requested == "disabled":
        return
    if requested != BACKEND_NAME:
        raise ValueError(f"Unknown Spry-UQ backend {requested!r}.")
    if not getattr(server_args, "spry_uq_library_path", None):
        raise ValueError(
            "--spry-uq-backend requires --spry-uq-library-path. The experimental "
            "overlay does not discover a library or modify site-packages."
        )
    config = getattr(server_args, "cuda_graph_config", None)
    if config is None or not (
        _phase_is_disabled(config, "decode") and _phase_is_disabled(config, "prefill")
    ):
        raise ValueError(
            "Spry-UQ requires both CUDA/HIP graph phases disabled. Start a fresh "
            "process after changing KV-cache representation."
        )
    blocked = list(_blocked_settings(server_args))
    if blocked:
        raise ValueError(
            "Spry-UQ supports only single-GPU ordinary causal attention; refused: "
            + ", ".join(blocked)
            + "."
        )


def validate_spry_uq_model_runner(model_runner: Any) -> None:
    """Validate model, dtype, device and layout after SGLang resolves them."""

    validate_spry_uq_server_args(model_runner.server_args)
    if getattr(model_runner, "is_draft_worker", False):
        raise ValueError("Spry-UQ does not support speculative draft workers.")
    if model_runner.dtype is not torch.float16:
        raise ValueError("Spry-UQ requires an FP16 model runtime.")
    if not torch.version.hip:
        raise ValueError("Spry-UQ is a HIP-only experimental backend.")
    props = torch.cuda.get_device_properties(model_runner.gpu_id)
    arch = str(getattr(props, "gcnArchName", ""))
    if arch.split(":", 1)[0] != "gfx1201":
        raise ValueError(
            "Spry-UQ is restricted to AMD RDNA4 gfx1201; detected " f"{arch or 'unknown'}.")
    config = model_runner.model_config
    architectures = tuple(getattr(config.hf_config, "architectures", ()) or ())
    allowed = {"LlamaForCausalLM", "Qwen3ForCausalLM"}
    if len(architectures) != 1 or architectures[0] not in allowed:
        raise ValueError(
            "Spry-UQ supports only ordinary causal LlamaForCausalLM or "
            "Qwen3ForCausalLM attention; refused architectures="
            f"{architectures!r}."
        )
    if getattr(model_runner, "use_mla_backend", False):
        raise ValueError("Spry-UQ does not support MLA cache layouts.")
    if getattr(model_runner, "mambaish_config", None) is not None:
        raise ValueError("Spry-UQ does not support recurrent or hybrid state models.")
    if getattr(model_runner, "is_hybrid_swa", False):
        raise ValueError("Spry-UQ does not support sliding-window attention pools.")
    if config.head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError(
            "Spry-UQ supports head dimensions 64, 128, and 256; "
            f"received {config.head_dim}."
        )
    if getattr(config, "v_head_dim", config.head_dim) != config.head_dim:
        raise ValueError("Spry-UQ requires equal K and V head dimensions.")
    model_runner.server_args.effective_spry_uq_backend = BACKEND_NAME


class SpryUQAttentionBackend:
    """Causal MHA/GQA backend consuming encoded cache storage directly."""

    needs_cpu_seq_lens = True
    supports_cuda_graph = False
    _extend_chunk = 32

    def __init__(self, model_runner: Any) -> None:
        validate_spry_uq_model_runner(model_runner)
        if not isinstance(model_runner.token_to_kv_pool, SpryUQTokenToKVPool):
            raise RuntimeError(
                "Spry-UQ attention requires its compressed KV pool; refusing an "
                "implicit dense-cache fallback."
            )
        self.pool = model_runner.token_to_kv_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.max_context_len = model_runner.model_config.context_len
        allocated_bytes = sum(cache.allocated_bytes for cache in self.pool.caches)
        model_runner.server_args.requested_spry_uq_backend = BACKEND_NAME
        model_runner.server_args.effective_spry_uq_backend = BACKEND_NAME
        model_runner.server_args.spry_uq_codec_version = CODEC_VERSION
        model_runner.server_args.spry_uq_qualification = "unqualified"
        model_runner.server_args.spry_uq_allocated_bytes = allocated_bytes
        logger.warning(
            "Spry-UQ experimental backend selected requested=%s effective=%s "
            "implementation=%s codec=%s storage=uint8 persistent_bytes=%d "
            "qualification=unqualified",
            BACKEND_NAME,
            BACKEND_NAME,
            IMPLEMENTATION,
            CODEC_VERSION,
            allocated_bytes,
        )

    def init_forward_metadata(self, forward_batch: Any) -> None:
        # SGLang calls this hook before each eager forward.  The slot table is
        # already owned and updated by its scheduler; no replacement metadata is
        # created here because graph capture is explicitly unavailable.
        return None

    def init_cuda_graph_state(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Spry-UQ explicitly refuses CUDA/HIP graph capture.")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Any,
        forward_batch: Any,
        save_kv_cache: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        if (
            any(value is not None for value in kwargs.values())
            or layer.is_cross_attention
            or layer.sliding_window_size > -1
            or layer.logit_cap != 0
            or layer.attn_type.value != "decoder"
            or layer.pos_encoding_mode != "NONE"
            or layer.use_irope
            or layer.xai_temperature_len != -1
            or layer.k_scale is not None
            or layer.v_scale is not None
            or layer.quant_method is not None
        ):
            raise ValueError("Spry-UQ supports only ordinary causal self-attention.")
        if not save_kv_cache or k is None or v is None:
            raise ValueError("Spry-UQ requires direct FP16 K/V cache writes for every forward.")
        if q.dtype is not torch.float16 or k.dtype is not torch.float16 or v.dtype is not torch.float16:
            raise ValueError("Spry-UQ attention requires FP16 Q/K/V tensors.")
        if q.device != k.device or q.device != v.device or not q.is_cuda:
            raise ValueError("Spry-UQ requires Q/K/V on the same HIP device.")
        if forward_batch.forward_mode.is_idle():
            return q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        mode = forward_batch.forward_mode
        if mode is ForwardMode.DECODE:
            return self.forward_decode(q, k, v, layer, forward_batch)
        if mode is ForwardMode.EXTEND:
            return self.forward_extend(q, k, v, layer, forward_batch)
        raise ValueError(f"Spry-UQ refuses forward mode {mode!r}.")

    @staticmethod
    def _host_lengths(value: Any, name: str) -> list[int]:
        if value is None:
            raise RuntimeError(f"Spry-UQ requires {name} host metadata in eager mode.")
        return [int(x) for x in value]

    def _write(self, layer: Any, loc: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        self.pool.set_kv_buffer(layer, loc, k, v)

    def forward_decode(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Any, forward_batch: Any
    ) -> torch.Tensor:
        q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim).contiguous()
        if q.shape[0] != forward_batch.batch_size:
            raise ValueError("Spry-UQ decode expects exactly one query token per request.")
        self._write(layer, forward_batch.out_cache_loc, k, v)
        lengths_host = self._host_lengths(forward_batch.seq_lens_cpu, "seq_lens_cpu")
        lengths = forward_batch.seq_lens[: q.shape[0]].to(torch.int64).contiguous()
        max_len = max(lengths_host[: q.shape[0]], default=0)
        if max_len <= 0:
            raise ValueError("Spry-UQ decode received an empty causal cache.")
        rows = self.req_to_token.index_select(0, forward_batch.req_pool_indices[: q.shape[0]])
        indices = rows[:, :max_len].to(torch.int64).contiguous()
        out = self.pool._cache(layer.layer_id).attention(q, indices, lengths, layer.scaling)
        return out.reshape(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Any, forward_batch: Any
    ) -> torch.Tensor:
        q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim).contiguous()
        self._write(layer, forward_batch.out_cache_loc, k, v)
        extend_lens = self._host_lengths(forward_batch.extend_seq_lens_cpu, "extend_seq_lens_cpu")
        prefix_lens = self._host_lengths(forward_batch.extend_prefix_lens_cpu, "extend_prefix_lens_cpu")
        seq_lens = self._host_lengths(forward_batch.seq_lens_cpu, "seq_lens_cpu")
        if not (len(extend_lens) == len(prefix_lens) == len(seq_lens) == forward_batch.batch_size):
            raise ValueError("Spry-UQ extend metadata does not match the request batch.")
        if sum(extend_lens) != q.shape[0]:
            raise ValueError("Spry-UQ extend query count does not match extend lengths.")
        output = q.new_empty((q.shape[0], layer.tp_q_head_num, layer.v_head_dim))
        cache = self.pool._cache(layer.layer_id)
        token_offset = 0
        for request_index, (extend_len, prefix_len, final_len) in enumerate(
            zip(extend_lens, prefix_lens, seq_lens)
        ):
            if extend_len < 1 or prefix_len < 0 or final_len != prefix_len + extend_len:
                raise ValueError("Spry-UQ received inconsistent causal extend lengths.")
            row = self.req_to_token[forward_batch.req_pool_indices[request_index], :final_len]
            for chunk_start in range(0, extend_len, self._extend_chunk):
                count = min(self._extend_chunk, extend_len - chunk_start)
                query_slice = q[token_offset + chunk_start : token_offset + chunk_start + count]
                # Each query has a different causal length but the same physical
                # page-table prefix.  The compressed kernel observes ``lengths``.
                indices = row.unsqueeze(0).expand(count, -1).contiguous().to(torch.int64)
                lengths = torch.arange(
                    prefix_len + chunk_start + 1,
                    prefix_len + chunk_start + count + 1,
                    dtype=torch.int64,
                    device=q.device,
                )
                output[token_offset + chunk_start : token_offset + chunk_start + count] = cache.attention(
                    query_slice, indices, lengths, layer.scaling
                )
            token_offset += extend_len
        return output.reshape(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
