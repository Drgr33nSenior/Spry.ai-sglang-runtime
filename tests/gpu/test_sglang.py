"""Owner-run SGLang producer/consumer qualification without model weights.

The gate is deliberately parsed before torch or SGLang imports. The fixture
uses a real patched ``ModelRunner._get_attention_backend`` dispatch, a real
SGLang ``RadixAttention`` layer, and a physical paged slot table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-authorized-gpu", action="store_true")
    parser.add_argument("--spry-uq-library", required=True)
    return parser.parse_args()


def _server_args(library: str) -> SimpleNamespace:
    disabled = SimpleNamespace(backend="disabled")
    return SimpleNamespace(
        spry_uq_backend="spry_uq_rdna4", spry_uq_library_path=library,
        cuda_graph_config=SimpleNamespace(decode=disabled, prefill=disabled),
        tp_size=1, pp_size=1, dp_size=1, dcp_size=1,
        enable_dp_attention=False, speculative_algorithm=None,
        disaggregation_mode="null", cpu_offload_gb=0,
        enable_hierarchical_cache=False, enable_lmcache=False,
        enable_pdmux=False, enable_two_batch_overlap=False,
        enable_mixed_chunk=False, is_embedding=False,
        prefill_only_disable_kv_cache=False, enable_memory_saver=False,
        enable_page_major_kv_layout=False, enable_unified_memory=False,
        kv_cache_dtype="auto", attention_backend="triton",
        prefill_attention_backend=None, decode_attention_backend=None,
    )


def _decode(encoded, head_dim, torch):
    """Independent FP4 E2M1 decode of the persistent byte representation."""
    magnitudes = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=torch.float32
    )
    groups = head_dim // 32
    data = encoded.detach().cpu().to(torch.int16).reshape(*encoded.shape[:-1], groups, 17)
    packed = data[..., :16]
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(*data.shape[:-1], 32)
    scales = torch.pow(2.0, data[..., 16].to(torch.float32) - 127.0).unsqueeze(-1)
    values = magnitudes[(codes & 7).long()] * scales
    return (torch.where((codes & 8) != 0, -values, values).to(torch.float16).float()
            .reshape(*encoded.shape[:-1], head_dim))


def _rotate_q(q, torch):
    values = q.float()
    width = 1
    while width < values.shape[-1]:
        blocks = values.reshape(*values.shape[:-1], -1, 2 * width)
        left, right = blocks[..., :width], blocks[..., width:]
        values = torch.cat((left + right, left - right), dim=-1).reshape_as(values)
        width *= 2
    return (values / math.sqrt(values.shape[-1])).to(torch.float16).float()


def _attention(q, keys, values, scale, torch, rotate):
    """Dense CPU oracle; ``keys`` and ``values`` have shape [tokens, heads, D]."""
    query = _rotate_q(q, torch) if rotate else q.float()
    output = torch.empty_like(query)
    q_heads, kv_heads = query.shape[0], keys.shape[1]
    for q_head in range(q_heads):
        kv_head = q_head // (q_heads // kv_heads)
        scores = (keys[:, kv_head] * query[q_head]).sum(dim=-1) * scale
        output[q_head] = torch.softmax(scores, dim=0) @ values[:, kv_head]
    return output


def _batch(mode, locations, sequence_length, prefix_length, torch):
    return SimpleNamespace(
        batch_size=1, forward_mode=mode, out_cache_loc=locations,
        req_pool_indices=torch.tensor([0], device="cuda", dtype=torch.int64),
        extend_seq_lens_cpu=[sequence_length - prefix_length],
        extend_prefix_lens_cpu=[prefix_length], seq_lens_cpu=[sequence_length],
        seq_lens=torch.tensor([sequence_length], device="cuda", dtype=torch.int32),
    )


def _qkv(tokens, torch):
    # Split views intentionally reproduce common fused-QKV non-contiguity.
    fused = torch.randn((tokens, 8 * 64), device="cuda", dtype=torch.float16)
    q, k, v = fused.split((4 * 64, 2 * 64, 2 * 64), dim=-1)
    return q, k.view(tokens, 2, 64), v.view(tokens, 2, 64)


def main() -> int:
    args = _arguments()
    if not args.owner_authorized_gpu:
        raise SystemExit("owner authorization required: pass --owner-authorized-gpu")
    library = Path(args.spry_uq_library).resolve(strict=True)

    import torch
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import KVCache
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner
    from spry_uq.sglang_backend import SpryUQAttentionBackend
    from spry_uq.sglang_pool import SpryUQTokenToKVPool

    if not torch.version.hip:
        raise SystemExit("requires a HIP PyTorch build")
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    if arch.split(":", 1)[0] != "gfx1201":
        raise SystemExit(f"requires gfx1201; observed {arch or 'unknown'}")

    torch.manual_seed(17)
    pool = SpryUQTokenToKVPool(
        size=64, page_size=4, dtype=torch.float16, head_num=2, head_dim=64,
        v_head_dim=64, layer_num=1, device="cuda", enable_memory_saver=False,
        library_path=str(library),
    )
    assert isinstance(pool, KVCache)
    assert pool.k_buffer[0].dtype is torch.uint8 and pool.v_buffer[0].dtype is torch.uint8
    allocator = PagedTokenToKVPoolAllocator(
        64, page_size=4, dtype=torch.float16, device="cuda", kvcache=pool, need_sort=False
    )
    allocated = allocator.alloc(36)
    assert allocated is not None and allocated.numel() == 36
    table = allocated.reshape(1, 36).to(torch.int32)
    server_args = _server_args(str(library))
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        head_dim=64, v_head_dim=64, context_len=36,
    )
    runner = SimpleNamespace(
        server_args=server_args, dtype=torch.float16, gpu_id=0,
        model_config=model_config, token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        is_draft_worker=False, use_mla_backend=False, mambaish_config=None,
        is_hybrid_swa=False,
    )
    # This is the patched production selector, not a handcrafted backend.
    backend = ModelRunner._get_attention_backend(runner)
    assert isinstance(backend, SpryUQAttentionBackend)
    assert server_args.requested_spry_uq_backend == "spry_uq_rdna4"
    assert server_args.effective_spry_uq_backend == "spry_uq_rdna4"
    assert server_args.spry_uq_allocated_bytes == sum(cache.allocated_bytes for cache in pool.caches)

    layer = RadixAttention(4, 64, 1.0 / 8.0, 2, 0)
    try:
        pool.get_kv_buffer(0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a dense KV writer was allowed to access compressed bytes")
    for invalid_indices in (allocated[:1].float(), allocated[:1].reshape(1, 1)):
        try:
            pool.invalidate(invalid_indices)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed slot-index dtype or rank was accepted")
    history = {}
    q, k, v = _qkv(33, torch)
    initial_q = q.detach().cpu().reshape(33, 4, 64)
    initial_output = backend.forward(
        q, k, v, layer, _batch(ForwardMode.EXTEND, table[0, :33], 33, 0, torch)
    )
    for index, slot in enumerate(table[0, :33].detach().cpu().tolist()):
        history[slot] = (k[index].detach().cpu().float(), v[index].detach().cpu().float())

    q, k, v = _qkv(1, torch)
    continuation_q = q.detach().cpu().reshape(4, 64)
    continuation_output = backend.forward(
        q, k, v, layer, _batch(ForwardMode.EXTEND, table[0, 33:34], 34, 33, torch)
    )
    history[int(table[0, 33].item())] = (k[0].detach().cpu().float(), v[0].detach().cpu().float())

    q, k, v = _qkv(1, torch)
    output = backend.forward(q, k, v, layer, _batch(ForwardMode.DECODE, table[0, 34:35], 35, 34, torch))
    history[int(table[0, 34].item())] = (k[0].detach().cpu().float(), v[0].detach().cpu().float())
    torch.cuda.synchronize()
    cache = pool._cache(0)
    assert cache.write_calls == 3 and cache.attention_calls == 4
    assert cache.allocated_bytes == cache.k.nbytes + cache.v.nbytes + cache.valid.nbytes
    assert all(bool(cache.valid[slot].item()) for slot in table[0, :35])

    slots = table[0, :35].detach().cpu().tolist()
    decoded_k, decoded_v = _decode(cache.k[slots], 64, torch), _decode(cache.v[slots], 64, torch)
    for token in (0, 32):
        expected = _attention(
            initial_q[token], decoded_k[: token + 1], decoded_v[: token + 1],
            layer.scaling, torch, rotate=True,
        )
        assert torch.allclose(
            initial_output[token].detach().cpu().reshape(4, 64).float(), expected,
            atol=2e-3, rtol=2e-3,
        )
    continuation_reference = _attention(
        continuation_q, decoded_k[:34], decoded_v[:34], layer.scaling, torch, rotate=True
    )
    assert torch.allclose(
        continuation_output.detach().cpu().reshape(4, 64).float(), continuation_reference,
        atol=2e-3, rtol=2e-3,
    )
    q_heads = q.reshape(1, 4, 64).detach().cpu()[0]
    decoded_reference = _attention(q_heads, decoded_k, decoded_v, layer.scaling, torch, rotate=True)
    actual = output.detach().cpu().reshape(4, 64).float()
    assert torch.allclose(actual, decoded_reference, atol=2e-3, rtol=2e-3)
    full_k = torch.stack([history[slot][0] for slot in slots])
    full_v = torch.stack([history[slot][1] for slot in slots])
    full_reference = _attention(q_heads, full_k, full_v, layer.scaling, torch, rotate=False)
    rms = torch.mean((actual - full_reference) ** 2).sqrt()
    assert float(rms) <= 0.15, f"quantized/full-precision RMS={float(rms):.6f}"

    # The final cache page is only partly used. Freeing its tail releases the
    # whole page, so the already-written sibling must be invalidated too.
    final_page = allocated[-4:]
    sibling = final_page[0]
    assert bool(cache.valid[sibling].item())
    allocator.free(final_page[1:])
    torch.cuda.synchronize()
    assert not bool(cache.valid[sibling].item())
    reused = allocator.alloc(4)
    assert reused is not None and reused.numel() == 4
    assert torch.equal(reused, final_page)
    table[0, :4] = reused.to(torch.int32)
    q, k, v = _qkv(4, torch)
    reused_output = backend.forward(
        q, k, v, layer, _batch(ForwardMode.EXTEND, reused.to(torch.int32), 4, 0, torch)
    )
    assert all(bool(cache.valid[slot].item()) for slot in reused)
    reused_k = _decode(cache.k[reused], 64, torch)
    reused_v = _decode(cache.v[reused], 64, torch)
    reused_reference = _attention(
        q.detach().cpu().reshape(4, 4, 64)[-1], reused_k, reused_v,
        layer.scaling, torch, rotate=True,
    )
    assert torch.allclose(
        reused_output[-1].detach().cpu().reshape(4, 64).float(), reused_reference,
        atol=2e-3, rtol=2e-3,
    )

    rejected = SimpleNamespace(**runner.__dict__)
    rejected.server_args = SimpleNamespace(**server_args.__dict__)
    rejected.server_args.tp_size = 2
    try:
        SpryUQAttentionBackend(rejected)
    except ValueError:
        pass
    else:
        raise AssertionError("tensor parallelism was not refused")
    rejected.server_args.tp_size = 1
    rejected.server_args.cuda_graph_config = SimpleNamespace(
        decode=SimpleNamespace(backend="disabled"), prefill=SimpleNamespace(backend="triton")
    )
    try:
        SpryUQAttentionBackend(rejected)
    except ValueError:
        pass
    else:
        raise AssertionError("graph capture was not refused")
    print(json.dumps({
        "schema_version": 1,
        "fixture": {"head_dim": 64, "query_heads": 4, "kv_heads": 2,
                    "page_size": 4, "prefill_tokens": 33, "model_weights": "none"},
        "cache": cache.evidence(),
        "final_decode_max_implementation_absolute_error": float(
            (actual - decoded_reference).abs().max()
        ),
        "final_decode_quantized_vs_dense_rms_error": float(rms),
        "qualification": "unqualified",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
