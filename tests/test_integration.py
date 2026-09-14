"""Static regression checks for the maintained SGLang overlay.

These checks run without torch or SGLang. They prove patch identity and wiring,
not SGLang execution; the owner-gated GPU test covers the real classes.
"""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "0001-spry-uq.patch"


class SGLangOverlayStaticTests(unittest.TestCase):
    def test_patch_is_pinned_and_strict(self) -> None:
        patch = PATCH.read_text(encoding="utf-8")
        self.assertIn("0b3bb0cbe31873994c9f989fddfe2f87ca839fdd", patch)
        self.assertIn("spry_uq_backend", patch)
        self.assertIn("SpryUQAttentionBackend", patch)
        self.assertIn("SpryUQTokenToKVPool", patch)
        self.assertIn("2 * head_num * 17", patch)
        self.assertIn('getattr(self._kvcache, "invalidate", None)', patch)
        self.assertIn("allocator/paged.py", patch)

    def test_overlay_has_real_storage_and_attention_calls(self) -> None:
        pool = (ROOT / "python" / "spry_uq" / "sglang_pool.py").read_text(
            encoding="utf-8"
        )
        backend = (ROOT / "python" / "spry_uq" / "sglang_backend.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("class SpryUQTokenToKVPool(KVCache)", pool)
        self.assertIn("CompressedCache(", pool)
        self.assertIn(".write(loc, cache_k.contiguous(), cache_v.contiguous())", pool)
        self.assertIn("cache.move(", pool)
        self.assertIn("def invalidate(self, loc: torch.Tensor)", pool)
        self.assertIn(".attention(q, indices, lengths, layer.scaling)", backend)
        self.assertIn("req_to_token", backend)
        self.assertIn("CUDA/HIP graph capture", backend)
        self.assertIn("requested_spry_uq_backend", backend)
        self.assertIn("ForwardMode.EXTEND", backend)


if __name__ == "__main__":
    unittest.main()
