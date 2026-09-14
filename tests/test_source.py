"""Source and execution-boundary regression checks; no Torch dependency."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SourceBoundaryTests(unittest.TestCase):
    def test_output_cannot_modify_input_or_installed_packages(self):
        spec = importlib.util.spec_from_file_location("prepare_sglang", ROOT / "tools/prepare_sglang.py")
        prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare)
        with self.assertRaisesRegex(ValueError, "existing artifacts"):
            prepare.validate_output(ROOT, ROOT)
        with self.assertRaisesRegex(ValueError, "pristine input"):
            prepare.validate_output(ROOT, ROOT / "build/must-not-create-nested-source")
        with self.assertRaisesRegex(ValueError, "installed Python"):
            prepare.validate_output(ROOT, ROOT.parent / "site-packages/spry-must-not-create")

    def test_patch_hash_is_verified_and_mismatch_refused(self):
        spec = importlib.util.spec_from_file_location("prepare_sglang", ROOT / "tools/prepare_sglang.py")
        prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare)
        lock = json.loads((ROOT / "sources.lock.json").read_text())
        self.assertEqual(len(prepare.verify_patches(lock)), 1)
        changed = copy.deepcopy(lock)
        name = changed["sglang"]["patches"][0]
        changed["sglang"]["patch_sha256"][name] = "0" * 64
        with self.assertRaisesRegex(ValueError, "patch SHA-256 mismatch"):
            prepare.verify_patches(changed)

    def test_runtime_import_does_not_load_gpu_packages(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import spry_uq, spry_uq.native, spry_uq.runtime; "
            "assert 'torch' not in sys.modules; assert 'sglang' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT / "python")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_owner_gpu_gate_precedes_library_or_torch_loading(self):
        result = subprocess.run(
            [sys.executable, "-I", "-B", str(ROOT / "tests/gpu/test_sglang.py"),
             "--spry-uq-library", "/nonexistent/spry-library.so"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("owner authorization required", result.stderr)
        self.assertNotIn("No module named", result.stderr)

    def test_wrong_source_revision_is_refused_without_output(self):
        output = ROOT / "build" / "must-not-be-created-by-rejected-source"
        self.assertFalse(output.exists())
        result = subprocess.run(
            [sys.executable, "-I", "-B", str(ROOT / "tools/prepare_sglang.py"),
             "--source", str(ROOT), "--output", str(output)],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("revision mismatch", result.stderr)
        self.assertFalse(output.exists())

    def test_lock_preserves_separate_runtime_and_hybrid_refusal(self):
        lock = json.loads((ROOT / "sources.lock.json").read_text())
        self.assertEqual(len(lock["sglang"]["commit"]), 40)
        self.assertEqual(len(lock["sglang"]["git_archive_sha256"]), 64)
        self.assertEqual(lock["installer_baseline"]["full_source_relationship"], "unknown")
        for model in lock["models_inspected"]:
            self.assertEqual(model["head_dim"], 256)
            self.assertTrue(model["runtime_support"].startswith("refused-hybrid"))


if __name__ == "__main__":
    unittest.main()
