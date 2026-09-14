#!/usr/bin/env python3
"""Verify a pinned source checkout and prepare a new local patched copy.

No networking, installation, GPU probing, or modification of the input tree.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(source, *args):
    return subprocess.run(
        ["git", "-C", str(source), *args], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_patches(lock):
    patches = []
    for name in lock["sglang"]["patches"]:
        patch = ROOT / name
        expected = lock["sglang"]["patch_sha256"].get(name)
        if not patch.is_file() or digest(patch) != expected:
            raise ValueError(f"maintained patch SHA-256 mismatch: {name}")
        patches.append(patch)
    return patches


def verify_source(source, lock):
    source = source.resolve(strict=True)
    expected = lock["sglang"]
    if git(source, "rev-parse", "HEAD").decode().strip() != expected["commit"]:
        raise ValueError("SGLang revision mismatch")
    if git(source, "rev-parse", "HEAD^{tree}").decode().strip() != expected["tree"]:
        raise ValueError("SGLang tree mismatch")
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("SGLang input must be pristine, including untracked files")
    git(source, "fsck", "--full", "--no-reflogs")
    archive_sha = hashlib.sha256(git(source, "archive", "HEAD")).hexdigest()
    if archive_sha != expected["git_archive_sha256"]:
        raise ValueError("SGLang source archive SHA-256 mismatch")
    for name, expected_sha in lock["installer_baseline"]["matching_public_source_files"].items():
        if digest(source / name) != expected_sha:
            raise ValueError(f"SGLang source file mismatch: {name}")
    return source


def validate_output(source, output):
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a new directory; existing artifacts are preserved")
    output = output.resolve()
    if source.resolve() in output.parents:
        raise ValueError("output must not be inside the pristine input checkout")
    if "site-packages" in output.parts or "dist-packages" in output.parts:
        raise ValueError("preparation into installed Python packages is refused")
    return output


def prepare(source, output):
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    source = verify_source(source, lock)
    output = validate_output(source, output)
    patches = verify_patches(lock)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A separate Git root makes patch paths unambiguous. --no-hardlinks avoids
    # shared mutable object files; this operation has no remote side effects.
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", "--no-checkout",
                    str(source), str(output)], check=True)
    git(output, "-c", "advice.detachedHead=false", "checkout", "--detach", lock["sglang"]["commit"])
    # Restore public source provenance on this generated checkout only.
    git(output, "remote", "set-url", "origin", lock["sglang"]["origin"])
    for patch in patches:
        git(output, "apply", "--check", "--whitespace=error-all", str(patch))
        git(output, "apply", "--whitespace=error-all", str(patch))
    destination = output / "python" / "spry_uq"
    if destination.exists():
        raise ValueError("upstream unexpectedly contains the overlay package")
    shutil.copytree(ROOT / "python" / "spry_uq", destination,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Use upstream's maintained non-CUDA package convention; do not resolve or
    # install dependencies here. Original metadata remains available upstream.
    shutil.copyfile(output / "python" / "pyproject_other.toml", output / "python" / "pyproject.toml")
    receipt = {
        "schema_version": 1,
        "qualification": "unqualified",
        "source": lock["sglang"],
        "source_lock_sha256": digest(ROOT / "sources.lock.json"),
        "patch_sha256": {p.name: digest(p) for p in patches},
        "overlay_sha256": {str(p.relative_to(destination)): digest(p)
                           for p in sorted(destination.rglob("*.py"))},
        "package_metadata_sha256": digest(output / "python" / "pyproject.toml"),
        "gpu_execution": "NOT RUN",
        "sglang_execution": "NOT RUN",
    }
    with (output / "spry-source.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(f"Prepared pinned experimental source: {output}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_only:
            lock = json.loads((ROOT / "sources.lock.json").read_text())
            verify_source(args.source, lock)
            verify_patches(lock)
            print("Pinned SGLang source and maintained patch integrity verified")
        elif args.output is None:
            parser.error("--output is required unless --verify-only is selected")
        else:
            prepare(args.source, args.output)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        # Do not echo arbitrary remote URLs or subprocess environment details.
        if isinstance(error, subprocess.CalledProcessError):
            parser.exit(1, f"source preparation failed: Git command exited {error.returncode}; artifacts retained\n")
        parser.exit(1, f"source preparation failed: {error}\n")


if __name__ == "__main__":
    main()
