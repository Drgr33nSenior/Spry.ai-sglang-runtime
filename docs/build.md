# Build and source preparation

## Preconditions

CPU checks require Python >=3.11, a C++17 compiler, CMake >=3.24 and Ninja.
They do not import Torch, initialize HIP, read kubeconfig, or use a GPU opt-in
environment variable. GPU checks require a separate explicit owner action.

The intended HIP compiler input is the installer's locked ROCm 10.0.0 gfx120X
distribution. Its origin and digest are in `sources.lock.json`. Use an already
prepared toolchain. Do not download or install that distribution with this
project's scripts. An upstream version tag does not identify the AMD image's
entire dependency set.

## Compile only

```sh
cmake --preset cpu -DPython3_EXECUTABLE=/absolute/path/to/python3
cmake --build --preset cpu
ctest --preset cpu

cmake --preset gfx1201 \
  -DCMAKE_HIP_COMPILER=/absolute/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/absolute/rocm \
  -DPython3_EXECUTABLE=/absolute/path/to/python3
cmake --build --preset gfx1201
```

The HIP target is exactly `gfx1201`. CMake must find a real HIP compiler and
HIP package configuration; there is no stub library or successful no-op target.
The library is `build/hip/libspry_uq_hip.so` on Linux. The build requires no
Torch headers or C++ Torch ABI. Python calls a small validated C ABI through
ctypes. CMake embeds a source/compiler/target identity in each library; record
the actual library SHA-256 as well, since a source identity is not a binary hash.

For generated-code inspection, use the prepared ROCm LLVM tools on the compiled
HIP object, without launching it:

```sh
/absolute/rocm/llvm/bin/llvm-objdump --offloading build/hip/libspry_uq_hip.so
/absolute/rocm/llvm/bin/llvm-objdump --disassemble-all --offloading build/hip/libspry_uq_hip.so
```

LLVM packaging varies: if this command lists only the host ELF, use that
toolchain's `clang-offload-bundler --list` / `--unbundle` on the HIP object and
disassemble the extracted gfx1201 code object. Retain compiler version, build
commands, code-object architecture and ISA privately. Inspect scratch/LDS and
register usage. A missing device disassembly is NOT RUN, not proof of a native
matrix instruction. This source intentionally requests neither scaled MFMA nor
native FP4 matrix execution.

## Prepare SGLang

Source retrieval is explicit and separate from preparation:

```sh
git clone --depth 1 --branch v0.5.15.post1 --single-branch \
  https://github.com/sgl-project/sglang.git .sources/upstream
python3 tools/prepare_sglang.py --source .sources/upstream --verify-only
python3 tools/prepare_sglang.py --source .sources/upstream --output .sources/candidate
```

Verification requires the exact revision/tree, a pristine input checkout,
Git object verification, and the locked SHA-256 of `git archive HEAD`.
It also checks the three installer source hashes and the maintained patch's
locked SHA-256. Update that hash deliberately when maintaining the patch;
preparation does not accept an unrecorded edit. A changed tag, modified
source, mismatched patch or existing output directory fails. Failed outputs
are retained; use a new output name after investigating. Preparation does not
install packages or touch site-packages.

The generated checkout uses upstream `pyproject_other.toml`, including its
`srt_hip` dependency convention. This project adds no production Python
dependency. The upstream metadata is pinned by the source revision but includes
some version ranges. A complete reproducible executable environment therefore
also requires an owner-recorded dependency inventory/artifact hashes from the
prepared environment; it is not provided by resolving those ranges afresh.
The installer image reports Python 3.14, Torch 2.12.0 and ROCm 10.0.0. These
are baseline build metadata, not observed local packages or qualification.

Use the generated checkout through `PYTHONPATH` with already prepared SGLang
HIP dependencies. Do not replace the installed package in place. Before owner
execution, verify imports resolve inside `.sources/candidate/python/sglang`
and retain `spry-source.json`, dependency versions, native build identity, binary
hash, model/config hashes and exact launch settings. `spry-source.json` records
preparation only; its NOT RUN fields must not be interpreted as execution.

## Owner-only synthetic execution

The optional test command and launch example are in
[qualification.md](qualification.md). They require explicit owner authorization.
Normal CTest never runs them. Missing GPU dependencies produce a failure or
NOT RUN status; they cannot become a successful GPU test.
