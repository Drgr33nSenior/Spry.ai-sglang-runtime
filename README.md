# Spry SGLang runtime experiment

An opt-in UltraQuant-derived KV-cache backend for gfx1201. The source implements
compressed allocation, quantizing writes, and attention that reads those same
compressed bytes. The initial HIP kernel unpacks one token at a time to FP16
and accumulates in FP32. It is a correctness-first scalar implementation.

This is a separate experimental runtime. It is **unqualified** on Radeon AI PRO
R9700. CPU numerical tests do not establish HIP compilation, SGLang execution,
model quality, memory savings, or speed. No GPU work is part of the default tests.

The maintained code is small C++/HIP and Python, plus one patch against SGLang
`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` (`v0.5.15.post1`). Preparation verifies
the complete upstream archive before applying the patch into a new checkout.
The upstream repository, licences and notices remain in that checkout.
Nothing modifies an installed SGLang package.

## Supported slice

- One gfx1201 GPU; FP16 runtime; ordinary causal Llama/Qwen3 MHA or GQA.
- Head dimensions 64, 128 and 256; contiguous per-token K/V, equal K/V dimensions.
- Prefill, continuation-prefill and decode using SGLang physical-slot indices;
  the attention consumer uses a separate causal prefix length for every query.
- Grouped E2M1 storage with 17 bytes per 32 channels, plus validity metadata.
  Persistent full-precision cache storage is not allocated.
- Explicit selection and library path; fresh process/cache for each format.
  The unchanged upstream backend remains the default.

Hybrid/recurrent models, MLA, sliding windows, cross attention, graphs,
speculative decoding, offload, KV transfer, tensor parallelism and multiple GPUs
are refused. Fused writers that request a dense KV tensor view are also refused.
The installer's two selected Qwen3.5-family models are hybrid and
are **not supported as complete models** by this first slice. Their full
attention dimension (256) and GQA ratios (4 and 6) inform native test fixtures.
All Q/K/V inputs must be finite with absolute value at most 256. These are
explicit numerical and implementation limits, not model qualification results.

## CPU tests and CLion

Use an already available C++17 compiler, CMake >=3.24, Ninja and Python >=3.11.
Python tests use only the standard library. No dependency installation occurs.

```sh
cmake --preset cpu -DPython3_EXECUTABLE=/absolute/path/to/python3
cmake --build --preset cpu
ctest --preset cpu
```

Open `CMakeLists.txt` in CLion and select the `cpu` preset. Set the interpreter
and local compiler in your own toolchain settings. This project does not change
`.idea` files. On macOS, CLion's bundled CMake/CTest can be used by absolute path
when they are absent from PATH. `CMakeUserPresets.json` is ignored for local
ROCm/compiler paths. No remote or SSH toolchain is activated automatically.

## HIP build and source integration

The [build guide](docs/build.md) gives explicit compile-only and source preparation
commands. A missing HIP compiler makes the HIP configuration fail. The
[codec specification](docs/codec.md) fixes rounding, layout and numerical
tolerances. The [source notes](docs/sources.md) distinguish the public source
from the installer's AMD image.

Only the owner may run the optional GPU tests and
[qualification procedure](docs/qualification.md). Keep results in private
evidence. Build success, dispatch counters and workload qualification are
separate observations. [Bridge handoff](docs/handoff.md) describes the bounded
evidence fields and remaining consumer work; Bridge remains GPU-free.

## Ownership and licences

`include/` and `native/` own the codec and runtime kernels. `python/spry_uq/`
owns bindings and SGLang adapters. `patches/` changes only the pinned SGLang
integration points. `tests/` contains permanent CPU regression tests and
explicit owner-run GPU checks. `sources.lock.json` records external identities.

The original MIT [LICENSE](LICENSE) is unchanged. SGLang is Apache-2.0; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This project does not relicense
SGLang or claim author implementation equivalence.
