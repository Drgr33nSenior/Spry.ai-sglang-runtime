# Owner-run qualification and rollback

No procedure here grants GPU, model-download, SSH, installer or deployment
authority. Run only within a separately authorized local test window, with
prepared tools and owner-staged model files. Do not inherit an environment
variable as authorization. Keep baseline and candidate artifacts and logs.

## 1. Source, compilation and synthetic numerical checks

Complete the [build guide](build.md). Retain the pinned source receipt, native
library SHA-256, embedded build identity, compiler version and device-code
inspection. Record build and numerical execution as separate levels. Use the
locked AMD image digest as the prepared dependency environment when available;
record an actual dependency inventory, not just its reported build versions.

After explicit GPU authorization, run the standalone synthetic test:

```sh
PYTHONPATH="$PWD/.sources/candidate/python" python3 -B tests/gpu/test_sglang.py \
  --owner-authorized-gpu \
  --spry-uq-library "$PWD/build/hip/libspry_uq_hip.so"
```

Without the CLI authorization flag, this program exits before importing Torch
or SGLang. It uses small synthetic tensors; it does not load model weights.
The prepared source must contain the maintained patch. A test failure is a
failure, not permission to change numerical tolerances or skip dispatch.

Check code/scale storage, producer and attention dispatch counters, actual
allocation, prefill/decode and prefix/append behavior. Compare GPU output with
dense attention using the same decoded quantized cache at atol=rtol=2e-3.
Separately report quantization error against unchanged dense attention. Reject
wrong devices/dtypes/layouts, malformed indices/scales, mixed formats and graphs.
GPU API validation uses synchronization and transient status storage; measure
its costs rather than excluding them from runtime comparisons.

## 2. Matched model runs

The current installer models are hybrid and refused. Do not substitute a
different model in a comparison and attribute the difference to this backend.
To qualify this initial slice, the owner must select an already staged,
supported dense Llama/Qwen3 configuration and fix its revision/config/weight
hashes. No generated model code or real tool actions may execute.

Run an unchanged dense-cache baseline and candidate as separate fresh processes,
one at a time, on the same physical GPU. Fix model weights, tokenizer, request
token IDs, seed, sampling, context, concurrency, dtype, memory budget, host
resources and warmup. Keep graph modes disabled in both for the controlled
comparison; separately retain the unchanged preferred baseline configuration
if its normal graph path matters to the product decision.

Candidate launch fragment, used with the prepared source and existing model:

```sh
PYTHONPATH="$PWD/.sources/candidate/python" python3 -m sglang.launch_server \
  --model-path /absolute/already-staged-supported-model \
  --dtype float16 --tp-size 1 --attention-backend triton \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  --spry-uq-backend spry_uq_rdna4 \
  --spry-uq-library-path "$PWD/build/hip/libspry_uq_hip.so"
```

The explicit experimental selector replaces the effective full-attention path;
the ordinary attention selector remains available to the unchanged baseline.
For the matched baseline, omit the two `--spry-uq-*` flags and use the same
remaining arguments. Capture requested and effective backend separately from
the cache's runtime evidence. Verify imported SGLang files and the native binary
match the prepared artifacts. Do not accept a loaded-library name as dispatch
evidence. Do not use `--trust-remote-code`.

Run at least two cold and two warm observations per configuration. Use new
private cache namespaces for cold runs and retain them; never delete an
existing model/compiler cache. Test short contexts first, then a fixed multi-turn
prefix-reuse workload with long enough contexts and concurrency to pressure the
cache while retaining owner-approved headroom. Preserve request order and token
budgets across runs. Include partial pages, continuation-prefill across chunk
boundaries, eviction and re-prefill. Report refusals and failed requests.

Compare existing quantized-cache backends that the exact baseline supports on
gfx1201 (for example FP8 only if source, range/scaling and execution have been
validated). Record unsupported options as refused/NOT RUN. Do not assume that
SGLang's existing FP4 pool or an FP8 checkpoint proves compatible attention
execution. This experiment does not change model-weight quantization.

## 3. Measurements and acceptance

Record before selecting a winner:

| Area | Required observations |
| --- | --- |
| Identity | GPU PCI/device/architecture, boot/driver/ROCm, source/patch/build/library hashes, model/config/tokenizer hashes, arguments, dependencies, seeds |
| Memory | Each GPU separately: allocated/reserved peak, persistent cache bytes including metadata/padding, temporary peaks, kernel scratch; host RSS/cgroup/pinned memory and shared memory separately |
| Runtime | TTFT, inter-token latency and p50/p95/p99, request latency, input/output throughput, completed/failed requests, cache hits, reuse, evictions, re-prefill and startup/warmup |
| Numerics | Same-decoded-value implementation errors; separate quantized-vs-dense errors, logits/divergence and non-finite counts |
| Quality | Representative retrieval exact match/accuracy plus fixed coding-answer assessment; retain prompts/results privately without executing generated code or tool actions |

Use allocator peak APIs, per-GPU telemetry and profiling to measure actual
temporary memory; `max_tracked_call_tensor_bytes` is only a partial accounting
of explicitly tracked tensors, not a total peak. Scalar thread arrays can spill
to GPU scratch. Native validation allocates a four-byte status value but its
allocator footprint can be larger. The current codec can clip a flat V=0.5
block to 0.375; synthetic random-fixture accuracy does not bound model errors.

Set application-quality and resource acceptance thresholds before the runs.
Report raw paired measurements and uncertainty. No speedup, safe memory
reduction, published-paper quality or deployment readiness follows from encoded
size arithmetic. Do not pool the two R9700 cards into one VRAM budget. This
single-GPU backend remains unavailable for TP=2 even after successful tests.

## 4. Rollback

Stop the candidate through normal owner process management, retaining its logs,
source, binary and caches. Start a fresh process using the unchanged baseline
source/image and arguments with the experimental selector omitted. Never reuse
the candidate's in-process KV representation or deserialize it as a dense cache.
No package replacement, global setting, installer patch or Bridge modification
is needed to roll back this local experiment.

Any source, compiler, driver, dependency, model, layout, backend or device change
invalidates the relevant qualification evidence. Successful report generation,
owner selection and workload qualification remain separate states.
