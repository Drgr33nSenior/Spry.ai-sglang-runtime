# Runtime evidence and future Bridge integration

Bridge remains GPU-free. This repository owns native compression, cache
allocation and SGLang dispatch. The installer owns its locks, images, deployment
and qualification policy. Neither reference repository is changed here.

Use `spry-source.json` from source preparation for source origin/revision,
source-lock hash, applied patch hashes, overlay hashes and package metadata.
Each runtime cache's `evidence()` returns codec, requested backend, effective
implementation, device index, compiled source identity, library SHA-256, layout,
actual allocated tensor bytes and successful native-call counters. This is
bounded process-local evidence, not an inference endpoint or approval record.
Attach the exact model/configuration identity and dependency inventory collected
by the owner. The cache cannot infer those from tensor shapes.

Keep these levels separate in the private result:

| Level | Evidence required |
| --- | --- |
| Reference/CPU correctness | Build command, permanent test outcomes and numerical bounds |
| gfx1201 compilation | Successful real compiler result and native artifact identity |
| Generated-code inspection | Extracted gfx1201 code object, ISA, register/scratch observations |
| GPU numerical execution | Owner-authorized executed kernel comparison and allocation measurements |
| SGLang integration execution | Actual pool allocation/write/backend dispatch and prefix/decode tests |
| Model/performance qualification | Matched baseline/candidate workloads and all quality/resource criteria |

Use supported/refused for the implementation's explicit capability decisions;
unknown for missing observations; unqualified for the candidate until the
owner's acceptance process is complete. An unexecuted test remains NOT RUN.
Preparation receipts never promote their fixed NOT RUN fields. Runtime counters
prove only completed native calls, not numerical correctness or quality.

Bridge's existing private performance bundle schema 1 can carry a bounded
`comparison` or `profile-status` result with `spec.json`, source/artifact hashes
and qualification state. Its allowlisted tools do not include this runtime.
Use the current owner-provisioned evidence ID plus digest workflow; do not add
an arbitrary path, GPU execution command, CGO binding or proxy endpoint.
Prompts, tokens, full traces, environment values and raw model outputs must
remain in private evidence, outside public status records.

Minimal follow-up work before installer/Bridge adoption:

1. Establish a complete reproducible source and dependency relationship for a
   selected candidate image; preserve upstream notices and record its digest.
2. Add and qualify hybrid-state integration for the selected installer models,
   and separately implement/qualify TP if dual-GPU support is wanted.
3. Complete the owner's hardware/model acceptance procedure and preserve
   separate per-GPU and host-memory accounting.
4. Extend the consumer contract only if structured codec/backend parsing or a
   typed runtime selection is needed. Keep absent measurements unknown and
   invalidate evidence when source/model/device/process identity changes.

Do not change the installer lock or present this overlay as a compatible
replacement merely because three source files match or its build succeeds.
