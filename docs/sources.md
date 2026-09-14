# Source relationship and implementation decisions

The initial repository revision is
`b4f00f55cf65df43fd951bd0608896f171e54871`, with only its MIT licence tracked.
The maintained source model is a pinned SGLang checkout plus a local overlay.
`sources.lock.json` records the exact commit, tree and archive SHA-256.

## Installer baseline

The read-only installer reference's current `versions.lock` is the baseline
authority. Its SHA-256 is recorded in this repository's lock. The image is
`rocm/sgl-dev:v0.5.15.post1-ubuntu24.04-py3.14-rocm10.0.0`, pinned by manifest,
index and config digests. Its documented install history identifies
`0.5.15.post1+amd.march.ge35a33b34a`.

The three locked source hashes for `server_args.py`,
`torch_compile_decoration.py`, and `qwen3_5.py` match the public SGLang revision
exactly. However, no reproducible complete mapping from the image source layer
to that upstream Git revision has been established. No full image was pulled,
assembled or executed here. The AMD image also reports different Torch build
metadata from the public default CUDA package metadata. This experiment is a
separate runtime, not an installer-compatible image replacement.

The pinned model config JSON files were retrieved without weights:

| Model | Full-attention shape | Other state | Runtime decision |
| --- | --- | --- | --- |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json) | D=256, Q=16, KV=4; 8 layers | 24 linear-attention layers, vision/gating | Refused as a complete model |
| [Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json) | D=256, Q=24, KV=4; 16 layers | 48 linear-attention layers, vision/gating; baseline TP=2 | Refused as a complete model |

The native D=256 tests exercise these full-attention head relationships. They
do not transform recurrent state into a KV cache or qualify either model.
The SGLang adapter accepts ordinary `LlamaForCausalLM` and `Qwen3ForCausalLM`
configurations only, subject to its other strict checks.

## Research boundary

Primary starting points were [UltraQuant v3](https://arxiv.org/html/2606.20474v3),
[AMD's RDNA4 matrix-core guide](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/),
[LLVM's AMDGPU guide](https://llvm.org/docs/AMDGPUUsage.html), the pinned
[SGLang source](https://github.com/sgl-project/sglang/tree/0b3bb0cbe31873994c9f989fddfe2f87ca839fdd),
and the [authors' AMD TurboQuant article](https://rocm.blogs.amd.com/artificial-intelligence/turboquant-vllm-agentic/README.html).
On 2026-09-14, the paper still described UltraQuant kernel release as future
work. Bounded searches of public author references and vLLM code found no
released UltraQuant implementation to pin. That is a bounded availability
finding, not proof that no code exists anywhere.

The codec uses the paper's storage ingredients; its full behavior is specified
in [codec.md](codec.md). This is neither the released Ultra-TQ codebook path
nor weight quantization. The paper's CDNA4 scaled-MFMA execution is not assumed
available on RDNA4. AMD documents gfx12 FP16 WMMA; this initial source uses
ordinary scalar HIP operations and software unpacking. The local Apple compiler
has no AMDGPU target, so no selected gfx1201 compiler/instruction combination
was verified locally. No paper performance or quality result is transferred to
this implementation.
