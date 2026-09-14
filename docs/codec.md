# Experimental codec: spry-uq-e2m1-v1

This specification fixes the numerical contract before kernel evaluation.
It describes a new UltraQuant-derived codec, not author-code equivalence.

## Provenance

[UltraQuant v3, sections 5 and A.2](https://arxiv.org/html/2606.20474v3)
specifies E2M1 KV codes, groups of 32, one UE8M0 scale byte per group,
`value = code * 2^(byte - 127)`, and
`E = round(log2(0.156 * absmax))`. Its CDNA4 execution uses scaled MFMA and
FP8 queries. Its quality experiments protect boundary layers. Ultra-TQ uses
different codebooks and is not this codec. Neither method quantizes weights.
The paper still describes the UltraQuant kernel release as future work.
The bounded author/public-repository search on 2026-09-14 found no released
UltraQuant encoder contract. No author implementation is copied here.

Packing, ties, zero/non-finite handling, exponent saturation, rotation signs,
and a complete RDNA4 execution contract are not established by released code.
The following choices are this project's specification.

## Storage and scalar arithmetic

The supported head dimensions are 64, 128 and 256. There are no dimension
tails. Each contiguous group of 32 has 16 code bytes followed by one scale
byte. Even-numbered channels occupy the low nibble. Bits 0–2 select
`[0, 0.5, 1, 1.5, 2, 3, 4, 6]`; bit 3 is the sign. All 16 patterns decode,
including negative zero. Encoders canonicalize quantized zero to positive zero.
Nearest-code rounding uses ties to the even magnitude-code index. Magnitudes
above six clip; values at or below the tie between zero and 0.5 round to zero.

Scale arithmetic uses the binary32 value of `0.156`. Round the base-two
logarithm to the nearest integer, ties to even, then clamp to [-127, 127].
Byte 255 is invalid. An all-zero group uses byte 127 and zero codes.
NaNs and infinities are rejected. Exponent-limit behavior and finite overflow
are tested by the native scalar codec; attention has a narrower input domain.
The scaled maximum is computed in binary32. Only an underflowed zero product
uses E=-127 directly; nonzero subnormal products use the normal logarithm rule.
The float decoder rejects code/scale pairs whose result exceeds finite FP32.
The CPU WHT refuses inputs with `D * maxabs > FLT_MAX` before changing output.

Each K or V head occupies `17 * D/32` bytes with no row padding. K and V
are separate contiguous uint8 tensors shaped `[slots, kv_heads, 17*D/32]`
per attention layer. Include every reserved slot, validity metadata, allocator
rounding, and transient tensor when reporting memory. Payload arithmetic alone
does not establish safe GPU memory savings.

## Rotation and attention adaptation

Apply the normalized Sylvester Walsh–Hadamard transform to each complete K
head after RoPE, and the identical transform to Q after RoPE. Construction is
the ordinary butterfly in increasing power-of-two order, divided by sqrt(D),
with no random signs or seeds. It is orthonormal and self-inverse. There is no
per-token norm folding. V remains in its original basis; this is a deliberate
asymmetric adaptation, not a claim about an unpublished author encoder.

The HIP path accepts contiguous FP16 Q/K/V with finite absolute values at
most 256. Reject larger values before writes or attention. Rotation is FP32;
rotated Q and decoded K/V are rounded to FP16 before multiplication. Products,
online stable softmax and output accumulation use FP32; output rounds to FP16.
The range limit prevents half overflow even after a D=256 rotation. Apply the
layer's positive finite attention scaling once to QK. Do not use FP8 or claim
native FP4 matrix execution. The kernel uses scalar HIP arithmetic and software
unpacking of one cache token at a time, with no expanded persistent cache.
Attention rejects scale bytes 141–255 before decoding; this conservative
restriction keeps every accepted FP4 code finite in half precision. The scalar
CPU codec accepts the larger finite FP32 subset described above.

GQA maps Q head `h` to KV head `h / (q_heads/kv_heads)`. Each query receives
an explicit ordered physical-slot list and a causal prefix length. These lists
can cover partial pages, prefix reuse and continuation-prefill. Duplicate write
slots, uninitialized reads, out-of-range slots and unsupported masks are errors.

## Numerical acceptance

Before evaluation: require bit-identical codec golden bytes and exhaustive code
decoding. Check WHT orthogonality against an independent dense sign matrix.
CPU attention exposes the FP32 result before final output rounding, so its
comparison with dense attention on the same decoded values uses absolute/relative
tolerance 2e-5. Comparing two already-rounded FP16 outputs at this tolerance
would turn tiny FP32 differences at a half rounding boundary into a false
implementation failure. GPU attention uses atol=2e-3, rtol=2e-3 to
account for half output and different FP32 accumulation order on tiny fixtures.
Do not increase these tolerances to resolve a failing implementation.

Separately report errors against unchanged dense full-precision attention.
For bounded deterministic small random fixtures, use an absolute RMS error
limit of 0.15 with unit-scale inputs; this is a regression bound, not a model
quality criterion. Golden single-token V cases isolate value reconstruction.
Neither codec round trips nor synthetic error bounds qualify model behavior.
