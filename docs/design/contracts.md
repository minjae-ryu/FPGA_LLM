# SmolLM2 C contracts (version 1)

Stage note (2026-09-23): the user authorized CUDA forward and a bounded CPU/GPU
speed comparison after correctness checks. The [CUDA plan](cuda-forward.md)
extends this CPU baseline. FP32 CUDA uses the same formats, tolerances and C
scoring; it has a separate device Model/Session ABI. Sensitive Q/K projections
use compensated binary32 accumulation; other GEMMs use pedantic cuBLAS. No
quantized CUDA or full evaluation is included in this comparison.

## Scope and numerical policy

Model `HuggingFaceTB/SmolLM2-135M`, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`: dim 576, intermediate 1536,
30 layers, 9 query heads, 3 KV heads, head dimension 64, vocabulary 49152,
maximum context 8192, RoPE theta 100000, RMS epsilon 1e-5, tied embeddings.
The source BF16 tensors are converted exactly to FP32. Q/K retain Hugging Face
split-half RoPE ordering. RMS reduction and model math use FP32; scoring uses
FP64 exp/log/log-sum-exp and is excluded from MathOps experiments.

Accuracy builds use `-O3 -fno-fast-math -ffp-contract=off`. Reference runs use
one thread, HF CPU FP32 and eager attention. OpenBLAS may use its own FMA kernels;
record its version/core configuration, CPU, compiler, and thread count.
Token IDs, integer quantization, and scoring masks must match exactly. FP32
logits must satisfy abs(error) <= 1e-3 + 1e-4 * abs(reference); average NLL
on the fixed probe must differ by at most 1e-4. Do not relax these thresholds.

## Model file

All integers/floats are little-endian. IEEE binary32 is required. A loader
rejects unsupported endian markers, dtype, rank, dimensions, duplicate/missing
tensors, overlapping regions, invalid offsets/lengths, and inconsistent config.
Files are written atomically. Model source and export hashes live in manifests.

The header is 256 bytes (zero-fill reserved fields):

| Offset | Encoding | Value |
|---|---|---|
| 0 | 8 bytes | `SML2C001` |
| 8 | u32 | version 1 |
| 12 | u32 | endian marker 0x01020304 |
| 16 | u32 | header size 256 |
| 20 | u32 | tensor count |
| 24 | 8 u32 | dim, intermediate, layers, heads, kv_heads, head_dim, vocab, max_context |
| 56 | 2 f32 | rope_theta, rms_epsilon |
| 64 | 2 u32 | tied_embeddings=1, group_size=64 |
| 72 | 40 bytes | ASCII source revision (no terminator) |

Immediately following are `tensor_count` descriptors of 128 bytes:

| Offset | Encoding | Value |
|---|---|---|
| 0 | 64 bytes | NUL-terminated HF tensor name |
| 64 | u32 | dtype: 1=FP32, 2=Q8_64 |
| 68 | u32 | rank 1 or 2 |
| 72 | 4 u64 | dimensions; unused entries zero |
| 104 | u64 | absolute data offset (64-byte aligned) |
| 112 | u64 | data byte count |
| 120 | 8 bytes | reserved zero |

Names follow HF (`model.embed_tokens.weight`, `model.norm.weight`, and each
layer's two norm weights plus self_attn q/k/v/o and mlp gate/up/down weights).
No lm_head tensor is stored: output and input embeddings share one tensor.
Weights are row-major [output,input]. Norms are FP32 in every configuration.

Q8 blocks are exactly 64 signed int8 values followed by one little-endian f32
scale (68 bytes); no padding. All matrices' K dimensions must be divisible by
64. A zero block has zero scale and all-zero integers. Otherwise
scale=FP32(max(abs(x))/127); q=clamp(round_ties_even(FP32(x/scale)),-127,127).
Reject nonfinite input or a nonzero block with unrepresentable zero scale.
Never automatically shrink the group size. Dequantization is FP32(q*scale).

## Runtime

`include/smollm.h` defines the public Model/Session interface. Model owns immutable
config/tensors; Session owns position, KV storage and reusable work buffers.
Sessions are independent. Reset invalidates cached tokens and returns position
to zero. Reject invalid token IDs and context overflow before mutating a call.
Prefill defaults to 128-token chunks; decode uses a single-token GEMV path.
`NONE`, `LAST`, `ALL` control output projection. Callbacks receive each requested
logit row and its absolute session position, valid only during the callback.

Prefill uses GEMM for all linear layers and attention QK/PV. GQA maps consecutive
groups of 3 query heads to one KV head. Causal masking uses absolute positions,
including existing cache. Chunk boundaries do not reset RoPE positions.
Validate chunks 1/127/128/129, mixed splits, reset, and context limits before Q8.

KernelOps exposes CPU FP32/Q8 GEMM and GEMV only. Q8 activation rows use the same
64-element K groups as weights. Each group's integer dot is int32, multiplied
by FP32 activation/weight scales, then accumulated in FP32 group order.
Embedding lookup dequantizes the selected row in W8A8 mode. Residuals, norms,
softmax, SiLU, and other nonlinear operations stay FP32.

MathOps exposes exp, reciprocal, rsqrt, and sin/cos with explicit call-site IDs
(attention softmax, SiLU, RMS, RoPE). Its CPU host pointers do not imply a future
CUDA ABI. Trace events include site, layer, token/position, head, shape, mask
semantics, input and output; large runs store aggregates only.

KV8 stores layer/token/KV-head groups (head dimension 64), quantizing K after
RoPE. The current chunk also passes through store/dequantization before QK/PV.
Reconstruct only the current layer into temporary FP32 buffers; never retain a
full FP32 shadow cache. Cache accounting distinguishes permanent/scratch bytes.

## Tokenizer

Implement the pinned tokenizer's Digits -> ByteLevel -> BPE pipeline, special
tokens, byte mapping and decode in C using PCRE2. General encoding does not add
BOS/EOS. Empty generation starts from token 0. The versioned binary format is
documented in `docs/design/tokenizer.md`.

## Reference and evaluation files

Token probes: 8-byte `SMTOK001`, u32 count, followed by count u32 token IDs.
Reference logits: raw little-endian FP32 [count,vocab], accompanied by a JSON
manifest with dimensions, revisions, tensor names and SHA256. Intermediate
references use raw FP32 and a shape/site manifest. C performs comparisons.

Prepared scoring data: 8-byte `SMEVAL01`, u32 kind (1=LM, 2=multiple choice),
u32 record_count. Each record starts with eight u32 fields: group, label,
choice, nchoices, token_count, score_start, score_end, norm_chars; then token IDs
as u32. `[score_start,score_end)` contains target token indices (start >=1).
The prediction for target j is output at j-1. Records reset cache/local position.
LM group/choice/label are zero, nchoices=1, norm_chars=0 (unused); normalization
uses target count.
MC records are contiguous by group and choice, with the same label/nchoices.
`norm_chars` is the number of Unicode characters in the original processed
choice, excluding the added delimiter and any moved context whitespace; never
its UTF-8 byte count or the token count. Validate every length and bound.

WikiText-2 raw test strings join with `\n\n`, no BOS/EOS, context 2048/stride 512.
Windows begin 0,512,... until the last target is covered. Score targets 1..end-1
on the first window and only globally new target indices thereafter. Thus every
global target 1..N-1 is scored once. Smoke limits input to the first 8192 tokens.

HellaSwag and PIQA validation use pinned lm-evaluation-harness preprocessing and
continuation-boundary handling; record source revision and exact transformations
in the data manifest. Smoke uses the first 32 original examples. Evaluate both
raw log-likelihood accuracy and character-normalized accuracy in C.

## Comparison and performance

Run FP32/FP32, W8A8/FP32, FP32/KV8, W8A8/KV8 on identical target tokens. Report
layer/logit error, FP64 KL, top-1 changes, PPL, and accuracy deltas. Replay linear
captures with weight-only, activation-only, and combined quantization; replay
nonlinear functions on identical FP32 inputs. Detailed traces use small probes.

Full final evaluation covers the entire split; smoke is not a replacement.
Summary JSONL/CSV includes model/data hash/revision, build, options, threads,
and scoring counts. Performance: prompts 128/512/2048, 128 decode tokens,
threads 1/12, two warmups and five measured repetitions, median. Match output
projection work across modes. Report loading, TTFT, prefill/decode throughput,
KV bytes, scratch bytes and peak RSS separately, in fresh processes per config.
