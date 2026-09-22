# Engine integration and numerical gates

The C model loader, sessions, OpenBLAS FP32 GEMM/GEMV, W8A8 GS64 kernels,
FP32/KV8 caches, MathOps, trace callbacks, FP64 scoring, sampling, and CLI are
integrated under `include/smollm.h`. Legacy sources remain byte-for-byte intact.

The loader validates all descriptors, expected HF names/shapes/dtypes, source
revision syntax, alignment, bounds, overlaps, nonfinite values, and canonical
Q8 blocks before publishing a model. Sessions validate complete token inputs
and context capacity before starting a call. Execution/callback failures require
reset before reuse, as documented in the public header.

## Observed FP32 gates

CPU FP32 eager HF references use the pinned original BF16 weights. With one
thread and the accuracy build:

| Probe | Tokens | Maximum logit absolute error | Mean NLL difference | Top-1 changes |
|---|---:|---:|---:|---:|
| Natural text | 57 | 8.34465e-5 | 5.76791e-7 | 0 |
| Longer text | 315 | 2.74658e-4 | 2.34769e-7 | 0 |

All values satisfy `atol=1e-3, rtol=1e-4`; NLL differences are below `1e-4`.
The longer reference also passed at 12 threads (maximum 2.00272e-4, mean NLL
difference 6.04505e-8). Raw command results live in `results/fp32-*.json*`.

Chunk sizes 1/127/128/129, mixed call boundaries, and reset passed on the
315-token real model probe. The independent synthetic HF eager test also covers
causal-prefix invariance, the exact context limit, invalid-token/context errors,
output NONE/LAST/ALL, and memory accounting. See `05-independent-review.md` for
fixed-input Q8/cache parity and the measured Q8 chunk sensitivity.

The C intermediate comparator passed all 21 selected layer/site aggregates.
Math replay reproduced all 19 function groups bit exactly. Linear replay passed
all 22 unquantized groups and reports separate weight-only, activation-only,
and combined quantization errors; see `06-replay.md`.

## Legacy and memory checks

`make run testcc` prints `ALL OK`. After downloading the official 260K legacy
test artifacts into ignored `test/`, both `test_all.py` tests passed, including
the exact 200-token expected sequence in C and Python.

The direct cache/kernel harness passed AddressSanitizer, UndefinedBehaviorSanitizer
and LeakSanitizer. LeakSanitizer needed an approved run outside the sandbox
because its process-tracing restriction blocked the initial invocation.

The first FP32 WikiText smoke run scored exactly 8191 targets across 13 windows,
mean NLL 2.7451189071 and PPL 15.5664647849. This is a smoke result. Full-split
results and the four-configuration performance matrix are a separate gate.
