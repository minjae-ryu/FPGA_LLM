# Task 06: trace comparison and replay

## Objective and prerequisites

Implement a strict streaming reader for `SMTRC001`, compare selected C
intermediates with independent Hugging Face references, replay default math on
captured identical inputs, attribute linear error to weight and activation
quantization, and compare complete trace streams across configurations.

Prerequisites were the version-1 model/numerical contract, the validated FP32
and Q8 model exports, the fixed 57-token probe and Hugging Face intermediates,
and runtime trace events containing exact tensor names and function inputs.

Owned files are `src/replay.c`, `tests/test_replay.py`,
`docs/design/trace.md`, and this task record. The orchestrator added
`src/replay.c` to `SM_COMMANDS` and routed `sm_command_replay` from `main`.
No shared API was changed by this worker.

## Focused verification

Strict compilation and synthetic tests:

```text
$ cc -std=c11 -O2 -Wall -Wextra -Werror -Wpedantic \
    -Iinclude $(pkg-config --cflags openblas libpcre2-8) \
    -c src/replay.c -o /tmp/replay.o
$ python3 -m pytest -q tests/test_replay.py
............                                                             [100%]
12 passed in 1.47s
```

The tests independently construct binary traces and cover bit-exact vector and
scalar math replay, trace-to-trace aggregation and header/count mismatches, all
21 intermediate reference streams, gate failure, FP32/weight-only/
activation-only/W8A8 replay, tied embedding output projection at absolute
position 56, strict option parsing, and malformed magic, partial headers,
unterminated sites, reserved bytes, inconsistent counts, truncated payloads,
and nonfinite output.

The integrated accuracy executable built successfully with `make smollm`.
Per task direction, the repository-wide suite was not run because two unrelated
Q8 chunk assertions were under separate investigation.

GCC `-fanalyzer` completed without findings. AddressSanitizer and
UndefinedBehaviorSanitizer completed the real 39 MiB compare/math parser path
and the two-trace 74-aggregate path without findings; leak detection was
disabled because LeakSanitizer is unavailable under the execution environment's
ptrace wrapper.

## Real 57-token probe

The FP32 trace was regenerated after exact tensor-name linear events were added:

```text
$ OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 build/smollm logits \
    --model models/smollm2-f32.bin \
    --tokens artifacts/reference/probe.tokens \
    --reference artifacts/reference/probe.logits \
    --trace artifacts/replay/probe-f32.smtrace --threads 1 --chunk 128
```

Observed: 8,664 records, 57 tokens, zero logit tolerance failures, zero top-1
changes, maximum logit absolute error `8.34465026855e-05`, logit RMSE
`1.10512430783e-05`, and mean-NLL delta `5.76791043118e-07`. The ignored trace
is 39 MiB plus a 1.3 MiB JSONL index.

Intermediate comparison:

```text
$ build/smollm replay --trace artifacts/replay/probe-f32.smtrace \
    --mode compare --reference-prefix artifacts/reference/probe
```

All 21 layer/site aggregates passed with zero tolerance failures and zero
top-1 changes. The largest absolute error was `0.00390625` in layer-14
`attn_residual` and `ffn_residual`; the largest aggregate RMSE was
`3.66294169064e-05` at layer-29 `ffn_residual`.

Default math replay:

```text
$ build/smollm replay --trace artifacts/replay/probe-f32.smtrace --mode math
```

All 19 site/layer aggregates reproduced exactly: zero changed bits, zero
maximum error, and zero RMSE. This includes vector SiLU reciprocals and RoPE's
sine-vector-then-cosine-vector layout.

Linear replay:

```text
$ OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 build/smollm replay \
    --trace artifacts/replay/probe-f32.smtrace --mode linear \
    --model models/smollm2-f32.bin --q8-model models/smollm2-q8.bin
```

All 22 tensor aggregates passed the FP32 capture gate, including the 49,152-row
tied embedding/output projection at absolute position 56. The largest FP32
capture error was `0.000259399414062` at layer-29 `mlp.down_proj.weight`, with
zero tolerance failures and zero top-1 changes. Across the descriptive
quantization variants, the largest absolute errors were `3.76098632812` for
weight-only, `10.2553710938` for activation-only, and `6.45349121094` for W8A8,
all at layer-29 `mlp.down_proj.weight`. These are reported measurements rather
than pass thresholds.

A matching Q8/FP32-KV trace was generated and compared with `--mode traces`.
The 8,664 headers matched exactly and produced 74 site/layer aggregates. The
logit aggregate had maximum absolute error `9.95070075989`, RMSE
`0.541707951662`, and 4 top-1 changes among 57 rows. The largest observed
intermediate maximum was `90.1706542969` at layer-29 `ffn_residual`. The mode
returned success because Q8 trace comparison is descriptive by contract.

## Limitations and next dependency

Detailed traces have no embedded model hash, record count, or whole-file
checksum. Replay instead requires matching model metadata where models are
used, validates every record, and detects physical truncation and excess data.
The compare mode is intentionally fixed to the pinned layers, sites, and widths.
Math exposes a code-level `SmMathOps` hook; it does not yet load approximation
plugins at runtime. Linear replay retains one dequantized matrix and bounded
row batches, but the tied output projection still requires roughly 108 MiB for
its dequantized FP32 weights.

The replay gates are complete. Quantized kernel and whole-model investigations
can now use identical captured inputs and the descriptive attribution reports
without changing tolerances.
