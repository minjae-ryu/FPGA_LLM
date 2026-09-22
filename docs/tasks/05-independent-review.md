# Independent runtime review

Reviewed the public runtime contract and the implementations in `model.c`,
`session.c`, `cache.c`, and `kernels.c`. Added an internal C harness plus Python
ABI cases without changing runtime APIs or numerical tolerances.

## Actionable finding

The initial loader accepted a noncanonical Q8 zero block. Replacing all 64
integers in a block with zero and setting its scale to `1.0f` still made
`sm_model_load` return success, although the format contract requires an
all-zero block to have zero scale. It also accepted an all-zero block with a
negative-zero scale.

The runtime owner fixed `src/model.c` to reject a sign-bit scale, a nonzero
integer with zero scale, `-128`, and an all-zero integer block with nonzero
scale. `test_loader_rejects_noncanonical_q8_blocks` now covers all four cases
and passes without an expected-failure marker.

No cache-addressing, GQA mapping, causal-mask, reset, or default Q8-kernel defect
was found.

## Direct kernel and cache evidence

`tests/test_cache_kernels.c` calls the internal cache API and public default
kernel function pointers directly. It checks:

- zero, positive/negative ties-to-even, two adjacent groups, `+/-127`, NaN,
  infinity, scale underflow, and a partial group;
- exact activation integer bytes and scales;
- Q8 GEMM and GEMV bit equality against a separate scalar `int32_t` dot with
  FP32 group-order scale accumulation;
- KV8 physical `[layer][token][kv_head][64]` block addressing for two layers,
  four tokens, and three KV heads;
- prefix bytes unchanged after storing a new chunk, then exact reconstruction
  of both the prefix and current chunk from independently chosen `q*scale`
  values;
- FP32 cache identity, layer isolation, byte accounting, and failed-store
  bounds with no mutation.

The harness is compiled by `tests/test_review_cases.py`, so it runs under the
normal pytest gate without a Makefile change. A standalone reproduction is:

```sh
make build/libsmollm.so
gcc -O2 -std=c11 -Wall -Wextra -Werror \
  -fno-fast-math -ffp-contract=off -Iinclude -Isrc \
  tests/test_cache_kernels.c -Lbuild \
  -Wl,-rpath,$PWD/build -lsmollm -o /tmp/test_cache_kernels
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/test_cache_kernels
```

Observed output: `cache/kernel independent checks passed`.

The Python trace test independently reconstructs every layer-0 attention row
for a five-token, three-query-head/one-KV-head fixture. It verifies the GQA head
mapping, softmax probabilities, exact zero future probabilities, and the
concatenated attention output. Repeating the identical call after reset is
bit-exact. Invalid token and capacity-overflow calls leave the session position
unchanged.

## Chunk-dependent quantization diagnosis

The prior synthetic failures compare different call shapes: chunk 1 uses GEMV
for linear rows, while chunk 127 uses GEMM. OpenBLAS reduction differences can
therefore change the FP32 input to a later quantizer. These are recorded rather
than treated as an arbitrary Q8 accuracy failure.

Command:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m pytest -q -s tests/test_review_cases.py
```

The test writes `artifacts/review/chunk_sensitivity.json`. For the 130-token
synthetic input, chunk 1 versus chunk 127 produced:

| Linear/cache | Maximum logit difference | Position, logit | Changed values |
|---|---:|---:|---:|
| FP32 / FP32 | 0.0004365444 | 105, 38 | 16,144 |
| Q8 / FP32 | 0.06068516 | 114, 111 | 1,347 |
| FP32 / KV8 | 0.01252508 | 65, 86 | 16,201 |
| Q8 / KV8 | 0.00007343292 | 111, 2 | 1,412 |

The FP32/FP32 result satisfies the unchanged project tolerance. No project Q8
end-to-end threshold exists, so the other rows are diagnostic values.

For Q8/FP32, the first captured difference is layer-1 attention at position 2,
`9.536743e-7`. At layer 1, position 114, the down-projection activation scales
are `0.7141950727` and `0.7139007449`; exactly one independently recomputed
activation integer changes by one bin. The subsequent Q8 projection amplifies
that discontinuity to the observed `0.06068516` logit difference.

For FP32/KV8, layer-0 post-RoPE K at position 0 first differs by
`7.6293945e-6`. The independent cache quantization has identical integer codes,
but its FP32 scales differ (`0.1314401031` versus `0.1314400882`). Thus the
restored cache differs before attention despite correct layout and storage.

Exact reset determinism and fixed-input quantizer/cache parity remain hard
gates. This diagnosis does not relax the FP32 tolerance.

## Validation and limits

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q -s tests/test_review_cases.py
7 passed in 1.01s

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 make check
110 passed in 6.72s
```

The full gate includes the independent HF eager synthetic FP32 comparison,
stable FP64 NLL, chunks 1/127/128/129, reset, output modes, cache accounting,
and loader corruption cases. The existing real 57-token and 315-token HF
FP32 references already pass their logit and NLL tolerances across the required
chunks; this review did not repeat a concurrent full-model run or perform any
throughput measurement.
