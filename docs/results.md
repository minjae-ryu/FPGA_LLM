# SmolLM2-135M C results

SmolLM2-135M Base revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, exported exactly from BF16 to FP32. The user requested a small first comparison capped at 30 minutes. All four configurations have smoke results and isolated prompt-128 / thread-1 performance measurements. Full-split accuracy and the complete 24-cell performance matrix are deferred; these smoke scores are not full-split results.

Executable SHA256: `32c208d95e7d399d60c7f87981833331dc0514eef05451e33a68609028d1f85e`. The source, compiler, CPU and library configuration are recorded in [engine-build.json](../manifests/engine-build.json). Inference, FP64 scoring, numerical comparisons and metric deltas run in C; this document only formats their recorded outputs.

Measurement budget: 1800 seconds, including earlier runs, counted conservatively from `2026-09-22T12:46:00+00:00`. Deadline: `2026-09-22T13:16:00+00:00`; measurements finished at `2026-09-22T13:04:33.121011+00:00`. The original matrix stopped after 11 complete cells, preserved in `benchmarks-extended-partial.jsonl/csv`. Two matching cells were reused and two KV8 cells added to form the balanced four-configuration small matrix. See [the plan](../manifests/measurement-plan.json).

## Smoke accuracy

Smoke uses 8,191 WikiText targets across 13 windows and the first 32 examples of each multiple-choice task. These are separate from the full splits. Normalized accuracy uses each original processed choice's Unicode character count. Deltas are candidate minus FP32 / FP32.

Execution options `(threads, chunk, context, concurrent processes)`: `[(3, 128, 2048, 4)]`. Evaluation timings are not used as performance measurements.

| WikiText configuration | Mean NLL | PPL | PPL delta |
| --- | --- | --- | --- |
| FP32 / FP32 | 2.74511891 | 15.566465 | baseline |
| W8A8 / FP32 | 2.74962092 | 15.636703 | +0.070238 |
| FP32 / KV8 | 2.74591668 | 15.578888 | +0.012423 |
| W8A8 / KV8 | 2.75155582 | 15.666988 | +0.100523 |

| Task | Configuration | Accuracy | Normalized accuracy | Accuracy delta (pp) | Normalized delta (pp) |
| --- | --- | --- | --- | --- | --- |
| hellaswag | FP32 / FP32 | 46.875000% | 40.625000% | baseline | baseline |
| hellaswag | W8A8 / FP32 | 46.875000% | 40.625000% | +0.000000 | +0.000000 |
| hellaswag | FP32 / KV8 | 46.875000% | 40.625000% | +0.000000 | +0.000000 |
| hellaswag | W8A8 / KV8 | 46.875000% | 40.625000% | +0.000000 | +0.000000 |
| piqa | FP32 / FP32 | 71.875000% | 68.750000% | baseline | baseline |
| piqa | W8A8 / FP32 | 71.875000% | 68.750000% | +0.000000 | +0.000000 |
| piqa | FP32 / KV8 | 71.875000% | 68.750000% | +0.000000 | +0.000000 |
| piqa | W8A8 / KV8 | 71.875000% | 68.750000% | +0.000000 | +0.000000 |

## Isolated performance

AMD Ryzen 9 7900, 12 physical cores / 24 logical CPUs, WSL2 Linux; OpenBLAS 0.3.20 (Zen), GCC 11.4.0. Each cell runs in a fresh process with two warmups and five measured repetitions; reported values are medians. No other project model execution or heavy test runs concurrently. Prompt IDs are deterministic; decode feeds fixed token 1729. Every mode performs LAST output projection for prefill and every decode token.

TTFT is warm model prefill including the last logits projection; it excludes tokenization, model/session loading and sampling. Loading is one observation per fresh process, with the operating system's file cache uncontrolled. These are CPU implementation measurements, not a hardware-independent claim about quantization.

### 1 thread(s)

| Configuration | Prompt | Prefill tok/s | Decode tok/s | TTFT (s) | Load (s) |
| --- | --- | --- | --- | --- | --- |
| FP32 / FP32 | 128 | 469.617 | 34.887 | 0.272563 | 0.072908 |
| W8A8 / FP32 | 128 | 129.726 | 70.049 | 0.986692 | 0.113150 |
| FP32 / KV8 | 128 | 453.598 | 35.597 | 0.282188 | 0.075457 |
| W8A8 / KV8 | 128 | 129.537 | 73.541 | 0.988136 | 0.095946 |

### Memory at prompt 128 + decode 128

Bytes below are permanent KV storage and session scratch, separately. Peak RSS is process-wide and includes touched model pages, library allocations and other overhead. Model files occupy 538,095,104 bytes (FP32) and 143,060,480 bytes (GS64 Q8). KV8 has no permanent FP32 shadow.

| Configuration | Threads | KV bytes | Scratch bytes | Peak RSS (KiB) |
| --- | --- | --- | --- | --- |
| FP32 / FP32 | 1 | 11796480 | 4173952 | 546816 |
| W8A8 / FP32 | 1 | 11796480 | 4382848 | 160512 |
| FP32 / KV8 | 1 | 3133440 | 4173952 | 538176 |
| W8A8 / KV8 | 1 | 3133440 | 4382848 | 152064 |

## Numerical error and attribution

FP32 logit tolerances remain `atol=1e-3, rtol=1e-4`; fixed-probe mean NLL difference must be at most `1e-4`. The 143-test suite passes. All 21 HF intermediate aggregates pass, and all 19 identical-input default-math aggregates have zero changed bits. There is no arbitrary accuracy pass/fail threshold for Q8.

| Probe tokens | Configuration | Max logit error | Logit RMSE | NLL delta vs HF | Mean KL vs HF | Top-1 changes |
| --- | --- | --- | --- | --- | --- | --- |
| 57 | FP32 / FP32 | 8.3446503e-05 | 1.1051243e-05 | +5.7679104e-07 | 1.8133253e-11 | 0 |
| 315 | FP32 / FP32 | 0.0002746582 | 1.5863038e-05 | +2.3476885e-07 | 6.6510591e-12 | 0 |
| 57 | W8A8 / FP32 | 9.9506893 | 0.54170799 | +0.10513782 | 0.04644756 | 4 |
| 315 | W8A8 / FP32 | 8.1311111 | 0.39406784 | +0.01033439 | 0.0080208745 | 3 |
| 57 | FP32 / KV8 | 0.44063568 | 0.095344673 | -0.002448439 | 0.00050101663 | 1 |
| 315 | FP32 / KV8 | 1.2010756 | 0.14315015 | +0.00091553582 | 0.00015365444 | 2 |
| 57 | W8A8 / KV8 | 9.9506893 | 0.54620131 | +0.098293548 | 0.047000808 | 9 |
| 315 | W8A8 / KV8 | 8.1311111 | 0.49583717 | +0.013512069 | 0.0083496489 | 5 |

### Identical-input linear replay

The last layer's MLP down projection illustrates the separate weight and activation quantization effects. All variants use identical captured FP32 inputs; errors are relative to the captured FP32 output. Their errors are not additive. All 22 tensors and four variants are retained in the aggregate files.

| Layer-29 down projection | Max absolute error | RMSE |
| --- | --- | --- |
| fp32_capture | 0.00025939941 | 7.0892015e-06 |
| weight_only | 3.7609863 | 0.16639991 |
| activation_only | 10.255371 | 0.35531477 |
| w8a8 | 6.4534912 | 0.38603277 |

### Accumulated trace error

These end-to-end trace differences use FP32 / FP32 as the baseline and sample layers 0, 14 and 29. The tensor scale changes between sites, so residual errors and normalized logit errors should be interpreted with their own scales.

| Configuration | Layer | Site | Max absolute error | RMSE |
| --- | --- | --- | --- | --- |
| W8A8 / FP32 | -1 | embedding | 0.0031190403 | 0.00079575588 |
| W8A8 / FP32 | 0 | ffn_residual | 0.34004974 | 0.027076048 |
| W8A8 / FP32 | 14 | ffn_residual | 127.16992 | 0.84660688 |
| W8A8 / FP32 | 29 | ffn_residual | 90.170654 | 1.7246216 |
| W8A8 / FP32 | -1 | final_norm | 4.2860241 | 0.11505936 |
| W8A8 / FP32 | -1 | logits | 9.9507008 | 0.54170795 |
| FP32 / KV8 | -1 | embedding | 0 | 0 |
| FP32 / KV8 | 0 | ffn_residual | 0.049581528 | 0.0028572777 |
| FP32 / KV8 | 14 | ffn_residual | 10.189453 | 0.082046088 |
| FP32 / KV8 | 29 | ffn_residual | 5.4014587 | 0.45236007 |
| FP32 / KV8 | -1 | final_norm | 0.45958471 | 0.02491837 |
| FP32 / KV8 | -1 | logits | 0.44065189 | 0.095345512 |
| W8A8 / KV8 | -1 | embedding | 0.0031190403 | 0.00079575588 |
| W8A8 / KV8 | 0 | ffn_residual | 0.34004974 | 0.027372054 |
| W8A8 / KV8 | 14 | ffn_residual | 127.16992 | 0.84862544 |
| W8A8 / KV8 | 29 | ffn_residual | 90.170654 | 1.756084 |
| W8A8 / KV8 | -1 | final_norm | 4.2860241 | 0.11621335 |
| W8A8 / KV8 | -1 | logits | 9.9507008 | 0.5462008 |

Q8 chunk differences are descriptive: small FP32 GEMM/GEMV changes can cross a quantizer boundary. Fixed-input integer quantization and cache gates still require exact agreement; see [the independent investigation](tasks/05-independent-review.md).

## Reproduction and files

Run `python3 scripts/render_results.py --scope small` after completing the selected commands in [validation.md](validation.md). It requires all cells in the selected scope, verifies result seals and coverage, calls C `compare-results`, and formats this report. [Evaluation and performance JSONL/CSV](../results/) include exact invocations, binary/model/data hashes, scoring counts, thread settings and raw timing samples. Full per-record scores, weights, datasets and traces remain ignored by Git.
