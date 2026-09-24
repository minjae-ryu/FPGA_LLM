# Task 12: FP32 CUDA forward and bounded CPU comparison

**Archived 2026-09-24:** the user stopped CUDA work and requested restoration
to the pre-CUDA CPU baseline. Five of six benchmark cells were saved; the
comparison is incomplete and must not be resumed automatically. No benchmark
process remained when checked. This branch preserves the work in progress.

The user authorized completing CUDA forward, validating correctness, then
measuring CPU versus CUDA. This supersedes the earlier no-new-benchmark scope
only for this bounded comparison. Full evaluation and the old quantized
performance matrix remain deferred. No agents, global configuration changes,
new toolkit, model downloads, or Python GPU inference were used.

## Implementation and contracts

- GPU embedding, sequential FP32 RMSNorm, split-half RoPE, Q/K/V/O and MLP,
  SiLU, residuals, final norm and tied LM head. Immutable weights, head-major
  KV and reusable scratch stay on the device.
- Session stream/cuBLAS ownership from C1 remains intact. Attention performs
  head-major KV scatter, GQA query packing, QK, causal FP32 softmax, PV and
  output scatter. It reuses one head's scores; no per-layer host activation
  round trips occur with trace disabled.
- CUDA `prefill`/`decode` implement NONE/LAST/ALL with synchronous completion,
  absolute callback positions, null-callback projection, preflight token/context
  validation and lazy logit allocation. Execution/callback errors poison the
  session until reset; reset clears status and invalidates old KV by position.
  Callbacks cannot reenter a session. Nonfinite residuals/logits report their
  location, including NONE and null-callback paths.
- Trace downloads host tensor rows only when requested. Linear inputs/outputs,
  norms, RoPE, attention probabilities/output, residuals, gate activation and
  logits are available. CUDA does not emit the CPU MathOps scalar replay sites;
  CPU `replay math` explicitly rejects a trace with no supported math records.
- `SmRun` dispatches without changing the CPU `SmSession` ABI. The CUDA CLI
  supports `--backend cpu|cuda` and `--device`; default remains CPU. CPU-only
  builds reject CUDA explicitly, Q8/KV8 are rejected before device allocation,
  and replay remains CPU-only. Metadata records the backend, device, upload and
  session setup times, device weights and CUDA compiler flags.
- Benchmarks return LAST logits to a host callback on both backends. CUDA
  stream completion and D2H are inside the timed operation. File validation,
  weight upload and session setup are separate. This is teacher-forced fixed
  token throughput, not tokenizer/sampler or variable-output generation timing.

## Numerical investigation

The initial all-cuBLAS implementation failed the synthetic HF gate at one of
38,400 logit elements for chunk 1 and 129. At token 195, logit 105 was
-1.3149245 versus HF FP32 -1.313784; the initial embedding and attention norm
matched CPU/HF exactly. Projection differences of a few ulps grew in the
attention softmax and residuals. CPU/GPU trace comparison localized this to
Q/K projection sensitivity rather than mask or cache layout.

For diagnosis only, a double-precision HF run gave -1.3143565002394553 at that
coordinate. A more accurate QK score reduction alone did not resolve the
end-to-end gate and was discarded. Fixing cuBLAS algorithm 0 did not change
the result and was also discarded.

The retained solution uses compensated **binary32** accumulation for Q/K
projections, with coalesced warp dot products and FP32 `fmaf` product residuals.
All values/accumulators/corrections remain FP32; there is no TF32, BF16 or FP64
production path. Other linear layers and attention QK/PV use pedantic cuBLAS.
This reduces the sensitive projection errors without relaxing the original
logit or NLL gates. It is an accuracy-first baseline, not a tuned attention or
decode implementation.

## Verification

- CPU suite: 150 passed, 10 explicit GPU skips (7.75 seconds).
- C0 pure-C/kernel/cuBLAS cubin and forced PTX tests passed during the full
  CUDA run. That initial run exposed the two synthetic failures described above.
- After the numerical fix, `make check-cuda-forward`: 9 passed, 2 C0 tests
  deselected (7.19 seconds). These include production C ABI ownership, every
  C1 rollback point, injected nonfinite residual/logit recovery, independent
  HF synthetic logits/NLL for chunk 1/127/128/129, mixed calls, reset, separate
  sessions, invalid tokens/context, callback failure/reentry and null callbacks.
- NumPy oracles on captured GPU inputs check linear, embedding, norms, RoPE,
  SiLU, causal softmax and GQA PV across a chunk boundary.
- Real model: both existing 57/315-token HF references pass for CUDA and
  CPU threads 1/12. CUDA maximum absolute errors are 0.000125885009766 and
  0.000375747680664; mean NLL differences are -6.12363684078e-7 and
  1.12152981424e-7. Both have zero top-1 changes.
- Existing C replay validates all 21 HF intermediate groups at layers 0/14/29.
  The real 315-token chunk and mixed-reset matrix passes all eight rows.
- Compute Sanitizer remains unavailable: the installed 2021.3.1 tool failed
  before its first instrumented API call in Task 11. It was not rerun here;
  no sanitizer cleanliness claim is made.

## Reproduce

Use approved host GPU access; the sandbox hides `/dev/dxg`. Keep the WSL CUDA
driver library first for these processes:

```sh
make smollm cuda build/cuda-lifecycle
make check
LD_LIBRARY_PATH=/usr/lib/wsl/lib make check-cuda-forward
LD_LIBRARY_PATH=/usr/lib/wsl/lib python3 scripts/run_cuda_comparison.py --phase correctness
LD_LIBRARY_PATH=/usr/lib/wsl/lib python3 scripts/run_cuda_comparison.py --phase bench
```

`make check-cuda` additionally repeats the C0 forced PTX path, whose first
cuBLAS JIT can take several minutes. Performance runs use the normal cubin path,
with forced/disabled PTX environment flags cleared.

The runner verifies pinned reference hashes and records executable/model/input
hashes. Benchmark phase refuses changed inputs, changed binary or altered gate
evidence. It runs serially in a fresh process per cell: prompts 128/512,
decode 128, context 1024, chunk 128, CPU threads 1/12 or CUDA, two warmups and
five measured repetitions. The reported statistic is the median. No other
model executions or tests ran during this measurement phase.

Results: `results/cuda-fp32/correctness.jsonl`,
`results/cuda-fp32/benchmark.jsonl`, and
`manifests/cuda-forward-comparison.json`. The final comparison report was not completed before the user stopped CUDA. Original CPU-stage results and C0/C1 provenance remain
historical; they are not relabeled as this new build's results.
