# C evaluation and performance commands

Implemented `sm_command_eval` and `sm_command_bench` behind the shared `SmRun`
CLI. The orchestrator wired them as `build/smollm eval` and
`build/smollm bench` and added both sources to the accuracy build.

## Evaluation

The evaluation interface is:

```sh
build/smollm eval \
  --model models/smollm2-f32.bin \
  --data data/wikitext2_smoke.smeval \
  --manifest manifests/data_wikitext2_smoke.json \
  --kv f32 --context 2048 --chunk 128 --threads 1 \
  --progress-every 100 \
  --scores results/scores/wikitext2-smoke-fp32_fp32.jsonl
```

The command reads and validates the complete `SMEVAL01` file before the first
model call. It rejects invalid magic, kind, record counts, lengths, trailing
bytes, token IDs, context overflow, score bounds, LM metadata, MC choice
metadata, non-contiguous groups, unordered choices, incomplete groups, and
inconsistent group labels/counts. Any validation, execution, callback, or output
failure returns nonzero without printing aggregate metrics. Optional score JSONL
is written through a temporary sibling and atomically renamed only after every
record succeeds.

Each record resets its session. Tokens before `score_start - 1` are prefetched
with `NONE`; inputs from `score_start - 1` through `score_end - 2` use `ALL`.
The callback checks every absolute position and scores target `position + 1`
with `sm_nll`. Callback and target counts must match before a record is accepted.
NLL and aggregate sums use FP64.

LM output reports exact record, input-token, target, NLL, mean-NLL, and
perplexity counts. MC output reports total continuation targets and Unicode
normalization characters plus raw log-likelihood and character-normalized
correct counts and accuracies. The normalized score is
`loglikelihood / norm_chars`; the prepared field represents Unicode code points
in the original processed choice and excludes the inserted delimiter or moved
context whitespace. Ties select the lowest choice index. Per-record JSONL keeps
the raw and normalized scores and both selected choices for paired comparisons.

Long evaluations print progress to stderr every 100 records and once at the
last record. `--progress-every 0` disables periodic reports while retaining the
final report.

## Performance

The fixed performance interface is:

```sh
build/smollm bench \
  --model models/smollm2-f32.bin --kv f32 \
  --context 2176 --chunk 128 --threads 1 \
  --prompt 2048 --decode 128 --warmup 2 --repetitions 5
```

Prompt sizes are restricted to 128, 512, or 2048. Each warmup and measured
repetition resets the session, prefills the deterministic prompt with `LAST`,
and decodes 128 deterministic tokens with `LAST`. This gives every configuration
the same output-projection workload. The JSON result includes every timing
sample, median TTFT/prefill/decode time, median-derived token rates, load time,
model/KV/scratch bytes, and peak RSS. No real-model performance measurement was
run during implementation.

`scripts/run_benchmarks.py` runs the four required configurations in serialized,
fresh child processes:

- FP32 weights / FP32 KV
- W8A8 GS64 weights / FP32 KV
- FP32 weights / KV8
- W8A8 GS64 weights / KV8

`eval --variant smoke|full|all` produces JSONL, CSV, and per-record scores for
WikiText-2, HellaSwag, and PIQA. `--configurations` selects any subset of the
four named configurations for either command and defaults to all four. `bench`
covers all three prompt sizes and thread counts 1 and 12. Rows include actual binary/model/data and manifest SHA256,
manifest revisions, Git state, exact invocation, compiler/build/OpenBLAS
metadata from C, and all run options.

Evaluation accepts `--jobs N` (default 1) and launches at most N independent C
subprocesses. Workers only execute their assigned cell and write its unique,
atomically published score file. The main Python process seals results and
updates the checkpoint, so concurrent completions cannot race checkpoint
writes. Output rows retain canonical plan order. Every row records
`evaluation_concurrency`, `timing_isolated`, and `timing_scope`; timing from a
run with `jobs > 1` is explicitly marked `concurrent_not_isolated` and is not an
isolated throughput measurement. Performance remains strictly serialized and
has no `--jobs` option.

Every finished matrix cell is checkpointed atomically. Its result is sealed by a
SHA256, and evaluation score files carry their own SHA256. Resume reuses a cell
only after checking the plan, executable, model, data, manifests, options,
sealed row, and score file. A changed plan is rejected; `--restart` explicitly
starts it again. This lets multi-hour full evaluations resume without mixing
measurements from different builds or inputs.

## Focused validation

The final focused command was:

```sh
python3 -m pytest -q tests/test_evaluation.py tests/test_export.py
```

The final runner-focused gate was
`python3 -m pytest -q tests/test_evaluation.py`: `17 passed in 1.15s`.
Evaluation coverage uses a compiled stub runtime
with analytic FP64 scores and checks LM masks/prefixes, raw versus Unicode-char
normalized MC accuracy, all current smoke-file metadata, malformed/truncated
inputs, execution failure cleanup, atomic score output, benchmark medians, and
resume/tamper/restart behavior. Export coverage remains included because the
synthetic loader fixture changed to the required head dimension 64.

`make smollm` then compiled the integrated CLI with the accuracy flags and no
warnings. A real C invocation on the small FP32 synthetic model also passed:

```text
records=2, targets=4, input_tokens=7
nll_sum=31.496903989619916
mean_nll=7.8742259974049791
perplexity=2628.6508201536581
```

Real checkpoint smoke/full evaluation and every performance matrix cell remain
unrun. Q8 end-to-end results must wait for the FP32 runtime gates and the current
independent Q8 review.
