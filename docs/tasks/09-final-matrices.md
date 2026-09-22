# Task 09: final build, evaluation and isolated performance

## Scope and inputs

Resume checkpoint `134321d78ecec1ccdee0cdd3377b677030af7905` using the existing
pinned model, tokenizer, HF references and six prepared evaluation files.
The repository-local `.agents/skills/numerical-bench/SKILL.md` governs this work.
No global skill/agent settings or project `.codex` files are changed.

Owned outputs: `manifests/engine-build.json`, aggregate `results/` files,
`scripts/render_results.py`,
this record, final results documentation, README and handoff status. The
runtime, public interfaces and data contracts remain at the validated checkpoint.

## Final build gate

On 2026-09-22, `make smollm check` found the executable up to date and passed
**143 tests in 8.72 seconds**, including the final JSON-parser regressions.

`python3 scripts/run_numerical.py` then completed on executable SHA256
`32c208d95e7d399d60c7f87981833331dc0514eef05451e33a68609028d1f85e`:

| Group | Aggregate rows |
|---|---:|
| Logits | 8 |
| Chunk/reset | 32 |
| HF intermediates | 21 |
| Identical-input math | 19 |
| Linear variants | 88 |
| Accumulated traces | 222 |

The FP32 gates pass at their original tolerances; all 19 math aggregates are
bit-exact. Q8 errors remain descriptive measurements with no new pass threshold.
The refreshed JSONL/CSV contains actual new executions, not rewritten old hashes.
`manifests/engine-build.json` records the binary, source-file hashes, compiler,
CPU, OpenBLAS, PCRE2 and validation evidence. Earlier legacy/sanitizer evidence is
explicitly identified as carried forward for unchanged source.

## User-requested scope and time limit

During the 24-cell performance run the user requested a smaller first
comparison, then capped total measurement time at **30 minutes**. The larger
run was interrupted with SIGINT after 11 complete cells. Those unmodified rows
are preserved in `results/benchmarks-extended-partial.jsonl/csv` and its original
resume checkpoint remains ignored by Git.

The balanced small matrix contains all four configurations, prompt 128,
decode 128, thread 1, two warmups and five measured repetitions. Two completed
matching cells were reused and the two missing KV8 cells ran alone in fresh
processes. `results/benchmarks-small.jsonl/csv` preserves each original result
seal and exact invocation. No model or runtime changes were made.

Accuracy is limited to the existing smoke splits: all four configurations on
WikiText (8,191 targets / 13 windows), HellaSwag (32 examples) and PIQA (32
examples). It runs jobs 4 / threads 3; concurrent evaluation timing is not a
performance measurement. FP32 57/315-token probes passed at threads 3 before
this evaluation: maximum logit errors `9.44137573242e-05` / `0.000351905822754`,
mean NLL deltas `-3.04112805034e-07` / `6.89774502088e-08`, zero top-1 changes.

`manifests/measurement-plan.json` counts all measurement time conservatively
from 2026-09-22 21:46 KST, before the original performance launch, and sets a
common 22:16 KST deadline. Each subsequent model command uses GNU `timeout`
with the remaining budget, sending SIGINT six seconds before the deadline and
allowing five seconds before SIGKILL. The default full split evaluation was
never started. The complete matrix remains deferred under the revised scope.

`scripts/render_results.py --scope small` will require all four small performance
cells and all twelve smoke evaluation cells before generating the report. It
checks result seals, binary hashes and dataset coverage, and delegates metric
deltas to C. `--scope full` separately requires the complete original matrices.

## Completed bounded comparison

All four small performance cells and all twelve smoke cells completed. The
last model execution finished at **2026-09-22 22:04:33.121 KST**, giving a
conservative measurement span of **1,113.121 seconds (18 minutes 33 seconds)**
against the 1,800-second limit. The timeout did not fire. No model executions
remain active.

`python3 scripts/render_results.py --scope small` passed coverage, hash and
result-seal checks, invoked C comparison for all nine candidate/task pairs,
and wrote `results/evaluation-smoke-deltas.jsonl/csv` plus `docs/results.md`.
All frozen source and executable hashes remain unchanged. The freshly rerun
numerical metrics match the checkpoint exactly after excluding provenance,
timing and RSS fields.

A negative formatting check also passed: `--scope full` rejected the absent
full evaluation file without replacing the completed small-scope report.
`git diff --check` found no whitespace errors.

WikiText smoke PPL is 15.566465 (FP32/FP32), 15.636703 (W8A8/FP32), 15.578888
(FP32/KV8), and 15.666988 (W8A8/KV8). Raw and normalized accuracies happen to
match across the four configurations on these 32-example HellaSwag and PIQA
subsets; this is not evidence that full-split accuracies are identical.

At prompt 128 / decode 128 / one thread, FP32/FP32 prefill and decode medians
are 469.617 and 34.887 tok/s; W8A8/KV8 gives 129.537 and 73.541 tok/s. In this
implementation the quantized path improves decode while slowing prefill.
Permanent KV storage for context 256 is 11,796,480 bytes (FP32) or 3,133,440
bytes (KV8). Loading, warm TTFT, RSS and scratch remain separate report fields.
