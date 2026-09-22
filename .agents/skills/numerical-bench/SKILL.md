---
name: numerical-bench
description: Prepare, run, and analyze the repository's WikiText-2, HellaSwag, PIQA, numerical-error, throughput, and memory comparisons.
---

Read `docs/design/contracts.md` and `docs/validation.md`. Validate token IDs and
scoring masks before scoring. C performs likelihood calculations in FP64.
Record model/data revisions and hashes, build, options, threads, and counts.

Smoke inputs are WikiText's first 8192 tokens and the first 32 validation
examples of each multiple-choice task. Final results require full splits in
all four linear/cache configurations. Never label smoke results as full results.

Run performance measurements alone, with two warmups and five repetitions,
report medians, and use a fresh process for each memory configuration. Preserve
summary JSONL/CSV and manifests in Git; keep large inputs and traces ignored.
