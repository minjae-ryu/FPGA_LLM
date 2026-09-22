# Task 08: C result comparison

## Objective

`smollm compare-results` compares the final evaluation JSONL produced by
`scripts/run_benchmarks.py`. It selects `fp32_fp32` as the baseline for every
`task` and `variant`, verifies that each candidate measured the same work, and
emits one JSON object per candidate. All metric differences are calculated in
C from the recorded aggregate values. The command does not rerun a model or
introduce an accuracy threshold for Q8 configurations.

Usage:

```text
build/smollm compare-results --input results/evaluation-full.jsonl
```

Input order does not affect output order. Results are sorted by task, variant,
and configuration before comparison. A group must contain exactly one
`fp32_fp32` result and at most one result for each of `w8a8_fp32`, `fp32_kv8`,
and `w8a8_kv8`.

## Validated input

Every row must be a JSON object with typed root fields. The common required
fields are `configuration`, `task`, `variant`, `data_sha256`, `model_revision`,
`model_sha256`, `binary_sha256`, `command`, `kind`, `record_count`,
`target_count`, `context`, `chunk`, `threads`, and `mean_nll`. `command` must be
`eval`. SHA-256 values are 64 lowercase hexadecimal characters and the pinned
model revision is 40 lowercase hexadecimal characters.

An LM row has `kind: "lm"` and a positive finite `perplexity`. A multiple
choice row has `kind: "multiple_choice"`, a positive `example_count`, and
finite `raw_accuracy` and `normalized_accuracy` in `[0,1]`. Metrics belonging
to the other kind are rejected rather than silently ignored.

Within a task and variant, the candidate and baseline must have identical:

- data SHA-256 and model revision
- benchmark kind, record count, and target count
- context, chunk, and thread settings
- example count for multiple choice tasks

Model and binary hashes can differ because the compared storage configurations
can use different model files and executables. Both sides' hashes are carried
into the output. This preserves source attribution without treating a required
configuration difference as an error.

The parser reads root fields structurally, so names inside runner metadata,
invocation arrays, or nested objects cannot replace comparison fields. It
rejects duplicate keys at every nesting level, malformed UTF-8, escapes and numbers,
nonfinite or out-of-range numeric values, invalid known-field types, trailing
data, embedded or escaped NUL bytes, nesting deeper than 32, objects with more than 4096
keys, lines over 1 MiB, and inputs over 4096 rows.

## Output and formulas

Each line contains task, variant, kind, baseline and candidate configuration,
both model SHA-256 values, the shared data SHA-256, both binary SHA-256 values,
the shared model revision, counts, and execution options. Deltas use
`candidate - baseline`:

```text
mean_nll_delta              = candidate_mean_nll - baseline_mean_nll
perplexity_absolute_delta   = candidate_perplexity - baseline_perplexity
perplexity_relative_delta   = candidate_perplexity / baseline_perplexity - 1
raw_accuracy_delta          = candidate_raw_accuracy - baseline_raw_accuracy
normalized_accuracy_delta   = candidate_normalized_accuracy
                              - baseline_normalized_accuracy
```

LM output includes the two perplexity deltas. Multiple choice output includes
the two accuracy deltas. All baseline and candidate scalar metrics are emitted
beside their deltas. Arithmetic that would produce a nonfinite JSON number is
rejected.

These values are measurements for review and downstream formatting. In
particular, the command applies no arbitrary pass/fail tolerance to weight or
KV quantization.

## Focused verification

The command compiles independently with strict C11 warnings and its synthetic
test suite requires no model execution:

```text
$ cc -std=c11 -Wall -Wextra -Werror -Wpedantic -Iinclude \
    src/compare_results.c /tmp/compare_harness.c -lm \
    -o /tmp/compare-results-test
$ python3 -m pytest -q tests/test_compare_results.py
.............................                                            [100%]
29 passed
```

The tests cover mathematically known LM and multiple choice deltas, complete
hash attribution, nested runner metadata, shuffled rows and groups, missing or
duplicate baselines, duplicate candidates, all cross-row compatibility fields,
nonfinite comparison arithmetic, malformed JSON, duplicate root and nested
keys, valid and malformed UTF-8, invalid types, NUL and size bounds, and exact
CLI option shape.
