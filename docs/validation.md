# Reproduction and numerical gates

Run commands from the repository root. OpenBLAS and PCRE2 development packages,
GCC, and a Python environment matching `requirements-smollm.txt` are required.
The original `requirements.txt` belongs to the unchanged legacy training path.
Keep preparation environments and downloaded files in repository-local `.venv/`,
`models/`, `data/`, or `artifacts/`. No global skill/agent installation is needed.

## Prepare pinned inputs

The preparation scripts accept explicit source paths; their defaults refer to
this machine's existing read-only HF cache. To use a clean checkout, download
`HuggingFaceTB/SmolLM2-135M` at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2` into `models/hf` using
`huggingface_hub.snapshot_download(..., revision=..., local_dir="models/hf")`.
The model exporter additionally verifies the exact config and weights hashes.

```sh
python3 scripts/export_smollm.py --source models/hf --dtype f32 --output models/smollm2-f32.bin --manifest manifests/model-f32.json
python3 scripts/export_smollm.py --source models/hf --dtype q8 --output models/smollm2-q8.bin --manifest manifests/model-q8.json
python3 scripts/export_tokenizer.py --tokenizer-json models/hf/tokenizer.json --output models/tokenizer.bin --manifest manifests/tokenizer.json
python3 scripts/prepare_reference.py --model-dir models/hf
python3 scripts/prepare_data.py --model-dir models/hf --task all --variant all
```

Data source paths/revisions, PIQA source archive, exact preprocessing and source
links are documented in [design/data.md](design/data.md). Pass `--wikitext`,
`--hellaswag`, `--piqa-data`, `--piqa-labels`, and `--piqa-script` when those source
files are outside the defaults. The manifests record file hashes and revisions.
Generated data includes both smoke and complete evaluation splits.

## Build and local checks

```sh
make smollm
make check
make run testcc
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q test_all.py
```

The last command downloads the official tiny legacy regression files if absent;
all generated files are ignored. The SmolLM2 suite uses small synthetic models
and the pinned tokenizer when available; skipped real-tokenizer tests are
explicitly reported. It covers malformed files, bit-exact quantization,
independent scalar integer kernels/cache, independent HF FP32 synthetic output,
causality/GQA, chunking, reset/context bounds, tokenizer, scoring and replay.

Accuracy flags are `-O3 -fno-fast-math -ffp-contract=off`; fast-math is rejected
at compile time. OpenBLAS's own FMA policy/core is recorded separately. Fix
threads to 1 for reference gates. Every CLI result records the actual build
flags, OpenBLAS configuration, selected threads and runtime options.

## Real-model FP32 gates

```sh
build/smollm logits --model models/smollm2-f32.bin --tokens artifacts/reference/probe.tokens --reference artifacts/reference/probe.logits --context 512 --trace traces/probe.smtrace
build/smollm logits --model models/smollm2-f32.bin --tokens artifacts/reference/long.tokens --reference artifacts/reference/long.logits --context 512
build/smollm chunks --model models/smollm2-f32.bin --tokens artifacts/reference/long.tokens --context 512
build/smollm replay --mode compare --trace traces/probe.smtrace --reference-prefix artifacts/reference/probe
build/smollm replay --mode math --trace traces/probe.smtrace
build/smollm replay --mode linear --trace traces/probe.smtrace --model models/smollm2-f32.bin --q8-model models/smollm2-q8.bin
```

Create `traces/` before writing there. Detailed trace/replay examples and format
are in [design/trace.md](design/trace.md). The CLI logit gate returns nonzero for
any failure of `abs(error) <= 1e-3 + 1e-4*abs(reference)` or a mean NLL difference
above `1e-4`. Q8 comparisons use `--report-only` to report observed differences;
this flag is never used to bypass FP32 validation. Q8 quantization integers and
cache behavior still have exact fixed-input gates. Changed GEMM/GEMV reduction
order can cross a quantizer threshold; the evidence is in task 05.

The complete four-configuration numerical and replay matrix, with hashes in every
summary row, is run by `python3 scripts/run_numerical.py`.

## Evaluation and isolated performance

```sh
python3 scripts/run_benchmarks.py eval --variant smoke
python3 scripts/run_benchmarks.py eval --variant full
python3 scripts/run_benchmarks.py bench
```

The recorded first comparison was reduced at the user's request to a total
30-minute measurement budget (including already completed measurements).
It uses four performance cells: prompt 128, decode 128, thread 1, two warmups,
five repetitions; accuracy uses only the existing smoke splits, jobs 4 and
threads 3. Both FP32 reference probes passed again at threads 3 before scoring.
The exact deadline, bounded commands and completion time are recorded in
`manifests/measurement-plan.json`. Full splits and the complete performance
matrix remain a later step.

To format the recorded small comparison, run:

```sh
python3 scripts/render_results.py --scope small
```

After all three complete matrices have been run, `--scope full` formats those
instead. The formatter requires complete coverage of its selected scope and
matching executable hashes, verifies the runner's result seals, calls C
`compare-results`, and writes delta JSONL/CSV plus `docs/results.md`. It formats
recorded C metrics without recalculating their differences in Python.

Evaluation defaults to all four configurations, thread 1, context 2048, chunk
128. `--configurations`, `--tasks`, and evaluation-only `--jobs` select a subset
or run independent accuracy jobs concurrently. Concurrent evaluation timing is
explicitly marked as non-isolated and is not a performance claim. Scoring is C
FP64; the Python runner validates provenance and launches processes.

The performance runner always executes serially: prompt 128/512/2048, decode
128, threads 1/12, two warmups, five repetitions, medians. Each cell gets a new
process. Do not run model preparation/inference or heavy tests concurrently.
LAST projection work is identical between configurations. Loading, TTFT,
prefill/decode throughput, cache/scratch bytes and peak RSS are separate fields.

The runner writes summary JSONL/CSV and safe resume checkpoints after each
completed cell. Resume requires matching executable/input/manifest/options and
score hashes. Full per-record scores and checkpoints are ignored; aggregate
results and reproducibility manifests belong in Git. Changes to a benchmark
binary require a new result stem or `--restart`.

## Generation

```sh
build/smollm generate --model models/smollm2-f32.bin --tokenizer models/tokenizer.bin --prompt "The purpose of a scientific experiment is" --steps 64
build/smollm generate --model models/smollm2-q8.bin --kv q8 --prompt "Once upon a time" --steps 64 --temperature 0.8 --seed 1
```

General tokenization adds no BOS/EOS. Only empty generation begins with token 0.
The model and tokenizer must have the same vocabulary. Greedy sampling is the
default; stochastic sampling is deterministic for a fixed seed and build.
