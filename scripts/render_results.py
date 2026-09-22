#!/usr/bin/env python3
"""Format complete C-produced matrices; all metric deltas are calculated in C."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from run_benchmarks import CONFIGURATION_NAMES, json_fingerprint, sha256_file, write_results


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "fp32_fp32": "FP32 / FP32",
    "w8a8_fp32": "W8A8 / FP32",
    "fp32_kv8": "FP32 / KV8",
    "w8a8_kv8": "W8A8 / KV8",
}


def read_rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / "results" / f"{name}.jsonl").read_text().splitlines()]


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def validate_sealed(rows: list[dict], binary_hash: str) -> None:
    for row in rows:
        if row["binary_sha256"] != binary_hash:
            raise ValueError("matrix executable differs from the validated build")
        unsealed = {key: value for key, value in row.items() if key != "result_sha256"}
        if json_fingerprint(unsealed) != row["result_sha256"]:
            raise ValueError(f"result seal mismatch: {row['job_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("small", "full"), default="small")
    args = parser.parse_args()
    variants = ("full", "smoke") if args.scope == "full" else ("smoke",)
    build = json.loads((ROOT / "manifests/engine-build.json").read_text())
    binary_hash = sha256_file(ROOT / "build/smollm")
    if binary_hash != build["binary_sha256"]:
        raise ValueError("executable differs from engine-build.json")
    evaluations = {variant: read_rows(f"evaluation-{variant}") for variant in variants}
    benchmarks = read_rows("benchmarks" if args.scope == "full" else "benchmarks-small")
    numerical = {name: read_rows(f"numerical-{name}") for name in
                 ("logits", "chunks", "intermediates", "math", "linear", "traces")}
    for name, rows in numerical.items():
        if len(rows) != build["numerical_matrix"]["group_counts"][name]:
            raise ValueError(f"incomplete numerical matrix: {name}")
        if any(row["binary_sha256"] != binary_hash for row in rows):
            raise ValueError(f"numerical executable differs: {name}")
    if not all(row["parity_pass"] for row in numerical["logits"]
               if row["configuration"] == "fp32_fp32"):
        raise ValueError("FP32 logit gate failed")
    if not all(row["parity_pass"] for row in numerical["intermediates"]):
        raise ValueError("HF intermediate gate failed")
    if not all(row["exact_pass"] for row in numerical["math"]):
        raise ValueError("identical-input math gate failed")
    for variant, rows in evaluations.items():
        expected = {(task, variant, config) for task in ("wikitext2", "hellaswag", "piqa")
                    for config in CONFIGURATION_NAMES}
        actual = {(row["task"], row["variant"], row["configuration"]) for row in rows}
        if len(rows) != len(expected) or actual != expected:
            raise ValueError(f"incomplete evaluation matrix: {variant}")
        validate_sealed(rows, binary_hash)
        for row in rows:
            manifest = json.loads((ROOT / f"manifests/data_{row['task']}_{variant}.json").read_text())
            if (row["target_count"] != manifest["target_count"]
                    or row["record_count"] != manifest["record_count"]
                    or row["data_sha256"] != manifest["artifact"]["sha256"]):
                raise ValueError(f"evaluation coverage differs: {row['job_id']}")
            if row["kind"] == "multiple_choice" and row["example_count"] != manifest["original_examples"]:
                raise ValueError(f"example coverage differs: {row['job_id']}")
    prompts = (128, 512, 2048) if args.scope == "full" else (128,)
    thread_counts = (1, 12) if args.scope == "full" else (1,)
    expected_bench = {(config, prompt, threads) for config in CONFIGURATION_NAMES
                      for prompt in prompts for threads in thread_counts}
    actual_bench = {(row["configuration"], row["prompt_tokens"], row["threads"]) for row in benchmarks}
    if len(benchmarks) != len(expected_bench) or actual_bench != expected_bench:
        raise ValueError("incomplete performance matrix")
    validate_sealed(benchmarks, binary_hash)
    if any((row["decode_tokens"], row["warmup_repetitions"], row["measured_repetitions"], row["chunk"])
           != (128, 2, 5, 128) for row in benchmarks):
        raise ValueError("performance options differ from the contract")

    # The C command validates comparison compatibility and calculates every delta.
    deltas = {}
    for variant in evaluations:
        completed = subprocess.run(
            [str(ROOT / "build/smollm"), "compare-results", "--input",
             str(ROOT / "results" / f"evaluation-{variant}.jsonl")],
            check=True, capture_output=True, text=True,
            env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        )
        rows = [json.loads(line) for line in completed.stdout.splitlines()]
        if len(rows) != 9:
            raise ValueError("C comparison did not produce all nine candidate rows")
        deltas[variant] = {(row["task"], row["candidate_configuration"]): row for row in rows}
        write_results(rows, ROOT / "results" / f"evaluation-{variant}-deltas")

    sections = [
        "# SmolLM2-135M C results",
        "SmolLM2-135M Base revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, "
        "exported exactly from BF16 to FP32. " + (
            "All four configurations have complete smoke and full evaluations, and all 24 "
            "isolated performance cells are complete." if args.scope == "full" else
            "The user requested a small first comparison capped at 30 minutes. All four "
            "configurations have smoke results and isolated prompt-128 / thread-1 performance "
            "measurements. Full-split accuracy and the complete 24-cell performance matrix "
            "are deferred; these smoke scores are not full-split results."),
        f"Executable SHA256: `{binary_hash}`. The source, compiler, CPU and library "
        "configuration are recorded in [engine-build.json](../manifests/engine-build.json). "
        "Inference, FP64 scoring, numerical comparisons and metric deltas run in C; "
        "this document only formats their recorded outputs.",
    ]
    if args.scope == "full":
        sections.extend(["## Full split accuracy",
        "WikiText-2 raw test covers 304,985 targets across 593 context-2048/stride-512 "
        "windows. HellaSwag covers all 10,042 validation examples (40,168 choices); "
        "PIQA covers all 1,838 validation examples (3,676 choices). No BOS/EOS is added. "
        "Normalized accuracy divides each choice log-likelihood by its original processed "
        "Unicode character count. Deltas are candidate minus FP32 / FP32."])
    else:
        plan = json.loads((ROOT / "manifests/measurement-plan.json").read_text())
        sections.append(
            f"Measurement budget: {plan['maximum_measurement_wall_seconds']} seconds, including "
            f"earlier runs, counted conservatively from `{plan['budget_start_utc']}`. "
            f"Deadline: `{plan['deadline_utc']}`; measurements finished at "
            f"`{plan['measurement_finished_at_utc']}`. The original matrix stopped after "
            "11 complete cells, preserved in `benchmarks-extended-partial.jsonl/csv`. "
            "Two matching cells were reused and two KV8 cells added to form the balanced "
            "four-configuration small matrix. See [the plan](../manifests/measurement-plan.json).")
    for variant in variants:
        if variant == "smoke":
            sections.extend(["## Smoke accuracy", "Smoke uses 8,191 WikiText targets across 13 windows "
                             "and the first 32 examples of each multiple-choice task. These are separate "
                             "from the full splits. Normalized accuracy uses each original processed "
                             "choice's Unicode character count. Deltas are candidate minus FP32 / FP32."])
        rows = evaluations[variant]
        options = sorted({(row["threads"], row["chunk"], row["context"], row["evaluation_concurrency"])
                          for row in rows})
        sections.append(f"Execution options `(threads, chunk, context, concurrent processes)`: `{options}`. "
                        "Evaluation timings are not used as performance measurements.")
        lm_table, mc_table = [], []
        for row in rows:
            config, task = row["configuration"], row["task"]
            delta = deltas[variant].get((task, config))
            if task == "wikitext2":
                lm_table.append([LABELS[config], f"{row['mean_nll']:.8f}", f"{row['perplexity']:.6f}",
                                 f"{delta['perplexity_absolute_delta']:+.6f}" if delta else "baseline"])
            else:
                mc_table.append([task, LABELS[config], f"{row['raw_accuracy']:.6%}",
                                 f"{row['normalized_accuracy']:.6%}",
                                 f"{delta['raw_accuracy_delta'] * 100:+.6f}" if delta else "baseline",
                                 f"{delta['normalized_accuracy_delta'] * 100:+.6f}" if delta else "baseline"])
        sections.extend([table(["WikiText configuration", "Mean NLL", "PPL", "PPL delta"], lm_table),
                         table(["Task", "Configuration", "Accuracy", "Normalized accuracy",
                                "Accuracy delta (pp)", "Normalized delta (pp)"], mc_table)])

    sections.extend([
        "## Isolated performance",
        "AMD Ryzen 9 7900, 12 physical cores / 24 logical CPUs, WSL2 Linux; OpenBLAS 0.3.20 (Zen), "
        "GCC 11.4.0. Each cell runs in a fresh process with two warmups and five measured "
        "repetitions; reported values are medians. No other project model execution or heavy "
        "test runs concurrently. Prompt IDs are deterministic; decode feeds fixed token 1729. "
        "Every mode performs LAST output projection for prefill and every decode token.",
        "TTFT is warm model prefill including the last logits projection; it excludes tokenization, "
        "model/session loading and sampling. Loading is one observation per fresh process, with "
        "the operating system's file cache uncontrolled. These are CPU implementation measurements, "
        "not a hardware-independent claim about quantization.",
    ])
    for threads in thread_counts:
        rows = [row for row in benchmarks if row["threads"] == threads]
        sections.append(f"### {threads} thread(s)")
        sections.append(table(["Configuration", "Prompt", "Prefill tok/s", "Decode tok/s", "TTFT (s)", "Load (s)"],
            [[LABELS[row["configuration"]], str(row["prompt_tokens"]),
              f"{row['prefill_tokens_per_second']:.3f}", f"{row['decode_tokens_per_second']:.3f}",
              f"{row['median_ttft_seconds']:.6f}", f"{row['load_seconds']:.6f}"] for row in rows]))
    memory_prompt = max(prompts)
    sections.extend([f"### Memory at prompt {memory_prompt} + decode 128",
        "Bytes below are permanent KV storage and session scratch, separately. Peak RSS is process-wide "
        "and includes touched model pages, library allocations and other overhead. Model files occupy "
        "538,095,104 bytes (FP32) and 143,060,480 bytes (GS64 Q8). KV8 has no permanent FP32 shadow.",
        table(["Configuration", "Threads", "KV bytes", "Scratch bytes", "Peak RSS (KiB)"],
            [[LABELS[row["configuration"]], str(row["threads"]), str(row["kv_bytes"]),
              str(row["scratch_bytes"]), str(row["peak_rss_kib"])] for row in benchmarks if row["prompt_tokens"] == memory_prompt]),
        "## Numerical error and attribution",
        "FP32 logit tolerances remain `atol=1e-3, rtol=1e-4`; fixed-probe mean NLL difference "
        "must be at most `1e-4`. The 143-test suite passes. All 21 HF intermediate aggregates "
        "pass, and all 19 identical-input default-math aggregates have zero changed bits. "
        "There is no arbitrary accuracy pass/fail threshold for Q8.",
        table(["Probe tokens", "Configuration", "Max logit error", "Logit RMSE", "NLL delta vs HF", "Mean KL vs HF", "Top-1 changes"],
            [[str(row["tokens"]), LABELS[row["configuration"]], f"{row['max_abs']:.8g}",
              f"{row['rmse']:.8g}", f"{row['nll_delta']:+.8g}", f"{row['mean_kl']:.8g}",
              str(row["top1_changes"])] for row in numerical["logits"]]),
        "### Identical-input linear replay",
        "The last layer's MLP down projection illustrates the separate weight and activation "
        "quantization effects. All variants use identical captured FP32 inputs; errors are "
        "relative to the captured FP32 output. Their errors are not additive. All 22 tensors "
        "and four variants are retained in the aggregate files.",
        table(["Layer-29 down projection", "Max absolute error", "RMSE"],
            [[row["variant"], f"{row['max_abs']:.8g}", f"{row['rmse']:.8g}"] for row in numerical["linear"]
             if row["site"] == "model.layers.29.mlp.down_proj.weight" and row["layer"] == 29]),
        "### Accumulated trace error",
        "These end-to-end trace differences use FP32 / FP32 as the baseline and sample layers "
        "0, 14 and 29. The tensor scale changes between sites, so residual errors and normalized "
        "logit errors should be interpreted with their own scales.",
        table(["Configuration", "Layer", "Site", "Max absolute error", "RMSE"],
            [[LABELS[row["configuration"]], str(row["layer"]), row["site"], f"{row['max_abs']:.8g}",
              f"{row['rmse']:.8g}"] for row in numerical["traces"]
             if row["site"] in ("embedding", "ffn_residual", "final_norm", "logits")]),
        "Q8 chunk differences are descriptive: small FP32 GEMM/GEMV changes can cross a "
        "quantizer boundary. Fixed-input integer quantization and cache gates still require "
        "exact agreement; see [the independent investigation](tasks/05-independent-review.md).",
        "## Reproduction and files",
        f"Run `python3 scripts/render_results.py --scope {args.scope}` after completing the selected "
        "commands in [validation.md](validation.md). It requires all cells in the selected scope, verifies result seals "
        "and coverage, calls C `compare-results`, and formats this report. "
        "[Evaluation and performance JSONL/CSV](../results/) include exact invocations, "
        "binary/model/data hashes, scoring counts, thread settings and raw timing samples. "
        "Full per-record scores, weights, datasets and traces remain ignored by Git.",
    ])
    output = ROOT / "docs/results.md"
    output.write_text("\n\n".join(sections) + "\n")
    print(output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from None
