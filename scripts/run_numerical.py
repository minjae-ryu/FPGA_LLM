#!/usr/bin/env python3
"""Launch C numerical comparisons and attach reproducibility metadata.

All error/NLL/KL/replay calculations are performed by the C commands. Python
only validates hashes, orchestrates them, and serializes their output records.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from run_benchmarks import checked_hash, git_metadata, load_json, write_results

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/smollm")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "traces/numerical")
    args = parser.parse_args()
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    binary_hash = checked_hash(args.binary, None, cache)
    provenance = {"binary_sha256": binary_hash, **git_metadata(), "threads": 1}
    models = {}
    for dtype in ("f32", "q8"):
        manifest = load_json(ROOT / f"manifests/model-{dtype}.json")
        path = ROOT / manifest["output"]["path"]
        models[dtype] = (path, checked_hash(path, manifest["output"]["sha256"], cache))
    references = {}
    for probe in ("probe", "long"):
        manifest = load_json(ROOT / f"manifests/reference_{probe}.json")
        references[probe] = manifest
        for item in [*manifest["artifacts"].values(), *manifest["intermediates"]]:
            checked_hash(ROOT / item["path"], item["sha256"], cache)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    groups = {"logits": [], "chunks": [], "intermediates": [], "math": [], "linear": [], "traces": []}

    def execute(group, command, **metadata):
        completed = subprocess.run([str(args.binary), *map(str, command)], cwd=ROOT,
                                   env=environment, stdout=subprocess.PIPE, text=True, check=True)
        records = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        if not records:
            raise RuntimeError(f"C command returned no records: {command}")
        for row in records:
            row.update(provenance)
            row.update(metadata)
            row["invocation"] = [str(args.binary), *map(str, command)]
            groups[group].append(row)

    configurations = (("fp32_fp32", "f32", "f32"), ("w8a8_fp32", "q8", "f32"),
                      ("fp32_kv8", "f32", "q8"), ("w8a8_kv8", "q8", "q8"))
    traces = {}
    for name, dtype, kv in configurations:
        model, model_hash = models[dtype]
        for probe in ("probe", "long"):
            ref = references[probe]
            command = ["logits", "--model", model, "--kv", kv, "--tokens", ROOT/ref["artifacts"]["tokens"]["path"],
                       "--reference", ROOT/ref["artifacts"]["logits"]["path"], "--context", "512", "--threads", "1"]
            if name != "fp32_fp32": command.append("--report-only")
            if probe == "probe":
                traces[name] = args.trace_dir / f"{name}.smtrace"
                command.extend(["--trace", traces[name]])
            execute("logits", command, configuration=name, probe=probe, model_sha256=model_hash,
                    tokens_sha256=ref["artifacts"]["tokens"]["sha256"],
                    reference_logits_sha256=ref["artifacts"]["logits"]["sha256"],
                    comparison_policy="fp32_gate" if name=="fp32_fp32" else "descriptive_q8_no_threshold")
        ref = references["long"]
        command = ["chunks", "--model", model, "--kv", kv, "--tokens", ROOT/ref["artifacts"]["tokens"]["path"],
                   "--context", "512", "--threads", "1"]
        if name != "fp32_fp32": command.append("--report-only")
        execute("chunks", command, configuration=name, model_sha256=model_hash,
                tokens_sha256=ref["artifacts"]["tokens"]["sha256"],
                comparison_policy="fp32_gate" if name=="fp32_fp32" else "descriptive_q8_no_threshold")
    trace = traces["fp32_fp32"]
    trace_hash = checked_hash(trace, None, cache)
    execute("intermediates", ["replay", "--mode", "compare", "--trace", trace,
                              "--reference-prefix", ROOT/"artifacts/reference/probe"], trace_sha256=trace_hash)
    execute("math", ["replay", "--mode", "math", "--trace", trace], trace_sha256=trace_hash)
    execute("linear", ["replay", "--mode", "linear", "--trace", trace,
                       "--model", models["f32"][0], "--q8-model", models["q8"][0]],
            trace_sha256=trace_hash, model_sha256=models["f32"][1], q8_model_sha256=models["q8"][1])
    for name, _, _ in configurations[1:]:
        execute("traces", ["replay", "--mode", "traces", "--trace", trace, "--other", traces[name]],
                configuration=name, baseline_trace_sha256=trace_hash,
                candidate_trace_sha256=checked_hash(traces[name], None, cache))
    # Detect a binary replaced midway through a long numerical run.
    if hashlib.sha256(args.binary.read_bytes()).hexdigest() != binary_hash:
        raise RuntimeError("the numerical executable changed during the run")
    for name, rows in groups.items():
        write_results(rows, args.output_dir/f"numerical-{name}")
    print(json.dumps({"status":"passed", "binary_sha256":binary_hash,
                      "groups": {name:len(rows) for name,rows in groups.items()}}))


if __name__ == "__main__":
    main()
