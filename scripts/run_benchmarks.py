#!/usr/bin/env python3
"""Run serialized four-configuration SmolLM2 evaluation or performance matrices."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = (
    ("fp32_fp32", "f32", "f32"),
    ("w8a8_fp32", "q8", "f32"),
    ("fp32_kv8", "f32", "q8"),
    ("w8a8_kv8", "q8", "q8"),
)
CONFIGURATION_NAMES = tuple(item[0] for item in CONFIGURATIONS)


def selected_configurations(args: argparse.Namespace) -> tuple[tuple[str, str, str], ...]:
    selected = set(args.configurations)
    return tuple(item for item in CONFIGURATIONS if item[0] in selected)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"manifest {path} is not a JSON object")
    return value


def checked_hash(path: Path, expected: str | None, cache: dict[Path, str]) -> str:
    resolved = path.resolve()
    if resolved not in cache:
        if not path.is_file():
            raise SystemExit(f"missing artifact: {path}")
        cache[resolved] = sha256_file(path)
    actual = cache[resolved]
    if expected is not None and actual != expected:
        raise SystemExit(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def git_metadata() -> dict[str, Any]:
    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments], text=True, capture_output=True, check=False
        )

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(status.stdout) if status.returncode == 0 else None,
    }


def run_json(command: list[str], environment: dict[str, str]) -> dict[str, Any]:
    # Keep stdout machine-readable while allowing long-running C progress on
    # stderr to reach the terminal immediately.
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        env=environment,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"command did not emit exactly one JSON line: {' '.join(command)}")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"command emitted invalid JSON: {' '.join(command)}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"command JSON is not an object: {' '.join(command)}")
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_results(rows: list[dict[str, Any]], stem: Path) -> tuple[Path, Path]:
    jsonl_path = stem.with_suffix(".jsonl")
    csv_path = stem.with_suffix(".csv")
    jsonl = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    fields = sorted({field for row in rows for field in row})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else value
                for field, value in row.items()
            }
        )
    atomic_text(jsonl_path, jsonl)
    atomic_text(csv_path, output.getvalue())
    return jsonl_path, csv_path


def json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def seal_result(row: dict[str, Any]) -> None:
    row["result_sha256"] = json_fingerprint(row)


def verify_result(row: dict[str, Any], job: dict[str, Any]) -> None:
    expected_result = row.get("result_sha256")
    unsealed = {key: value for key, value in row.items() if key != "result_sha256"}
    if not isinstance(expected_result, str) or json_fingerprint(unsealed) != expected_result:
        raise SystemExit(f"checkpoint result hash mismatch for {job['job_id']}")
    expected_fields = {
        "job_id": job["job_id"],
        "run_fingerprint": job["fingerprint"],
        "configuration": job["configuration"],
        "invocation": job["command"],
        "model_sha256": job["model"]["sha256"],
    }
    for field, expected in expected_fields.items():
        if row.get(field) != expected:
            raise SystemExit(f"checkpoint field {field} differs for {job['job_id']}")


def load_checkpoint(path: Path, plan_sha256: str, restart: bool) -> dict[str, dict[str, Any]]:
    if restart or not path.exists():
        return {}
    checkpoint = load_json(path)
    if checkpoint.get("plan_sha256") != plan_sha256:
        raise SystemExit(
            f"checkpoint plan differs from this invocation: {path}; use --restart to start a new matrix"
        )
    rows = checkpoint.get("completed")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise SystemExit(f"invalid checkpoint rows: {path}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        job_id = row.get("job_id")
        if not isinstance(job_id, str) or job_id in result:
            raise SystemExit(f"invalid or duplicate checkpoint job: {path}")
        result[job_id] = row
    return result


def save_checkpoint(
    path: Path, plan_sha256: str, plan: list[dict[str, Any]], completed: dict[str, dict[str, Any]]
) -> None:
    ordered = [completed[item["job_id"]] for item in plan if item["job_id"] in completed]
    value = {"format": "smollm-matrix-checkpoint-v1", "plan_sha256": plan_sha256, "completed": ordered}
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def model_information(args: argparse.Namespace, cache: dict[Path, str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    required_storage = {storage for _, storage, _ in selected_configurations(args)}
    for storage, model_path, manifest_path in (
        ("f32", args.f32_model, args.f32_manifest),
        ("q8", args.q8_model, args.q8_manifest),
    ):
        if storage not in required_storage:
            continue
        manifest = load_json(manifest_path)
        expected = manifest.get("output", {}).get("sha256")
        if not isinstance(expected, str):
            raise SystemExit(f"model manifest lacks output SHA256: {manifest_path}")
        result[storage] = {
            "path": model_path,
            "sha256": checked_hash(model_path, expected, cache),
            "manifest_path": manifest_path,
            "manifest_sha256": checked_hash(manifest_path, None, cache),
            "revision": manifest.get("source", {}).get("revision"),
        }
    return result


def provenance(
    *,
    args: argparse.Namespace,
    command: list[str],
    configuration: str,
    model: dict[str, Any],
    binary_sha256: str,
    git: dict[str, Any],
) -> dict[str, Any]:
    return {
        "configuration": configuration,
        "invocation": command,
        "binary_path": str(args.binary),
        "binary_sha256": binary_sha256,
        "model_sha256": model["sha256"],
        "model_manifest": str(model["manifest_path"]),
        "model_manifest_sha256": model["manifest_sha256"],
        "manifest_model_revision": model["revision"],
        "orchestrator_python": platform.python_version(),
        "orchestrator_platform": platform.platform(),
        **git,
    }


def evaluation(args: argparse.Namespace) -> list[dict[str, Any]]:
    cache: dict[Path, str] = {}
    binary_sha256 = checked_hash(args.binary, None, cache)
    models = model_information(args, cache)
    git = git_metadata()
    variants = ("smoke", "full") if args.variant == "all" else (args.variant,)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plan: list[dict[str, Any]] = []
    for variant in variants:
        for task in args.tasks:
            data_path = args.data_dir / f"{task}_{variant}.smeval"
            data_manifest_path = args.manifest_dir / f"data_{task}_{variant}.json"
            data_manifest = load_json(data_manifest_path)
            expected = data_manifest.get("artifact", {}).get("sha256")
            if not isinstance(expected, str):
                raise SystemExit(f"data manifest lacks artifact SHA256: {data_manifest_path}")
            data_sha256 = checked_hash(data_path, expected, cache)
            data_manifest_sha256 = checked_hash(data_manifest_path, None, cache)
            for name, storage, kv in selected_configurations(args):
                model = models[storage]
                score_path = output_dir / "scores" / f"{task}-{variant}-{name}.jsonl"
                command = [
                    str(args.binary), "eval", "--model", str(model["path"]),
                    "--data", str(data_path), "--manifest", str(data_manifest_path),
                    "--kv", kv, "--context", str(args.context), "--chunk", str(args.chunk),
                    "--threads", str(args.threads), "--progress-every", str(args.progress_every),
                    "--scores", str(score_path),
                ]
                job_id = f"{task}:{variant}:{name}"
                identity = {
                    "job_id": job_id,
                    "command": command,
                    "binary_sha256": binary_sha256,
                    "model_sha256": model["sha256"],
                    "model_manifest_sha256": model["manifest_sha256"],
                    "data_sha256": data_sha256,
                    "data_manifest_sha256": data_manifest_sha256,
                    "evaluation_concurrency": args.jobs,
                }
                plan.append(
                    {
                        "job_id": job_id,
                        "fingerprint": json_fingerprint(identity),
                        "command": command,
                        "configuration": name,
                        "model": model,
                        "data_sha256": data_sha256,
                        "data_manifest_sha256": data_manifest_sha256,
                        "data_revision": data_manifest.get("source_revision"),
                        "task": task,
                        "variant": variant,
                        "score_path": score_path,
                    }
                )
    if args.dry_run:
        for job in plan:
            print(json.dumps({"job_id": job["job_id"], "command": job["command"]}))
        return []

    plan_sha256 = json_fingerprint(
        [{"job_id": job["job_id"], "fingerprint": job["fingerprint"]} for job in plan]
    )
    stem = output_dir / (args.output_stem or f"evaluation-{args.variant}")
    checkpoint_path = output_dir / f".{stem.name}.checkpoint.json"
    completed = load_checkpoint(checkpoint_path, plan_sha256, args.restart)
    pending: list[dict[str, Any]] = []
    for job in plan:
        cached = completed.get(job["job_id"])
        if cached is not None:
            verify_result(cached, job)
            if cached.get("data_sha256") != job["data_sha256"] or cached.get(
                "data_manifest_sha256"
            ) != job["data_manifest_sha256"]:
                raise SystemExit(f"checkpoint data hashes differ for {job['job_id']}")
            expected_score_hash = cached.get("scores_sha256")
            if not isinstance(expected_score_hash, str):
                raise SystemExit(f"checkpoint lacks score hash for {job['job_id']}")
            checked_hash(job["score_path"], expected_score_hash, cache)
            continue

        pending.append(job)

    def run_job(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        job["score_path"].parent.mkdir(parents=True, exist_ok=True)
        environment = dict(
            os.environ,
            OPENBLAS_NUM_THREADS=str(args.threads),
            OMP_NUM_THREADS=str(args.threads),
        )
        return job, run_json(job["command"], environment)

    def finish_job(job: dict[str, Any], metric: dict[str, Any]) -> None:
        metric.update(
            provenance(
                args=args,
                command=job["command"],
                configuration=job["configuration"],
                model=job["model"],
                binary_sha256=binary_sha256,
                git=git,
            )
        )
        metric.update(
            {
                "job_id": job["job_id"],
                "run_fingerprint": job["fingerprint"],
                "task": job["task"],
                "variant": job["variant"],
                "data_sha256": job["data_sha256"],
                "data_manifest_sha256": job["data_manifest_sha256"],
                "manifest_data_revision": job["data_revision"],
                "scores_path": str(job["score_path"]),
                "scores_sha256": checked_hash(job["score_path"], None, cache),
                "evaluation_concurrency": args.jobs,
                "timing_isolated": args.jobs == 1,
                "timing_scope": "isolated" if args.jobs == 1 else "concurrent_not_isolated",
            }
        )
        seal_result(metric)
        completed[job["job_id"]] = metric
        save_checkpoint(checkpoint_path, plan_sha256, plan, completed)

    if args.jobs == 1:
        for job in pending:
            finished_job, metric = run_job(job)
            finish_job(finished_job, metric)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs, thread_name_prefix="smollm-eval") as executor:
            futures: dict[Future[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]] = {
                executor.submit(run_job, job): job for job in pending
            }
            try:
                for future in as_completed(futures):
                    finished_job, metric = future.result()
                    finish_job(finished_job, metric)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    rows = [completed[job["job_id"]] for job in plan]
    write_results(rows, stem)
    return rows


def performance(args: argparse.Namespace) -> list[dict[str, Any]]:
    cache: dict[Path, str] = {}
    binary_sha256 = checked_hash(args.binary, None, cache)
    models = model_information(args, cache)
    git = git_metadata()
    plan: list[dict[str, Any]] = []
    for name, storage, kv in selected_configurations(args):
        model = models[storage]
        for prompt in args.prompts:
            for threads in args.thread_counts:
                command = [
                    str(args.binary),
                    "bench",
                    "--model",
                    str(model["path"]),
                    "--kv",
                    kv,
                    "--context",
                    str(prompt + args.decode),
                    "--chunk",
                    str(args.chunk),
                    "--threads",
                    str(threads),
                    "--prompt",
                    str(prompt),
                    "--decode",
                    str(args.decode),
                    "--warmup",
                    str(args.warmup),
                    "--repetitions",
                    str(args.repetitions),
                ]
                job_id = f"{name}:prompt{prompt}:threads{threads}"
                identity = {
                    "job_id": job_id,
                    "command": command,
                    "binary_sha256": binary_sha256,
                    "model_sha256": model["sha256"],
                    "model_manifest_sha256": model["manifest_sha256"],
                }
                plan.append(
                    {
                        "job_id": job_id,
                        "fingerprint": json_fingerprint(identity),
                        "command": command,
                        "configuration": name,
                        "model": model,
                        "threads": threads,
                    }
                )
    if args.dry_run:
        for job in plan:
            print(json.dumps({"job_id": job["job_id"], "command": job["command"]}))
        return []

    plan_sha256 = json_fingerprint(
        [{"job_id": job["job_id"], "fingerprint": job["fingerprint"]} for job in plan]
    )
    stem = args.output_dir / (args.output_stem or "benchmarks")
    checkpoint_path = args.output_dir / f".{stem.name}.checkpoint.json"
    completed = load_checkpoint(checkpoint_path, plan_sha256, args.restart)
    rows: list[dict[str, Any]] = []
    for job in plan:
        cached = completed.get(job["job_id"])
        if cached is not None:
            verify_result(cached, job)
            rows.append(cached)
            continue
        environment = dict(
            os.environ,
            OPENBLAS_NUM_THREADS=str(job["threads"]),
            OMP_NUM_THREADS=str(job["threads"]),
        )
        metric = run_json(job["command"], environment)
        metric.update(
            provenance(
                args=args,
                command=job["command"],
                configuration=job["configuration"],
                model=job["model"],
                binary_sha256=binary_sha256,
                git=git,
            )
        )
        metric.update({"job_id": job["job_id"], "run_fingerprint": job["fingerprint"]})
        seal_result(metric)
        completed[job["job_id"]] = metric
        rows.append(metric)
        save_checkpoint(checkpoint_path, plan_sha256, plan, completed)
    write_results(rows, stem)
    return rows


def common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--binary", type=Path, default=ROOT / "build/smollm")
    parser.add_argument("--f32-model", type=Path, default=ROOT / "models/smollm2-f32.bin")
    parser.add_argument("--q8-model", type=Path, default=ROOT / "models/smollm2-q8.bin")
    parser.add_argument("--f32-manifest", type=Path, default=ROOT / "manifests/model-f32.json")
    parser.add_argument("--q8-manifest", type=Path, default=ROOT / "manifests/model-q8.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=CONFIGURATION_NAMES,
        default=CONFIGURATION_NAMES,
        help="matrix configurations to run (default: all four)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--restart", action="store_true", help="ignore an existing matrix checkpoint"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("eval", help="run serialized scoring evaluations")
    common_options(evaluate)
    evaluate.add_argument("--data-dir", type=Path, default=ROOT / "data")
    evaluate.add_argument("--manifest-dir", type=Path, default=ROOT / "manifests")
    evaluate.add_argument("--variant", choices=("smoke", "full", "all"), default="smoke")
    evaluate.add_argument("--tasks", nargs="+", choices=("wikitext2", "hellaswag", "piqa"), default=("wikitext2", "hellaswag", "piqa"))
    evaluate.add_argument("--context", type=int, default=2048)
    evaluate.add_argument("--threads", type=int, default=1)
    evaluate.add_argument(
        "--jobs", type=int, default=1, help="concurrent evaluation subprocesses"
    )
    evaluate.add_argument("--progress-every", type=int, default=100)
    evaluate.add_argument("--output-stem")

    bench = commands.add_parser("bench", help="run serialized fresh-process performance matrix")
    common_options(bench)
    bench.add_argument("--prompts", nargs="+", type=int, choices=(128, 512, 2048), default=(128, 512, 2048))
    bench.add_argument("--thread-counts", nargs="+", type=int, default=(1, 12))
    bench.add_argument("--decode", type=int, default=128)
    bench.add_argument("--warmup", type=int, default=2)
    bench.add_argument("--repetitions", type=int, default=5)
    bench.add_argument("--output-stem")
    arguments = parser.parse_args()
    positive = [arguments.chunk]
    if arguments.command == "eval":
        positive.extend((arguments.context, arguments.threads, arguments.jobs))
        if arguments.progress_every < 0:
            parser.error("--progress-every must be nonnegative")
        if len(set(arguments.tasks)) != len(arguments.tasks):
            parser.error("--tasks must not contain duplicates")
    else:
        positive.extend((*arguments.thread_counts, arguments.decode, arguments.repetitions))
        if arguments.warmup < 0:
            parser.error("--warmup must be nonnegative")
        if len(set(arguments.prompts)) != len(arguments.prompts) or len(set(arguments.thread_counts)) != len(arguments.thread_counts):
            parser.error("--prompts and --thread-counts must not contain duplicates")
    if any(value <= 0 for value in positive):
        parser.error("sizes, jobs, and thread counts must be positive")
    if len(set(arguments.configurations)) != len(arguments.configurations):
        parser.error("--configurations must not contain duplicates")
    return arguments


def main() -> int:
    arguments = parse_args()
    try:
        rows = evaluation(arguments) if arguments.command == "eval" else performance(arguments)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if not arguments.dry_run:
        print(f"completed {len(rows)} fresh-process runs in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
