from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]

HARNESS = r'''
#include "sm_cli.h"
#include <stdio.h>
#include <string.h>

int sm_command_compare_results(int argc,char **argv);

void sm_json_string(FILE *out,const char *value) {
    fputc('"',out);
    for (const unsigned char *p=(const unsigned char *)value;*p;p++) {
        if (*p=='"'||*p=='\\') { fputc('\\',out);fputc(*p,out); }
        else if (*p<32) fprintf(out,"\\u%04x",*p);
        else fputc(*p,out);
    }
    fputc('"',out);
}

int main(int argc,char **argv) {
    return sm_command_compare_results(argc,argv);
}
'''


@pytest.fixture(scope="module")
def cli(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("compare-results-cli")
    source = directory / "harness.c"
    binary = directory / "compare-results-cli"
    source.write_text(HARNESS, encoding="utf-8")
    subprocess.run(
        [
            "gcc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wpedantic",
            "-I",
            str(ROOT / "include"),
            str(ROOT / "src/compare_results.c"),
            str(source),
            "-lm",
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary


def result(
    configuration: str = "fp32_fp32",
    *,
    kind: str = "lm",
    task: str = "wikitext2",
    variant: str = "smoke",
    **updates: object,
) -> dict[str, object]:
    configurations = {
        "fp32_fp32": ("1" * 64, "a" * 64),
        "w8a8_fp32": ("2" * 64, "b" * 64),
        "fp32_kv8": ("3" * 64, "c" * 64),
        "w8a8_kv8": ("4" * 64, "d" * 64),
    }
    model_sha, binary_sha = configurations[configuration]
    row: dict[str, object] = {
        "configuration": configuration,
        "task": task,
        "variant": variant,
        "data_sha256": "e" * 64,
        "model_revision": "f" * 40,
        "model_sha256": model_sha,
        "binary_sha256": binary_sha,
        "command": "eval",
        "kind": kind,
        "record_count": 5,
        "target_count": 100,
        "context": 2048,
        "chunk": 128,
        "threads": 1,
        "mean_nll": 2.0,
    }
    if kind == "lm":
        row["perplexity"] = math.exp(2.0)
    else:
        row.update(example_count=5, raw_accuracy=0.6, normalized_accuracy=0.4)
    row.update(updates)
    return row


def invoke(
    cli: Path,
    tmp_path: Path,
    rows: list[dict[str, object]] | None = None,
    *,
    raw: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    input_path = tmp_path / "results.jsonl"
    if raw is None:
        raw = b"".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in (rows or [])
        )
    input_path.write_bytes(raw)
    return subprocess.run(
        [str(cli), "compare-results", "--input", str(input_path)],
        text=True,
        capture_output=True,
        check=False,
    )


def output_rows(completed: subprocess.CompletedProcess[str]) -> list[dict[str, object]]:
    assert completed.returncode == 0, completed.stderr
    return [json.loads(line) for line in completed.stdout.splitlines()]


def test_lm_delta_and_attribution_are_computed_in_c(cli: Path, tmp_path: Path) -> None:
    baseline = result(
        mean_nll=2.5,
        perplexity=10.0,
        metadata={"task": "nested-decoy", "items": [1, {"chunk": 999}, "한글 a\"b"]},
    )
    candidate = result(
        "w8a8_kv8",
        mean_nll=2.75,
        perplexity=12.5,
        metadata=[{"data_sha256": "0" * 64}, True, None],
    )

    rows = output_rows(invoke(cli, tmp_path, [candidate, baseline]))

    assert len(rows) == 1
    row = rows[0]
    assert row["command"] == "compare-results"
    assert row["task"] == "wikitext2"
    assert row["variant"] == "smoke"
    assert row["baseline_configuration"] == "fp32_fp32"
    assert row["candidate_configuration"] == "w8a8_kv8"
    assert row["baseline_model_sha256"] == "1" * 64
    assert row["candidate_model_sha256"] == "4" * 64
    assert row["baseline_binary_sha256"] == "a" * 64
    assert row["candidate_binary_sha256"] == "d" * 64
    assert row["data_sha256"] == "e" * 64
    assert row["model_revision"] == "f" * 40
    assert row["mean_nll_delta"] == pytest.approx(0.25)
    assert row["perplexity_absolute_delta"] == pytest.approx(2.5)
    assert row["perplexity_relative_delta"] == pytest.approx(0.25)


def test_mc_accuracy_deltas_and_sorted_groups(cli: Path, tmp_path: Path) -> None:
    baseline = result(
        kind="multiple_choice",
        task="piqa",
        variant="full",
        mean_nll=1.5,
        raw_accuracy=0.625,
        normalized_accuracy=0.5,
    )
    candidate = result(
        "fp32_kv8",
        kind="multiple_choice",
        task="piqa",
        variant="full",
        mean_nll=1.25,
        raw_accuracy=0.75,
        normalized_accuracy=0.375,
    )
    other_baseline = result(task="wikitext2", variant="full")
    other_candidate = result("w8a8_fp32", task="wikitext2", variant="full", perplexity=8.0)

    rows = output_rows(invoke(cli, tmp_path, [other_candidate, candidate, other_baseline, baseline]))

    assert [(row["task"], row["candidate_configuration"]) for row in rows] == [
        ("piqa", "fp32_kv8"),
        ("wikitext2", "w8a8_fp32"),
    ]
    row = rows[0]
    assert row["example_count"] == 5
    assert row["mean_nll_delta"] == pytest.approx(-0.25)
    assert row["raw_accuracy_delta"] == pytest.approx(0.125)
    assert row["normalized_accuracy_delta"] == pytest.approx(-0.125)
    assert "perplexity_relative_delta" not in row


@pytest.mark.parametrize(
    ("candidate_update", "message"),
    [
        ({"data_sha256": "0" * 64}, "data_sha256"),
        ({"model_revision": "0" * 40}, "model_revision"),
        ({"record_count": 6}, "record_count"),
        ({"target_count": 99}, "target_count"),
        ({"context": 1024}, "context"),
        ({"chunk": 64}, "chunk"),
        ({"threads": 2}, "threads"),
        ({"perplexity": "not-a-number"}, "field must be a number"),
    ],
)
def test_rejects_incompatible_or_invalid_rows(
    cli: Path,
    tmp_path: Path,
    candidate_update: dict[str, object],
    message: str,
) -> None:
    candidate = result("w8a8_fp32")
    candidate.update(candidate_update)
    completed = invoke(cli, tmp_path, [result(), candidate])
    assert completed.returncode != 0
    assert message in completed.stderr
    assert completed.stdout == ""


def test_rejects_mc_example_count_mismatch(cli: Path, tmp_path: Path) -> None:
    baseline = result(kind="multiple_choice")
    candidate = result("w8a8_fp32", kind="multiple_choice", example_count=4)
    completed = invoke(cli, tmp_path, [candidate, baseline])
    assert completed.returncode != 0
    assert "example_count" in completed.stderr


def test_rejects_benchmark_kind_mismatch(cli: Path, tmp_path: Path) -> None:
    baseline = result()
    candidate = result("w8a8_fp32", kind="multiple_choice")
    completed = invoke(cli, tmp_path, [candidate, baseline])
    assert completed.returncode != 0
    assert "kind" in completed.stderr


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([result("w8a8_fp32")], "0 fp32_fp32 baselines"),
        ([result(), result()], "2 fp32_fp32 baselines"),
        ([result(), result("w8a8_fp32"), result("w8a8_fp32")], "duplicate configuration"),
    ],
)
def test_requires_one_baseline_and_unique_candidates(
    cli: Path,
    tmp_path: Path,
    rows: list[dict[str, object]],
    message: str,
) -> None:
    completed = invoke(cli, tmp_path, rows)
    assert completed.returncode != 0
    assert message in completed.stderr
    assert completed.stdout == ""


@pytest.mark.parametrize(
    "raw",
    [
        b'{"configuration":"fp32_fp32",}\n',
        b'{"x":1,"x":2}\n',
        b'{"metadata":{"x":1,"x":2}}\n',
        b'[]\n',
        b'{"mean_nll":NaN}\n',
        b'{"mean_nll":1e999}\n',
        b'{"context":"2048"}\n',
        b'{"x":[1,2,]}\n',
        b'{"x":"bad\\q"}\n',
        b'{"x":"bad\\u0000value"}\n',
        b'{"x":"\xc0\x80"}\n',
    ],
)
def test_rejects_malformed_json_duplicate_keys_and_invalid_types(
    cli: Path, tmp_path: Path, raw: bytes
) -> None:
    completed = invoke(cli, tmp_path, raw=raw)
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "line 1" in completed.stderr


def test_rejects_nonfinite_comparison_arithmetic(cli: Path, tmp_path: Path) -> None:
    baseline = result(perplexity=1e-300)
    candidate = result("w8a8_fp32", perplexity=1e308)
    completed = invoke(cli, tmp_path, [baseline, candidate])
    assert completed.returncode != 0
    assert "arithmetic is nonfinite" in completed.stderr
    assert completed.stdout == ""


def test_rejects_empty_nul_and_oversize_input(cli: Path, tmp_path: Path) -> None:
    assert invoke(cli, tmp_path, raw=b"\n \t\n").returncode != 0
    nul = invoke(cli, tmp_path, raw=b'{}\x00\n')
    assert nul.returncode != 0
    assert "not text" in nul.stderr
    oversize = invoke(cli, tmp_path, raw=b" " * (1024 * 1024 + 1) + b"\n")
    assert oversize.returncode != 0
    assert "too long" in oversize.stderr


def test_cli_shape_is_exact(cli: Path, tmp_path: Path) -> None:
    completed = subprocess.run([str(cli), "compare-results"], text=True, capture_output=True)
    assert completed.returncode != 0
    assert "usage:" in completed.stderr
    completed = subprocess.run(
        [str(cli), "compare-results", "--wrong", str(tmp_path / "x")],
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "usage:" in completed.stderr
