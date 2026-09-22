from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_smollm
from smollm_format import parse_model


def trace_record(
    site: str,
    output,
    *,
    input_values=(),
    layer: int = -1,
    head: int = -1,
    position: int = 0,
    rows: int = 1,
    columns: int | None = None,
    causal: int = -1,
) -> bytes:
    encoded = site.encode("ascii")
    assert 0 < len(encoded) < 64
    inputs = np.asarray(input_values, dtype="<f4").reshape(-1)
    outputs = np.asarray(output, dtype="<f4").reshape(-1)
    if columns is None:
        assert outputs.size % rows == 0
        columns = outputs.size // rows
    header = bytearray(128)
    header[: len(encoded)] = encoded
    struct.pack_into("<iiQIIi", header, 64, layer, head, position, rows, columns, causal)
    struct.pack_into("<QQ", header, 96, inputs.size, outputs.size)
    return bytes(header) + inputs.tobytes() + outputs.tobytes()


def write_trace(path: Path, records: list[bytes]) -> None:
    path.write_bytes(b"SMTRC001" + b"".join(records))


def run(cli: Path, *arguments: str) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    return subprocess.run(
        [str(cli), "replay", *map(str, arguments)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )


def json_lines(process: subprocess.CompletedProcess) -> list[dict]:
    return [json.loads(line) for line in process.stdout.splitlines()]


@pytest.fixture(scope="module")
def replay_cli(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("replay-build")
    wrapper = directory / "main.c"
    executable = directory / "replay-cli"
    wrapper.write_text(
        '#include "sm_cli.h"\nint main(int argc,char **argv) '
        "{ return sm_command_replay(argc,argv); }\n"
    )
    cflags = subprocess.check_output(
        ["pkg-config", "--cflags", "openblas", "libpcre2-8"], text=True
    ).split()
    libraries = subprocess.check_output(
        ["pkg-config", "--libs", "openblas", "libpcre2-8"], text=True
    ).split()
    sources = [
        wrapper,
        ROOT / "src/replay.c",
        ROOT / "src/cli_common.c",
        ROOT / "src/model.c",
        ROOT / "src/kernels.c",
        ROOT / "src/math_ops.c",
        ROOT / "src/session.c",
        ROOT / "src/cache.c",
        ROOT / "src/scoring.c",
    ]
    subprocess.run(
        [
            "cc",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wpedantic",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-fopenmp",
            "-I",
            str(ROOT / "include"),
            *cflags,
            *map(str, sources),
            *libraries,
            "-lm",
            "-fopenmp",
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


def exact_math_records() -> list[bytes]:
    libm = ctypes.CDLL(None)
    for name in ("expf", "sqrtf", "sinf", "cosf"):
        function = getattr(libm, name)
        function.argtypes = [ctypes.c_float]
        function.restype = ctypes.c_float

    def f32(value) -> np.float32:
        return np.float32(value)

    rms_input = f32(4.25)
    rms_output = f32(1.0) / f32(libm.sqrtf(float(rms_input)))
    softmax_input = np.array([-3.5, -1.0, 0.0], dtype=np.float32)
    softmax_output = np.array([libm.expf(float(value)) for value in softmax_input], dtype=np.float32)
    softmax_sum = f32(softmax_output.sum(dtype=np.float32))
    reciprocal = f32(1.0) / softmax_sum
    silu_input = np.array([-2.25, 0.75], dtype=np.float32)
    silu_output = np.array([libm.expf(float(value)) for value in silu_input], dtype=np.float32)
    silu_denominator = np.asarray(1.0 + silu_output, dtype=np.float32)
    silu_reciprocal = np.asarray(np.float32(1.0) / silu_denominator, dtype=np.float32)
    angles = np.array([0.0, 0.125, 3.0], dtype=np.float32)
    sincos = np.array(
        [libm.sinf(float(value)) for value in angles]
        + [libm.cosf(float(value)) for value in angles],
        dtype=np.float32,
    )
    return [
        trace_record("rms.rsqrt", [rms_output], input_values=[rms_input], layer=0),
        trace_record("softmax.exp", softmax_output, input_values=softmax_input, layer=0, head=1, causal=1),
        trace_record("softmax.reciprocal", [reciprocal], input_values=[softmax_sum], layer=0, head=1, causal=1),
        trace_record("silu.exp", silu_output, input_values=silu_input, layer=14),
        trace_record("silu.reciprocal", silu_reciprocal, input_values=silu_denominator, layer=14),
        trace_record("rope.sincos", sincos, input_values=angles, layer=29),
    ]


def test_math_replay_is_bit_exact(replay_cli, tmp_path):
    trace = tmp_path / "math.trace"
    write_trace(trace, exact_math_records())
    process = run(replay_cli, "--trace", trace, "--mode", "math")
    assert process.returncode == 0, process.stderr
    reports = json_lines(process)
    assert len(reports) == 6
    assert all(report["exact_pass"] for report in reports)
    assert all(report["bit_changes"] == 0 for report in reports)

    records = exact_math_records()
    damaged = bytearray(records[0])
    damaged[-1] ^= 1
    records[0] = bytes(damaged)
    write_trace(trace, records)
    process = run(replay_cli, "--trace", trace, "--mode", "math")
    assert process.returncode == 2
    assert any(not report["exact_pass"] for report in json_lines(process))


def test_trace_to_trace_aggregates_without_q8_gate(replay_cli, tmp_path):
    first = tmp_path / "first.trace"
    second = tmp_path / "second.trace"
    common = [
        trace_record("attn_norm", [1.0, 2.0, 3.0], layer=0),
        trace_record("logits", [0.0, 4.0, 1.0, 2.0], position=2),
    ]
    changed = [
        trace_record("attn_norm", [1.25, 2.0, 3.0], layer=0),
        trace_record("logits", [0.0, 1.0, 5.0, 2.0], position=2),
    ]
    write_trace(first, common)
    write_trace(second, changed)
    process = run(replay_cli, "--trace", first, "--mode", "traces", "--other", second)
    assert process.returncode == 0, process.stderr
    reports = json_lines(process)
    assert {report["site"] for report in reports} == {"attn_norm", "logits"}
    logits = next(report for report in reports if report["site"] == "logits")
    assert logits["top1_changes"] == 1
    assert logits["max_abs"] == 4.0

    write_trace(second, changed[:1])
    process = run(replay_cli, "--trace", first, "--mode", "traces", "--other", second)
    assert process.returncode == 1
    assert "record counts differ" in process.stderr


def compare_fixture(tmp_path: Path, mismatch: bool = False) -> tuple[Path, Path]:
    sites = ["attn_norm", "q_rope", "k_rope", "attention", "attn_residual", "ffn_norm", "ffn_residual"]
    records = []
    prefix = tmp_path / ("bad-reference" if mismatch else "reference")
    for layer in (0, 14, 29):
        for site_index, site in enumerate(sites):
            width = 192 if site == "k_rope" else 576
            values = (np.arange(width, dtype=np.float32) * np.float32(0.001) + layer + site_index)
            reference = values.copy()
            if mismatch and layer == 14 and site == "attention":
                values[3] += np.float32(0.1)
            Path(f"{prefix}.layer{layer}.{site}.f32").write_bytes(reference.astype("<f4").tobytes())
            records.append(trace_record(site, values, layer=layer))
    trace = tmp_path / ("bad-compare.trace" if mismatch else "compare.trace")
    write_trace(trace, records)
    return trace, prefix


def test_compare_mode_checks_all_reference_sites(replay_cli, tmp_path):
    trace, prefix = compare_fixture(tmp_path)
    process = run(replay_cli, "--trace", trace, "--mode", "compare", "--reference-prefix", prefix)
    assert process.returncode == 0, process.stderr
    reports = json_lines(process)
    assert len(reports) == 21
    assert all(report["parity_pass"] for report in reports)

    trace, prefix = compare_fixture(tmp_path, mismatch=True)
    process = run(replay_cli, "--trace", trace, "--mode", "compare", "--reference-prefix", prefix)
    assert process.returncode == 2
    failed = [report for report in json_lines(process) if not report["parity_pass"]]
    assert [(report["layer"], report["site"]) for report in failed] == [(14, "attention")]


@pytest.fixture(scope="module")
def synthetic_models(tmp_path_factory):
    directory = tmp_path_factory.mktemp("replay-models")
    fp32 = directory / "f32.bin"
    q8 = directory / "q8.bin"
    export_smollm.export_synthetic(fp32, "f32")
    export_smollm.export_synthetic(q8, "q8")
    return fp32, q8


def linear_trace(path: Path, model: Path, *, corrupt: bool = False) -> None:
    parsed = parse_model(model)
    records = []
    for index, (name, layer, position) in enumerate(
        [
            ("model.layers.0.self_attn.q_proj.weight", 0, 0),
            ("model.embed_tokens.weight", -1, 56),
        ]
    ):
        descriptor = next(item for item in parsed.descriptors if item.name == name)
        with model.open("rb") as stream:
            stream.seek(descriptor.offset)
            weights = np.frombuffer(stream.read(descriptor.nbytes), dtype="<f4").reshape(descriptor.shape)
        inputs = np.linspace(-1.0, 1.0, descriptor.shape[1], dtype=np.float32)
        output = np.asarray(weights @ inputs, dtype=np.float32)
        if corrupt and index == 0:
            output[0] += np.float32(0.25)
        records.append(
            trace_record(name, output, input_values=inputs, layer=layer, position=position)
        )
    write_trace(path, records)


def test_linear_replay_attributes_q8_error(replay_cli, synthetic_models, tmp_path):
    fp32, q8 = synthetic_models
    trace = tmp_path / "linear.trace"
    linear_trace(trace, fp32)
    process = run(
        replay_cli,
        "--trace",
        trace,
        "--mode",
        "linear",
        "--model",
        fp32,
        "--q8-model",
        q8,
    )
    assert process.returncode == 0, process.stderr
    reports = json_lines(process)
    assert len(reports) == 8
    assert [report["variant"] for report in reports[:4]] == [
        "fp32_capture",
        "weight_only",
        "activation_only",
        "w8a8",
    ]
    assert reports[4]["site"] == "model.embed_tokens.weight"
    assert reports[4]["layer"] == -1
    assert reports[0]["parity_pass"]
    assert "parity_pass" not in reports[1]
    assert reports[3]["max_abs"] >= 0.0

    linear_trace(trace, fp32, corrupt=True)
    process = run(
        replay_cli,
        "--trace",
        trace,
        "--mode",
        "linear",
        "--model",
        fp32,
        "--q8-model",
        q8,
    )
    assert process.returncode == 2
    assert not json_lines(process)[0]["parity_pass"]


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("magic", "magic"),
        ("partial-header", "truncated trace header"),
        ("site", "invalid trace site"),
        ("reserved", "reserved"),
        ("count", "metadata"),
        ("payload", "truncated trace payload"),
        ("nonfinite", "nonfinite trace output"),
    ],
)
def test_strict_trace_parser_rejects_malformed(replay_cli, tmp_path, mutation, expected):
    valid = bytearray(b"SMTRC001" + trace_record("rms.rsqrt", [0.5], input_values=[4.0], layer=0))
    if mutation == "magic":
        valid[0] ^= 1
    elif mutation == "partial-header":
        valid = valid[:20]
    elif mutation == "site":
        valid[8:72] = b"x" * 64
    elif mutation == "reserved":
        valid[8 + 112] = 1
    elif mutation == "count":
        struct.pack_into("<Q", valid, 8 + 104, 2)
    elif mutation == "payload":
        valid = valid[:-1]
    elif mutation == "nonfinite":
        struct.pack_into("<I", valid, len(valid) - 4, 0x7FC00000)
    trace = tmp_path / f"{mutation}.trace"
    trace.write_bytes(valid)
    process = run(replay_cli, "--trace", trace, "--mode", "math")
    assert process.returncode == 1
    assert expected in process.stderr


def test_replay_options_are_strict(replay_cli, tmp_path):
    trace = tmp_path / "empty.trace"
    write_trace(trace, [])
    process = run(replay_cli, "--trace", trace, "--mode", "math", "--wat", "value")
    assert process.returncode == 1
    assert "unknown replay option" in process.stderr
    process = run(replay_cli, "--trace", trace, "--trace", trace, "--mode", "math")
    assert process.returncode == 1
    assert "duplicate replay option" in process.stderr
