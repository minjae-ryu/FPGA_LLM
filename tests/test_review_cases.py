"""Independent cache/kernel/runtime review cases.

These tests intentionally avoid assigning an end-to-end Q8 accuracy threshold.
They gate exact integer/cache behavior and record chunk-dependent Q8 sensitivity
separately under ``artifacts/review``.
"""

from __future__ import annotations

import ctypes as ct
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_smollm as exporter
from smollm_format import DTYPE_Q8_64, parse_model


os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


class Error(ct.Structure):
    _fields_ = [("message", ct.c_char * 256)]


class Trace(ct.Structure):
    _fields_ = [
        ("site", ct.c_char_p),
        ("layer", ct.c_int),
        ("head", ct.c_int),
        ("position", ct.c_size_t),
        ("rows", ct.c_size_t),
        ("columns", ct.c_size_t),
        ("causal", ct.c_int),
        ("input", ct.POINTER(ct.c_float)),
        ("output", ct.POINTER(ct.c_float)),
        ("input_count", ct.c_size_t),
        ("output_count", ct.c_size_t),
    ]


LOGIT_CALLBACK = ct.CFUNCTYPE(
    ct.c_int,
    ct.c_void_p,
    ct.c_size_t,
    ct.POINTER(ct.c_float),
    ct.c_size_t,
)
TRACE_CALLBACK = ct.CFUNCTYPE(None, ct.c_void_p, ct.POINTER(Trace))


@pytest.fixture(scope="module")
def review_lib():
    subprocess.run(
        ["make", "build/libsmollm.so"], cwd=ROOT, check=True, text=True
    )
    library = ct.CDLL(str(ROOT / "build/libsmollm.so"))
    library.sm_model_load.argtypes = [
        ct.c_char_p,
        ct.POINTER(ct.c_void_p),
        ct.POINTER(Error),
    ]
    library.sm_model_free.argtypes = [ct.c_void_p]
    library.sm_session_create.argtypes = [
        ct.c_void_p,
        ct.c_int,
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(ct.c_void_p),
        ct.POINTER(Error),
    ]
    library.sm_session_free.argtypes = [ct.c_void_p]
    library.sm_session_reset.argtypes = [ct.c_void_p]
    library.sm_session_position.argtypes = [ct.c_void_p]
    library.sm_session_position.restype = ct.c_size_t
    library.sm_session_set_trace.argtypes = [ct.c_void_p, TRACE_CALLBACK, ct.c_void_p]
    library.sm_prefill.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_uint32),
        ct.c_size_t,
        ct.c_int,
        LOGIT_CALLBACK,
        ct.c_void_p,
        ct.POINTER(Error),
    ]
    return library


@pytest.fixture(scope="module")
def synthetic_models(tmp_path_factory):
    output = tmp_path_factory.mktemp("review-models")
    for dtype in ("f32", "q8"):
        exporter.export_synthetic(output / f"{dtype}.bin", dtype)
    return output


def load_model(library, path: Path):
    model = ct.c_void_p()
    error = Error()
    result = library.sm_model_load(
        str(path).encode(), ct.byref(model), ct.byref(error)
    )
    assert result == 0, error.message.decode()
    assert model.value
    return model


def create_session(library, model, *, kv: int, chunk: int, context: int = 384):
    session = ct.c_void_p()
    error = Error()
    result = library.sm_session_create(
        model, kv, context, chunk, ct.byref(session), ct.byref(error)
    )
    assert result == 0, error.message.decode()
    assert session.value
    return session


def run_forward(library, session, tokens, mode=2):
    rows = []
    positions = []

    @LOGIT_CALLBACK
    def receive(_context, position, values, width):
        rows.append(np.ctypeslib.as_array(values, shape=(width,)).copy())
        positions.append(position)
        return 0

    token_array = np.ascontiguousarray(tokens, dtype=np.uint32)
    error = Error()
    result = library.sm_prefill(
        session,
        token_array.ctypes.data_as(ct.POINTER(ct.c_uint32)),
        token_array.size,
        mode,
        receive,
        None,
        ct.byref(error),
    )
    assert result == 0, error.message.decode()
    return np.asarray(rows, dtype=np.float32), positions


def test_internal_cache_and_default_kernel_harness(review_lib, tmp_path):
    compiler = os.environ.get("CC", "gcc")
    executable = tmp_path / "test-cache-kernels"
    subprocess.run(
        [
            compiler,
            "-O2",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-Iinclude",
            "-Isrc",
            "tests/test_cache_kernels.c",
            "-Lbuild",
            f"-Wl,-rpath,{ROOT / 'build'}",
            "-lsmollm",
            "-o",
            str(executable),
        ],
        cwd=ROOT,
        check=True,
        text=True,
    )
    result = subprocess.run(
        [str(executable)], cwd=ROOT, check=True, text=True, capture_output=True
    )
    assert result.stdout.strip() == "cache/kernel independent checks passed"


@pytest.mark.parametrize(
    "mutation",
    ("all_zero_positive_scale", "all_zero_negative_zero", "nonzero_zero_scale", "minus_128"),
)
def test_loader_rejects_noncanonical_q8_blocks(
    review_lib, synthetic_models, tmp_path, mutation
):
    source = synthetic_models / "q8.bin"
    parsed = parse_model(source)
    descriptor = next(item for item in parsed.descriptors if item.dtype == DTYPE_Q8_64)
    data = bytearray(source.read_bytes())
    offset = descriptor.offset
    if mutation == "all_zero_positive_scale":
        data[offset : offset + 64] = bytes(64)
        struct.pack_into("<f", data, offset + 64, 1.0)
    elif mutation == "all_zero_negative_zero":
        data[offset : offset + 64] = bytes(64)
        struct.pack_into("<I", data, offset + 64, 0x80000000)
    elif mutation == "nonzero_zero_scale":
        data[offset] = 1
        struct.pack_into("<f", data, offset + 64, 0.0)
    else:
        data[offset] = 0x80
        struct.pack_into("<f", data, offset + 64, 1.0)
    path = tmp_path / f"{mutation}.bin"
    path.write_bytes(data)

    model = ct.c_void_p()
    error = Error()
    result = review_lib.sm_model_load(
        str(path).encode(), ct.byref(model), ct.byref(error)
    )
    assert result != 0
    assert not model.value
    assert error.message


def test_causal_gqa_cache_reset_and_input_bounds(review_lib, synthetic_models):
    config = exporter.synthetic_config()
    model = load_model(review_lib, synthetic_models / "f32.bin")
    session = create_session(review_lib, model, kv=1, chunk=5, context=8)
    events = {}

    @TRACE_CALLBACK
    def collect(_context, event_pointer):
        event = event_pointer.contents
        site = event.site.decode()
        if site in {"q_rope", "k_rope", "v", "attention", "attention.probabilities"}:
            output = np.ctypeslib.as_array(
                event.output, shape=(event.output_count,)
            ).copy()
            events[(site, event.layer, event.head, event.position)] = (
                output,
                event.causal,
            )

    tokens = np.asarray([3, 10, 17, 24, 31], dtype=np.uint32)
    review_lib.sm_session_set_trace(session, collect, None)
    try:
        first, positions = run_forward(review_lib, session, tokens)
        assert positions == list(range(tokens.size))

        # The synthetic fixture has three query heads and one KV head. Recompute
        # every causal attention row independently; all query heads must map to
        # that one KV head while retaining separate queries.
        for position in range(tokens.size):
            query = events[("q_rope", 0, -1, position)][0].reshape(
                config.heads, config.head_dim
            )
            keys = np.stack(
                [events[("k_rope", 0, -1, index)][0] for index in range(position + 1)]
            )
            values = np.stack(
                [events[("v", 0, -1, index)][0] for index in range(position + 1)]
            )
            actual_attention = events[("attention", 0, -1, position)][0].reshape(
                config.heads, config.head_dim
            )
            for head in range(config.heads):
                scores = (
                    keys.astype(np.float64) @ query[head].astype(np.float64)
                ) / math.sqrt(config.head_dim)
                scores -= scores.max()
                probabilities = np.exp(scores)
                probabilities /= probabilities.sum()
                expected = probabilities @ values.astype(np.float64)
                np.testing.assert_allclose(
                    actual_attention[head], expected, atol=3e-5, rtol=3e-5
                )

                traced, causal = events[
                    ("attention.probabilities", 0, head, position)
                ]
                assert causal == 1
                np.testing.assert_allclose(
                    traced[: position + 1], probabilities, atol=2e-6, rtol=2e-6
                )
                assert np.count_nonzero(traced[position + 1 :]) == 0

        review_lib.sm_session_reset(session)
        second, second_positions = run_forward(review_lib, session, tokens)
        assert second_positions == positions
        np.testing.assert_array_equal(second, first)

        old_position = review_lib.sm_session_position(session)
        assert old_position == 5
        callback = LOGIT_CALLBACK(lambda *_args: 0)
        error = Error()
        invalid = np.asarray([config.vocab], dtype=np.uint32)
        result = review_lib.sm_prefill(
            session,
            invalid.ctypes.data_as(ct.POINTER(ct.c_uint32)),
            1,
            0,
            callback,
            None,
            ct.byref(error),
        )
        assert result != 0
        assert review_lib.sm_session_position(session) == old_position
        overflow = np.zeros(4, dtype=np.uint32)
        result = review_lib.sm_prefill(
            session,
            overflow.ctypes.data_as(ct.POINTER(ct.c_uint32)),
            overflow.size,
            0,
            callback,
            None,
            ct.byref(error),
        )
        assert result != 0
        assert review_lib.sm_session_position(session) == old_position
    finally:
        review_lib.sm_session_free(session)
        review_lib.sm_model_free(model)


def _run_with_trace(library, model_path, *, kv, chunk, tokens, wanted):
    model = load_model(library, model_path)
    session = create_session(library, model, kv=kv, chunk=chunk)
    events = {}

    @TRACE_CALLBACK
    def collect(_context, event_pointer):
        event = event_pointer.contents
        site = event.site.decode()
        key = (site, event.layer, event.head, event.position)
        if key in wanted:
            output = np.ctypeslib.as_array(
                event.output, shape=(event.output_count,)
            ).copy()
            input_values = (
                np.ctypeslib.as_array(
                    event.input, shape=(event.input_count,)
                ).copy()
                if event.input_count
                else None
            )
            events[key] = (input_values, output)

    library.sm_session_set_trace(session, collect, None)
    try:
        logits, positions = run_forward(library, session, tokens)
    finally:
        library.sm_session_free(session)
        library.sm_model_free(model)
    return logits, positions, events


def _independent_q8(values):
    groups = np.asarray(values, dtype=np.float32).reshape(-1, 64)
    scales = np.max(np.abs(groups), axis=1).astype(np.float32) / np.float32(127.0)
    quantized = np.zeros_like(groups, dtype=np.int8)
    nonzero = scales != 0
    quantized[nonzero] = np.clip(
        np.rint(groups[nonzero] / scales[nonzero, None]), -127, 127
    ).astype(np.int8)
    return quantized, scales


def test_chunk_quantization_sensitivity_is_diagnostic(
    review_lib, synthetic_models
):
    tokens = (np.arange(130, dtype=np.uint32) * 11 + 1) % 128
    wanted = {
        ("attention", 1, -1, 2),
        ("model.layers.1.mlp.down_proj.weight", 1, -1, 114),
        ("k_rope", 0, -1, 0),
    }
    report = {
        "tokens": int(tokens.size),
        "baseline_chunk": 1,
        "candidate_chunk": 127,
        "threads": {
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "configurations": {},
    }
    outputs = {}
    traces = {}
    for dtype, kv, label in (
        ("f32", 1, "f32_kvf32"),
        ("q8", 1, "q8_kvf32"),
        ("f32", 2, "f32_kv8"),
        ("q8", 2, "q8_kv8"),
    ):
        model_path = synthetic_models / f"{dtype}.bin"
        baseline, base_positions, base_trace = _run_with_trace(
            review_lib,
            model_path,
            kv=kv,
            chunk=1,
            tokens=tokens,
            wanted=wanted,
        )
        candidate, candidate_positions, candidate_trace = _run_with_trace(
            review_lib,
            model_path,
            kv=kv,
            chunk=127,
            tokens=tokens,
            wanted=wanted,
        )
        assert base_positions == candidate_positions == list(range(tokens.size))
        assert np.isfinite(baseline).all() and np.isfinite(candidate).all()
        difference = np.abs(candidate - baseline)
        maximum_index = np.unravel_index(np.argmax(difference), difference.shape)
        report["configurations"][label] = {
            "max_abs": float(difference[maximum_index]),
            "max_position": int(maximum_index[0]),
            "max_logit": int(maximum_index[1]),
            "changed_values": int(np.count_nonzero(difference)),
        }
        outputs[label] = (baseline, candidate)
        traces[label] = (base_trace, candidate_trace)

    # Only the pure FP32 configuration has a project accuracy threshold.
    fp32_baseline, fp32_candidate = outputs["f32_kvf32"]
    assert np.all(
        np.abs(fp32_candidate - fp32_baseline)
        <= 1e-3 + 1e-4 * np.abs(fp32_baseline)
    )

    q8_base, q8_candidate = traces["q8_kvf32"]
    attention_key = ("attention", 1, -1, 2)
    attention_delta = np.max(
        np.abs(q8_candidate[attention_key][1] - q8_base[attention_key][1])
    )
    down_key = ("model.layers.1.mlp.down_proj.weight", 1, -1, 114)
    base_q, base_scale = _independent_q8(q8_base[down_key][0])
    candidate_q, candidate_scale = _independent_q8(q8_candidate[down_key][0])
    report["q8_kvf32_diagnosis"] = {
        "first_observed_attention_delta_layer1_position2": float(attention_delta),
        "down_projection_position114_q_code_changes": int(
            np.count_nonzero(base_q != candidate_q)
        ),
        "down_projection_position114_scales": [
            float(base_scale[0]),
            float(candidate_scale[0]),
        ],
    }

    f32_kv8_base, f32_kv8_candidate = traces["f32_kv8"]
    key = ("k_rope", 0, -1, 0)
    base_k_q, base_k_scale = _independent_q8(f32_kv8_base[key][1])
    candidate_k_q, candidate_k_scale = _independent_q8(
        f32_kv8_candidate[key][1]
    )
    report["f32_kv8_diagnosis"] = {
        "k_rope_layer0_position0_max_abs": float(
            np.max(np.abs(f32_kv8_candidate[key][1] - f32_kv8_base[key][1]))
        ),
        "k_q_code_changes": int(np.count_nonzero(base_k_q != candidate_k_q)),
        "k_scales": [float(base_k_scale[0]), float(candidate_k_scale[0])],
    }

    output_dir = ROOT / "artifacts/review"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chunk_sensitivity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
