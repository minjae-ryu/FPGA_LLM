from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_smollm as exporter  # noqa: E402
from smollm_format import (  # noqa: E402
    DESCRIPTOR_SIZE,
    DTYPE_F32,
    DTYPE_Q8_64,
    FormatError,
    HEADER_SIZE,
    MODEL_REVISION,
    ModelConfig,
    decode_q8_groups,
    encode_q8_groups,
    expected_tensor_specs,
    parse_model,
    quantize_groups_f32,
    validate_safetensor_schema,
)


FIXTURES = json.loads((ROOT / "tests/fixtures/quantization.json").read_text(encoding="utf-8"))


def _fixture_values(case: dict[str, object]) -> np.ndarray:
    values = np.full(int(case["length"]), case.get("fill", 0.0), dtype=np.float32)
    for index, value in case.get("overrides", {}).items():
        values[int(index)] = np.float32(value)
    for index, bits in case.get("overrides_f32_bits", {}).items():
        values[int(index)] = np.array([int(bits, 16)], dtype=np.uint32).view(np.float32)[0]
    special = {"nan": np.nan, "+inf": np.inf, "-inf": -np.inf}
    for index, value in case.get("overrides_special", {}).items():
        values[int(index)] = special[value]
    if values.size % 64:
        return values
    return values.reshape(-1, 64)


@pytest.mark.parametrize("case", FIXTURES["valid"], ids=lambda case: case["name"])
def test_quantization_matches_independent_fixtures(case: dict[str, object]) -> None:
    values = _fixture_values(case)
    quantized, scales = quantize_groups_f32(values)
    expected = np.full(values.size, case["expected"]["q_fill"], dtype=np.int8)
    for index, value in case["expected"]["q_overrides"].items():
        expected[int(index)] = value
    expected_scale_bits = np.array(
        [int(value, 16) for value in case["expected"]["scale_bits"]], dtype=np.uint32
    )
    np.testing.assert_array_equal(quantized.reshape(-1), expected)
    np.testing.assert_array_equal(scales.reshape(-1).view(np.uint32), expected_scale_bits)

    payload = encode_q8_groups(values)
    decoded_q, decoded_scales = decode_q8_groups(payload)
    np.testing.assert_array_equal(decoded_q.reshape(-1), expected)
    np.testing.assert_array_equal(decoded_scales.view(np.uint32), expected_scale_bits)


@pytest.mark.parametrize("case", FIXTURES["invalid"], ids=lambda case: case["name"])
def test_quantization_rejects_invalid_fixtures(case: dict[str, object]) -> None:
    with pytest.raises(FormatError, match=case["error"]):
        quantize_groups_f32(_fixture_values(case))


def test_quantization_requires_fp32() -> None:
    with pytest.raises(FormatError, match="dtype float32"):
        quantize_groups_f32(np.zeros((1, 64), dtype=np.float64))


@pytest.mark.parametrize("dtype", ["f32", "q8"])
def test_synthetic_export_layout_payloads_and_manifest(tmp_path: Path, dtype: str) -> None:
    output = tmp_path / f"synthetic-{dtype}.bin"
    manifest_path = tmp_path / f"synthetic-{dtype}.json"
    config, source, verification = exporter.export_synthetic(output, dtype)
    manifest = exporter.write_manifest(
        manifest_path,
        output,
        dtype,
        config,
        source,
        verification,
        ["synthetic-test"],
    )
    parsed = parse_model(output, expected_config=config)
    specs = expected_tensor_specs(config)

    assert parsed.revision == exporter.SYNTHETIC_REVISION
    assert [descriptor.name for descriptor in parsed.descriptors] == [spec.name for spec in specs]
    assert all(descriptor.offset % 64 == 0 for descriptor in parsed.descriptors)
    assert parsed.size % 64 == 0
    assert verification["status"] == "passed"
    assert verification["f32_values_verified"] > 0
    if dtype == "f32":
        assert all(descriptor.dtype == DTYPE_F32 for descriptor in parsed.descriptors)
        assert verification["q8_values_verified"] == 0
    else:
        assert all(
            descriptor.dtype == (DTYPE_F32 if len(spec.shape) == 1 else DTYPE_Q8_64)
            for descriptor, spec in zip(parsed.descriptors, specs)
        )
        assert verification["q8_values_verified"] > 0

    assert manifest["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest["source"]["generator"] == exporter.SYNTHETIC_GENERATOR
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_synthetic_f32_payload_is_exact_bf16_to_f32(tmp_path: Path) -> None:
    output = tmp_path / "synthetic-f32.bin"
    config, _, _ = exporter.export_synthetic(output, "f32")
    tensors = exporter.synthetic_tensors(config)
    parsed = parse_model(output, expected_config=config)
    with output.open("rb") as handle:
        for descriptor in parsed.descriptors:
            handle.seek(descriptor.offset)
            actual = handle.read(descriptor.nbytes)
            expected = tensors[descriptor.name].float().numpy().astype("<f4", copy=False).tobytes()
            assert actual == expected, descriptor.name


def test_cli_derives_manifest_path_when_omitted(tmp_path: Path) -> None:
    output = tmp_path / "fixture.bin"
    assert exporter.main(["--synthetic", "--output", str(output), "--dtype", "q8"]) == 0
    manifest = output.with_suffix(".json")
    assert manifest.is_file()
    assert json.loads(manifest.read_text(encoding="utf-8"))["output"]["sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()


def test_failed_export_does_not_replace_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing.bin"
    output.write_bytes(b"keep me")
    config = exporter.synthetic_config()

    def fail(_name: str) -> torch.Tensor:
        raise FormatError("injected source failure")

    with pytest.raises(FormatError, match="injected source failure"):
        exporter._write_model(output, config, exporter.SYNTHETIC_REVISION, "f32", fail)
    assert output.read_bytes() == b"keep me"
    assert not list(tmp_path.glob(".*.tmp"))


def test_parser_rejects_reserved_bytes_and_bad_lengths(tmp_path: Path) -> None:
    output = tmp_path / "synthetic.bin"
    config, _, _ = exporter.export_synthetic(output, "f32")
    data = bytearray(output.read_bytes())
    data[112] = 1
    bad_reserved = tmp_path / "bad-reserved.bin"
    bad_reserved.write_bytes(data)
    with pytest.raises(FormatError, match="reserved"):
        parse_model(bad_reserved, expected_config=config)

    data = bytearray(output.read_bytes())
    first_nbytes_offset = HEADER_SIZE + 112
    old_nbytes = struct.unpack_from("<Q", data, first_nbytes_offset)[0]
    struct.pack_into("<Q", data, first_nbytes_offset, old_nbytes + 4)
    bad_length = tmp_path / "bad-length.bin"
    bad_length.write_bytes(data)
    with pytest.raises(FormatError, match="byte count"):
        parse_model(bad_length, expected_config=config)


def test_schema_rejects_missing_extra_shape_and_dtype() -> None:
    config = exporter.synthetic_config()
    specs = expected_tensor_specs(config)
    shapes = {spec.name: spec.shape for spec in specs}
    dtypes = {spec.name: torch.bfloat16 for spec in specs}
    keys = list(shapes)
    assert validate_safetensor_schema(keys, shapes, dtypes, config, bfloat16_dtype=torch.bfloat16) == specs

    with pytest.raises(FormatError, match="missing=.*model.norm.weight"):
        validate_safetensor_schema(keys[:-1], shapes, dtypes, config, bfloat16_dtype=torch.bfloat16)

    extra_shapes = dict(shapes, unexpected=(1,))
    extra_dtypes = dict(dtypes, unexpected=torch.bfloat16)
    with pytest.raises(FormatError, match="extra=.*unexpected"):
        validate_safetensor_schema(
            [*keys, "unexpected"], extra_shapes, extra_dtypes, config, bfloat16_dtype=torch.bfloat16
        )

    wrong_shapes = dict(shapes)
    wrong_shapes[keys[0]] = (1, config.dim)
    with pytest.raises(FormatError, match="has shape"):
        validate_safetensor_schema(keys, wrong_shapes, dtypes, config, bfloat16_dtype=torch.bfloat16)

    wrong_dtypes = dict(dtypes)
    wrong_dtypes[keys[0]] = torch.float32
    with pytest.raises(FormatError, match="not BF16"):
        validate_safetensor_schema(keys, shapes, wrong_dtypes, config, bfloat16_dtype=torch.bfloat16)


def test_config_rejects_inconsistent_gqa_and_unquantizable_k() -> None:
    base = {
        "hidden_size": 192,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "vocab_size": 128,
        "max_position_embeddings": 384,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
    }
    assert ModelConfig.from_hf(base, require_pinned=False) == exporter.synthetic_config()
    with pytest.raises(FormatError, match="divisible by num_key_value_heads"):
        ModelConfig.from_hf(dict(base, num_key_value_heads=2), require_pinned=False)
    with pytest.raises(FormatError, match="matrix K dimensions"):
        ModelConfig.from_hf(dict(base, intermediate_size=96), require_pinned=False)
    with pytest.raises(FormatError, match="tie_word_embeddings"):
        ModelConfig.from_hf(dict(base, tie_word_embeddings=False), require_pinned=False)


def test_pinned_config_requires_architecture_and_numerical_fields() -> None:
    pinned = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "hidden_act": "silu",
        "hidden_size": 576,
        "intermediate_size": 1536,
        "max_position_embeddings": 8192,
        "model_type": "llama",
        "num_attention_heads": 9,
        "num_hidden_layers": 30,
        "num_key_value_heads": 3,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-5,
        "rope_interleaved": False,
        "rope_scaling": None,
        "rope_theta": 100000,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": 49152,
    }
    config = ModelConfig.from_hf(pinned, require_pinned=True)
    assert len(expected_tensor_specs(config)) == 272
    with pytest.raises(FormatError, match="torch_dtype"):
        ModelConfig.from_hf(dict(pinned, torch_dtype="float32"), require_pinned=True)
    missing = dict(pinned)
    del missing["rope_interleaved"]
    with pytest.raises(FormatError, match="rope_interleaved"):
        ModelConfig.from_hf(missing, require_pinned=True)


def test_header_revision_is_exactly_pinned_length() -> None:
    assert len(MODEL_REVISION.encode("ascii")) == 40
    assert DESCRIPTOR_SIZE == 128
