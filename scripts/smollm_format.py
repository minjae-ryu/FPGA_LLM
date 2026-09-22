"""SmolLM2 model-format definitions and exact GS64 quantization helpers.

This module intentionally contains no Transformers dependency.  The exporter and
tests share these structural checks, while the C loader remains the independent
consumer of the on-disk contract in docs/design/contracts.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Iterable, Mapping, Sequence

import numpy as np


MAGIC = b"SML2C001"
VERSION = 1
ENDIAN_MARKER = 0x01020304
HEADER_SIZE = 256
DESCRIPTOR_SIZE = 128
DTYPE_F32 = 1
DTYPE_Q8_64 = 2
GROUP_SIZE = 64
Q8_BLOCK_SIZE = 68

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
CONFIG_SHA256 = "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843"
SAFETENSORS_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"


class FormatError(ValueError):
    """An input or output violates the version-one model contract."""


def _require_plain_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FormatError(f"config field {field!r} must be an integer")
    if value <= 0 or value > 0xFFFFFFFF:
        raise FormatError(f"config field {field!r} is outside u32 range")
    return value


def _require_finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormatError(f"config field {field!r} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise FormatError(f"config field {field!r} must be finite and positive")
    return result


@dataclass(frozen=True)
class ModelConfig:
    dim: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    max_context: int
    rope_theta: float
    rms_epsilon: float
    tied_embeddings: int = 1
    group_size: int = GROUP_SIZE

    @classmethod
    def from_hf(cls, raw: Mapping[str, object], *, require_pinned: bool) -> "ModelConfig":
        required = {
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "tie_word_embeddings",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise FormatError(f"config is missing required fields: {', '.join(missing)}")

        dim = _require_plain_int(raw["hidden_size"], "hidden_size")
        heads = _require_plain_int(raw["num_attention_heads"], "num_attention_heads")
        kv_heads = _require_plain_int(raw["num_key_value_heads"], "num_key_value_heads")
        if dim % heads:
            raise FormatError("hidden_size must be divisible by num_attention_heads")
        if heads % kv_heads:
            raise FormatError("num_attention_heads must be divisible by num_key_value_heads")
        if raw["tie_word_embeddings"] is not True:
            raise FormatError("tie_word_embeddings must be true")

        config = cls(
            dim=dim,
            intermediate=_require_plain_int(raw["intermediate_size"], "intermediate_size"),
            layers=_require_plain_int(raw["num_hidden_layers"], "num_hidden_layers"),
            heads=heads,
            kv_heads=kv_heads,
            head_dim=dim // heads,
            vocab=_require_plain_int(raw["vocab_size"], "vocab_size"),
            max_context=_require_plain_int(raw["max_position_embeddings"], "max_position_embeddings"),
            rope_theta=_require_finite_float(raw["rope_theta"], "rope_theta"),
            rms_epsilon=_require_finite_float(raw["rms_norm_eps"], "rms_norm_eps"),
        )
        config.validate()
        if require_pinned:
            config._validate_pinned(raw)
        return config

    def validate(self) -> None:
        if self.head_dim * self.heads != self.dim:
            raise FormatError("head_dim * heads must equal dim")
        if self.heads % self.kv_heads:
            raise FormatError("heads must be divisible by kv_heads")
        if self.dim % self.group_size or self.intermediate % self.group_size:
            raise FormatError("all matrix K dimensions must be divisible by group_size")
        if self.tied_embeddings != 1:
            raise FormatError("only tied embeddings are supported")
        if self.group_size != GROUP_SIZE:
            raise FormatError("only GS64 is supported")
        for name, value in (("rope_theta", self.rope_theta), ("rms_epsilon", self.rms_epsilon)):
            encoded = np.float32(value)
            if not np.isfinite(encoded) or encoded <= np.float32(0.0):
                raise FormatError(f"{name} cannot be represented as a positive finite f32")

    def _validate_pinned(self, raw: Mapping[str, object]) -> None:
        expected_config = ModelConfig(
            dim=576,
            intermediate=1536,
            layers=30,
            heads=9,
            kv_heads=3,
            head_dim=64,
            vocab=49152,
            max_context=8192,
            rope_theta=100000.0,
            rms_epsilon=1e-5,
        )
        if self != expected_config:
            raise FormatError(f"config does not match pinned {MODEL_ID} revision")
        pinned_fields = {
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": False,
            "hidden_act": "silu",
            "model_type": "llama",
            "pretraining_tp": 1,
            "rope_interleaved": False,
            "rope_scaling": None,
            "torch_dtype": "bfloat16",
            "use_cache": True,
        }
        for field, expected in pinned_fields.items():
            if field not in raw:
                raise FormatError(f"pinned config is missing field {field!r}")
            if raw[field] != expected or type(raw[field]) is not type(expected):
                raise FormatError(f"pinned config field {field!r} has an unexpected value")

    def header_values(self) -> tuple[int, ...]:
        return (
            self.dim,
            self.intermediate,
            self.layers,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.vocab,
            self.max_context,
        )


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]

    @property
    def rank(self) -> int:
        return len(self.shape)


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    dtype: int
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclass(frozen=True)
class ParsedModel:
    config: ModelConfig
    revision: str
    descriptors: tuple[TensorDescriptor, ...]
    size: int


def read_config(path: Path, *, require_pinned: bool) -> tuple[ModelConfig, dict[str, object]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormatError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise FormatError("config root must be an object")
    return ModelConfig.from_hf(raw, require_pinned=require_pinned), raw


def expected_tensor_specs(config: ModelConfig) -> tuple[TensorSpec, ...]:
    dim = config.dim
    kv_dim = config.kv_heads * config.head_dim
    specs: list[TensorSpec] = [TensorSpec("model.embed_tokens.weight", (config.vocab, dim))]
    for layer in range(config.layers):
        prefix = f"model.layers.{layer}"
        specs.extend(
            (
                TensorSpec(f"{prefix}.input_layernorm.weight", (dim,)),
                TensorSpec(f"{prefix}.self_attn.q_proj.weight", (dim, dim)),
                TensorSpec(f"{prefix}.self_attn.k_proj.weight", (kv_dim, dim)),
                TensorSpec(f"{prefix}.self_attn.v_proj.weight", (kv_dim, dim)),
                TensorSpec(f"{prefix}.self_attn.o_proj.weight", (dim, dim)),
                TensorSpec(f"{prefix}.post_attention_layernorm.weight", (dim,)),
                TensorSpec(f"{prefix}.mlp.gate_proj.weight", (config.intermediate, dim)),
                TensorSpec(f"{prefix}.mlp.up_proj.weight", (config.intermediate, dim)),
                TensorSpec(f"{prefix}.mlp.down_proj.weight", (dim, config.intermediate)),
            )
        )
    specs.append(TensorSpec("model.norm.weight", (dim,)))
    for spec in specs:
        encoded = spec.name.encode("ascii")
        if len(encoded) > 63:
            raise FormatError(f"tensor name is too long for descriptor: {spec.name}")
    return tuple(specs)


def output_dtype(spec: TensorSpec, export_dtype: str) -> int:
    if export_dtype not in {"f32", "q8"}:
        raise FormatError(f"unsupported export dtype {export_dtype!r}")
    return DTYPE_Q8_64 if export_dtype == "q8" and spec.rank == 2 else DTYPE_F32


def tensor_nbytes(spec: TensorSpec, dtype: int) -> int:
    elements = math.prod(spec.shape)
    if dtype == DTYPE_F32:
        return elements * 4
    if dtype == DTYPE_Q8_64:
        if spec.rank != 2 or spec.shape[1] % GROUP_SIZE:
            raise FormatError(f"Q8 tensor {spec.name} has an invalid shape")
        return (elements // GROUP_SIZE) * Q8_BLOCK_SIZE
    raise FormatError(f"unsupported dtype code {dtype}")


def align64(value: int) -> int:
    return (value + 63) & ~63


def make_descriptors(config: ModelConfig, export_dtype: str) -> tuple[TensorDescriptor, ...]:
    specs = expected_tensor_specs(config)
    cursor = align64(HEADER_SIZE + DESCRIPTOR_SIZE * len(specs))
    descriptors: list[TensorDescriptor] = []
    for spec in specs:
        dtype = output_dtype(spec, export_dtype)
        nbytes = tensor_nbytes(spec, dtype)
        descriptors.append(TensorDescriptor(spec.name, dtype, spec.shape, cursor, nbytes))
        cursor = align64(cursor + nbytes)
    return tuple(descriptors)


def encode_header(config: ModelConfig, tensor_count: int, revision: str) -> bytes:
    try:
        encoded_revision = revision.encode("ascii")
    except UnicodeEncodeError as exc:
        raise FormatError("source revision must be ASCII") from exc
    if len(encoded_revision) != 40:
        raise FormatError("source revision must be exactly 40 ASCII bytes")
    header = bytearray(HEADER_SIZE)
    header[:8] = MAGIC
    struct.pack_into("<IIII", header, 8, VERSION, ENDIAN_MARKER, HEADER_SIZE, tensor_count)
    struct.pack_into("<8I", header, 24, *config.header_values())
    struct.pack_into("<2f", header, 56, config.rope_theta, config.rms_epsilon)
    struct.pack_into("<2I", header, 64, config.tied_embeddings, config.group_size)
    header[72:112] = encoded_revision
    return bytes(header)


def encode_descriptor(descriptor: TensorDescriptor) -> bytes:
    name = descriptor.name.encode("ascii")
    if not name or len(name) > 63 or b"\0" in name:
        raise FormatError(f"invalid descriptor name {descriptor.name!r}")
    if len(descriptor.shape) not in (1, 2):
        raise FormatError(f"invalid rank for {descriptor.name}")
    dimensions = descriptor.shape + (0,) * (4 - len(descriptor.shape))
    result = bytearray(DESCRIPTOR_SIZE)
    result[: len(name)] = name
    struct.pack_into("<II", result, 64, descriptor.dtype, len(descriptor.shape))
    struct.pack_into("<4Q", result, 72, *dimensions)
    struct.pack_into("<QQ", result, 104, descriptor.offset, descriptor.nbytes)
    return bytes(result)


def quantize_groups_f32(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize [..., 64] FP32 groups with contract-defined FP32 operations."""

    groups = np.asarray(values)
    if groups.dtype != np.float32:
        raise FormatError("quantization input must have dtype float32")
    if groups.ndim < 1 or groups.shape[-1] != GROUP_SIZE:
        raise FormatError("quantization input must consist of complete groups of 64")
    if not np.isfinite(groups).all():
        raise FormatError("quantization input contains a nonfinite value")

    maxima = np.max(np.abs(groups), axis=-1).astype(np.float32, copy=False)
    scales = np.divide(maxima, np.float32(127.0), dtype=np.float32)
    invalid_zero = (maxima != np.float32(0.0)) & (scales == np.float32(0.0))
    if np.any(invalid_zero):
        raise FormatError("a nonzero Q8 group has an unrepresentable zero scale")

    quotients = np.zeros_like(groups, dtype=np.float32)
    nonzero = scales != np.float32(0.0)
    np.divide(groups, scales[..., None], out=quotients, where=nonzero[..., None])
    rounded = np.rint(quotients)
    clipped = np.clip(rounded, np.float32(-127.0), np.float32(127.0))
    quantized = clipped.astype(np.int8)
    return quantized, scales


def encode_q8_groups(values: np.ndarray) -> bytes:
    quantized, scales = quantize_groups_f32(values)
    flat_q = quantized.reshape(-1, GROUP_SIZE)
    flat_scales = scales.reshape(-1)
    blocks = np.empty((flat_q.shape[0], Q8_BLOCK_SIZE), dtype=np.uint8)
    blocks[:, :GROUP_SIZE] = flat_q.view(np.uint8)
    blocks[:, GROUP_SIZE:] = flat_scales.astype("<f4", copy=False).view(np.uint8).reshape(-1, 4)
    return blocks.tobytes(order="C")


def decode_q8_groups(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(payload) % Q8_BLOCK_SIZE:
        raise FormatError("Q8 payload is not a whole number of blocks")
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(-1, Q8_BLOCK_SIZE)
    quantized = blocks[:, :GROUP_SIZE].copy().view(np.int8)
    scales = blocks[:, GROUP_SIZE:].copy().reshape(-1, 4).view("<f4").reshape(-1)
    return quantized, scales


def parse_model(path: Path, *, expected_config: ModelConfig | None = None) -> ParsedModel:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            header = handle.read(HEADER_SIZE)
            if len(header) != HEADER_SIZE:
                raise FormatError("truncated model header")
            if header[:8] != MAGIC:
                raise FormatError("unsupported model magic")
            version, endian, header_size, tensor_count = struct.unpack_from("<IIII", header, 8)
            if version != VERSION:
                raise FormatError("unsupported model version")
            if endian != ENDIAN_MARKER:
                raise FormatError("unsupported endian marker")
            if header_size != HEADER_SIZE:
                raise FormatError("unsupported header size")
            if tensor_count == 0 or tensor_count > 1_000_000:
                raise FormatError("invalid tensor count")
            values = struct.unpack_from("<8I", header, 24)
            rope_theta, rms_epsilon = struct.unpack_from("<2f", header, 56)
            tied, group_size = struct.unpack_from("<2I", header, 64)
            config = ModelConfig(*values, rope_theta, rms_epsilon, tied, group_size)
            config.validate()
            if expected_config is not None:
                expected_header_config = ModelConfig(
                    *expected_config.header_values(),
                    float(np.float32(expected_config.rope_theta)),
                    float(np.float32(expected_config.rms_epsilon)),
                    expected_config.tied_embeddings,
                    expected_config.group_size,
                )
                if config != expected_header_config:
                    raise FormatError("model header config is inconsistent with source config")
            try:
                revision = header[72:112].decode("ascii")
            except UnicodeDecodeError as exc:
                raise FormatError("source revision is not ASCII") from exc
            if header[112:] != bytes(HEADER_SIZE - 112):
                raise FormatError("model header reserved bytes are nonzero")

            raw_descriptors = handle.read(tensor_count * DESCRIPTOR_SIZE)
            if len(raw_descriptors) != tensor_count * DESCRIPTOR_SIZE:
                raise FormatError("truncated descriptor table")
    except OSError as exc:
        raise FormatError(f"cannot read model {path}: {exc}") from exc

    descriptors: list[TensorDescriptor] = []
    names: set[str] = set()
    previous_end = align64(HEADER_SIZE + tensor_count * DESCRIPTOR_SIZE)
    for index in range(tensor_count):
        raw = raw_descriptors[index * DESCRIPTOR_SIZE : (index + 1) * DESCRIPTOR_SIZE]
        name_field = raw[:64]
        terminator = name_field.find(b"\0")
        if terminator <= 0 or any(name_field[terminator:]):
            raise FormatError(f"descriptor {index} has an invalid name field")
        try:
            name = name_field[:terminator].decode("ascii")
        except UnicodeDecodeError as exc:
            raise FormatError(f"descriptor {index} name is not ASCII") from exc
        if name in names:
            raise FormatError(f"duplicate tensor descriptor {name}")
        names.add(name)
        dtype, rank = struct.unpack_from("<II", raw, 64)
        dimensions = struct.unpack_from("<4Q", raw, 72)
        offset, nbytes = struct.unpack_from("<QQ", raw, 104)
        if rank not in (1, 2) or any(dim == 0 for dim in dimensions[:rank]):
            raise FormatError(f"tensor {name} has invalid rank or dimensions")
        if any(dimensions[rank:]):
            raise FormatError(f"tensor {name} has nonzero unused dimensions")
        if any(raw[120:]):
            raise FormatError(f"tensor {name} has nonzero descriptor reserved bytes")
        descriptor = TensorDescriptor(name, dtype, tuple(dimensions[:rank]), offset, nbytes)
        expected_bytes = tensor_nbytes(TensorSpec(name, descriptor.shape), dtype)
        if nbytes != expected_bytes:
            raise FormatError(f"tensor {name} has an inconsistent byte count")
        if offset % 64:
            raise FormatError(f"tensor {name} data offset is not 64-byte aligned")
        if offset < previous_end:
            raise FormatError(f"tensor {name} overlaps preceding file data")
        if offset + nbytes > size:
            raise FormatError(f"tensor {name} extends past end of file")
        previous_end = offset + nbytes
        descriptors.append(descriptor)

    expected_specs = expected_tensor_specs(config)
    expected_by_name = {spec.name: spec for spec in expected_specs}
    if len(descriptors) != len(expected_specs):
        raise FormatError("tensor count is inconsistent with config")
    if tuple(descriptor.name for descriptor in descriptors) != tuple(spec.name for spec in expected_specs):
        missing = sorted(expected_by_name.keys() - names)
        extra = sorted(names - expected_by_name.keys())
        detail = f"; missing={missing}, extra={extra}" if missing or extra else ""
        raise FormatError(f"tensor descriptors are missing, extra, or out of canonical order{detail}")
    for descriptor, spec in zip(descriptors, expected_specs):
        if descriptor.shape != spec.shape:
            raise FormatError(f"tensor {descriptor.name} shape is inconsistent with config")
        if descriptor.dtype == DTYPE_Q8_64 and descriptor.shape[1] % GROUP_SIZE:
            raise FormatError(f"tensor {descriptor.name} has an invalid Q8 K dimension")
        if descriptor.dtype == DTYPE_Q8_64 and descriptor.shape == (config.dim,):
            raise FormatError(f"norm tensor {descriptor.name} may not be quantized")

    if size != align64(previous_end):
        raise FormatError("model has trailing or missing final alignment bytes")
    return ParsedModel(config, revision, tuple(descriptors), size)


def validate_safetensor_schema(
    keys: Iterable[str],
    shapes: Mapping[str, Sequence[int]],
    dtypes: Mapping[str, object],
    config: ModelConfig,
    *,
    bfloat16_dtype: object,
) -> tuple[TensorSpec, ...]:
    specs = expected_tensor_specs(config)
    expected = {spec.name: spec for spec in specs}
    actual_keys = set(keys)
    missing = sorted(expected.keys() - actual_keys)
    extra = sorted(actual_keys - expected.keys())
    if missing or extra:
        raise FormatError(f"safetensors key mismatch: missing={missing}, extra={extra}")
    for spec in specs:
        if tuple(shapes[spec.name]) != spec.shape:
            raise FormatError(
                f"safetensors tensor {spec.name} has shape {tuple(shapes[spec.name])}, expected {spec.shape}"
            )
        if dtypes[spec.name] != bfloat16_dtype:
            raise FormatError(f"safetensors tensor {spec.name} is not BF16")
    return specs
