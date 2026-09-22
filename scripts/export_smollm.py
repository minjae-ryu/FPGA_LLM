#!/usr/bin/env python3
"""Export the pinned SmolLM2 BF16 checkpoint to the SML2C001 format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Callable, Mapping

import numpy as np
import safetensors
from safetensors import safe_open
import torch

from smollm_format import (
    CONFIG_SHA256,
    DESCRIPTOR_SIZE,
    DTYPE_F32,
    DTYPE_Q8_64,
    FormatError,
    GROUP_SIZE,
    HEADER_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    ModelConfig,
    SAFETENSORS_SHA256,
    TensorDescriptor,
    align64,
    encode_descriptor,
    encode_header,
    encode_q8_groups,
    expected_tensor_specs,
    make_descriptors,
    parse_model,
    read_config,
    validate_safetensor_schema,
)


SYNTHETIC_REVISION = "0000000000000000000000000000000000000001"
SYNTHETIC_GENERATOR = "smollm-loader-fixture-v1"
Q8_GROUP_CHUNK = 4096


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )


def _write_zeroes(handle, count: int) -> None:
    zeroes = bytes(64 * 1024)
    while count:
        amount = min(count, len(zeroes))
        handle.write(zeroes[:amount])
        count -= amount


def _f32_array(tensor: torch.Tensor, name: str) -> np.ndarray:
    if tensor.dtype != torch.bfloat16:
        raise FormatError(f"source tensor {name} is not BF16")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    result = tensor.to(dtype=torch.float32).numpy()
    if not np.isfinite(result).all():
        raise FormatError(f"source tensor {name} contains a nonfinite value")
    return result


def _write_tensor(handle, tensor: torch.Tensor, descriptor: TensorDescriptor) -> None:
    values = _f32_array(tensor, descriptor.name)
    if descriptor.dtype == DTYPE_F32:
        handle.write(values.astype("<f4", copy=False).tobytes(order="C"))
        return
    if descriptor.dtype != DTYPE_Q8_64:
        raise FormatError(f"unsupported output dtype for {descriptor.name}")
    groups = values.reshape(-1, GROUP_SIZE)
    for start in range(0, groups.shape[0], Q8_GROUP_CHUNK):
        handle.write(encode_q8_groups(groups[start : start + Q8_GROUP_CHUNK]))


def _write_model(
    output: Path,
    config: ModelConfig,
    revision: str,
    export_dtype: str,
    get_tensor: Callable[[str], torch.Tensor],
) -> tuple[TensorDescriptor, ...]:
    descriptors = make_descriptors(config, export_dtype)
    temporary_name: str | None = None
    try:
        with _atomic_writer(output) as handle:
            temporary_name = handle.name
            os.fchmod(handle.fileno(), 0o644)
            handle.write(encode_header(config, len(descriptors), revision))
            for descriptor in descriptors:
                handle.write(encode_descriptor(descriptor))
            _write_zeroes(handle, align64(handle.tell()) - handle.tell())
            for descriptor in descriptors:
                if handle.tell() > descriptor.offset:
                    raise FormatError(f"internal offset error before {descriptor.name}")
                _write_zeroes(handle, descriptor.offset - handle.tell())
                tensor = get_tensor(descriptor.name)
                if tuple(tensor.shape) != descriptor.shape:
                    raise FormatError(f"source tensor {descriptor.name} changed shape during export")
                _write_tensor(handle, tensor, descriptor)
                if handle.tell() != descriptor.offset + descriptor.nbytes:
                    raise FormatError(f"internal byte-count error after {descriptor.name}")
            _write_zeroes(handle, align64(handle.tell()) - handle.tell())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return descriptors


def _check_source_tensor(tensor: torch.Tensor, descriptor: TensorDescriptor) -> np.ndarray:
    if tuple(tensor.shape) != descriptor.shape:
        raise FormatError(f"source tensor {descriptor.name} changed shape during verification")
    return _f32_array(tensor, descriptor.name)


def verify_payloads(
    output: Path,
    config: ModelConfig,
    revision: str,
    get_tensor: Callable[[str], torch.Tensor],
) -> dict[str, object]:
    parsed = parse_model(output, expected_config=config)
    if parsed.revision != revision:
        raise FormatError("exported source revision does not match requested revision")
    f32_values = 0
    q8_values = 0
    with output.open("rb") as handle:
        for descriptor in parsed.descriptors:
            values = _check_source_tensor(get_tensor(descriptor.name), descriptor)
            handle.seek(descriptor.offset)
            if descriptor.dtype == DTYPE_F32:
                expected = values.astype("<f4", copy=False).tobytes(order="C")
                actual = handle.read(descriptor.nbytes)
                if actual != expected:
                    raise FormatError(f"FP32 payload verification failed for {descriptor.name}")
                f32_values += values.size
            elif descriptor.dtype == DTYPE_Q8_64:
                groups = values.reshape(-1, GROUP_SIZE)
                for start in range(0, groups.shape[0], Q8_GROUP_CHUNK):
                    expected = encode_q8_groups(groups[start : start + Q8_GROUP_CHUNK])
                    actual = handle.read(len(expected))
                    if actual != expected:
                        raise FormatError(f"Q8 payload verification failed for {descriptor.name}")
                q8_values += values.size
            else:
                raise FormatError(f"unsupported exported dtype for {descriptor.name}")
    return {
        "method": "byte_exact_recompute_from_source_bf16",
        "tensor_count": len(parsed.descriptors),
        "f32_values_verified": f32_values,
        "q8_values_verified": q8_values,
        "status": "passed",
    }


def _snapshot_schema(snapshot: Path) -> tuple[ModelConfig, Path, Path]:
    if not snapshot.is_dir():
        raise FormatError(f"source snapshot is not a directory: {snapshot}")
    config_path = snapshot / "config.json"
    weights_path = snapshot / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FormatError("source snapshot must contain config.json and model.safetensors")
    config_hash = sha256_file(config_path)
    weights_hash = sha256_file(weights_path)
    if config_hash != CONFIG_SHA256:
        raise FormatError(f"config SHA256 does not match pinned revision: {config_hash}")
    if weights_hash != SAFETENSORS_SHA256:
        raise FormatError(f"model.safetensors SHA256 does not match pinned revision: {weights_hash}")
    config, _ = read_config(config_path, require_pinned=True)
    return config, config_path, weights_path


def export_snapshot(snapshot: Path, output: Path, export_dtype: str) -> tuple[ModelConfig, dict[str, object], dict[str, object]]:
    config, config_path, weights_path = _snapshot_schema(snapshot)
    try:
        with safe_open(weights_path, framework="pt", device="cpu") as source:
            keys = list(source.keys())
            shapes: dict[str, tuple[int, ...]] = {}
            dtypes: dict[str, torch.dtype] = {}
            for name in keys:
                tensor = source.get_tensor(name)
                shapes[name] = tuple(tensor.shape)
                dtypes[name] = tensor.dtype
            validate_safetensor_schema(keys, shapes, dtypes, config, bfloat16_dtype=torch.bfloat16)
            _write_model(output, config, MODEL_REVISION, export_dtype, source.get_tensor)
            verification = verify_payloads(output, config, MODEL_REVISION, source.get_tensor)
    except (OSError, RuntimeError) as exc:
        raise FormatError(f"cannot read safetensors checkpoint: {exc}") from exc
    source_record = {
        "kind": "huggingface_snapshot",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "config": {"filename": config_path.name, "sha256": CONFIG_SHA256},
        "weights": {"filename": weights_path.name, "sha256": SAFETENSORS_SHA256},
    }
    return config, source_record, verification


def synthetic_config() -> ModelConfig:
    return ModelConfig(
        dim=192,
        intermediate=64,
        layers=2,
        heads=3,
        kv_heads=1,
        head_dim=64,
        vocab=128,
        max_context=384,
        rope_theta=10000.0,
        rms_epsilon=1e-5,
    )


def synthetic_tensors(config: ModelConfig) -> Mapping[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for tensor_index, spec in enumerate(expected_tensor_specs(config)):
        count = int(np.prod(spec.shape))
        positions = torch.arange(count, dtype=torch.int64)
        values = ((positions * 37 + tensor_index * 19) % 257 - 128).to(torch.float32) / 128.0
        result[spec.name] = values.reshape(spec.shape).to(torch.bfloat16)
    return result


def export_synthetic(output: Path, export_dtype: str) -> tuple[ModelConfig, dict[str, object], dict[str, object]]:
    config = synthetic_config()
    tensors = synthetic_tensors(config)
    _write_model(output, config, SYNTHETIC_REVISION, export_dtype, tensors.__getitem__)
    verification = verify_payloads(output, config, SYNTHETIC_REVISION, tensors.__getitem__)
    source_record = {
        "kind": "synthetic",
        "generator": SYNTHETIC_GENERATOR,
        "revision": SYNTHETIC_REVISION,
    }
    return config, source_record, verification


def _config_manifest(config: ModelConfig) -> dict[str, object]:
    return {
        "dim": config.dim,
        "intermediate": config.intermediate,
        "layers": config.layers,
        "heads": config.heads,
        "kv_heads": config.kv_heads,
        "head_dim": config.head_dim,
        "vocab": config.vocab,
        "max_context": config.max_context,
        "rope_theta_f32": float(np.float32(config.rope_theta)),
        "rms_epsilon_f32": float(np.float32(config.rms_epsilon)),
        "tied_embeddings": config.tied_embeddings,
        "group_size": config.group_size,
    }


def write_manifest(
    path: Path,
    output: Path,
    export_dtype: str,
    config: ModelConfig,
    source_record: dict[str, object],
    verification: dict[str, object],
    command: list[str],
) -> dict[str, object]:
    parsed = parse_model(output, expected_config=config)
    manifest = {
        "format": "SML2C001",
        "format_version": 1,
        "dtype": export_dtype,
        "source": source_record,
        "config": _config_manifest(config),
        "output": {
            "path": output.as_posix(),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "tensor_count": len(parsed.descriptors),
        },
        "verification": verification,
        "reproduction": {
            "command": command,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            os.fchmod(handle.fileno(), 0o644)
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="pinned Hugging Face snapshot directory")
    source.add_argument("--synthetic", action="store_true", help="export the deterministic small GQA/GS64 fixture")
    parser.add_argument("--output", type=Path, required=True, help="output SML2C001 model file")
    parser.add_argument("--dtype", choices=("f32", "q8"), required=True, help="matrix storage type")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="reproducibility manifest path (default: replace the output suffix with .json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(sys.argv[1:] if argv is None else argv)
    command = ["python3", "scripts/export_smollm.py", *(sys.argv[1:] if argv is None else argv)]
    manifest_path = arguments.manifest or arguments.output.with_suffix(".json")
    try:
        if arguments.synthetic:
            config, source_record, verification = export_synthetic(arguments.output, arguments.dtype)
        else:
            config, source_record, verification = export_snapshot(arguments.source, arguments.output, arguments.dtype)
        manifest = write_manifest(
            manifest_path,
            arguments.output,
            arguments.dtype,
            config,
            source_record,
            verification,
            command,
        )
    except FormatError as exc:
        print(f"export_smollm.py: error: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {arguments.output} ({manifest['output']['bytes']} bytes, "
        f"sha256={manifest['output']['sha256']})"
    )
    print(f"verification: {verification['status']} ({verification['method']})")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
