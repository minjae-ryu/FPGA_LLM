#!/usr/bin/env python3
"""Generate pinned Hugging Face CPU/FP32/eager numerical references.

Large binary outputs are written below ``artifacts/reference`` (ignored by
Git).  Small, reproducible manifests are written below ``manifests``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import tempfile
from pathlib import Path
from typing import Any

# Constrain native libraries before importing NumPy or PyTorch. These are
# process-local settings and do not modify the user's environment.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
VOCAB_SIZE = 49152
CAPTURE_LAYERS = (0, 14, 29)
TOKEN_MAGIC = b"SMTOK001"
DEFAULT_MODEL_DIR = Path(
    "/home/rmj/.cache/huggingface/hub/"
    "models--HuggingFaceTB--SmolLM2-135M/snapshots/"
    + MODEL_REVISION
)

PROBE_TEXT = (
    "On a quiet morning, the engineer opened the lab window and watched rain "
    "gather on the glass. She wrote down one careful question: could a small, "
    "efficient machine learn to reason from clear examples? Then she tested it, "
    "measured every result, and saved the evidence for the next person."
)

_LONG_PARAGRAPH = (
    "A careful systems experiment begins with a fixed input and a clear contract. "
    "The engineer records every revision, checks each token, and compares the "
    "same outputs at every boundary. When a result changes, the trace should "
    "reveal where and why. This paragraph extends the prompt across several "
    "processing chunks while keeping the language natural and deterministic."
)
LONG_TEXT = " ".join([_LONG_PARAGRAPH] * 5)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.NamedTemporaryFile(
        mode="wb", prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False
    )


def atomic_write_bytes(path: Path, data: bytes) -> None:
    temp = _atomic_path(path)
    temp_path = Path(temp.name)
    try:
        with temp:
            temp.write(data)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def atomic_write_array(path: Path, array: np.ndarray) -> None:
    little = np.ascontiguousarray(array, dtype="<f4")
    temp = _atomic_path(path)
    temp_path = Path(temp.name)
    try:
        with temp:
            little.tofile(temp)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload)


def write_tokens(path: Path, token_ids: list[int]) -> None:
    if any(token < 0 or token >= VOCAB_SIZE for token in token_ids):
        raise ValueError("token ID outside pinned vocabulary")
    body = struct.pack("<8sI", TOKEN_MAGIC, len(token_ids))
    body += struct.pack(f"<{len(token_ids)}I", *token_ids)
    atomic_write_bytes(path, body)


def fp64_mean_nll(logits: np.ndarray, token_ids: list[int]) -> float:
    """Mean next-token NLL, with all reductions explicitly in FP64."""
    if logits.shape != (len(token_ids), VOCAB_SIZE):
        raise ValueError(f"unexpected logits shape {logits.shape}")
    if len(token_ids) < 2:
        raise ValueError("NLL requires at least one real target")
    rows = logits[:-1].astype(np.float64, copy=False)
    maxima = rows.max(axis=1)
    logsumexp = maxima + np.log(np.exp(rows - maxima[:, None]).sum(axis=1))
    targets = np.asarray(token_ids[1:], dtype=np.int64)
    target_logits = rows[np.arange(targets.size), targets]
    return float(np.sum(logsumexp - target_logits, dtype=np.float64) / targets.size)


def install_capture_hooks(model: torch.nn.Module):
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def save(site: str, tensor: torch.Tensor) -> None:
        captures[site] = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()

    for layer_index in CAPTURE_LAYERS:
        layer = model.model.layers[layer_index]

        def norm_hook(_module, _args, output, *, layer_index=layer_index):
            save(f"layer{layer_index}.attn_norm", output[0])

        def attention_pre_hook(
            module, _args, kwargs, *, layer_index=layer_index
        ):
            hidden = kwargs["hidden_states"]
            cos, sin = kwargs["position_embeddings"]
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            query = module.q_proj(hidden).view(shape).transpose(1, 2)
            key = module.k_proj(hidden).view(shape).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            save(f"layer{layer_index}.q_rope", query[0].transpose(0, 1))
            save(f"layer{layer_index}.k_rope", key[0].transpose(0, 1))

        def attention_hook(_module, args, *, layer_index=layer_index):
            # Input to o_proj is the concatenated-head attention output.
            save(f"layer{layer_index}.attention", args[0][0])

        def attn_residual_hook(_module, args, *, layer_index=layer_index):
            save(f"layer{layer_index}.attn_residual", args[0][0])

        def ffn_norm_hook(_module, _args, output, *, layer_index=layer_index):
            save(f"layer{layer_index}.ffn_norm", output[0])

        def layer_hook(_module, _args, output, *, layer_index=layer_index):
            # LlamaDecoderLayer returns the residual after attention and MLP.
            save(f"layer{layer_index}.ffn_residual", output[0])

        handles.extend(
            [
                layer.input_layernorm.register_forward_hook(norm_hook),
                layer.self_attn.register_forward_pre_hook(
                    attention_pre_hook, with_kwargs=True
                ),
                layer.self_attn.o_proj.register_forward_pre_hook(attention_hook),
                layer.post_attention_layernorm.register_forward_pre_hook(
                    attn_residual_hook
                ),
                layer.post_attention_layernorm.register_forward_hook(ffn_norm_hook),
                layer.register_forward_hook(layer_hook),
            ]
        )
    return captures, handles


def source_hashes(model_dir: Path) -> dict[str, str]:
    names = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
    )
    result = {}
    for name in names:
        path = model_dir / name
        if path.exists():
            result[name] = sha256_file(path)
    return result


def library_versions() -> dict[str, str]:
    import tokenizers

    return {
        "numpy": np.__version__,
        "python": platform.python_version(),
        "tokenizers": tokenizers.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def system_info() -> dict[str, Any]:
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    try:
        numpy_build = np.show_config(mode="dicts")
    except TypeError:  # NumPy before show_config(mode="dicts")
        numpy_build = "unavailable on this NumPy version"
    return {
        "cpu_model": cpu_model,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_compiler": platform.python_compiler(),
        "numpy_build": numpy_build,
        "torch_build": torch.__config__.show(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def generate_one(
    *,
    name: str,
    text: str,
    model: torch.nn.Module,
    tokenizer,
    model_dir: Path,
    output_dir: Path,
    manifest_dir: Path,
    capture_intermediates: bool,
) -> dict[str, Any]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if name == "probe" and not 32 <= len(token_ids) <= 64:
        raise ValueError(f"probe must contain 32..64 tokens, got {len(token_ids)}")
    if name == "long" and not 280 <= len(token_ids) <= 400:
        raise ValueError(f"long probe must contain 280..400 tokens, got {len(token_ids)}")

    token_path = output_dir / f"{name}.tokens"
    logits_path = output_dir / f"{name}.logits"
    write_tokens(token_path, token_ids)

    captures: dict[str, torch.Tensor] = {}
    handles = []
    if capture_intermediates:
        captures, handles = install_capture_hooks(model)
    try:
        with torch.inference_mode():
            inputs = torch.tensor([token_ids], dtype=torch.long, device="cpu")
            output = model(input_ids=inputs, use_cache=False, return_dict=True)
            logits = output.logits[0].detach().to(torch.float32).cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()

    atomic_write_array(logits_path, logits)
    artifact_entries: dict[str, dict[str, Any]] = {
        "tokens": {
            "path": str(token_path),
            "format": "SMTOK001 + little-endian u32 count + u32 token IDs",
            "shape": [len(token_ids)],
            "sha256": sha256_file(token_path),
        },
        "logits": {
            "path": str(logits_path),
            "tensor": "logits",
            "dtype": "little-endian float32",
            "shape": [len(token_ids), VOCAB_SIZE],
            "sha256": sha256_file(logits_path),
        },
    }

    intermediate_entries = []
    for site, tensor in sorted(captures.items()):
        path = output_dir / f"{name}.{site}.f32"
        array = tensor.numpy()
        atomic_write_array(path, array)
        intermediate_entries.append(
            {
                "site": site,
                "tensor": site,
                "path": str(path),
                "dtype": "little-endian float32",
                "shape": list(array.shape),
                "sha256": sha256_file(path),
            }
        )

    manifest = {
        "format_version": 1,
        "reference": name,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "execution": {
            "attention": model.config._attn_implementation,
            "device": "cpu",
            "dtype": "float32",
            "interop_threads": torch.get_num_interop_threads(),
            "threads": torch.get_num_threads(),
        },
        "source_tensor_dtype": "bfloat16",
        "conversion": "load pinned BF16 tensors as CPU float32; BF16 values are exactly representable",
        "tokenization": {"add_special_tokens": False, "bos": False, "eos": False},
        "text": text,
        "token_count": len(token_ids),
        "target_count": len(token_ids) - 1,
        "mean_nll_fp64": fp64_mean_nll(logits, token_ids),
        "artifacts": artifact_entries,
        "intermediates": intermediate_entries,
        "capture_semantics": {
            "attn_norm": "input RMSNorm output",
            "q_rope": "HF split-half rotary query, [token,query_head,head_dim]",
            "k_rope": "HF split-half rotary key, [token,kv_head,head_dim]",
            "attention": "pre-o-proj concatenated-head attention output",
            "attn_residual": "hidden state after attention residual add",
            "ffn_norm": "post-attention RMSNorm output",
            "ffn_residual": "decoder-layer output after MLP residual add",
        },
        "chunk_comparisons": (
            {"sizes": [1, 127, 128, 129], "reference": "same full-sequence logits"}
            if name == "long"
            else None
        ),
        "source_files_sha256": source_hashes(model_dir),
        "libraries": library_versions(),
        "system": system_info(),
    }
    manifest_path = manifest_dir / f"reference_{name}.json"
    atomic_write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reference"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--which", choices=("probe", "long", "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.model_dir / "model.safetensors").is_file():
        raise SystemExit(f"pinned model snapshot is missing: {args.model_dir}")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.to(device="cpu", dtype=torch.float32)
    model.eval()
    if model.config.vocab_size != VOCAB_SIZE:
        raise RuntimeError(f"unexpected vocabulary: {model.config.vocab_size}")
    if model.config._attn_implementation != "eager":
        raise RuntimeError("Hugging Face model did not select eager attention")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("model contains a non-FP32 parameter")

    requested = ("probe", "long") if args.which == "all" else (args.which,)
    for name in requested:
        manifest = generate_one(
            name=name,
            text=PROBE_TEXT if name == "probe" else LONG_TEXT,
            model=model,
            tokenizer=tokenizer,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            manifest_dir=args.manifest_dir,
            capture_intermediates=name == "probe",
        )
        print(
            f"{name}: {manifest['token_count']} tokens, "
            f"mean NLL {manifest['mean_nll_fp64']:.9f}"
        )


if __name__ == "__main__":
    main()
