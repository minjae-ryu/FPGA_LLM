#!/usr/bin/env python3
"""Export the pinned SmolLM2 tokenizer to the defensive C runtime format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path


MAGIC = b"SMTOK001"
VERSION = 1
ENDIAN = 0x01020304
HEADER_SIZE = 256
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


def byte_alphabet() -> tuple[dict[int, str], dict[str, int]]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(0xA1, 0xAC + 1))
    visible += list(range(0xAE, 0xFF + 1))
    chars = visible[:]
    extra = 0
    for value in range(256):
        if value not in visible:
            visible.append(value)
            chars.append(256 + extra)
            extra += 1
    encode = dict(zip(visible, map(chr, chars), strict=True))
    return encode, {character: value for value, character in encode.items()}


def decode_vocab_token(token: str, inverse: dict[str, int]) -> bytes:
    try:
        return bytes(inverse[character] for character in token)
    except KeyError as exc:
        raise ValueError(f"vocabulary token contains non-ByteLevel character {exc.args[0]!r}") from exc


def align8(buffer: bytearray) -> None:
    buffer.extend(b"\0" * (-len(buffer) % 8))


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def build_export(source: Path) -> tuple[bytes, dict[str, object]]:
    source_bytes = source.read_bytes()
    data = json.loads(source_bytes)
    if data.get("version") != "1.0":
        raise ValueError("expected tokenizer.json version 1.0")

    pre = data.get("pre_tokenizer")
    expected_pre = {
        "type": "Sequence",
        "pretokenizers": [
            {"type": "Digits", "individual_digits": True},
            {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
        ],
    }
    if pre != expected_pre or data.get("normalizer") is not None:
        raise ValueError("tokenizer preprocessing is not pinned Digits -> ByteLevel")

    model = data.get("model", {})
    if any(
        model.get(key) != expected
        for key, expected in {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
        }.items()
    ):
        raise ValueError("unsupported BPE configuration")

    vocab_by_text = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab_by_text, dict) or not isinstance(merges, list):
        raise ValueError("missing BPE vocabulary or merges")
    vocab_size = len(vocab_by_text)
    vocab: list[str | None] = [None] * vocab_size
    for token, token_id in vocab_by_text.items():
        if not isinstance(token, str) or not isinstance(token_id, int):
            raise ValueError("invalid vocabulary entry")
        if token_id < 0 or token_id >= vocab_size or vocab[token_id] is not None:
            raise ValueError(f"invalid or duplicate vocabulary id {token_id}")
        vocab[token_id] = token
    if any(token is None for token in vocab):
        raise ValueError("vocabulary ids are not dense")
    concrete_vocab = [token for token in vocab if token is not None]

    encoder, inverse = byte_alphabet()
    decoded_vocab = [decode_vocab_token(token, inverse) for token in concrete_vocab]
    byte_ids: list[int] = []
    for value in range(256):
        symbol = encoder[value]
        # This tokenizer was trained with a restricted initial alphabet. The
        # tokenizers BPE implementation drops an absent byte when unk_token is
        # null; UINT32_MAX records that behavior explicitly.
        byte_ids.append(vocab_by_text.get(symbol, 0xFFFFFFFF))

    merge_records: list[tuple[int, int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for rank, merge in enumerate(merges):
        if not isinstance(merge, str):
            raise ValueError(f"merge {rank} is not a string")
        parts = merge.split(" ")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"merge {rank} does not contain two symbols")
        left, right = parts
        result = left + right
        try:
            record = (vocab_by_text[left], vocab_by_text[right], vocab_by_text[result])
        except KeyError as exc:
            raise ValueError(f"merge {rank} references absent token {exc.args[0]!r}") from exc
        if record[:2] in seen_pairs:
            raise ValueError(f"duplicate merge pair at rank {rank}")
        seen_pairs.add(record[:2])
        merge_records.append(record)

    special_records: list[tuple[int, bytes]] = []
    seen_special_ids: set[int] = set()
    seen_special_text: set[bytes] = set()
    for item in data.get("added_tokens", []):
        if not item.get("special"):
            continue
        if any(item.get(key) != False for key in ("single_word", "lstrip", "rstrip", "normalized")):
            raise ValueError("unsupported special-token matching flags")
        token_id = item.get("id")
        content = item.get("content")
        if not isinstance(token_id, int) or not isinstance(content, str):
            raise ValueError("invalid special token")
        encoded = content.encode("utf-8")
        if token_id < 0 or token_id >= vocab_size or vocab_by_text.get(content) != token_id:
            raise ValueError(f"special token {content!r} disagrees with vocabulary")
        if token_id in seen_special_ids or encoded in seen_special_text or not encoded:
            raise ValueError("duplicate or empty special token")
        seen_special_ids.add(token_id)
        seen_special_text.add(encoded)
        special_records.append((token_id, encoded))

    sections: list[tuple[int, int, int, int]] = []
    output = bytearray(b"\0" * HEADER_SIZE)

    align8(output)
    start = len(output)
    for token, decoded in zip(concrete_vocab, decoded_vocab, strict=True):
        encoded = token.encode("utf-8")
        output += struct.pack("<II", len(encoded), len(decoded))
        output += encoded
        output += decoded
    sections.append((start, len(output) - start, vocab_size, 0))

    align8(output)
    start = len(output)
    for record in merge_records:
        output += struct.pack("<III", *record)
    sections.append((start, len(output) - start, len(merge_records), 12))

    align8(output)
    start = len(output)
    for token_id, content in special_records:
        output += struct.pack("<II", token_id, len(content))
        output += content
    sections.append((start, len(output) - start, len(special_records), 0))

    align8(output)
    start = len(output)
    output += struct.pack("<256I", *byte_ids)
    sections.append((start, len(output) - start, 256, 4))

    revision_bytes = REVISION.encode("ascii")
    if len(revision_bytes) != 40:
        raise AssertionError("revision must occupy exactly 40 bytes")
    source_hash = hashlib.sha256(source_bytes).digest()
    payload_hash = hashlib.sha256(output[HEADER_SIZE:]).digest()
    struct.pack_into("<8sIIIIQIIII", output, 0, MAGIC, VERSION, ENDIAN,
                     HEADER_SIZE, len(sections), len(output), vocab_size,
                     len(merge_records), len(special_records), 1)
    output[48:88] = revision_bytes
    output[88:120] = source_hash
    output[120:152] = payload_hash
    for index, descriptor in enumerate(sections):
        struct.pack_into("<QQII", output, 152 + 24 * index, *descriptor)

    manifest: dict[str, object] = {
        "format": {"magic": MAGIC.decode("ascii"), "version": VERSION},
        "model": "HuggingFaceTB/SmolLM2-135M",
        "revision": REVISION,
        "source": str(source),
        "source_sha256": source_hash.hex(),
        "export_sha256": hashlib.sha256(output).hexdigest(),
        "payload_sha256": payload_hash.hex(),
        "file_bytes": len(output),
        "vocab_size": vocab_size,
        "merge_count": len(merge_records),
        "special_count": len(special_records),
    }
    return bytes(output), manifest


def atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    arguments = parser.parse_args()

    contents, manifest = build_export(arguments.tokenizer_json)
    atomic_write(arguments.output, contents)
    if arguments.manifest:
        rendered = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        atomic_write(arguments.manifest, rendered)
    print(
        f"wrote {arguments.output} ({len(contents)} bytes, "
        f"sha256={manifest['export_sha256']})"
    )


if __name__ == "__main__":
    main()
