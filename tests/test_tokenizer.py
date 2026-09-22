from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import random
import struct
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(
    "/home/rmj/.cache/huggingface/hub/"
    "models--HuggingFaceTB--SmolLM2-135M/snapshots/"
    "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
)
TOKENIZER_JSON = SNAPSHOT / "tokenizer.json"


def _load_export_module():
    spec = importlib.util.spec_from_file_location(
        "export_tokenizer", ROOT / "scripts/export_tokenizer.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CRuntime:
    def __init__(self, library: Path, tokenizer_binary: Path):
        self.lib = ctypes.CDLL(str(library))
        self.lib.sm_tokenizer_load.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.sm_tokenizer_load.restype = ctypes.c_int
        self.lib.sm_tokenizer_free.argtypes = [ctypes.c_void_p]
        self.lib.sm_tokenizer_vocab_size.argtypes = [ctypes.c_void_p]
        self.lib.sm_tokenizer_vocab_size.restype = ctypes.c_uint32
        self.lib.sm_tokenizer_encode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.sm_tokenizer_encode.restype = ctypes.c_int
        self.lib.sm_tokenizer_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.sm_tokenizer_decode.restype = ctypes.c_int
        self.lib.sm_tokenizer_decode_token.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.sm_tokenizer_decode_token.restype = ctypes.c_int
        self.lib.sm_tokenizer_buffer_free.argtypes = [ctypes.c_void_p]
        self.handle = ctypes.c_void_p()
        error = ctypes.create_string_buffer(512)
        status = self.lib.sm_tokenizer_load(
            str(tokenizer_binary).encode(), ctypes.byref(self.handle), error, len(error)
        )
        assert status == 0, error.value.decode()

    def close(self) -> None:
        if self.handle:
            self.lib.sm_tokenizer_free(self.handle)
            self.handle = ctypes.c_void_p()

    def encode_bytes(self, data: bytes) -> list[int]:
        storage = (ctypes.c_uint8 * len(data)).from_buffer_copy(data) if data else None
        output = ctypes.POINTER(ctypes.c_uint32)()
        count = ctypes.c_size_t()
        error = ctypes.create_string_buffer(512)
        status = self.lib.sm_tokenizer_encode(
            self.handle,
            storage,
            len(data),
            ctypes.byref(output),
            ctypes.byref(count),
            error,
            len(error),
        )
        if status != 0:
            raise ValueError(error.value.decode())
        result = [output[index] for index in range(count.value)]
        self.lib.sm_tokenizer_buffer_free(output)
        return result

    def decode(self, ids: list[int]) -> bytes:
        storage = (ctypes.c_uint32 * len(ids))(*ids) if ids else None
        output = ctypes.POINTER(ctypes.c_uint8)()
        count = ctypes.c_size_t()
        error = ctypes.create_string_buffer(512)
        status = self.lib.sm_tokenizer_decode(
            self.handle,
            storage,
            len(ids),
            ctypes.byref(output),
            ctypes.byref(count),
            error,
            len(error),
        )
        if status != 0:
            raise ValueError(error.value.decode())
        result = ctypes.string_at(output, count.value) if count.value else b""
        self.lib.sm_tokenizer_buffer_free(output)
        return result

    def decode_id(self, token_id: int) -> bytes:
        output = ctypes.POINTER(ctypes.c_uint8)()
        count = ctypes.c_size_t()
        error = ctypes.create_string_buffer(512)
        status = self.lib.sm_tokenizer_decode_token(
            self.handle,
            token_id,
            ctypes.byref(output),
            ctypes.byref(count),
            error,
            len(error),
        )
        if status != 0:
            raise ValueError(error.value.decode())
        result = ctypes.string_at(output, count.value) if count.value else b""
        self.lib.sm_tokenizer_buffer_free(output)
        return result


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    pytest.importorskip("tokenizers")
    if not TOKENIZER_JSON.exists():
        pytest.skip(f"pinned tokenizer snapshot is absent: {TOKENIZER_JSON}")
    directory = tmp_path_factory.mktemp("sm-tokenizer")
    binary = directory / "tokenizer.bin"
    manifest = directory / "manifest.json"
    library = directory / "libsm_tokenizer.so"
    cli = directory / "tokenizer_cli"
    subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/export_tokenizer.py"),
            "--tokenizer-json",
            str(TOKENIZER_JSON),
            "--output",
            str(binary),
            "--manifest",
            str(manifest),
        ],
        check=True,
    )
    pcre_cflags = subprocess.check_output(
        ["pkg-config", "--cflags", "libpcre2-8"], text=True
    ).split()
    pcre_libs = subprocess.check_output(
        ["pkg-config", "--libs", "libpcre2-8"], text=True
    ).split()
    common = ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "include")]
    subprocess.run(
        common
        + ["-fPIC", "-shared"]
        + pcre_cflags
        + [str(ROOT / "src/tokenizer.c")]
        + pcre_libs
        + ["-o", str(library)],
        check=True,
    )
    subprocess.run(
        common
        + pcre_cflags
        + [str(ROOT / "src/tokenizer.c"), str(ROOT / "tests/tokenizer_cli.c")]
        + pcre_libs
        + ["-o", str(cli)],
        check=True,
    )
    runtime = CRuntime(library, binary)
    from tokenizers import Tokenizer

    reference = Tokenizer.from_file(str(TOKENIZER_JSON))
    yield {
        "binary": binary,
        "manifest": manifest,
        "library": library,
        "cli": cli,
        "runtime": runtime,
        "reference": reference,
    }
    runtime.close()


def _reference_ids(reference, text: str) -> list[int]:
    return reference.encode(text, add_special_tokens=False).ids


def test_export_is_deterministic_and_pinned(built, tmp_path):
    second = tmp_path / "second.bin"
    subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/export_tokenizer.py"),
            "--tokenizer-json",
            str(TOKENIZER_JSON),
            "--output",
            str(second),
        ],
        check=True,
    )
    original = built["binary"].read_bytes()
    assert second.read_bytes() == original
    manifest = json.loads(built["manifest"].read_text())
    assert manifest["source_sha256"] == hashlib.sha256(TOKENIZER_JSON.read_bytes()).hexdigest()
    assert manifest["export_sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["vocab_size"] == 49152
    assert manifest["merge_count"] == 48900
    assert manifest["special_count"] == 17
    assert original[:8] == b"SMTOK001"
    assert struct.unpack_from("<I", original, 8)[0] == 1


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello, world! SmolLM2's tokenizer isn't GPT-2 exactly.",
        "안녕하세요. 한국어 토크나이저 검증입니다.",
        "中文、日本語、العربية、हिन्दी、русский、Ελληνικά",
        "🙂🚀 café naïve — déjà vu",
        "0123456789 １２３４５ ١٢٣٤ १२३४ ²³ ⅣⅤ ①② 𝟠𝟡",
        "I'm we're they'll she'd I'M 'S 'LL 'VE 'RE",
        "  leading\tspaces\n\ntrailing  \r\n",
        "a\x00b\x04c\x06d\x1de",
        "<|endoftext|><|im_start|>x<|im_end|>",
        "left<repo_name><reponame><file_sep><filename>right",
        "<gh_stars><issue_start><issue_comment><issue_closed>",
        "<jupyter_start><jupyter_text><jupyter_code><jupyter_output>"
        "<jupyter_script><empty_output>",
    ],
)
def test_curated_encode_exact(built, text):
    assert built["runtime"].encode_bytes(text.encode()) == _reference_ids(
        built["reference"], text
    )


def test_every_unicode_number_is_individual(built):
    text = "A1２٣०𝟠²Ⅳ①B"
    assert built["runtime"].encode_bytes(text.encode()) == _reference_ids(
        built["reference"], text
    )


def test_randomized_encode_exact(built):
    rng = random.Random(0x5A0112)
    alphabet = list("abcXYZ .,!?\n\r\t'0129") + [
        "안",
        "녕",
        "中",
        "文",
        "١",
        "２",
        "²",
        "Ⅳ",
        "🙂",
        "é",
        "\x00",
        "\x04",
        "<|endoftext|>",
        "<|im_start|>",
        "<repo_name>",
    ]
    for _ in range(2000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(80)))
        assert built["runtime"].encode_bytes(text.encode()) == _reference_ids(
            built["reference"], text
        )


def test_decode_all_vocabulary_ids(built):
    exporter = _load_export_module()
    _, inverse = exporter.byte_alphabet()
    source = json.loads(TOKENIZER_JSON.read_text())
    vocab = [None] * len(source["model"]["vocab"])
    for token, token_id in source["model"]["vocab"].items():
        vocab[token_id] = token
    specials = {
        item["id"]: item["content"].encode()
        for item in source["added_tokens"]
        if item["special"]
    }
    reference = built["reference"]
    for token_id, token in enumerate(vocab):
        expected = specials.get(
            token_id, bytes(inverse[character] for character in token)
        )
        actual = built["runtime"].decode_id(token_id)
        assert actual == expected
        assert actual.decode("utf-8", errors="replace") == reference.decode(
            [token_id], skip_special_tokens=False
        )


def test_sequence_decode_and_specials(built):
    texts = [
        "Hello\x00 world",
        "안녕하세요 🙂",
        "a<|endoftext|><|im_start|>b<|im_end|>",
        "0123 １２٣",
    ]
    for text in texts:
        ids = _reference_ids(built["reference"], text)
        expected = built["reference"].decode(ids, skip_special_tokens=False).encode()
        assert built["runtime"].decode(ids) == expected
    assert built["runtime"].decode([]) == b""


def test_invalid_input_and_token_id_are_rejected(built):
    with pytest.raises(ValueError, match="valid UTF-8"):
        built["runtime"].encode_bytes(b"\xff")
    with pytest.raises(ValueError, match="out of range"):
        built["runtime"].decode([49152])


def test_cli_binary_protocol_and_embedded_nul(built):
    data = b"a\x00b<|endoftext|>c"
    encoded = subprocess.run(
        [str(built["cli"]), "encode", str(built["binary"])],
        input=data,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    count = struct.unpack_from("<I", encoded)[0]
    ids = list(struct.unpack_from(f"<{count}I", encoded, 4))
    assert ids == built["runtime"].encode_bytes(data)
    decoded = subprocess.run(
        [str(built["cli"]), "decode", str(built["binary"])],
        input=encoded,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    assert decoded == data


def test_loader_rejects_malformed_files(built, tmp_path):
    original = built["binary"].read_bytes()

    def rejected(name: str, blob: bytes, expected: bytes) -> None:
        path = tmp_path / name
        path.write_bytes(blob)
        process = subprocess.run(
            [str(built["cli"]), "info", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.returncode != 0
        assert expected in process.stderr

    rejected("truncated.bin", original[:-1], b"length")
    changed = bytearray(original)
    changed[300] ^= 1
    rejected("checksum.bin", changed, b"checksum")
    changed = bytearray(original)
    changed[0] ^= 1
    rejected("magic.bin", changed, b"header")
    changed = bytearray(original)
    struct.pack_into("<Q", changed, 24, len(changed) + 1)
    rejected("file-length.bin", changed, b"length")
    changed = bytearray(original)
    struct.pack_into("<Q", changed, 152, 128)
    rejected("section.bin", changed, b"section")
    changed = bytearray(original)
    changed[88] ^= 1
    rejected("source.bin", changed, b"revision")

    changed = bytearray(original)
    cursor = struct.unpack_from("<Q", changed, 152)[0]
    records = []
    for _ in range(19):
        encoded_length, decoded_length = struct.unpack_from("<II", changed, cursor)
        data_start = cursor + 8
        records.append((data_start, encoded_length, decoded_length))
        cursor = data_start + encoded_length + decoded_length
    first, second = records[17], records[18]
    assert first[1:] == second[1:] == (1, 1)
    changed[second[0] : second[0] + 2] = changed[first[0] : first[0] + 2]
    changed[120:152] = hashlib.sha256(changed[256:]).digest()
    rejected("duplicate-vocab.bin", changed, b"duplicate vocabulary")
