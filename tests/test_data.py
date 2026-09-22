import hashlib
import json
import math
import struct
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from scripts.prepare_data import (
    DEFAULT_MODEL_DIR,
    EVAL_MAGIC,
    LM_KIND,
    MC_KIND,
    Record,
    encode_pair,
    hellaswag_preprocess,
    multiple_choice_records,
    parse_smeval,
    serialize_smeval,
    wikitext_windows,
)


def test_smeval_round_trip_and_strict_framing():
    records = [Record(0, 0, 0, 1, (7, 8, 9), 1, 3, 0)]
    payload = serialize_smeval(LM_KIND, records)
    assert payload[:8] == EVAL_MAGIC
    assert parse_smeval(payload) == (LM_KIND, records)
    with pytest.raises(ValueError, match="truncated"):
        parse_smeval(payload[:-1])
    with pytest.raises(ValueError, match="trailing"):
        parse_smeval(payload + b"\0")
    with pytest.raises(ValueError, match="magic"):
        parse_smeval(b"BADMAGIC" + payload[8:])


@pytest.mark.parametrize(
    "record, message",
    [
        (Record(0, 0, 0, 1, (1, 2), 0, 2, 0), "bounds"),
        (Record(0, 0, 0, 1, (1, 49152), 1, 2, 0), "vocabulary"),
        (Record(0, 0, 0, 2, (1, 2), 1, 2, 0), "LM metadata"),
    ],
)
def test_lm_record_validation(record, message):
    with pytest.raises(ValueError, match=message):
        serialize_smeval(LM_KIND, [record])


def test_multiple_choice_group_invariants():
    valid = [
        Record(0, 1, 0, 2, (1, 2), 1, 2, 4),
        Record(0, 1, 1, 2, (1, 3), 1, 2, 5),
    ]
    assert parse_smeval(serialize_smeval(MC_KIND, valid)) == (MC_KIND, valid)
    with pytest.raises(ValueError, match="ordered"):
        serialize_smeval(MC_KIND, list(reversed(valid)))
    with pytest.raises(ValueError, match="incomplete"):
        serialize_smeval(MC_KIND, valid[:1])
    bad_label = [valid[0], Record(0, 0, 1, 2, (1, 3), 1, 2, 5)]
    with pytest.raises(ValueError, match="inconsistent"):
        serialize_smeval(MC_KIND, bad_label)


def test_wikitext_windows_cover_each_global_target_exactly_once():
    token_ids = list(range(5000))
    records = wikitext_windows(token_ids)
    seen = []
    for window_index, record in enumerate(records):
        start = window_index * 512
        seen.extend(start + local for local in range(record.score_start, record.score_end))
    assert seen == list(range(1, len(token_ids)))
    assert len(seen) == len(set(seen))
    assert [record.score_start for record in records[:3]] == [1, 1536, 1536]


def test_wikitext_smoke_geometry():
    records = wikitext_windows(list(range(8192)))
    assert len(records) == 13
    assert sum(record.target_count for record in records) == 8191
    assert len(records[0].token_ids) == 2048
    assert len(records[-1].token_ids) == 2048


@pytest.fixture(scope="module")
def tokenizer():
    if not DEFAULT_MODEL_DIR.is_dir():
        pytest.skip("pinned tokenizer snapshot is not cached")
    return AutoTokenizer.from_pretrained(DEFAULT_MODEL_DIR, local_files_only=True)


def test_lm_eval_boundary_and_unicode_character_count(tokenizer):
    context = "Question: café  "
    continuation = " 답변🙂"
    token_ids, score_start, moved = encode_pair(tokenizer, context, continuation)
    assert moved == "  " + continuation
    assert score_start >= 1
    assert score_start < len(token_ids)
    records = multiple_choice_records(
        tokenizer, [(context, ["답변🙂", "plain answer"], 0)]
    )
    assert records[0].norm_chars == len("답변🙂")
    assert records[0].norm_chars != len("답변🙂".encode("utf-8"))


def test_hellaswag_preprocessing_matches_pinned_transform():
    source = "  Cook [title]: [noise]Mix  the batter.  "
    assert hellaswag_preprocess(source) == "Cook. : Mix the batter."


def test_generated_artifact_hashes_when_present():
    root = Path(__file__).resolve().parents[1]
    for manifest_path in sorted((root / "manifests").glob("data_*.json")):
        manifest = json.loads(manifest_path.read_text())
        artifact_path = root / manifest["artifact"]["path"]
        if not artifact_path.is_file():
            continue
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert digest == manifest["artifact"]["sha256"]
        kind, records = parse_smeval(artifact_path.read_bytes())
        assert kind == manifest["kind"]
        assert len(records) == manifest["record_count"]
        assert sum(record.target_count for record in records) == manifest["target_count"]


def test_generated_reference_hashes_and_shapes_when_present():
    root = Path(__file__).resolve().parents[1]
    for manifest_path in sorted((root / "manifests").glob("reference_*.json")):
        manifest = json.loads(manifest_path.read_text())
        token_info = manifest["artifacts"]["tokens"]
        token_path = root / token_info["path"]
        if not token_path.is_file():
            continue
        token_payload = token_path.read_bytes()
        magic, count = struct.unpack_from("<8sI", token_payload)
        assert magic == b"SMTOK001"
        assert count == manifest["token_count"]
        assert len(token_payload) == 12 + 4 * count
        assert hashlib.sha256(token_payload).hexdigest() == token_info["sha256"]

        float_artifacts = [manifest["artifacts"]["logits"]] + manifest["intermediates"]
        for info in float_artifacts:
            path = root / info["path"]
            assert path.stat().st_size == 4 * math.prod(info["shape"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
