#!/usr/bin/env python3
"""Prepare pinned WikiText-2, HellaSwag, and PIQA scoring binaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import tempfile
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq
import pyarrow
import tokenizers
import transformers
from transformers import AutoTokenizer


MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
VOCAB_SIZE = 49152
MAX_CONTEXT = 8192
EVAL_MAGIC = b"SMEVAL01"
LM_KIND = 1
MC_KIND = 2
LM_EVAL_REVISION = "d6de81643928d653435c431bae19945d41d32520"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
HELLASWAG_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
PIQA_REVISION = "2e8ac2dffd59bac8c3c6714948f4c551a0848bb0"

HF_CACHE = Path.home() / ".cache/huggingface"
DEFAULT_MODEL_DIR = (
    HF_CACHE
    / "hub/models--HuggingFaceTB--SmolLM2-135M/snapshots"
    / MODEL_REVISION
)
DEFAULT_WIKITEXT = (
    HF_CACHE
    / "hub/datasets--Salesforce--wikitext/snapshots"
    / WIKITEXT_REVISION
    / "wikitext-2-raw-v1/test-00000-of-00001.parquet"
)
DEFAULT_HELLASWAG = (
    HF_CACHE
    / "hub/datasets--Rowan--hellaswag/snapshots"
    / HELLASWAG_REVISION
    / "data/validation-00000-of-00001.parquet"
)
DEFAULT_PIQA_DIR = (
    HF_CACHE
    / "datasets/downloads/extracted/7b6f535dbe59b28b7e097b137617754056edd3cf9cbf05c960d10228ffa5a944"
    / "physicaliqa-train-dev"
)
DEFAULT_PIQA_SCRIPT = (
    HF_CACHE
    / "hub/datasets--ybisk--piqa/snapshots"
    / PIQA_REVISION
    / "piqa.py"
)


@dataclass(frozen=True)
class Record:
    group: int
    label: int
    choice: int
    nchoices: int
    token_ids: tuple[int, ...]
    score_start: int
    score_end: int
    norm_chars: int

    @property
    def target_count(self) -> int:
        return self.score_end - self.score_start


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = tempfile.NamedTemporaryFile(
        mode="wb", prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False
    )
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


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def validate_records(kind: int, records: Sequence[Record]) -> None:
    if kind not in (LM_KIND, MC_KIND):
        raise ValueError(f"unsupported SMEVAL kind {kind}")
    if not records:
        raise ValueError("SMEVAL file must contain records")
    for index, record in enumerate(records):
        count = len(record.token_ids)
        if count < 2:
            raise ValueError(f"record {index}: token_count must be at least two")
        if count > MAX_CONTEXT:
            raise ValueError(f"record {index}: token_count exceeds model context")
        if not 1 <= record.score_start < record.score_end <= count:
            raise ValueError(f"record {index}: invalid scoring bounds")
        if any(token < 0 or token >= VOCAB_SIZE for token in record.token_ids):
            raise ValueError(f"record {index}: token outside pinned vocabulary")
        fields = (
            record.group,
            record.label,
            record.choice,
            record.nchoices,
            count,
            record.score_start,
            record.score_end,
            record.norm_chars,
        )
        if any(field < 0 or field > 0xFFFFFFFF for field in fields):
            raise ValueError(f"record {index}: field outside u32")
        if kind == LM_KIND:
            if (record.group, record.label, record.choice, record.nchoices) != (0, 0, 0, 1):
                raise ValueError(f"record {index}: invalid LM metadata")
            if record.norm_chars != 0:
                raise ValueError(f"record {index}: LM norm_chars must be zero")
        else:
            if record.nchoices < 2 or record.choice >= record.nchoices:
                raise ValueError(f"record {index}: invalid MC choice metadata")
            if record.label >= record.nchoices:
                raise ValueError(f"record {index}: invalid MC label")
            if record.norm_chars == 0:
                raise ValueError(f"record {index}: MC continuation is empty")

    if kind == MC_KIND:
        expected_group = 0
        offset = 0
        while offset < len(records):
            first = records[offset]
            if first.group != expected_group:
                raise ValueError("MC groups must be contiguous and zero-based")
            end = offset + first.nchoices
            group = records[offset:end]
            if len(group) != first.nchoices:
                raise ValueError(f"group {first.group}: incomplete choices")
            if [record.choice for record in group] != list(range(first.nchoices)):
                raise ValueError(f"group {first.group}: choices are not ordered")
            if any(
                record.group != first.group
                or record.label != first.label
                or record.nchoices != first.nchoices
                for record in group
            ):
                raise ValueError(f"group {first.group}: inconsistent metadata")
            offset = end
            expected_group += 1


def serialize_smeval(kind: int, records: Sequence[Record]) -> bytes:
    validate_records(kind, records)
    output = bytearray(struct.pack("<8sII", EVAL_MAGIC, kind, len(records)))
    for record in records:
        output.extend(
            struct.pack(
                "<8I",
                record.group,
                record.label,
                record.choice,
                record.nchoices,
                len(record.token_ids),
                record.score_start,
                record.score_end,
                record.norm_chars,
            )
        )
        output.extend(struct.pack(f"<{len(record.token_ids)}I", *record.token_ids))
    return bytes(output)


def parse_smeval(payload: bytes) -> tuple[int, list[Record]]:
    if len(payload) < 16:
        raise ValueError("truncated SMEVAL header")
    magic, kind, record_count = struct.unpack_from("<8sII", payload)
    if magic != EVAL_MAGIC:
        raise ValueError("invalid SMEVAL magic")
    records = []
    offset = 16
    for index in range(record_count):
        if len(payload) - offset < 32:
            raise ValueError(f"record {index}: truncated header")
        fields = struct.unpack_from("<8I", payload, offset)
        offset += 32
        group, label, choice, nchoices, count, score_start, score_end, norm_chars = fields
        token_bytes = 4 * count
        if count > VOCAB_SIZE * 1024 or len(payload) - offset < token_bytes:
            raise ValueError(f"record {index}: truncated or unreasonable tokens")
        token_ids = struct.unpack_from(f"<{count}I", payload, offset)
        offset += token_bytes
        records.append(
            Record(
                group,
                label,
                choice,
                nchoices,
                token_ids,
                score_start,
                score_end,
                norm_chars,
            )
        )
    if offset != len(payload):
        raise ValueError("trailing bytes after SMEVAL records")
    validate_records(kind, records)
    return kind, records


def wikitext_windows(
    token_ids: Sequence[int], *, context: int = 2048, stride: int = 512
) -> list[Record]:
    if context < 2 or stride < 1 or stride >= context:
        raise ValueError("invalid context/stride")
    if len(token_ids) < 2:
        raise ValueError("WikiText input has no targets")
    records = []
    start = 0
    last_scored = 0
    while last_scored < len(token_ids) - 1:
        window = tuple(token_ids[start : start + context])
        score_start = 1 if start == 0 else last_scored + 1 - start
        if not 1 <= score_start < len(window):
            raise ValueError("window geometry did not add targets")
        score_end = len(window)
        records.append(Record(0, 0, 0, 1, window, score_start, score_end, 0))
        last_scored = start + score_end - 1
        start += stride
    validate_records(LM_KIND, records)
    return records


def hellaswag_preprocess(text: str) -> str:
    """EleutherAI harness hellaswag/utils.py preprocessing at the pinned revision."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def encode_pair(tokenizer, context: str, continuation: str):
    """Mirror TemplateLM._encode_pair for its causal backend."""
    if not context:
        raise ValueError("MC context cannot be empty")
    trailing_spaces = len(context) - len(context.rstrip())
    if trailing_spaces:
        continuation = context[-trailing_spaces:] + continuation
        context = context[:-trailing_spaces]
    whole_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    continuation_ids = whole_ids[len(context_ids) :]
    if not context_ids or not continuation_ids:
        raise ValueError("empty context or continuation tokenization")
    # lm-eval concatenates these two lists even when a BPE merge changes the
    # final context token at the boundary; preserve that behavior exactly.
    return context_ids + continuation_ids, len(context_ids), continuation


def multiple_choice_records(
    tokenizer, examples: Iterable[tuple[str, Sequence[str], int]]
) -> list[Record]:
    records: list[Record] = []
    for group, (context, choices, label) in enumerate(examples):
        nchoices = len(choices)
        if not 0 <= label < nchoices:
            raise ValueError(f"group {group}: invalid label {label}")
        for choice_index, choice in enumerate(choices):
            continuation = " " + choice
            token_ids, score_start, _moved_continuation = encode_pair(
                tokenizer, context, continuation
            )
            records.append(
                Record(
                    group=group,
                    label=label,
                    choice=choice_index,
                    nchoices=nchoices,
                    token_ids=tuple(token_ids),
                    score_start=score_start,
                    score_end=len(token_ids),
                    # lm-eval process_results computes completion_len from the
                    # processed choice, excluding its target delimiter.
                    norm_chars=len(choice),
                )
            )
    validate_records(MC_KIND, records)
    return records


def load_wikitext(path: Path) -> list[str]:
    table = pq.read_table(path, columns=["text"])
    return table.column("text").to_pylist()


def load_hellaswag(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def load_piqa(data_path: Path, label_path: Path) -> list[dict[str, Any]]:
    lines = data_path.read_text(encoding="utf-8").splitlines()
    labels = label_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(labels):
        raise ValueError("PIQA examples and labels have different lengths")
    result = []
    for line, label in zip(lines, labels, strict=True):
        row = json.loads(line)
        row["label"] = int(label)
        result.append(row)
    return result


def hellaswag_examples(rows: Sequence[dict[str, Any]]):
    for row in rows:
        context = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        query = hellaswag_preprocess(row["activity_label"] + ": " + context)
        choices = [hellaswag_preprocess(ending) for ending in row["endings"]]
        yield query, choices, int(row["label"])


def piqa_examples(rows: Sequence[dict[str, Any]]):
    for row in rows:
        yield (
            f"Question: {row['goal']}\nAnswer:",
            [row["sol1"], row["sol2"]],
            int(row["label"]),
        )


def write_dataset(
    *,
    task: str,
    variant: str,
    kind: int,
    records: Sequence[Record],
    output_dir: Path,
    manifest_dir: Path,
    source_revision: str,
    source_files: Sequence[Path],
    original_examples: int | None,
    total_input_tokens: int | None = None,
) -> dict[str, Any]:
    payload = serialize_smeval(kind, records)
    output_path = output_dir / f"{task}_{variant}.smeval"
    atomic_write(output_path, payload)
    parsed_kind, parsed_records = parse_smeval(output_path.read_bytes())
    if parsed_kind != kind or parsed_records != list(records):
        raise RuntimeError("SMEVAL write/read verification failed")

    target_count = sum(record.target_count for record in records)
    manifest: dict[str, Any] = {
        "format": "SMEVAL01",
        "format_version": 1,
        "task": task,
        "variant": variant,
        "kind": kind,
        "model": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "tokenization": {"add_special_tokens": False, "bos": False, "eos": False},
        "libraries": {
            "pyarrow": pyarrow.__version__,
            "python": platform.python_version(),
            "tokenizers": tokenizers.__version__,
            "transformers": transformers.__version__,
        },
        "source_revision": source_revision,
        "source_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_files
        ],
        "artifact": {
            "path": str(output_path),
            "bytes": len(payload),
            "sha256": sha256_file(output_path),
        },
        "record_count": len(records),
        "target_count": target_count,
        "original_examples": original_examples,
        "total_input_tokens": total_input_tokens,
    }
    if task == "wikitext2":
        manifest.update(
            {
                "join": "\\n\\n",
                "context": 2048,
                "stride": 512,
                "target_coverage": (
                    "global token indices 1..N-1 exactly once; first window scores "
                    "1..end and later windows only new targets"
                ),
            }
        )
    else:
        preprocessing = (
            {
                "context": "ctx_a + ' ' + ctx_b.capitalize()",
                "query": "preprocess(activity_label + ': ' + context)",
                "choices": "preprocess(each ending)",
                "preprocess": [
                    "strip",
                    "replace ' [title]' with '. '",
                    "remove non-greedy bracketed spans with \\[.*?\\]",
                    "replace each occurrence of two spaces with one space",
                ],
            }
            if task == "hellaswag"
            else {
                "query": "'Question: ' + goal + '\\nAnswer:'",
                "choices": "[sol1, sol2]",
            }
        )
        manifest.update(
            {
                "lm_evaluation_harness_revision": LM_EVAL_REVISION,
                "lm_evaluation_harness_files": {
                    "boundary": (
                        "lm_eval/api/model.py TemplateLM._encode_pair; move trailing "
                        "context spaces, tokenize context+continuation and context, then split"
                    ),
                    "normalization": (
                        "lm_eval/api/task.py process_results; completion_len is "
                        "Python len() of each processed choice"
                    ),
                    "task": f"lm_eval/tasks/{task}/{task}.yaml",
                    "hellaswag_preprocess": (
                        "lm_eval/tasks/hellaswag/utils.py" if task == "hellaswag" else None
                    ),
                },
                "target_delimiter": " ",
                "zero_shot": True,
                "preprocessing": preprocessing,
                "normalization": (
                    "Python len() Unicode code points in the processed choice text; "
                    "exclude target delimiter and moved context whitespace"
                ),
                "metrics": ["raw_loglikelihood_accuracy", "character_normalized_accuracy"],
            }
        )
    manifest_path = manifest_dir / f"data_{task}_{variant}.json"
    atomic_json(manifest_path, manifest)
    return manifest


def prepare_wikitext(args, tokenizer, variant: str):
    rows = load_wikitext(args.wikitext)
    joined = "\n\n".join(rows)
    token_ids = tokenizer.encode(joined, add_special_tokens=False)
    selected = token_ids[:8192] if variant == "smoke" else token_ids
    records = wikitext_windows(selected)
    return write_dataset(
        task="wikitext2",
        variant=variant,
        kind=LM_KIND,
        records=records,
        output_dir=args.output_dir,
        manifest_dir=args.manifest_dir,
        source_revision=WIKITEXT_REVISION,
        source_files=[args.wikitext],
        original_examples=len(rows),
        total_input_tokens=len(selected),
    )


def prepare_hellaswag(args, tokenizer, variant: str):
    rows = load_hellaswag(args.hellaswag)
    selected = rows[:32] if variant == "smoke" else rows
    records = multiple_choice_records(tokenizer, hellaswag_examples(selected))
    return write_dataset(
        task="hellaswag",
        variant=variant,
        kind=MC_KIND,
        records=records,
        output_dir=args.output_dir,
        manifest_dir=args.manifest_dir,
        source_revision=HELLASWAG_REVISION,
        source_files=[args.hellaswag],
        original_examples=len(selected),
    )


def prepare_piqa(args, tokenizer, variant: str):
    rows = load_piqa(args.piqa_data, args.piqa_labels)
    selected = rows[:32] if variant == "smoke" else rows
    records = multiple_choice_records(tokenizer, piqa_examples(selected))
    return write_dataset(
        task="piqa",
        variant=variant,
        kind=MC_KIND,
        records=records,
        output_dir=args.output_dir,
        manifest_dir=args.manifest_dir,
        source_revision=PIQA_REVISION,
        source_files=[args.piqa_script, args.piqa_data, args.piqa_labels],
        original_examples=len(selected),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("wikitext2", "hellaswag", "piqa", "all"), default="all")
    parser.add_argument("--variant", choices=("smoke", "full", "all"), default="all")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--wikitext", type=Path, default=DEFAULT_WIKITEXT)
    parser.add_argument("--hellaswag", type=Path, default=DEFAULT_HELLASWAG)
    parser.add_argument("--piqa-data", type=Path, default=DEFAULT_PIQA_DIR / "dev.jsonl")
    parser.add_argument("--piqa-labels", type=Path, default=DEFAULT_PIQA_DIR / "dev-labels.lst")
    parser.add_argument("--piqa-script", type=Path, default=DEFAULT_PIQA_SCRIPT)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [args.model_dir / "tokenizer.json"]
    if args.task in ("wikitext2", "all"):
        required.append(args.wikitext)
    if args.task in ("hellaswag", "all"):
        required.append(args.hellaswag)
    if args.task in ("piqa", "all"):
        required.extend((args.piqa_script, args.piqa_data, args.piqa_labels))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing pinned local inputs:\n" + "\n".join(missing))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    # WikiText is tokenized as one stream and windowed only afterwards.
    # Suppress the tokenizer's model-input warning; no oversized tensor is run.
    tokenizer.model_max_length = 1 << 60
    tasks = ("wikitext2", "hellaswag", "piqa") if args.task == "all" else (args.task,)
    variants = ("smoke", "full") if args.variant == "all" else (args.variant,)
    functions = {
        "wikitext2": prepare_wikitext,
        "hellaswag": prepare_hellaswag,
        "piqa": prepare_piqa,
    }
    for task in tasks:
        for variant in variants:
            manifest = functions[task](args, tokenizer, variant)
            print(
                f"{task}/{variant}: {manifest['record_count']} records, "
                f"{manifest['target_count']} targets, "
                f"{manifest['artifact']['sha256']}"
            )


if __name__ == "__main__":
    main()
