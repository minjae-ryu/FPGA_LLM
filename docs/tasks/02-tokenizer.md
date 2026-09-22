# Task 02: independent SmolLM2 tokenizer

## Objective and prerequisite

Implement the pinned tokenizer independently in C, establish exact token-ID
parity before model-logit validation, and provide deterministic export and a
defensive loader. The prerequisite was the version-1 contract in
`docs/design/contracts.md`; model numerical gates do not depend on tokenizer
work until this exact-ID gate passes.

## Inputs, skill, and interfaces

- Model: `HuggingFaceTB/SmolLM2-135M`
- Revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- Cached input: `/home/rmj/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/tokenizer.json`
- Input SHA-256: `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`
- Local skill: `.agents/skills/smollm-port/SKILL.md`
- External C dependency: PCRE2 10.39 through `pkg-config libpcre2-8`
- Independent reference: Python `tokenizers` 0.22.2 loaded directly from the
  pinned JSON

The worker-owned public interface is the opaque `SmTokenizer` in
`include/sm_tokenizer.h`. Load/free, vocabulary size, encode, one-token decode,
sequence decode, and returned-buffer free are separate calls. Encoding accepts
explicit bytes plus length and adds no BOS/EOS. No shared model API or build
file was changed by this task.

Owned source and records are `include/sm_tokenizer.h`, `src/tokenizer.c`,
`scripts/export_tokenizer.py`, `tests/test_tokenizer.py`,
`tests/tokenizer_cli.c`, `docs/design/tokenizer.md`, this task record, and
`manifests/tokenizer.json`. `models/tokenizer.bin` is generated and ignored.

## Commands and observed results

Exporter and artifact:

```text
$ python3 scripts/export_tokenizer.py --tokenizer-json /home/rmj/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/tokenizer.json --output models/tokenizer.bin --manifest manifests/tokenizer.json
wrote models/tokenizer.bin (1683200 bytes, sha256=c2145cd851d7d1614d206f825f805262c97c0b81c63f987ab5ec2552da066ad2)
```

Strict standalone compilation and a known probe:

```text
$ cc -std=c11 -O2 -Wall -Wextra -Werror -Iinclude $(pkg-config --cflags libpcre2-8) src/tokenizer.c tests/tokenizer_cli.c $(pkg-config --libs libpcre2-8) -o /tmp/tokenizer-build/tokenizer_cli
$ /tmp/tokenizer-build/tokenizer_cli info models/tokenizer.bin
49152
$ printf 'Hello world' | /tmp/tokenizer-build/tokenizer_cli encode models/tokenizer.bin | od -An -tu4
          2      19556        905
```

Independent parity and malformed-file suite:

```text
$ python3 -m pytest -q tests/test_tokenizer.py
.....................                                                    [100%]
21 passed in 1.59s
```

The suite checks exact IDs on curated English, Korean, multilingual, emoji,
combining-character, all Unicode number-category examples, lower/upper-case
contractions, varied whitespace, every special token and adjacency, embedded
NUL and omitted control-byte symbols, empty input, and 2,000 seeded randomized
strings. A separate development differential ran 5,000 seeded random strings
with exact parity. It verifies exact raw decode bytes and independent Hugging
Face decode semantics for all 49,152 IDs. It also exercises sequence decode,
invalid UTF-8/IDs, CLI binary I/O, deterministic export, and rejection of
truncation, bad physical length, checksum corruption, bad magic, invalid section
bounds, and wrong source provenance. No token-ID mismatches remain.

Generated artifact hashes:

- `models/tokenizer.bin`: `c2145cd851d7d1614d206f825f805262c97c0b81c63f987ab5ec2552da066ad2`
- payload: `da27898fd735238c5c7b41acc836dd1ded2755f828047f7bcc2e1a2c2a489dbe`
- `manifests/tokenizer.json`: `a5f80d3484e3fb6dc2c93d357f5e41775f86f9159fd294b8f4b8616fbf38d77c`

## Edge cases, limitations, and next dependency

The C API deliberately rejects invalid UTF-8 because PCRE2 Unicode properties
define this text pipeline. Valid text can contain NUL and all Unicode scalar
values. The trained BPE alphabet omits 21 raw byte symbols; matching Hugging
Face therefore loses those bytes when they occur in a ByteLevel regex token.
An individual token can decode to incomplete UTF-8, so decode is byte-oriented.

The test suite needs the pinned cached snapshot and Python `tokenizers` for the
independent reference. The C runtime itself needs only libc and PCRE2. Runtime
matching scans the 17 special tokens linearly; this is bounded for the pinned
format and is not intended as a general mutable-tokenizer implementation.

The exact-token gate is complete. Model reference probes and prepared scoring
records may now consume `models/tokenizer.bin` through the public API.

A `-Wpedantic` strict build also passed. AddressSanitizer and UndefinedBehavior
Sanitizer passed loader, multilingual/special/control-byte encode, and decode
smokes with leak detection disabled because LeakSanitizer is unavailable under
the execution environment's ptrace wrapper.

The repository-level build was already current:

```text
$ make smollm
make: Nothing to be done for 'smollm'.
```

`make check` ran 73 tests in 4.63 seconds. All tokenizer tests passed, while two
concurrently owned runtime chunk-equivalence cases failed: Q8/FP32-KV had 368
of 38,400 elements outside tolerance with maximum absolute difference
0.06068516, and FP32/KV8 had 503 of 38,400 with maximum absolute difference
0.01252508. The final summary was `2 failed, 71 passed`. These failures are in
`tests/test_runtime.py::test_chunk_reset_modes_bounds_and_cache_accounting` and
are outside tokenizer-owned code; no tokenizer claim depends on them.
