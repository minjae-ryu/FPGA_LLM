# SmolLM2 tokenizer contract

This runtime implements the tokenizer from `HuggingFaceTB/SmolLM2-135M` at
revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. The pinned
`tokenizer.json` SHA-256 is
`9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`.
The exporter rejects another pipeline or BPE configuration, and the loader
rejects another source hash or revision.

## Encoding and decoding

Added special tokens are extracted first using literal, non-stripping matches.
At the same byte position, the longest match wins. General text then runs the
configured sequence:

1. PCRE2 `\p{N}` isolates every Unicode number code point. This includes the
   `Nd`, `Nl`, and `No` classes, rather than ASCII digits alone.
2. Each resulting span runs the ByteLevel expression
   `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
   with UTF and Unicode-property modes. No prefix space is inserted.
3. UTF-8 bytes in each regex token map through the GPT-2 byte alphabet. BPE
   repeatedly applies the lowest-ranked adjacent merge, merging every
   non-overlapping occurrence from left to right.

The trained vocabulary omits 21 byte-alphabet symbols. They correspond to raw
bytes 4, 6, 19, 20, 22, 29, 192, 193, 241, 242, and 245 through 255. The
Hugging Face BPE model has no unknown token, so an omitted byte contributes no
token. It is removed inside its ByteLevel regex token before BPE; regex-token
boundaries still prevent merges. The binary represents an omitted mapping as
`UINT32_MAX` and the loader verifies that the symbol is truly absent.

`sm_tokenizer_encode` accepts an explicit byte pointer and length, so embedded
NUL is supported. Input must be valid UTF-8; malformed or truncated UTF-8 is an
error. It returns only pipeline token IDs and never inserts BOS or EOS. Empty
input returns zero IDs. The caller releases returned storage with
`sm_tokenizer_buffer_free`.

Decode maps ordinary vocabulary symbols back to their exact ByteLevel bytes.
Special-token IDs emit their literal UTF-8 content, so sequence decode does not
silently skip special tokens. Individual vocabulary IDs can decode to an
incomplete UTF-8 byte sequence; this is intentional, and the byte pointer plus
length remains lossless. Invalid IDs fail before output is allocated.

The loaded tokenizer is immutable. Encode allocates its own PCRE2 match state,
so distinct threads may use one tokenizer concurrently as long as it is not
freed during a call.

## Binary format `SMTOK001`

All integers are unsigned little-endian. Offsets are absolute. Variable records
are packed without padding inside a section; section starts are 8-byte aligned,
and inter-section padding is zero. The header is 256 bytes:

| Offset | Encoding | Meaning |
|---:|---|---|
| 0 | 8 bytes | magic `SMTOK001` |
| 8 | u32 | format version 1 |
| 12 | u32 | endian marker `0x01020304` |
| 16 | u32 | header size 256 |
| 20 | u32 | section count 4 |
| 24 | u64 | exact file size |
| 32 | u32 | vocabulary count |
| 36 | u32 | merge count |
| 40 | u32 | special-token count |
| 44 | u32 | flags, exactly 1 (`Digits` individual mode) |
| 48 | 40 bytes | pinned ASCII source revision |
| 88 | 32 bytes | raw SHA-256 of source `tokenizer.json` |
| 120 | 32 bytes | raw SHA-256 of bytes `[256, file_size)` |
| 152 | 4 descriptors | vocabulary, merges, specials, byte alphabet |
| 248 | 8 bytes | reserved zero |

Each 24-byte descriptor is `u64 offset, u64 byte_length, u32 count, u32
entry_size`. Variable sections use `entry_size=0`.

The vocabulary has one record per ID in ascending order. A record is `u32
encoded_length, u32 decoded_length`, followed by the encoded UTF-8 BPE symbol
and decoded bytes. The loader independently verifies the UTF-8, byte-alphabet
mapping, and decoded value.

The merge section has fixed 12-byte records in rank order: `u32 left_id, u32
right_id, u32 result_id`. The loader rejects duplicate pairs and verifies that
the result's encoded symbol is the concatenation of the inputs.

The special-token section has `u32 id, u32 content_length`, followed by literal
UTF-8 content. IDs and content must be unique and agree with both forms of the
vocabulary record.

The byte-alphabet section contains 256 u32 vocabulary IDs in raw-byte order.
An absent alphabet symbol is `UINT32_MAX`.

The loader caps the file at 256 MiB and validates the complete physical length,
magic, version, endian marker, pinned provenance, checksum, count relationships,
section bounds/order/non-overlap, zero padding, exact section consumption,
every ID, and every cross-record invariant before returning a handle. The
exporter writes through a temporary file, flushes it, and atomically replaces
the destination.

## Commands

Export the pinned artifact:

```sh
python3 scripts/export_tokenizer.py \
  --tokenizer-json /home/rmj/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/tokenizer.json \
  --output models/tokenizer.bin \
  --manifest manifests/tokenizer.json
```

Compile the independent test CLI:

```sh
mkdir -p build
cc -std=c11 -O2 -Wall -Wextra -Werror -Iinclude \
  $(pkg-config --cflags libpcre2-8) \
  src/tokenizer.c tests/tokenizer_cli.c \
  $(pkg-config --libs libpcre2-8) \
  -o build/tokenizer_cli
```

The CLI's `encode` operation reads raw stdin and writes a little-endian u32
count followed by u32 IDs. `decode` consumes that representation and writes raw
decoded bytes. `decode-id ID` writes one token's bytes, and `info` prints the
vocabulary size.
