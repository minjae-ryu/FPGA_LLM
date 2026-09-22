# SmolLM2 model export

Implemented the version-one `SML2C001` exporter for the pinned
`HuggingFaceTB/SmolLM2-135M` revision. The production path accepts only the
exact config and safetensors source hashes recorded below. It validates the HF
configuration, the complete 272-name tensor set, every BF16 dtype and every
shape before writing. Output is written to a temporary sibling, flushed, and
atomically renamed.

## Interface

FP32 production export:

```sh
python3 scripts/export_smollm.py \
  --source /home/rmj/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --output models/smollm2-f32.bin \
  --dtype f32 \
  --manifest manifests/model-f32.json
```

Change `--dtype f32` and the two destination names to `q8`,
`models/smollm2-q8.bin`, and `manifests/model-q8.json` for GS64 Q8 matrices.
Norm vectors remain FP32. `--synthetic` may replace `--source SNAPSHOT` to
produce a deterministic two-layer loader fixture with dim 192, three query
heads, one KV head, head dimension 64, context 384, and K dimensions divisible
by 64.

The source hashes are:

- `config.json`: `1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843`
- `model.safetensors`: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`

Each manifest records both source hashes, the output hash and size, the exact
command, tool versions, config, tensor count, and value counts covered by the
post-write verification.

## Quantization reference

`tests/fixtures/quantization.json` is an independently specified binary32
reference. It covers zero blocks, positive and negative ties-to-even, two group
boundaries, and a real clamp case where a subnormal rounded scale makes
`x / scale` equal to `127.583336f`. Invalid references cover NaN, both
infinities, incomplete groups, and a nonzero group whose scale underflows to
zero. The exporter computes the maximum, scale, quotient, ties-to-even rounding,
and clamp as FP32 operations and writes each block as 64 signed bytes followed
by one little-endian FP32 scale.

## Validation and artifacts

The focused gate was run with:

```sh
python3 -m pytest -q tests/test_export.py
```

Result at export completion: `20 passed in 0.85s`. This includes format layout and corruption checks,
strict config/tensor-schema failures, atomic replacement failure, both synthetic
storage modes, exact FP32 payloads, and all independent quantization cases.

The repository gates available at integration time also passed:

```text
make check       -> 30 passed in 2.69s
make run testcc  -> ALL OK
```

Both production commands above were run twice. Their output hashes remained
identical:

| file | bytes | SHA256 | verified values |
|---|---:|---|---:|
| `models/smollm2-f32.bin` | 538,095,104 | `2d2367f33b1e87706b959fa8ec1de4b3d04572c5d5bad9b0fd6ecbd905fc7849` | 134,515,008 FP32 |
| `models/smollm2-q8.bin` | 143,060,480 | `b9ab3f3690b6648fe3ba2c1b75e1cb9af93b54d6f4ce9ad93867d20c2e61873d` | 35,136 FP32 + 134,479,872 Q8 |

The built-in post-write pass reopens the model, validates all descriptors, and
byte-compares every payload against a fresh source conversion or quantization.
An additional standalone checker parsed the FP32 header/descriptors with
`struct`, read each tensor directly with `safetensors.safe_open`, and compared
the raw uint32 arrays. Result:

```text
independent raw-descriptor FP32 bit comparison: PASS; tensors=272; values=134515008; mismatches=0
```

A separate raw Q8 structural scan checked every block and FP32 norm payload:

```text
independent Q8 structural scan: PASS; blocks=2101248; q8_values=134479872; f32_values=35136; zero_scales=0; bad_zero_blocks=0; minus128=0
```

Small ignored loader fixtures are available in `artifacts/export/`:

| file | bytes | SHA256 |
|---|---:|---|
| `smollm-synthetic-f32.bin` | 1,186,304 | `03869267c518fee4415efd77de88ce96f7937e5a92b0f3f356f3183211a14e53` |
| `smollm-synthetic-q8.bin` | 320,000 | `aa651b0322e3dcd1b9c58bafa5322cb7df12da1832c0d261dde8aed309a109cd` |

The real model binaries and synthetic artifacts are ignored by Git. The two
production manifests are tracked.

## Current boundary

This task establishes the FP32 export gate and exact Q8 encoding. It does not
claim C loader, kernel, logit, or end-to-end Q8 parity; those checks depend on
the shared C runtime and begin only after its FP32 gates pass. The production
CLI intentionally supports the single pinned, unsharded safetensors checkpoint.
