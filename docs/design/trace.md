# Trace replay and comparison

Detailed traces are little-endian streaming files. They are intended for small,
fixed probes and store the exact FP32 inputs and outputs needed to reproduce one
runtime operation. Generated traces, indexes, and replay reports belong under
the ignored `artifacts/` tree.

## `SMTRC001` stream

The file begins with the 8-byte magic `SMTRC001`. Records continue to physical
EOF; there is no implicit padding or trailing index. Every record has a
128-byte header followed immediately by `input_count` float32 values and then
`output_count` float32 values:

| Offset | Encoding | Meaning |
|---:|---|---|
| 0 | 64 bytes | NUL-terminated site name; unused bytes zero |
| 64 | i32 | layer, or -1 outside decoder layers |
| 68 | i32 | attention head, or -1 |
| 72 | u64 | absolute token position |
| 80 | u32 | rows |
| 84 | u32 | columns |
| 88 | i32 | causal-mask marker: 1 or -1 for no mask |
| 92 | 4 bytes | reserved zero |
| 96 | u64 | input float count |
| 104 | u64 | output float count |
| 112 | 16 bytes | reserved zero |

`output_count` must equal `rows * columns`. Inputs and outputs must be finite.
The replay reader validates the complete magic, header terminator and padding,
reserved fields, signed metadata, count arithmetic, payload lengths, and clean
EOF while streaming. A record may contain at most 16,777,216 input or output
floats. This limit prevents a malformed count from requesting an unbounded
allocation and is well above the pinned model's largest record of 49,152
floats.

The writer records only layers 0, 14, and 29 plus layer-independent events.
Linear sites use exact Hugging Face tensor names. Each row is a record whose
input count is K and output shape is `[1,N]`. The tied
`model.embed_tokens.weight` site is the output projection; its positions are
the absolute output-token positions.

Function records have these layouts:

| Site | Input | Output |
|---|---|---|
| `rms.rsqrt` | one variance | one reciprocal square root |
| `softmax.exp` | shifted logits | elementwise exponentials |
| `softmax.reciprocal` | one sum | one reciprocal |
| `silu.exp` | negated gate values | elementwise exponentials |
| `silu.reciprocal` | `1 + exp(-x)` values | elementwise reciprocals |
| `rope.sincos` | angle vector | all sine values, then all cosine values |

## Replay modes

All modes write one JSON object per aggregate to stdout. Exit status 0 means the
input was valid and every applicable gate passed, 1 means malformed input or an
I/O/model/configuration error, and 2 means an enforced numerical gate failed.

### Hugging Face intermediates

```sh
build/smollm replay \
  --trace artifacts/replay/probe-f32.smtrace \
  --mode compare \
  --reference-prefix artifacts/reference/probe
```

`compare` consumes the seven sites `attn_norm`, `q_rope`, `k_rope`,
`attention`, `attn_residual`, `ffn_norm`, and `ffn_residual` at layers 0, 14,
and 29. It requires all 21 streams in absolute position order. Reference paths
are `<prefix>.layer<L>.<site>.f32`, with no header. Reference files must end
exactly after the last compared row. Reports include maximum absolute error,
RMSE, top-1 changes, and failures of
`abs(error) <= 1e-3 + 1e-4 * abs(reference)`. Any tolerance failure returns 2.

### Math functions

```sh
build/smollm replay --trace artifacts/replay/probe-f32.smtrace --mode math
```

`math` applies `sm_default_math()` to captured identical inputs. The replay
implementation takes an `SmMathOps` argument internally, which is the extension
point for a future configurable approximation. The current exact defaults must
reproduce every output bit. Reports include bit changes and numerical errors;
any bit change returns 2.

### Linear quantization attribution

```sh
build/smollm replay \
  --trace artifacts/replay/probe-f32.smtrace \
  --mode linear \
  --model models/smollm2-f32.bin \
  --q8-model models/smollm2-q8.bin
```

The model configurations, revisions, tensor names, shapes, and storage dtypes
must agree. Replay uses the captured FP32 activation rows for four variants:

- `fp32_capture`: original FP32 weights and inputs through the default BLAS
  kernel, compared with the captured output using the FP32 tolerance gate.
- `weight_only`: Q8 weights dequantized to FP32, with the original FP32 input,
  compared with the FP32 replay result.
- `activation_only`: the input quantized and dequantized by the runtime GS64
  functions, with original FP32 weights, compared with FP32 replay.
- `w8a8`: original Q8 blocks and runtime Q8 activation/kernel path, compared
  with FP32 replay.

Q8 variants report maximum absolute error, RMSE, and top-1 changes. They have no
arbitrary pass threshold. Only failure of the `fp32_capture` check returns 2.
Consecutive rows for one tensor are grouped in batches of at most 64. Buffers
and a dequantized weight matrix are reused, while every captured row is replayed
through the default BLAS GEMV and Q8 GEMV paths.

### Trace-to-trace comparison

```sh
build/smollm replay \
  --trace artifacts/replay/probe-f32.smtrace \
  --mode traces \
  --other artifacts/replay/probe-q8.smtrace
```

Both streams are parsed independently. Record headers and order must match
exactly. Outputs are aggregated by site and layer with maximum absolute error,
RMSE, and top-1 changes. This descriptive comparison does not assign a Q8
threshold or return 2 for numerical differences.
