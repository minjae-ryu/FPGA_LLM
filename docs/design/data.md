# Reference and evaluation data

All generated binaries are little-endian. They live under ignored `data/` and
`artifacts/reference/` directories; their small JSON manifests are tracked.
Generation reads pinned local Hugging Face snapshots and never changes a global
cache or configuration.

## Hugging Face references

`scripts/prepare_reference.py` loads
`HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
on CPU as FP32 and forces eager attention and one native thread. The checkpoint
contains only BF16 tensors. Casting them to FP32 is exact because every BF16
value is representable in binary32.

The fixed natural-language probe has 57 tokens. Its `.tokens` file is the
`SMTOK001` header, a `u32` count, and that many `u32` IDs. Its logits are raw
FP32 `[57,49152]`. Selected rows at layers 0, 14, and 29 are also raw FP32:

| Site | Shape | Meaning |
|---|---:|---|
| `attn_norm` | `[token,576]` | input RMSNorm output |
| `q_rope` | `[token,9,64]` | query after split-half RoPE |
| `k_rope` | `[token,3,64]` | key after split-half RoPE |
| `attention` | `[token,576]` | concatenated heads before `o_proj` |
| `attn_residual` | `[token,576]` | state after attention residual add |
| `ffn_norm` | `[token,576]` | post-attention RMSNorm output |
| `ffn_residual` | `[token,576]` | layer output after MLP residual add |

The long natural-language probe has 315 tokens. Its full-sequence logits are
the common reference for C prefill chunk sizes 1, 127, 128, and 129. No separate
HF chunk execution is used because that would introduce a second cache path
into the independent reference.

Mean next-token NLL scores the real targets `tokens[1:]` from logit rows
`0..count-2`. Logits are first widened to FP64; max, exp, sum, log, target
subtraction, and the final mean are all FP64 operations.

Run:

```sh
python3 scripts/prepare_reference.py --which all
```

## Evaluation records

An `SMEVAL01` file begins with the 8-byte magic, a `u32` kind (`1` LM, `2`
multiple choice), and a `u32` record count. Each record contains eight `u32`
fields followed by `token_count` `u32` IDs:

```text
group label choice nchoices token_count score_start score_end norm_chars
```

`[score_start,score_end)` names target token indices; output row `j-1` predicts
target `j`. The preparer rejects zero-context targets, out-of-range bounds or
tokens, inconsistent choice groups, incomplete groups, and trailing bytes.

WikiText-2 uses the raw test strings joined by two newlines and encodes without
BOS or EOS. Windows have context 2048 and stride 512. The first window scores
local targets 1 through its end; each later window scores only global targets
not covered earlier. This covers global target indices `1..N-1` once each.
The smoke input truncates the joined token stream to its first 8192 IDs.

Multiple-choice prompts reproduce the zero-shot task definitions in
EleutherAI/lm-evaluation-harness commit
[`d6de81643928d653435c431bae19945d41d32520`](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520).
HellaSwag uses the pinned
[`process_docs`](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/hellaswag/utils.py)
transformation. PIQA uses `Question: {goal}\nAnswer:` and the two solution
strings from the pinned
[`piqa.yaml`](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/piqa/piqa.yaml).
Both insert the default single-space target delimiter and reproduce
[`TemplateLM._encode_pair`](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/api/model.py):
move trailing context spaces to the continuation, tokenize the whole string and
the context, then split at the context-token count. Matching harness
`process_results`, `norm_chars` is Python `len(choice)` on the processed choice
text. It counts Unicode code points and excludes the inserted target delimiter
and any whitespace moved from the context. The first 32 original validation
examples form each smoke set.

Run:

```sh
python3 scripts/prepare_data.py --task all --variant all
pytest -q tests/test_data.py
```

The PIQA dataset revision is pinned to
`ybisk/piqa@2e8ac2dffd59bac8c3c6714948f4c551a0848bb0`. That repository stores a
dataset loader rather than Parquet, so preparation reads the loader's cached
canonical `dev.jsonl` and `dev-labels.lst`; their content hashes are in every
PIQA manifest. The pinned harness commit now names `baber/piqa` in its YAML.
Only its prompt and scoring transformations are adopted, while the
dataset source remains pinned separately.
