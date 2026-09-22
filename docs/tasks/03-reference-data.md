# Reference and evaluation data

Implemented independent Hugging Face CPU FP32/eager references and pinned
evaluation inputs for SmolLM2-135M. Generated binaries remain ignored; tracked
manifests contain their hashes, shapes, source hashes, revisions, library and
machine details, thread settings, and exact preprocessing rules.

## Reference evidence

Command:

```sh
python3 scripts/prepare_reference.py --which all
```

The cached checkpoint contains 272 BF16 tensors and no other tensor dtype. The
script loaded them as FP32, selected eager attention, and reported one intra-op
and one inter-op thread. Observed results:

| Reference | Tokens | Targets | FP64 mean NLL | Token SHA256 | Logit SHA256 |
|---|---:|---:|---:|---|---|
| probe | 57 | 56 | 4.090981860112246 | `88d223f9f600a238e541743196efb0eadd156c4ab5c812df958a823aadf39216` | `6ba6cb197a8be66f49526e2f37e0aec63e96de2f09d5d93bec37feaec045d2da` |
| long | 315 | 314 | 1.0517954907452423 | `9da1f7ccc3bf03b9b09784abb26fccf5c35e46ae57149908becbf0c6091cca93` | `15bee88c6399fa88ae9b98c052832721ae69ece2650b6ae4c49c0f6b030e6472` |

The probe also produced 21 intermediate tensors: seven sites at each of layers
0, 14, and 29. Every generated tensor was checked for the manifest shape and
finite FP32 values. The long logits are marked for chunk-size comparisons at
1, 127, 128, and 129.

## Data evidence

Command:

```sh
python3 scripts/prepare_data.py --task all --variant all
```

Observed results:

| Input | Records | Scored targets | SHA256 |
|---|---:|---:|---|
| WikiText-2 smoke | 13 | 8,191 | `f3c5643db867be07cfda9053617abbfb2f7a4c3938cfef61ca6ea3945a24cff2` |
| WikiText-2 full | 593 | 304,985 | `4db5c38759a2a0f75701d20948879ca5254712b2505f1814e802180e07fee1f5` |
| HellaSwag smoke | 128 | 1,623 | `d4868c537197fe947b9fa1cfff5aec19e75deb665786c08e91f80a0caef843a0` |
| HellaSwag full | 40,168 | 1,189,452 | `a586a2f4395d819fae5ceaca99484f81a66a7fbefbe0710b5575f44767b1b7ca` |
| PIQA smoke | 64 | 1,416 | `2a854a3b5561f8f895ee317664cc1b3beca67dff1fbd6df1dd1c8fce17589cd2` |
| PIQA full | 3,676 | 84,004 | `ce62ee6175460293a62a965c8903a93719986d29c5382fffd3e082ca5cbb2351` |

WikiText has 304,986 input tokens and therefore exactly 304,985 real
next-token targets. HellaSwag contains 10,042 original examples with four
records each. PIQA contains 1,838 original examples with two records each.

Validation command and result:

```text
python3 -m pytest -q tests/test_data.py
11 passed in 2.52s
```

The tests cover strict headers and EOF, malformed bounds and token IDs,
contiguous choice groups, Unicode character counts, harness boundary behavior,
the HellaSwag transformation, exact WikiText target coverage without duplicates,
the 8192-token smoke geometry, and generated artifact hashes/counts. Character
normalization matches harness `process_results`: it uses the Unicode length of
the processed choice and excludes the target delimiter and moved whitespace.

## Provenance and limits

Dataset revisions are WikiText-2
`b08601e04326c79dfdd32d625aee71d232d685c3`, HellaSwag
`218ec52e09a7e7462a5400043bb9a69a41d06b76`, and PIQA
`2e8ac2dffd59bac8c3c6714948f4c551a0848bb0`. Multiple-choice prompt,
preprocessing, target delimiter, and boundary handling are pinned to
EleutherAI/lm-evaluation-harness
`d6de81643928d653435c431bae19945d41d32520`. Source-file content hashes are in
the data manifests.

The PIQA repository snapshot contains a loader rather than data files. The
script reads the canonical cached `dev.jsonl` and `dev-labels.lst` downloaded by
that loader and records both hashes. The pinned harness revision now names
`baber/piqa`; its prompt/scoring transformation is used with the separately
pinned requested `ybisk/piqa` content.

This task did not run C parity, chunked-runtime comparison, perplexity,
multiple-choice accuracy, quantized comparisons, or performance benchmarks;
their consumers were not yet integrated. No full-model execution overlapped the
serialized HF reference run, and no throughput claim is made here.
