---
name: smollm-port
description: Implement or verify this repository's SmolLM2 checkpoint conversion, tokenizer, FP32 engine, and chunked prefill against pinned Hugging Face references.
---

Read `docs/design/contracts.md` from the repository root before modifying a
format or computation. Keep HF Q/K split-half layout. Preserve BF16 values
exactly when exporting FP32. Do not reuse legacy model/tokenizer assumptions.

Run `make smollm` and `make check` for local contracts. For real-model parity,
use the reproducible commands in `docs/validation.md`; compare tokens exactly,
then intermediate tensors and logits, then chunk equivalence. A failed earlier
gate blocks later numerical claims. Record observed errors and artifact hashes
in the task record. Do not silently replace a revision or relax tolerances.
