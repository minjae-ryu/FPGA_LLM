---
name: kernel-parity
description: Change and validate FP32 or W8A8 GS64 kernels, KV8 cache storage, and function replay in the SmolLM2 C engine.
---

Use `docs/design/contracts.md` as the sole numerical contract. Validate exact
integer quantization against independently prepared Python vectors before
testing linear replay or end-to-end quantization. Run `make check` and the
replay/parity commands in `docs/validation.md`. Include zero, ties, clamp,
nonfinite, group boundaries, reset, and same-chunk cache reconstruction cases.

Use identical FP32 inputs for function replay. Distinguish weight-only,
activation-only, and combined linear error from accumulated model error.
Report Q8 accuracy changes without inventing a pass threshold. Hand-written AVX,
fixed-point scales, Taylor replacements, and CUDA are outside this stage.
