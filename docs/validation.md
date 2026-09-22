# Validation gates

The implementation task records will add exact commands as executables become
available. No gate is considered passed until its command and observed result
are recorded. Run accuracy checks with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.

1. `make run testcc`: unchanged legacy C build/tests.
2. `make smollm check`: format, quantization, tokenizer and runtime unit gates.
3. Export the pinned BF16 model to FP32 and verify every value exactly.
4. Generate independent HF CPU FP32/eager probes; compare C tokenizer IDs,
   intermediate tensors, full logits and mean NLL.
5. Check chunk sizes 1/127/128/129, mixed splits, cache reset and context boundary.
6. Check quantization integers, linear replay, same-chunk KV reconstruction.
7. Validate prepared target masks/counts and continuation boundaries, run smoke
   and full four-configuration evaluations, then serialized performance runs.

Keep reproducibility manifests and summary results in Git; generated binaries,
weights, full traces, data, environments and downloads belong in ignored folders.
