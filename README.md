# SmolLM2 C numerical experiments

Current work: [CUDA forward-only implementation plan](docs/design/cuda-forward.md).
The CUDA backend is not implemented yet. Further benchmark/full evaluation work
has been stopped at the user's request; the existing CPU results remain available.
See [the current handoff](docs/HANDOFF.md) to resume at CUDA C0/C1.

A CPU C engine for SmolLM2-135M Base, comparing FP32, W8A8 GS64, and Q8 KV caches
before future nonlinear-function approximations. The fixed model revision is
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Python prepares weights, data and HF
references; C runs inference, scoring, error comparison and replay.

```sh
make smollm
make check
build/smollm generate --model models/smollm2-f32.bin --prompt "Once upon a time" --steps 32
```

OpenBLAS and PCRE2 are required. Follow [reproduction and validation](docs/validation.md)
to export the pinned model/tokenizer and prepare evaluation data before generation.

The engine includes chunked prefill GEMM, single-token GEMV, GQA, split-half RoPE,
independent Model/Session ownership, swappable CPU KernelOps/MathOps, GS64 integer
kernels, compact KV8 storage, and trace/replay interfaces. C evaluators cover
WikiText-2 raw test, HellaSwag and PIQA validation with FP64 likelihood scoring.

The [first measured comparison](docs/results.md) covers all four configurations:
four isolated performance cells (prompt 128, decode 128, thread 1) and twelve
smoke evaluation cells. The user capped measurement time at 30 minutes; this
run finished in about 18 minutes 33 seconds, including prior measurements counted
conservatively. The full 24-cell performance matrix and full evaluation splits
remain deferred. All 143 tests and the final-build numerical gates passed.

- [Numerical and API contracts](docs/design/contracts.md)
- [Tokenizer format](docs/design/tokenizer.md), [data preparation](docs/design/data.md), [trace/replay](docs/design/trace.md)
- [Engine validation](docs/tasks/07-engine.md), [bounded measurement evidence](docs/tasks/09-final-matrices.md), and [summary data](results/)
- [Repository-local agent rules](AGENTS.md) and [.agents/skills](.agents/skills/)
- [Source attribution](docs/provenance.md) and [legacy documentation](docs/llama2c-README.md)

The legacy `run.c` and `runq.c` remain unchanged and build with `make run`.
`make testcc` and `test_all.py` retain their original regression paths. Model
weights, prepared data, full traces, environments and build outputs are ignored.
