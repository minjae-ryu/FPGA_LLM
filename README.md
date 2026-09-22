# SmolLM2 C numerical experiments

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

- [Numerical and API contracts](docs/design/contracts.md)
- [Tokenizer format](docs/design/tokenizer.md), [data preparation](docs/design/data.md), [trace/replay](docs/design/trace.md)
- [Validation evidence](docs/tasks/07-engine.md) and [summary results](results/)
- [Repository-local agent rules](AGENTS.md) and [.agents/skills](.agents/skills/)
- [Source attribution](docs/provenance.md) and [legacy documentation](docs/llama2c-README.md)

The legacy `run.c` and `runq.c` remain unchanged and build with `make run`.
`make testcc` and `test_all.py` retain their original regression paths. Model
weights, prepared data, full traces, environments and build outputs are ignored.
