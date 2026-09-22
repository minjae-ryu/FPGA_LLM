# SmolLM2 C numerical experiments

CPU C baseline for SmolLM2-135M Base, comparing FP32, W8A8 GS64, and Q8 KV
caches before future nonlinear-function approximations. OpenBLAS and PCRE2 are
required. Python prepares models, data, and references; C executes and scores.

- Numerical/API contracts: [docs/design/contracts.md](docs/design/contracts.md)
- Reproduction and gates: [docs/validation.md](docs/validation.md)
- Repository-local agent rules: [AGENTS.md](AGENTS.md)
- Upstream attribution: [docs/provenance.md](docs/provenance.md)
- Legacy documentation: [docs/llama2c-README.md](docs/llama2c-README.md)

Implementation and measurements are recorded in `docs/tasks/` and `results/`.
Legacy executables remain available through `make run` and `make testcc`.
