# Project instructions

All project skills and agent instructions live in this repository. Do not install
or modify global skills, global agent settings, or a project `.codex` directory.

## Current stage (2026-09-23)

The user authorized completing FP32 CUDA forward and a bounded CPU/CUDA speed
comparison after correctness gates. Read `docs/HANDOFF.md`,
`docs/tasks/12-cuda-forward-comparison.md` and `docs/design/cuda-forward.md`.
The initial forward and C CLI dispatch are implemented. Keep FP32 numerical
contracts; do not automatically resume full evaluation or the old quantized
performance matrix. WSL GPU execution requires host device access and
`LD_LIBRARY_PATH=/usr/lib/wsl/lib`; see Task 11 for diagnosis. CUDA sanitizer
validation remains unavailable with the installed tool. Earlier CUDA-exclusion
statements describe the superseded CPU stage.

## Coordination

- Orchestrator: `gpt-6-astra`, reasoning `max`, selected by the session owner.
- Workers and reviewers: `gpt-5.6-sol`, reasoning `xhigh`; specify both when
  spawning. At most one orchestrator and three workers are active.
- Independent work may be delegated. Each task specifies its objective,
  prerequisites, inputs, skill, owned files, acceptance tests, and report format.
- The orchestrator owns public APIs, binary contracts, build files, and Git
  integration. Assign disjoint file ownership; consult before API changes.
- Review diffs and actual test evidence before integrating. Prefer independently
  generated numerical references. Never run performance measurements alongside
  other model executions or heavy tests.
- Task records belong in `docs/tasks/`; role guidance is in `docs/agents/`.

## Implementation and validation

Read `docs/design/contracts.md` before changing runtime behavior. Python prepares
models/data/HF references. Inference, scoring, replay, and comparison run in C.
Keep upstream `run.c`, `runq.c` and their tests intact as legacy paths.

Use `.agents/skills/smollm-port` for model/tokenizer work,
`.agents/skills/kernel-parity` for kernels/cache, and
`.agents/skills/numerical-bench` for evaluation and measurements.

Numerical gates are dependencies, not optional end-of-project checks. Never
relax a tolerance to hide a discrepancy. Report unavailable checks explicitly.
Do not commit model weights, datasets, full traces, environments, or builds.

## Git

`origin` is `minjae-ryu/FPGA_LLM`; `upstream` is `karpathy/llama2.c`.
Preserve history. No force pushes. Work on stage branches, test, integrate, and
push accepted changes. Keep normal work at `/home/rmj/FPGA_LLM/llama2.c`.
