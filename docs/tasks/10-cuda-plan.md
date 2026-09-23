# Task 10: CUDA planning and session handoff

Date: 2026-09-23. This task produces a detailed plan; it does not implement CUDA.

The user first requested continuation, then stopped further measurements and
redirected the work to CUDA forward-only integration. The final instruction was
to write a very detailed plan and move to a new session. No new performance or
model execution had started before that redirection.

Read inputs: `AGENTS.md`, CPU numerical contracts and validation documentation,
repository-local `smollm-port` / `kernel-parity` instructions, public and internal
headers, the CPU session, CLI ownership and build rules. The user's explicit
CUDA instruction supersedes the old CPU-stage CUDA exclusion. No skill or agent
configuration outside this repository was changed; no subagents were created.

Read-only environment checks found RTX 4070 Ti (SM 8.9), driver 591.86, total
12,282 MiB VRAM, nvcc 11.5.119, and CUDA/cuBLAS 11 libraries in the loader list.
The compiler does not list sm_89. CUDA execution, library resolution and numerical
parity remain unverified. The plan starts with an sm_86 cubin + compute_86 PTX
compatibility check rather than assuming a toolkit upgrade is necessary.

Deliverables:

- `docs/design/cuda-forward.md`: detailed C ABI, ownership, resident memory,
  transfers, FP32 operations, errors, callbacks, trace, build and C0–C6 gates.
- `docs/HANDOFF.md`: next-session entry point with the latest user scope.
- `docs/CPU_HANDOFF_2026-09-22.md`: exact preservation of the previous handoff.
- `manifests/cuda-environment.json`: observed environment, commands, untested items.
- `manifests/handoff.json`: current task and next actions; CPU evidence retained.
- Stage notes in AGENTS, README, contracts and validation to prevent accidental
  resumption of superseded benchmark work.

No runtime/build source or weights were changed, no packages installed, and no
CUDA build, kernel, correctness suite or model measurement was executed here.
The previous 143-test CPU result remains historical evidence, not a new run.
Next session starts C0/C1. Future execution evidence belongs in task 11 onward.

Documentation checks passed: local Markdown links and code fences, JSON parsing,
byte-for-byte preservation of the old CPU handoff, and `git diff --check`.
The CPU executable and every source/build file listed in engine-build.json
still match their recorded SHA256 values. These are document/hash checks, not
a new execution of the CPU or CUDA correctness suites.
