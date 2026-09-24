# Task 13: restore the pre-CUDA CPU baseline

User scope (2026-09-24): set CUDA aside, return to the pre-CUDA Git state and
write a handoff. This cancels the prior CUDA implementation/comparison task.

- Preserved uncommitted CUDA forward and incomplete comparison work in
  `archive/cuda-paused-2026-09-24`, commit `2d5ab02`.
- The comparison contains five of six planned cells. Its archive manifest now
  says interrupted, not complete. No CUDA comparison process remained when
  checked outside the sandbox (`ps -C smollm-cuda` and exact runner lookup).
- Started `stage/restore-cpu-2026-09-24` from main and applied
  `git revert --no-commit 2201faf 2655aeb`. Before new handoff edits, the staged
  tree matched `26ff391` exactly. No reset/force push or history deletion.
- Restored all runtime sources, public headers, Makefile, scripts and tests to
  `26ff391`. CUDA-only tracked files are absent from the current CPU tree.
- Moved ignored `build/`, `.cache/cuda`, and CUDA probe trace/index to
  `artifacts/cuda-paused-2026-09-24/`. Model/data/HF/CPU evidence stays in place.
  CUDA generated files are local-only; the archive branch contains code and
  summary evidence, not large builds/traces/cache.
- Rebuilt the CPU targets in a fresh `build/` and ran `make smollm check`:
  **143 passed in 8.46s**. All 20 source/build hashes in `engine-build.json`
  match. CPU executable SHA256 equals the pre-CUDA executable:
  `32c208d95e7d399d60c7f87981833331dc0514eef05451e33a68609028d1f85e`.
- No new model benchmark, full evaluation or GPU execution during rollback.
  Historical CPU numerical/sanitizer/performance evidence is preserved and
  not represented as a fresh run.

The current instructions, README and handoff state that CUDA is inactive and
neither the archived CUDA comparison nor the old CPU matrices should resume
automatically. The old CPU handoff is preserved verbatim as
`docs/CPU_HANDOFF_2026-09-22.md`; the new entry point is `docs/HANDOFF.md`.

CPU baseline commit: `26ff391`; engine source baseline: `134321d`.
Git HEAD is a new restoration/documentation commit retaining all prior history.
See `manifests/cpu-restore.json` for machine-readable validation and archive refs.
