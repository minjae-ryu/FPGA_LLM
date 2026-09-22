# Worker task contract

Use the repository's AGENTS.md and the named local skill. Write only assigned
files. Report proposed interface changes to the orchestrator before editing
shared headers. Do not commit or push; the orchestrator reviews and integrates.

Every assignment and completion record in `docs/tasks/` includes:

1. Objective and prerequisite numerical gate.
2. Exact inputs and fixed revisions, skill, owned files, and shared interfaces.
3. Commands actually executed, their results, and generated artifact paths.
4. Numerical maxima/counts, unresolved failures, limitations, and next dependency.

Reference-generation workers must use HF/PyTorch independently of C arithmetic.
Review workers should check numerical meaning and failure cases, not just style.
