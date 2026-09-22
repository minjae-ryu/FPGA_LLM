# Repository setup

Replaced the tracked FPGA files in a worktree based on original origin/main.
Imported llama2.c 350e04fe35433e6d2941dce5a1f53308f87058eb without altering
legacy sources. Added repository-local agent rules/skills and numerical contracts.

Validation: `make run testcc` passed; `git diff --check` passed. Skill frontmatter
and names validated. Engine, tokenizer, model, and benchmark gates remain pending.
