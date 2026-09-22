# Provenance

The initial FPGA_LLM history remains the parent of the replacement commit
(`8051b2ab782a4908dea802dded821c2cc4d765f3`). The previous Verilog sources are
available in that history. No history was rewritten.

Legacy llama2.c files come from https://github.com/karpathy/llama2.c at
`350e04f` (the full commit is recorded in the setup manifest). Their MIT LICENSE
is retained at the repository root. The original README is preserved at
`docs/llama2c-README.md`. `run.c` and `runq.c` remain independent legacy programs.

SmolLM2 weights/tokenizer: `HuggingFaceTB/SmolLM2-135M`, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Downloaded artifacts retain their
upstream license and are excluded from Git.
