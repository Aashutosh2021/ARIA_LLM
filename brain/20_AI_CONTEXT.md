# AI Context

A compressed memory context cheat-sheet designed for future incoming AI agents.

## Core Context Essentials

- **What is this project?** A from-scratch PyTorch transformer implementation with custom classes (`GPT`, `TransformerBlock`, `MultiHeadAttention`, `FeedForward`, `RMSNorm`, `BPETokenizer`, `Trainer`, etc.) designed to run SFT and pretraining loops completely from scratch.
- **Where are the core weights?** 
  - Standard training saves checkpoints under `checkpoints/best.pt` and `checkpoints/last.pt`.
  - **How is data structured?** Conversational text files (`data/conversations.txt` and `data/dialogs.txt`) are formatted using turn prefixes:
  ```text
  USER: <query>
  AIRA: <response>
  <|endoftext|>
  ```
- **What are the critical configuration parameters?** Hand-written GPT parameters are controlled by `configs/model.yaml` (embedding dimension, layer count, attention heads, ffn multiplier).

## Common Gotchas & Lessons
- **Newline bug:** Do not use `\s+` replacement during dataset cleaning. It converts newlines to spaces, flattening dialogue structures. Use `[ \t]+` to preserve `\n` boundaries.
- **BPE vs Char tokenizer:** Classical character tokenizers are simple but result in large token lists. Use BPE tokenizers with defined target vocabulary sizes for real pretraining or SFT runs.
