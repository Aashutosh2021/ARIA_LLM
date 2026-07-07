# AI Context

A compressed memory context cheat-sheet designed for future incoming AI agents.

## Core Context Essentials

- **What is this project?** A from-scratch PyTorch transformer implementation with custom classes (`GPT`, `TransformerBlock`, `MultiHeadAttention`, `FeedForward`, `RMSNorm`, `BPETokenizer`, `Trainer`, etc.) that also has import mechanisms to load real pre-trained weights from `Qwen2.5-0.5B-Instruct` into our handwritten classes.
- **Where are the core weights?** 
  - Standard training saves checkpoints under `checkpoints/best.pt` and `checkpoints/last.pt`.
  - Qwen converted weights are saved under `checkpoints/qwen2.5-0.5b-instruct.pt`.
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
- **Qwen loading keys:** Qwen has specific state dict keys and expects Rotary embeddings (RoPE), RMSNorm, SwiGLU activation, and Grouped-Query Attention (GQA).
