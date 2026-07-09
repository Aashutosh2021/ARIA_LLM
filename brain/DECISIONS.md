# Architectural Decisions

This document records the major design choices and trade-offs made during the development of ARIA-LLM.

## 1. Custom Handwriting Model Architecture (No HF Models)
- **Decision:** Build the entire model block logic (`model/`) using raw PyTorch classes.
- **Trade-off:** Requires writing all normalization, attention, and block math manually, which is more error-prone than using HF `transformers`. However, it fulfills the core project objective: running a custom, fully local model that is simple to study and modify.

## 2. Converting Pretrained Qwen Weights (Removed)
- **Decision:** Originally, Qwen weight conversion was supported to bootstrap testing of RoPE, RMSNorm, SwiGLU, and GQA blocks.
- **Trade-off:** This has been removed to keep the repository clean, 100% locally-trained, and dedicated exclusively to custom, from-scratch models.

## 3. Preserving Newlines in Dataset Cleaner
- **Decision:** Update the cleaner script (`dataset/cleaner.py`) regex to `[ \t]+` instead of `\s+` to prevent stripping newlines.
- **Trade-off:** The corpus files are slightly larger because of newlines, but it ensures that the BPE tokenizer learns `USER:` and `AIRA:` boundaries, preventing the model from generating repeating loops.
