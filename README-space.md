---
title: ARIA-LLM
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# ARIA-LLM

Chat with a GPT-style transformer built **from scratch** — RoPE, RMSNorm,
SwiGLU, and Grouped-Query Attention are all hand-written (see `model/`).

Two selectable weight sets run on the same architecture:

- **ARIA + Qwen2.5 weights** — real Qwen2.5-0.5B-Instruct weights loaded into
  this hand-written architecture, proving the forward pass is numerically
  correct.
- **ARIA (fine-tuned)** — the same architecture fine-tuned on the project's own
  conversational corpus.

Weights are hosted in a companion Hugging Face model repo and downloaded at
startup (set the `HF_MODEL_REPO` Space secret). Runs on the free CPU tier.

Built by Aashutosh.
