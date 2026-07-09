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

Chat with **ARIA**, a GPT-style transformer built entirely from scratch — RoPE,
RMSNorm, SwiGLU, and Grouped-Query Attention are all hand-written (see
`model/`). It's trained from random initialization with its own byte-level BPE
tokenizer. No third-party model runs at inference.

ARIA was trained via **knowledge distillation**: a larger model generated a
conversational dataset, and ARIA learned from that data from scratch — so the
weights are entirely its own.

The weights (`aria.pt`) and tokenizer (`aria_vocab.json`) are hosted in a
companion Hugging Face model repo and downloaded at startup (set the
`HF_MODEL_REPO` Space secret). Runs on the free CPU tier.

Built by Aashutosh.
