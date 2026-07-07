# File Index

An annotated reference of key files in the ARIA-LLM repository.

## Root Directory

- [chat.py](file:///d:/Personal%20Project/ARIA-LLM/chat.py) — Interactive console REPL for local ARIA model.
- [chat_ai.py](file:///d:/Personal%20Project/ARIA-LLM/chat_ai.py) — Backup/alternative CLI chat interface.
- [chat_qwen.py](file:///d:/Personal%20Project/ARIA-LLM/chat_qwen.py) — Primary interactive demo executing real Qwen2.5 weights inside ARIA architecture.
- [train.py](file:///d:/Personal%20Project/ARIA-LLM/train.py) — Main training pipeline launcher.
- [train_chat.py](file:///d:/Personal%20Project/ARIA-LLM/train_chat.py) — Script to run conversational training.
- [conftest.py](file:///d:/Personal%20Project/ARIA-LLM/conftest.py) — Pytest configuration.
- [requirements.txt](file:///d:/Personal%20Project/ARIA-LLM/requirements.txt) — Python package dependencies.

---

## model/

- [model/gpt.py](file:///d:/Personal%20Project/ARIA-LLM/model/gpt.py) — Top-level model container, weights loading classmethod, forward pass.
- [model/transformer_block.py](file:///d:/Personal%20Project/ARIA-LLM/model/transformer_block.py) — Individual transformer layer binding self-attention and MLP/SwiGLU.
- [model/multi_head_attention.py](file:///d:/Personal%20Project/ARIA-LLM/model/multi_head_attention.py) — MHA/GQA projection and scale calculations.
- [model/feed_forward.py](file:///d:/Personal%20Project/ARIA-LLM/model/feed_forward.py) — Standard MLP and SwiGLU implementations.
- [model/rope.py](file:///d:/Personal%20Project/ARIA-LLM/model/rope.py) — Rotary position embedding math (cis vectors).
- [model/rmsnorm.py](file:///d:/Personal%20Project/ARIA-LLM/model/rmsnorm.py) — Root Mean Square Normalization.
- [model/layer_norm.py](file:///d:/Personal%20Project/ARIA-LLM/model/layer_norm.py) — Layer normalization module.
- [model/embedding.py](file:///d:/Personal%20Project/ARIA-LLM/model/embedding.py) — Token embedding layer.
- [model/position.py](file:///d:/Personal%20Project/ARIA-LLM/model/position.py) — Learned absolute positional encoding.
- [model/attention.py](file:///d:/Personal%20Project/ARIA-LLM/model/attention.py) — Dot product attention execution.
- [model/mask.py](file:///d:/Personal%20Project/ARIA-LLM/model/mask.py) — Casual mask utility.
- [model/init.py](file:///d:/Personal%20Project/ARIA-LLM/model/init.py) — GPT weights initialization functions.

---

## dataset/

- [dataset/cleaner.py](file:///d:/Personal%20Project/ARIA-LLM/dataset/cleaner.py) — Text cleaner (email/URL/html removal, spaces cleaning).
- [dataset/loader.py](file:///d:/Personal%20Project/ARIA-LLM/dataset/loader.py) — Dataset parser (txt, csv, json, jsonl loader).
- [dataset/preprocess.py](file:///d:/Personal%20Project/ARIA-LLM/dataset/preprocess.py) — Binds tokenizer and cleaner; constructs train/val sequences.
- [dataset/dataset.py](file:///d:/Personal%20Project/ARIA-LLM/dataset/dataset.py) — Custom PyTorch dataset `LLMDataset`.

---

## tokenizer/

- [tokenizer/bpe_tokenizer.py](file:///d:/Personal%20Project/ARIA-LLM/tokenizer/bpe_tokenizer.py) — Custom Byte Pair Encoding (BPE) implementation.
- [tokenizer/word_tokenizer.py](file:///d:/Personal%20Project/ARIA-LLM/tokenizer/word_tokenizer.py) — Basic word/whitespace tokenizer.
- [tokenizer/char_tokenizer.py](file:///d:/Personal%20Project/ARIA-LLM/tokenizer/char_tokenizer.py) — Character-level tokenizer.
- [tokenizer/vocab.py](file:///d:/Personal%20Project/ARIA-LLM/tokenizer/vocab.py) — Vocabulary manager.
- [tokenizer/special_tokens.py](file:///d:/Personal%20Project/ARIA-LLM/tokenizer/special_tokens.py) — Definition of special tokens (PAD, UNK, BOS, EOS, MASK).

---

## inference/

- [inference/generator.py](file:///d:/Personal%20Project/ARIA-LLM/inference/generator.py) — Autoregressive generation manager.
- [inference/sampler.py](file:///d:/Personal%20Project/ARIA-LLM/inference/sampler.py) — Sampler wrapper dispatching temperatures/top-k/top-p.
- [inference/beam_search.py](file:///d:/Personal%20Project/ARIA-LLM/inference/beam_search.py) — Beam search algorithm.
- [inference/topp.py](file:///d:/Personal%20Project/ARIA-LLM/inference/topp.py) — Nucleus (top-p) sampling filters.
- [inference/topk.py](file:///d:/Personal%20Project/ARIA-LLM/inference/topk.py) — Top-k sampling filters.
- [inference/temperature.py](file:///d:/Personal%20Project/ARIA-LLM/inference/temperature.py) — Logits scaling factor implementation.

---

## training/

- [training/trainer.py](file:///d:/Personal%20Project/ARIA-LLM/training/trainer.py) — Main training epoch executor.
- [training/checkpoint.py](file:///d:/Personal%20Project/ARIA-LLM/training/checkpoint.py) — Checkpoint saver/loader.
- [training/loss.py](file:///d:/Personal%20Project/ARIA-LLM/training/loss.py) — Cross entropy language modeling loss.
- [training/optimizer.py](file:///d:/Personal%20Project/ARIA-LLM/training/optimizer.py) — AdamW builder.
- [training/scheduler.py](file:///d:/Personal%20Project/ARIA-LLM/training/scheduler.py) — Warmup cosine learning rate scheduler.
- [training/logger.py](file:///d:/Personal%20Project/ARIA-LLM/training/logger.py) — Run progress logger.

---

## scripts/

- [scripts/prepare_conversations.py](file:///d:/Personal%20Project/ARIA-LLM/scripts/prepare_conversations.py) — Synthetic multi-template conversational corpus generator.
- [scripts/import_qwen.py](file:///d:/Personal%20Project/ARIA-LLM/scripts/import_qwen.py) — State dict converter for Qwen2.5 weights.
- [scripts/format_dialogs.py](file:///d:/Personal%20Project/ARIA-LLM/scripts/format_dialogs.py) — Tab-separated list formatter.
