# Project Memory

## Project Identity
- **Name:** ARIA-LLM (Alibaba-inspired Custom Architecture Large Language Model)
- **Concept:** A PyTorch-based, completely "from-scratch" implementation of a GPT transformer model. It does not rely on third-party libraries (like Hugging Face `transformers`) for the model architecture, but contains a script to convert and load real pretrained weights from Qwen2.5-0.5B-Instruct into its own hand-written classes.

## Development History

### 1. From-Scratch GPT Architecture
- Standard GPT transformer built with parameters loaded from local YAML configs.
- Custom embedding layer, positional encoding, and multi-head attention.
- Designed to run training locally on raw text corpora.

### 2. Conversational Finetuning Attempt
- The user wanted ARIA to act as a friendly local assistant without Qwen weights initially.
- We generated synthetic training pairs (greetings, math, etc.) to convert standard causal training into conversational formatting (`USER: ...\nAIRA: ...\n<|endoftext|>`).
- **Critical Bug Discovered:** The dataset preprocess cleaner (`dataset/cleaner.py`) was stripping all newlines (`\n`) using `\s+` replacement. This caused dialogue turns to squash together (e.g. `USER: hi AIRA: hello`), rendering the model unable to learn conversational boundaries.
- **Fix:** Corrected cleaner regex to `[ \t]+`, preserving newlines. Retrained for 10/20 epochs, which successfully allowed `chat.py` to prompt the model with newlines and clean up output.

### 3. Converted Qwen Weights Integration
- The user requested the ability to load a real instruct model while keeping the hand-written transformer.
- Wrote [import_qwen.py](file:///d:/Personal%20Project/ARIA-LLM/scripts/import_qwen.py) to convert Hugging Face `Qwen/Qwen2.5-0.5B-Instruct` state dict variables (mapping Qwen keys to ARIA-LLM keys) and save them to `checkpoints/qwen2.5-0.5b-instruct.pt`.
- Modified [chat_qwen.py](file:///d:/Personal%20Project/ARIA-LLM/chat_qwen.py) to stream responses from ARIA's handwritten architecture running the loaded Qwen weights.
- Implemented Rotary Position Embedding (RoPE), RMSNorm, SwiGLU FFN multiplier, and Grouped-Query Attention (GQA) in [gpt.py](file:///d:/Personal%20Project/ARIA-LLM/model/gpt.py) to fully support Qwen2.5's modern parameters.

### 4. Dialogs Dataset Formatting
- The user asked to format `data/dialogs.txt` (a tab-separated conversational Q&A dataset of 3,725 rows) into the formatted structure of `data/conversations.txt` (using `USER:`, `AIRA:`, and `<|endoftext|>`).
- Created `scripts/format_dialogs.py` which split by tab and formatted the file directly.
