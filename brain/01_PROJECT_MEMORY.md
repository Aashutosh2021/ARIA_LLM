# Project Memory

## Project Identity
- **Name:** ARIA-LLM (Custom Architecture Large Language Model)
- **Concept:** A PyTorch-based, completely "from-scratch" implementation of a GPT transformer model. It does not rely on third-party libraries (like Hugging Face `transformers`) for the model architecture.

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

### 3. Converted Qwen Weights Integration (Removed)
- An experimental script was originally created to convert Hugging Face Qwen2.5-0.5B-Instruct weights into our custom format. This was later removed to keep the repository strictly focused on 100% locally-trained, from-scratch models.

### 4. Dialogs Dataset Formatting
- The user asked to format `data/dialogs.txt` (a tab-separated conversational Q&A dataset of 3,725 rows) into the formatted structure of `data/conversations.txt` (using `USER:`, `AIRA:`, and `<|endoftext|>`).
- Created `scripts/format_dialogs.py` which split by tab and formatted the file directly.
