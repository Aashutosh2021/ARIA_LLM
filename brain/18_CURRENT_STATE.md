# Current State

Current implementation progress of ARIA-LLM.

## Implementation Checklist

- [x] Custom PyTorch Transformer Block Architecture
- [x] Rotary Position Embeddings (RoPE)
- [x] RMSNorm & LayerNorm Support
- [x] SwiGLU Gated Linear FeedForward Unit
- [x] Custom BPE Tokenizer Trainer & Encoder
- [x] Autoregressive Logits Sampler (Temperature, Top-k, Top-p)
- [x] Checkpoint Saving and Configurations Serialization
- [x] Conversational Data Preprocess Cleaner (Newlines preserving fix)
- [x] Dialogs Text Dataset Formatter (`scripts/format_dialogs.py`)
- [x] Permanent AI Brain Documentation Registry (`brain/` directory)

---

## Active Status Summary

### 1. Custom Model Training (`train.py`)
- **Status:** Functional.
- **Dataset:** Trains on BPE tokenized sequences from custom text documents.
- **Save Path:** Output checkpoint states stored in `checkpoints/`.

### 2. Local Model Chat (`chat.py`)
- **Status:** Functional.
- **Model:** Loads `checkpoints/best.pt` models trained locally.
- **Dialogue Structure:** Turn-based `USER:\nAIRA:` formatting.
