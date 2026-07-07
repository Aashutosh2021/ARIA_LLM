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
- [x] Qwen2.5-0.5B-Instruct Weight Import Script
- [x] Converted Weights Streaming Chat Interface (`chat_qwen.py`)
- [x] Dialogs Text Dataset Formatter (`scripts/format_dialogs.py`)
- [x] Permanent AI Brain Documentation Registry (`brain/` directory)

---

## Active Status Summary

### 1. Converted Weights Chat (`chat_qwen.py`)
- **Status:** Fully Functional.
- **Model:** Qwen2.5-0.5B-Instruct weights loaded into handwritten RoPE + GQA + SwiGLU + RMSNorm layers.
- **Inference:** Fast local GPU execution. Streams tokens sequentially with repetition penalty.

### 2. Custom Model Training (`train.py`)
- **Status:** Functional.
- **Dataset:** Trains on BPE tokenized sequences from custom text documents.
- **Save Path:** Output checkpoint states stored in `checkpoints/`.

### 3. Local Model Chat (`chat.py`)
- **Status:** Functional.
- **Model:** Loads `checkpoints/best.pt` models trained locally.
- **Dialogue Structure:** Turn-based `USER:\nAIRA:` formatting.
