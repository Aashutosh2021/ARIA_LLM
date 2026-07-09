# External Services

ARIA-LLM runs locally with no external API at train or inference time. Two
offline, one-off exceptions pull assets over HTTPS:

## External Dependencies

### 1. KaggleHub (via `kagglehub` package)
- **Purpose:** Used to pull external training assets or dataset corpora.
- **Access Protocol:** Standard HTTPS API connections.

### 2. Hugging Face Hub (via `transformers`) — distillation only
- **Purpose:** `scripts/distill_generate.py` downloads Qwen2.5-0.5B once to
  **generate training text**. Qwen acts as a data teacher; its weights are
  never loaded into ARIA. Not used during ARIA training or chat.
