# Global Variables

ARIA-LLM contains several global constants, defaults, and configuration options:

## 1. CLI Execution Constants

- **`DEFAULT_SYSTEM`** (`chat_qwen.py`):
  Initial Qwen prompt instructing the assistant on its persona:
  `"You are ARIA, a helpful and friendly AI assistant running entirely on a transformer architecture built from scratch. Answer clearly and concisely."`
- **`QWEN_ID`** (`chat_qwen.py`):
  The Hugging Face repo ID: `"Qwen/Qwen2.5-0.5B-Instruct"`.
- **`IM_END`** (`chat_qwen.py`):
  The Qwen-specific termination token ID: `151645`.
- **`BANNER`** (`chat.py` / `chat_qwen.py`):
  Console ASCII art banner displayed at REPL boot.

---

## 2. Tokenizer Constants (`tokenizer/special_tokens.py`)

Special identifiers used across preprocessors and custom tokenizers:
- **`PAD_TOKEN`**: `"<PAD>"` (Token ID: `0`)
- **`UNK_TOKEN`**: `"<UNK>"` (Token ID: `1`)
- **`BOS_TOKEN`**: `"<BOS>"` (Token ID: `2`)
- **`EOS_TOKEN`**: `"<EOS>"` (Token ID: `3`)
- **`MASK_TOKEN`**: `"<MASK>"` (Token ID: `4`)

---

## 3. Training & Seeding State (`utils/seed.py`)

- **`set_seed(seed: int)`**:
  Globally synchronizes randomized states across python `random`, `numpy`, and `torch` library layers (including CUDA backend determinism settings) to ensure reproducibility:
  ```python
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  ```
