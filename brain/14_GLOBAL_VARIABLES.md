# Global Variables

ARIA-LLM contains several global constants, defaults, and configuration options:

## 1. CLI Execution Constants

- **`BANNER`** (`chat.py`):
  Console ASCII art banner displayed at REPL boot.
- **`USER_TOKEN`**, **`BOT_TOKEN`**, **`END_TOKEN`** (`train_chat.py`):
  Special tokens used to wrap conversational dialogue turns (`<|user|>`, `<|bot|>`, `<|endoftext|>`).

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
