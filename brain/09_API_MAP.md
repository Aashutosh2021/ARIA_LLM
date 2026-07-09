# API Map

ARIA-LLM exposes several CLI configurations and model initializers as public APIs.

## 1. CLI Configurations

### `train.py`
- **`--data`**: Path to the text/json/jsonl/csv corpus. Default: `data/tiny_corpus.txt`.
- **`--epochs`**: Override the training epochs count.
- **`--batch-size`**: Override batch size.
- **`--seq-len`**: Sequence context length window size. Default: `64`.
- **`--tokenizer`**: Kind of tokenizer (`word`, `bpe`, `char`). Default: `word`.
- **`--vocab-size`**: Target vocabulary BPE token count. Default: `8000`.
- **`--device`**: Target device execution (`auto`, `cpu`, `cuda`, `mps`).
- **`--output-dir`**: Output directory for checkpoints. Default: `checkpoints`.
- **`--seed`**: Seed configuration. Default: `42`.

### `chat.py`
- **`--checkpoint`**: Checkpoint path. Default: `checkpoints/ARIA_model.pt`.
- **`--vocab`**: Vocabulary file path. Default: `checkpoints/vocab.json`.
- **`--max-new-tokens`**: Maximum autoregressive generation length. Default: `64`.
- **`--temperature`**: Sampling temperature scale. Default: `0.7`.
- **`--top-k`**: Top-k sampling threshold. Default: `40`.
- **`--top-p`**: Nucleus sampling probability threshold. Default: `0.95`.
- **`--greedy`**: Enable greedy decoding.
- **`--history-turns`**: Number of conversational turns to preserve. Default: `3`.
- **`--device`**: Execution target device. Default: `auto`.

---

## 2. In-Chat REPL Command API

`chat.py` interprets the following command prefixes:
- **`/exit` or `/quit`**: Terminates program loop.
- **`/help`**: Prints active console configurations.
- **`/temp <val>`**: Dynamically updates sampling temperature (e.g. `/temp 0.8`).
- **`/reset`**: Clears the conversation history state.
- **`/greedy`**: Toggles greedy decoding on/off.
- **`/settings`**: Prints the current generation settings.
- **`/tokens <val>`**: Dynamically updates max new tokens.
- **`/topk <val>`**: Dynamically updates top-k.
- **`/topp <val>`**: Dynamically updates top-p.

---

