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

### `chat_qwen.py`
- **`--checkpoint`**: Checkpoint path. Default: `checkpoints/qwen2.5-0.5b-instruct.pt`.
- **`--device`**: Execution target device. Default: `auto`.
- **`--system`**: Core assistant system instructions.
- **`--max-new-tokens`**: Maximum autoregressive generation length. Default: `256`.
- **`--temperature`**: Sampling temperature scale. Default: `0.7`.
- **`--top-p`**: Nucleus sampling probability threshold. Default: `0.9`.
- **`--repetition-penalty`**: Discourage generated repetitions. Default: `1.1`.
- **`--max-context`**: Context window boundary. Default: `1536`.

---

## 2. In-Chat REPL Command API

Both `chat.py` and `chat_qwen.py` interpret the following command prefixes:
- **`/exit` or `/quit`**: Terminates program loop.
- **`/help`**: Prints active console configurations.
- **`/temp <val>`**: Dynamically updates sampling temperature (e.g. `/temp 0.8`).
- **`/reset`** *(Qwen only)*: Clears the conversation history state.
- **`/system <text>`** *(Qwen only)*: Updates the system instructions prompt and clears context.

---

## 3. Converted Weights Model Import API (`scripts/import_qwen.py`)

Handles automated translation of state dictionary keys from Hugging Face model keys to custom ARIA-LLM keys:
- **Query mapping:** `model.layers.{i}.self_attn.q_proj` -> `layers.{i}.attention.q_proj`
- **Key mapping:** `model.layers.{i}.self_attn.k_proj` -> `layers.{i}.attention.k_proj`
- **Value mapping:** `model.layers.{i}.self_attn.v_proj` -> `layers.{i}.attention.v_proj`
- **Output proj mapping:** `model.layers.{i}.self_attn.o_proj` -> `layers.{i}.attention.proj`
- **MLP gate maps (SwiGLU):**
  - `model.layers.{i}.mlp.gate_proj` -> `layers.{i}.ffn.w1`
  - `model.layers.{i}.mlp.down_proj` -> `layers.{i}.ffn.w2`
  - `model.layers.{i}.mlp.up_proj` -> `layers.{i}.ffn.w3`
- **RMSNorm maps:**
  - `model.layers.{i}.input_layernorm` -> `layers.{i}.ln1`
  - `model.layers.{i}.post_attention_layernorm` -> `layers.{i}.ln2`
  - `model.norm` -> `ln_f`
- **Embed/Output Head maps:**
  - `model.embed_tokens` -> `token_embedding.embedding`
  - `lm_head` -> `lm_head`
