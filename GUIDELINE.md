# ðŸ“˜ ARIA-LLM â€” Usage Guideline

This guide explains how to **chat**, **train**, and **test** ARIA-LLM.

There are **two ways to chat**, for two different goals:

| Script          | What it is | Quality | Use it whenâ€¦ |
|-----------------|-----------|---------|--------------|
| **`chat_ai.py`**  | A small **pretrained instruction-tuned** model (Qwen2.5) | âœ… Real conversation, answers questions | You want a usable chatbot right now |
| **`chat.py`**     | The **from-scratch** ARIA model you train yourself | âœï¸ Text continuation, learns from your data | You want to see the hand-built model work |

> **Why two?** Training a genuinely conversational model from scratch needs
> massive data + compute. `chat_ai.py` gives you real conversation today by
> standing on a small pretrained model. `chat.py` is the educational core of
> this project â€” every layer is hand-written â€” and it *continues* text in the
> style of whatever corpus you train it on.

---

## 1. Setup (one time)

```bash
pip install -r requirements.txt
```

**For GPU training** (recommended â€” you have an NVIDIA GPU), install a CUDA
build of PyTorch that matches your driver, e.g. CUDA 12.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Check it worked:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 2. ðŸ’¬ Just want to chat? â†’ `chat_ai.py` (real conversation)

```bash
python chat_ai.py
```

- First run downloads a ~1 GB model (Qwen2.5-0.5B-Instruct).
- Then you get a real assistant that answers questions and remembers the
  conversation.

Example:
```
you> hi, my name is Aashutosh
aira> Hello Aashutosh! How can I assist you today?
you> what is 15 times 3?
aira> 15 times 3 is 45.
```

Options:
```bash
python chat_ai.py --model qwen1.5b            # bigger/better (~3 GB)
python chat_ai.py --device cpu                # no GPU
python chat_ai.py --system "You are a pirate who loves math."
```

In-chat commands: `/reset`, `/system <text>`, `/temp 0.7`, `/help`, `/exit`.

This is the mode to use if your goal is **normal conversation**.

---

## 3. ðŸ§  The from-scratch model â†’ train, then `chat.py`

This is the hand-built model. You must **train it first**; then you can chat
with (really, *prompt*) it. It continues text rather than answering.

### 3a. Train on real English (TinyStories) with GPU + BPE

```bash
# Download ~20MB of simple English stories
python scripts/prepare_data.py

# Train on GPU with the byte-level BPE tokenizer (no more <UNK>!)
python train.py --data data/tinystories.txt --tokenizer bpe \
    --vocab-size 4000 --seq-len 128 --batch-size 32 --epochs 3 \
    --device cuda --output-dir checkpoints_ts
```

This trains a from-scratch GPT on real English in a few minutes on your GPU.
Checkpoints and tokenizer are saved to `checkpoints_ts/`.

### 3b. Chat with your trained model

```bash
python chat.py --checkpoint checkpoints_ts/best.pt
```

Start a sentence and let it continue, e.g. `once upon a time there was a`.

Because it learned from stories, it will continue in that style. It is far
more coherent than the original toy model, but it is **not** a Q&A chatbot â€”
for that use `chat_ai.py` (Section 2).

### One-off generation (non-interactive)

```bash
python generate.py --checkpoint checkpoints_ts/best.pt \
    --prompt "the little robot" --top-k 40 --max-new-tokens 60
```

---

## 4. Why the old model said `<UNK>` (and why it's fixed)

The first version used a **word tokenizer** trained on a tiny corpus, so its
vocabulary was ~80 words. Any unknown word (like "hello") became `<UNK>`.

The new **byte-level BPE tokenizer** (`--tokenizer bpe`) represents text as
bytes, so **every possible character/word is representable â€” no `<UNK>` ever.**
Train with `--tokenizer bpe` (as in Section 3a) to use it.

Tokenizer options for `train.py`:

| `--tokenizer` | Description | Vocab | Best for |
|---------------|-------------|-------|----------|
| `word` | Whole-word (original) | small | tiny demos |
| `char` | One token per character | tiny | very small data |
| `bpe`  | Byte-level BPE (recommended) | `--vocab-size` | real text |

---

## 5. Training options reference (`train.py`)

| Flag              | Meaning                                  | Default              |
|-------------------|------------------------------------------|----------------------|
| `--data`          | Dataset path (`.txt`/`.json`/`.jsonl`/`.csv`) | `data/tiny_corpus.txt` |
| `--tokenizer`     | `word` / `char` / `bpe`                  | `word`               |
| `--vocab-size`    | BPE target vocab size                    | `8000`               |
| `--seq-len`       | Context window (tokens)                  | `64`                 |
| `--batch-size`    | Sequences per step                       | from `training.yaml` |
| `--epochs`        | Passes over the data                     | from `training.yaml` |
| `--device`        | `auto` / `cpu` / `cuda` / `mps`          | `auto`               |
| `--output-dir`    | Where checkpoints go                     | `checkpoints`        |

Tune the model size in `configs/model.yaml` (`embedding_dim`, `num_layers`,
`num_heads`) and training in `configs/training.yaml`.

---

## 6. Run the tests

```bash
pytest tests/test_pipeline.py -q
```

Checks model shapes, the causal-mask invariant, tokenizer round-trips
(including BPE with unicode/emoji and no `<UNK>`), that a training step lowers
the loss, and that generation produces text. Expected: **16 passed**.

---

## 7. Tuning generation (both `chat.py` and `generate.py`)

| You wantâ€¦                    | Tryâ€¦                                   |
|------------------------------|----------------------------------------|
| Safer, more predictable text | lower temperature (`0.5`), or `/greedy` |
| More varied, creative text   | higher temperature (`0.9â€“1.1`)         |
| Avoid rare gibberish         | `top_k 40` or `top_p 0.9`              |
| Longer replies               | `--max-new-tokens 80` / `/tokens 80`   |

---

## 8. Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA: False` | Install the CUDA torch build (Section 1). |
| `Checkpoint not found` | Train first (Section 3a). |
| `No module named 'transformers'` | `pip install -r requirements.txt` (needed for `chat_ai.py`). |
| `operator torchvision::nms does not exist` | Version mismatch â€” `pip uninstall torchvision` (not needed here). |
| Out of GPU memory | Lower `--batch-size` (e.g. 16) or `--seq-len`. |
| From-scratch output still simple | Train longer / on more data / bigger model. It will never match `chat_ai.py`. |

---

## 9. What each model can and cannot do

**`chat_ai.py` (pretrained Qwen)**
âœ… Real conversation, answers questions, remembers context, does simple math
âŒ Not built from scratch (uses a pretrained model)
âŒ Small (0.5B) â€” not as smart as ChatGPT

**`chat.py` (from-scratch ARIA)**
âœ… 100% hand-implemented â€” every layer
âœ… Learns grammar/style from your training data
âœ… With BPE + TinyStories, writes coherent simple English
âŒ Continues text; does not "answer" questions
âŒ Only as capable as its data + size allow on your hardware


