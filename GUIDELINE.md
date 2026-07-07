# ARIA-LLM - Usage Guideline

This guide explains how to **chat**, **train**, and **test** ARIA-LLM.

There are **three ways to chat**, for three different goals:

| Script | What it is | Quality | Use it when... |
|--------|-----------|---------|----------------|
| **`chat_qwen.py`** | **Our from-scratch architecture** loaded with **real Qwen2.5 weights** | Real conversation, on YOUR model code | You want the best demo (resume / project) |
| **`chat_ai.py`** | A pretrained model via HuggingFace directly | Real conversation, minimal code | You just want a chatbot fast |
| **`chat.py`** | The from-scratch model **you train** yourself | Text continuation, learns from your data | You want to watch the hand-built model learn |

> **Why three?** Training a genuinely conversational model *from random
> initialization* needs massive data and compute, which consumer hardware
> cannot provide. So the project offers three angles:
>
> - **`chat_qwen.py`** is the star. The transformer is 100% our own code
>   (RoPE, RMSNorm, SwiGLU, Grouped-Query Attention), and we load the real
>   pretrained weights of Qwen2.5-0.5B into it. Real conversation, running on
>   the architecture we built. This is what proves the architecture is correct.
> - **`chat_ai.py`** loads a pretrained model the normal way (via
>   `transformers`) - the simplest path to a working chatbot.
> - **`chat.py`** is the educational core - every layer hand-written - and it
>   *continues* text in the style of whatever corpus you train it on.

---

## 1. Setup (one time)

```bash
pip install -r requirements.txt
```

**For GPU** (recommended - you have an NVIDIA GPU), install a CUDA build of
PyTorch. This project was set up with CUDA 12.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Check it worked:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 2. Best demo -> `chat_qwen.py` (real conversation on our architecture)

This is the headline: our hand-written transformer running **real** Qwen2.5
pretrained weights. Only the weights come from Qwen; every line of the forward
pass (attention, RoPE, RMSNorm, SwiGLU, GQA) is our own code in `model/`.

```bash
# One-time: download Qwen2.5-0.5B-Instruct and convert its weights to our format
python scripts/import_qwen.py

# Chat
python chat_qwen.py
```

Example:

```text
you> Hi! My name is Aashutosh.
aira> Hello! Aashutosh. How can I help you today?
you> What is 12 times 8?
aira> 12 times 8 is 96.
you> What was my name?
aira> Your name is Aashutosh.
```

Options and in-chat commands:

```bash
python chat_qwen.py --device cpu                 # run on CPU (slower)
python chat_qwen.py --temperature 0.7            # sampling temperature
python chat_qwen.py --max-new-tokens 300         # longer replies
```

`/reset` (clear history), `/system <text>` (set persona), `/temp <value>`,
`/help`, `/exit`.

> **How it works (the resume talking point):** `scripts/import_qwen.py` maps
> Qwen2.5's weight names onto our module names (e.g. `self_attn.q_proj` ->
> our `attention.q_proj`, `mlp.gate_proj` -> our `ffn.gate_proj`) and saves a
> checkpoint. Because our architecture computes exactly what Qwen does, the
> weights "just work" and produce coherent conversation.

---

## 3. Simplest chatbot -> `chat_ai.py` (pretrained, via transformers)

```bash
python chat_ai.py                      # Qwen2.5-0.5B-Instruct
python chat_ai.py --model qwen1.5b     # bigger / better (~3 GB)
python chat_ai.py --device cpu
```

Same in-chat commands as above. This path uses the `transformers` library
directly (not our model code) - handy as a baseline / fallback.

---

## 4. The from-scratch model you train -> `chat.py`

This is the educational core. You **train** the model first, then chat with
(really, *prompt*) it. It continues text rather than answering questions.

### 4a. Train on real English (TinyStories) with GPU + BPE

```bash
python scripts/prepare_data.py         # download ~20MB of simple English stories

python train.py --data data/tinystories.txt --tokenizer bpe \
    --vocab-size 4000 --seq-len 128 --batch-size 32 --epochs 3 \
    --device cuda --output-dir checkpoints_ts
```

Checkpoints and tokenizer are saved to `checkpoints_ts/`.

### 4b. Chat with your trained model

```bash
python chat.py --checkpoint checkpoints_ts/best.pt
```

Start a sentence and let it continue, e.g. `once upon a time there was a`.
It writes coherent simple English, but it is **not** a Q&A chatbot - for that
use `chat_qwen.py` (Section 2).

### One-off generation (non-interactive)

```bash
python generate.py --checkpoint checkpoints_ts/best.pt \
    --prompt "the little robot" --top-k 40 --max-new-tokens 60
```

---

## 5. Why the old model said `<UNK>` (and why it's fixed)

The first version used a **word tokenizer** with a tiny vocabulary, so any word
it never saw (like "hello") became `<UNK>`. The **byte-level BPE tokenizer**
(`--tokenizer bpe`) represents text as raw bytes, so **every** character, word,
or emoji is representable - **no `<UNK>` ever**.

| `--tokenizer` | Description | Best for |
|---------------|-------------|----------|
| `word` | Whole-word (original) | tiny demos |
| `char` | One token per character | very small data |
| `bpe`  | Byte-level BPE (recommended) | real text |

---

## 6. Training options reference (`train.py`)

| Flag | Meaning | Default |
|------|---------|---------|
| `--data` | Dataset path (`.txt`/`.json`/`.jsonl`/`.csv`) | `data/tiny_corpus.txt` |
| `--tokenizer` | `word` / `char` / `bpe` | `word` |
| `--vocab-size` | BPE target vocab size | `8000` |
| `--seq-len` | Context window (tokens) | `64` |
| `--batch-size` | Sequences per step | from `training.yaml` |
| `--epochs` | Passes over the data | from `training.yaml` |
| `--device` | `auto` / `cpu` / `cuda` / `mps` | `auto` |
| `--output-dir` | Where checkpoints go | `checkpoints` |

---

## 7. Run the tests

```bash
pytest tests/test_pipeline.py -q      # 16 passed
```

Checks model shapes, the causal-mask invariant, tokenizer round-trips
(including BPE with unicode/emoji and no `<UNK>`), that a training step lowers
the loss, and that generation produces text.

---

## 8. Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA: False` | Install the CUDA torch build (Section 1). |
| `Checkpoint not found: checkpoints/qwen2.5-...` | Run `python scripts/import_qwen.py` first. |
| `Checkpoint not found` (chat.py) | Train first (Section 4a). |
| `No module named 'transformers'` | `pip install -r requirements.txt`. |
| `operator torchvision::nms does not exist` | `pip uninstall torchvision` (not needed here; it conflicts with the CUDA torch build). |
| `value cannot be converted to type c10::Half` | Old bug - update; the attention mask now uses a dtype-safe value. |
| Out of GPU memory | Lower `--batch-size` / `--seq-len`, or use `--device cpu`. |

---

## 9. What each mode can and cannot do

**`chat_qwen.py`** - our architecture + real Qwen weights
- Real conversation, answers questions, remembers context, simple math
- Proves the from-scratch architecture (RoPE/RMSNorm/SwiGLU/GQA) is correct
- Uses pretrained *weights* (only the numbers are Qwen's; the code is ours)

**`chat_ai.py`** - pretrained via transformers
- Real conversation with the least code
- Not our model code - a baseline

**`chat.py`** - the from-scratch trained model
- 100% hand-built and trained by you
- Continues text; does not "answer" questions
- Only as capable as its data + size allow
