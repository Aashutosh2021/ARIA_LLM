# ðŸ¤– AIRA-LLM

> **AIRA-LLM** is a GPT-style Large Language Model built completely from scratch using Python and PyTorch.
> The goal of this project is to understand and implement every core component of a modern decoder-only language model instead of relying on existing pretrained models or high-level libraries.

---

# âš¡ Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt
```

There are **two ways to chat** (see GUIDELINE.md for full details):

### ðŸ’¬ Real conversation (pretrained)
```bash
python chat_ai.py          # small instruction-tuned model (Qwen2.5), answers questions
```

### ðŸ§  The from-scratch model (train it, then continue text)
```bash
# GPU build of torch (NVIDIA): pip install torch --index-url https://download.pytorch.org/whl/cu128
python scripts/prepare_data.py                                  # ~20MB TinyStories
python train.py --data data/tinystories.txt --tokenizer bpe \
    --vocab-size 4000 --seq-len 128 --batch-size 32 --epochs 3 \
    --device cuda --output-dir checkpoints_ts
python chat.py --checkpoint checkpoints_ts/best.pt              # continue text
```

Run the tests:
```bash
pytest tests/test_pipeline.py -q      # 16 passed
```

> **`chat_ai.py`** gives a genuinely conversational assistant by using a small
> pretrained model. **`chat.py`** uses the 100%-from-scratch AIRA model â€” it
> *continues* text in the style of its training data. The from-scratch model
> now uses a **byte-level BPE tokenizer** (no more `<UNK>`) and trains on real
> English (TinyStories) on the GPU.

---

# ðŸ“– Project Vision

Most tutorials simply load a pretrained model like GPT-2 or Llama.

This project follows a completely different philosophy.

Instead of importing an existing language model, every important component is implemented manually, including:

* Vocabulary Builder
* Tokenizer
* Dataset Pipeline
* Embedding Layer
* Positional Encoding
* Attention Mechanism
* Multi-Head Attention
* Transformer Blocks
* GPT Architecture
* Training Pipeline
* Inference Engine

The objective is not only to build a working language model, but also to deeply understand how modern LLMs work internally.

---

# ðŸ§­ Two Ways to Use AIRA-LLM

The project ships **two chat experiences** that serve two different goals.
Understanding the difference is key to the whole project.

### 1. The From-Scratch Model â€” `chat.py`

The heart of the project. A GPT you **train yourself**, where every layer
(tokenizer, attention, transformer blocks, training loop) is hand-implemented.

* It is a **text-continuation** model: you give it the start of a sentence and
  it predicts what comes next, in the style of its training data.
* It is **not** an instruction-following assistant â€” it completes text, it does
  not "answer" questions.
* Its intelligence is bounded by its size and its data. Trained on TinyStories
  it writes coherent simple English; it will never match ChatGPT.

### 2. Real Conversation â€” `chat_ai.py`

A practical, genuinely conversational assistant built on a small **pretrained
instruction-tuned model** (Qwen2.5). It answers questions and remembers the
conversation.

* Use this when you want a **usable chatbot right now**.
* It is intentionally **not from scratch** â€” it exists to show the contrast
  between a hand-built continuation model and a real instruction-tuned LLM.

| | `chat.py` (from scratch) | `chat_ai.py` (pretrained) |
|---|---|---|
| **Built by** | You, every layer | Loaded pretrained |
| **Behaviour** | Continues text | Answers & converses |
| **Purpose** | Learning how LLMs work | A usable chatbot |
| **Quality** | As good as your data + size | Coherent out of the box |

---

# ðŸŽ¯ Goals

* Learn how LLMs work from scratch
* Build every module manually
* Create a scalable architecture
* Progress from a small educational model to a real GPT model
* Eventually train a custom language model with millions of parameters

---

# ðŸ›  Technology Stack

* Python 3.x
* PyTorch (CPU **and** CUDA / GPU training)
* NumPy
* Hugging Face Transformers *(only for the optional pretrained chat mode)*
* JSON / YAML (configs & tokenizer files)
* Git

> Every core LLM component is written from scratch. `transformers` is used
> **only** by the optional conversational mode (`chat_ai.py`), never by the
> from-scratch model.

---

# ðŸ“ Project Structure

```text
AIRA-LLM/
â”‚
â”œâ”€â”€ configs/                # YAML configs (model / training / dataset)
â”‚   â”œâ”€â”€ dataset.yaml
â”‚   â”œâ”€â”€ model.yaml
â”‚   â””â”€â”€ training.yaml
â”‚
â”œâ”€â”€ data/                   # Training corpora
â”‚   â”œâ”€â”€ tiny_corpus.txt     #   bundled toy corpus
â”‚   â””â”€â”€ tinystories.txt     #   downloaded real English (via scripts/prepare_data.py)
â”‚
â”œâ”€â”€ dataset/                # Data pipeline: load â†’ clean â†’ tokenize â†’ sequences
â”‚   â”œâ”€â”€ cleaner.py
â”‚   â”œâ”€â”€ dataset.py
â”‚   â”œâ”€â”€ loader.py
â”‚   â””â”€â”€ preprocess.py
â”‚
â”œâ”€â”€ tokenizer/              # Tokenizers (all from scratch)
â”‚   â”œâ”€â”€ base_tokenizer.py   #   abstract interface
â”‚   â”œâ”€â”€ word_tokenizer.py   #   whole-word
â”‚   â”œâ”€â”€ char_tokenizer.py   #   character-level
â”‚   â”œâ”€â”€ bpe_tokenizer.py    #   byte-level BPE  (no <UNK>)
â”‚   â”œâ”€â”€ vocab.py
â”‚   â””â”€â”€ special_tokens.py
â”‚
â”œâ”€â”€ model/                  # The GPT architecture (hand-built)
â”‚   â”œâ”€â”€ embedding.py        #   token embedding
â”‚   â”œâ”€â”€ position.py         #   sinusoidal positional encoding
â”‚   â”œâ”€â”€ attention.py        #   scaled dot-product attention
â”‚   â”œâ”€â”€ multi_head_attention.py
â”‚   â”œâ”€â”€ mask.py             #   causal + padding masks
â”‚   â”œâ”€â”€ feed_forward.py
â”‚   â”œâ”€â”€ layer_norm.py
â”‚   â”œâ”€â”€ transformer_block.py
â”‚   â”œâ”€â”€ init.py             #   GPT-2 style weight init
â”‚   â”œâ”€â”€ gpt.py              #   full model (weight tying, from_config)
â”‚   â”œâ”€â”€ rope.py             #   (planned) rotary embeddings
â”‚   â””â”€â”€ cache.py            #   (planned) KV cache
â”‚
â”œâ”€â”€ training/               # Training loop
â”‚   â”œâ”€â”€ trainer.py          #   train/val loop, grad clip, checkpoints
â”‚   â”œâ”€â”€ loss.py             #   cross-entropy
â”‚   â”œâ”€â”€ optimizer.py        #   AdamW
â”‚   â”œâ”€â”€ scheduler.py        #   warmup + cosine
â”‚   â”œâ”€â”€ checkpoint.py       #   save / load / resume
â”‚   â””â”€â”€ logger.py           #   JSONL logging
â”‚
â”œâ”€â”€ inference/              # Text generation
â”‚   â”œâ”€â”€ generator.py        #   autoregressive decode loop
â”‚   â”œâ”€â”€ sampler.py          #   combines the strategies below
â”‚   â”œâ”€â”€ temperature.py
â”‚   â”œâ”€â”€ topk.py
â”‚   â”œâ”€â”€ topp.py
â”‚   â””â”€â”€ beam_search.py
â”‚
â”œâ”€â”€ utils/                  # Helpers: device, seeding, metrics, config loader
â”‚   â”œâ”€â”€ device.py
â”‚   â”œâ”€â”€ seed.py
â”‚   â”œâ”€â”€ metrics.py          #   perplexity, running averages
â”‚   â””â”€â”€ helper.py           #   YAML config + tokenizer loading
â”‚
â”œâ”€â”€ scripts/
â”‚   â””â”€â”€ prepare_data.py     # Download / prepare the TinyStories corpus
â”‚
â”œâ”€â”€ tests/
â”‚   â””â”€â”€ test_pipeline.py    # Assertion-based test suite (16 tests)
â”‚
â”œâ”€â”€ train.py                # â–º Train the from-scratch model
â”œâ”€â”€ generate.py             # â–º One-off text generation
â”œâ”€â”€ chat.py                 # â–º Interactive REPL for the from-scratch model
â”œâ”€â”€ chat_ai.py              # â–º Real conversation (pretrained Qwen2.5)
â”œâ”€â”€ conftest.py
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ GUIDELINE.md            # Full how-to-use guide
â””â”€â”€ README.md
```

---

# ðŸ§© Project Workflow

```text
Raw Dataset
      â”‚
      â–¼
Dataset Loader
      â”‚
      â–¼
Dataset Cleaner
      â”‚
      â–¼
Tokenizer
      â”‚
      â–¼
Vocabulary
      â”‚
      â–¼
Token IDs
      â”‚
      â–¼
Training Samples
      â”‚
      â–¼
Embedding Layer
      â”‚
      â–¼
Positional Encoding
      â”‚
      â–¼
Multi-Head Self Attention
      â”‚
      â–¼
Feed Forward Network
      â”‚
      â–¼
Transformer Blocks
      â”‚
      â–¼
GPT Model
      â”‚
      â–¼
Training
      â”‚
      â–¼
Inference
      â”‚
      â–¼
Generated Text
```

---

# ðŸ“š Tokenizer Module

A tokenizer converts raw text into numerical token IDs the network can
process (and back again). Three tokenizers are implemented **from scratch**,
each a different trade-off:

| Tokenizer | How it splits text | Vocabulary | Handles unseen words? |
|-----------|--------------------|------------|-----------------------|
| **Word** | by whitespace/words | one ID per word | âŒ â†’ `<UNK>` |
| **Char** | one ID per character | tiny (all chars) | âœ… (any character) |
| **BPE** (byte-level) | learned sub-word merges over raw bytes | configurable (e.g. 4kâ€“8k) | âœ… **never `<UNK>`** |

### The `<UNK>` problem â€” and why byte-level BPE solves it

A **word** tokenizer only knows the words it saw in training. Anything new
(like `hello` if it never appeared) becomes the "unknown" token `<UNK>`, and
the model can neither understand nor produce it.

**Byte-level BPE** fixes this at the root:

1. Any text is first turned into its raw **UTF-8 bytes** (values 0â€“255).
   Because all 256 byte values are always in the vocabulary, **every possible
   character â€” any language, punctuation, even emoji â€” is representable.**
2. Training then learns **merges**: the most frequent adjacent pair of tokens
   is repeatedly fused into a new token, growing the vocabulary from 256 bytes
   up to the target size. This is the same core idea used by GPT-2 / GPT-4.

Result: **there is no such thing as an unknown token.** This is the single
biggest quality improvement for the from-scratch model.

### Reserved special tokens

| Token    | ID | Meaning |
| -------- | -: | ------- |
| `<PAD>`  |  0 | padding |
| `<UNK>`  |  1 | unknown (word tokenizer only) |
| `<BOS>`  |  2 | beginning of sequence |
| `<EOS>`  |  3 | end of sequence |
| `<MASK>` |  4 | masked position |

---

# ðŸ“š Dataset Module

The dataset pipeline performs:

* Loading datasets
* Cleaning text
* Building vocabulary
* Tokenizing text
* Creating input-target sequences
* Creating PyTorch Dataset objects

Pipeline:

```text
Text

â†“

Clean

â†“

Tokenize

â†“

Vocabulary

â†“

Encode

â†“

Sequences

â†“

PyTorch Dataset
```

---

# ðŸ§  Model Architecture

Current model consists of:

### Token Embedding

Converts token IDs into dense vectors.

---

### Positional Encoding

Adds positional information to embeddings.

---

### Attention (Causal + Padding Masks)

Implements scaled dot-product attention. A **causal mask** ensures each
position can only attend to earlier positions (so the model cannot "cheat" by
looking at future tokens), and an optional **padding mask** lets it ignore
`<PAD>` positions in batched sequences.

---

### Multi-Head Attention

Allows the model to attend to multiple representation subspaces in parallel.

---

### Feed Forward Network

Processes token representations independently after attention.

---

### Layer Normalization

Improves training stability.

---

### Transformer Block (Pre-Norm)

Combines the pieces using a **pre-norm** design (LayerNorm applied *before*
each sub-layer, as in GPT-2), which trains more stably:

* Multi-Head Attention
* Residual Connections
* Feed Forward Network
* Layer Normalization

---

### GPT Model

Stacks multiple Transformer Blocks to perform next-token prediction. It also
includes two modern touches:

* **Weight Tying** â€” the input embedding matrix and the output projection
  share the same weights. This saves parameters and typically improves
  quality.
* **GPT-2 style Weight Initialization** â€” weights start from a small normal
  distribution, with residual projections scaled by `1/âˆš(2Â·n_layers)` to keep
  activations stable as depth grows.

The model trained on TinyStories is a **~4.7M parameter** GPT that reaches a
validation perplexity of ~8.6 and writes coherent simple English.

---

# ðŸ‹ Training Pipeline

A complete, working training loop. The model learns by **next-token
prediction**: given a sequence, predict the following token, and minimise the
error over millions of examples.

The training module includes:

* **Cross-Entropy Loss** â€” next-token prediction objective
* **AdamW Optimizer** â€” with weight decay
* **Warmup + Cosine** learning-rate schedule â€” LR ramps up, then decays smoothly
* **Gradient Clipping** â€” prevents unstable updates
* **Validation loop** â€” tracks loss & **perplexity** on held-out data
* **Checkpointing** â€” saves `best.pt` / `last.pt`, resumable (model + optimizer + step)
* **Logging** â€” human-readable + JSONL for later inspection
* **CPU and GPU (CUDA)** training â€” auto-detects the device

Perplexity (â‰ˆ how "surprised" the model is by the next word; lower is better)
dropping over epochs is the signal that learning is working:

```text
epoch 1 â†’ perplexity 17.7
epoch 2 â†’ perplexity  9.8
epoch 3 â†’ perplexity  8.6
```

---

# ðŸ” Inference

Generation is **autoregressive**: the model predicts one token, appends it,
and feeds the result back in to predict the next â€” repeating until the text is
complete. *How* the next token is chosen is the **decoding strategy**:

* **Greedy Decoding** â€” always take the most likely token (deterministic)
* **Temperature Sampling** â€” flatten/sharpen the probabilities (creativity dial)
* **Top-k Sampling** â€” sample only from the `k` most likely tokens
* **Top-p (Nucleus) Sampling** â€” sample from the smallest set covering probability `p`
* **Beam Search** â€” keep several candidate sequences and pick the best overall

These are exposed through three entry points:

* **`generate.py`** â€” one-off text generation from a prompt
* **`chat.py`** â€” interactive REPL for the from-scratch model (text continuation)
* **`chat_ai.py`** â€” real conversation via a pretrained instruction-tuned model

---

# ðŸ§ª Testing

An **assertion-based test suite** (`tests/test_pipeline.py`, 16 tests, run with
`pytest`) verifies the system actually behaves correctly â€” not just that it runs:

* Model output shapes and the **causal-mask invariant** (no attention to the future)
* Weight tying is active
* Tokenizer **round-trips** â€” including BPE on unicode/emoji with **no `<UNK>`**
* A training step **reduces the loss** (learning really happens)
* End-to-end **generation** produces text

Alongside these, each module also has its own script-style smoke test.

---

# ðŸš€ Current Progress

## Completed & Working

* Project Structure & Configuration Files
* Word Tokenizer + Character Tokenizer
* Vocabulary Builder with reserved special tokens
* Dataset pipeline (Loader, Cleaner, Preprocessor, PyTorch Dataset)
* Token Embedding + Sinusoidal Positional Encoding
* Scaled Dot-Product Attention (causal)
* Multi-Head Self Attention with padding-mask support
* Feed Forward Network + Layer Normalization
* Pre-Norm Transformer Block
* GPT model with **weight tying** and GPT-2 style **weight init**
* Cross-Entropy Loss, AdamW Optimizer
* **Warmup + Cosine** learning-rate scheduler
* **Full training loop** (grad clipping, validation, checkpointing, logging)
* **Inference**: greedy, temperature, top-k, top-p, beam search
* End-to-end **`train.py`** and **`generate.py`** entry points
* Utilities: device selection, seeding, perplexity metrics, config loader
* **Assertion-based test suite** (`tests/test_pipeline.py`)

* **Byte-level BPE tokenizer** (from scratch, no `<UNK>`)
* **GPU training** (CUDA) on real English (TinyStories)
* **Conversational chat mode** (`chat_ai.py`, pretrained instruction-tuned model)
* Interactive REPL for the from-scratch model (`chat.py`)

## In Progress / Planned

* RoPE, RMSNorm, KV cache (stubs present under `model/`)
* Mixed precision + gradient accumulation
* Larger from-scratch model / longer training
* Instruction fine-tuning of the from-scratch model

---

# ðŸ“Œ Upcoming Features

*(Already done: Character & byte-level BPE tokenizers, GPT-2 style decoder,
causal self-attention, weight tying, text generation, chat interfaces, GPU
training.)*

Still planned:

* RMSNorm
* Rotary Positional Embeddings (RoPE) â€” stub in `model/rope.py`
* KV Cache â€” stub in `model/cache.py`
* Flash Attention
* Mixed Precision Training
* Gradient Accumulation
* Distributed Training
* Instruction Fine-tuning of the from-scratch model
* Web Interface / REST API
* Quantization & Model Export

---

# ðŸ—º Roadmap

### âœ… Version 0.1 â€” done

* Learning implementation
* Basic GPT prototype
* Complete data pipeline

---

### âœ… Version 0.2 â€” done

* GPT-2 architecture (pre-norm, weight tying, GPT-2 init)
* Decoder-only model
* Better attention (causal + padding mask)
* Byte-level BPE tokenizer

---

### âœ… Version 0.3 â€” done

* Training on TinyStories (on GPU)
* First coherent text generation
* Interactive chat (`chat.py`) + conversational mode (`chat_ai.py`)

---

### ðŸ”œ Version 0.4 â€” in progress

* Larger from-scratch model (scale beyond ~5M parameters)
* Longer training / bigger corpus
* RoPE, RMSNorm, KV cache

---

### ðŸŽ¯ Version 1.0 â€” goal

* Production-ready AIRA-LLM
* Instruction fine-tuning of the from-scratch model
* Documentation
* API
* Web UI

---

# ðŸŽ“ Learning Objectives

This project aims to understand:

* Tokenization â€” word, character, and **byte-level BPE** (and the `<UNK>` problem)
* Embeddings & positional encoding
* Self Attention â€” including **causal masking**
* Multi Head Attention
* Transformer Architecture (pre-norm blocks, residuals)
* GPT Architecture â€” weight tying, GPT-2 initialization
* Language Modeling (next-token prediction) & **perplexity**
* Training Deep Neural Networks â€” schedules, gradient clipping, checkpointing
* Text Generation â€” greedy, temperature, top-k, top-p, beam search
* The difference between a **from-scratch continuation model** and a
  **pretrained instruction-tuned chatbot**

---

# ðŸ¤ Contributions

Contributions are welcome.

If you would like to improve the project:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

# ðŸ“„ License

This project is intended for educational and research purposes.

---

# ðŸ‘¨â€ðŸ’» Author

**Aashutosh Kumar**

**Project Name:** AIRA-LLM

Building a Large Language Model completely from scratch to understand every layer of modern AI systems.


