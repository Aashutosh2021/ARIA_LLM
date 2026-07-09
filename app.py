"""
ARIA-LLM — public chat demo (Gradio, CPU-friendly).

Serves ARIA: a GPT-style transformer built entirely from scratch (RoPE,
RMSNorm, SwiGLU, GQA in model/), trained from random initialization on a
conversational corpus. Its own byte-level BPE tokenizer, its own weights.
No third-party model is loaded at inference time.

Weights + tokenizer are NOT committed (large). At startup the app looks under
deploy/ locally first, then falls back to downloading from a Hugging Face
model repo (set HF_MODEL_REPO). The deploy bundle is two files:
    aria.pt         -- {"config", "model_state_dict"}
    aria_vocab.json -- the BPE tokenizer vocabulary

Run locally:
    python app.py
"""

import os

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
import gradio as gr

from model.gpt import GPT
from utils.helper import load_tokenizer


HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")

WEIGHTS_LOCAL = "deploy/aria.pt"
VOCAB_LOCAL = "deploy/aria_vocab.json"
WEIGHTS_HF = "aria.pt"
VOCAB_HF = "aria_vocab.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve(local_path, hf_filename):
    """Local file if present, else download from HF, else None."""
    if os.path.exists(local_path):
        return local_path
    if HF_MODEL_REPO:
        try:
            # pyrefly: ignore [missing-import]
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=HF_MODEL_REPO, filename=hf_filename)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not fetch {hf_filename}: {e}")
            return None
    return None


def load_aria():
    """Load ARIA (weights + BPE tokenizer). Returns (model, tokenizer, cfg)."""
    wpath = _resolve(WEIGHTS_LOCAL, WEIGHTS_HF)
    vpath = _resolve(VOCAB_LOCAL, VOCAB_HF)
    if wpath is None or vpath is None:
        raise gr.Error(
            "ARIA weights/tokenizer not found. Place deploy/aria.pt and "
            "deploy/aria_vocab.json locally, or set HF_MODEL_REPO."
        )

    ckpt = torch.load(wpath, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    state = ckpt.get("model_state_dict") or ckpt.get("model_state")

    tokenizer = load_tokenizer(config.get("tokenizer_type", "bpe"), vpath)
    config["vocab_size"] = len(tokenizer)

    model = GPT.from_config(config)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return model, tokenizer, config


MODEL, TOKENIZER, CONFIG = None, None, None

U = "<|user|>"
B = "<|bot|>"
END = "<|endoftext|>"


@torch.no_grad()
def generate_stream(model, tokenizer, ids, max_new_tokens, temperature,
                    top_k, top_p, stop_strings):
    """Yield the decoded reply incrementally; stop at any marker."""
    max_len = getattr(model, "max_sequence_length", 256)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []

    for _ in range(max_new_tokens):
        context = tokens[:, -max_len:]
        logits, _ = model(context)
        next_logits = logits[:, -1, :].float()

        if temperature <= 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            if top_k and top_k > 0:
                k = min(top_k, next_logits.size(-1))
                thresh = torch.topk(next_logits, k, dim=-1).values[..., -1, None]
                next_logits = next_logits.masked_fill(next_logits < thresh, float("-inf"))
            if top_p < 1.0:
                sl, si = torch.sort(next_logits, descending=True)
                cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sl[remove] = float("-inf")
                next_logits = sl.scatter(1, si, sl)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated.append(next_token.item())
        tokens = torch.cat([tokens, next_token], dim=1)

        text = tokenizer.decode(generated)
        cut = text
        stop = False
        for s in stop_strings:
            if s in cut:
                cut = cut.split(s)[0]
                stop = True
        yield cut.strip()
        if stop:
            return


def respond(user_msg, history, temperature, max_new_tokens):
    global MODEL, TOKENIZER, CONFIG
    if MODEL is None:
        MODEL, TOKENIZER, CONFIG = load_aria()

    history = history or []
    convo = ""
    for u, b in history[-3:]:
        convo += f"{U} {u}\n{B} {b}\n{END}\n"
    convo += f"{U} {user_msg}\n{B}"
    ids = TOKENIZER.encode(convo)

    last = ""
    for partial in generate_stream(
        MODEL, TOKENIZER, ids,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=40, top_p=0.95,
        stop_strings=[END, U, B],
    ):
        last = partial
        yield partial
    if not last:
        yield "..."


ABOUT = """
### ARIA-LLM
A GPT-style transformer built **entirely from scratch** — RoPE, RMSNorm,
SwiGLU, and Grouped-Query Attention are all hand-written (see `model/`), and
the model is trained from random initialization with its own byte-level BPE
tokenizer. No third-party model runs here.

Training used **knowledge distillation**: a larger model generated a
conversational dataset, and ARIA learned from that data from scratch — so the
weights are entirely ARIA's own.

Running on a free CPU tier, so replies stream a little slowly. Built by Aashutosh.
"""


def build_ui():
    with gr.Blocks(title="ARIA-LLM") as demo:
        gr.Markdown("# 🤖 ARIA-LLM — a from-scratch conversational model")
        with gr.Row():
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
            max_new_tokens = gr.Slider(16, 200, value=80, step=8, label="Max new tokens")
        gr.ChatInterface(
            fn=respond,
            additional_inputs=[temperature, max_new_tokens],
            examples=[["Hello, who are you?"], ["What can you do?"],
                      ["Tell me a fun fact."], ["I'm feeling bored."]],
        )
        gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    build_ui().queue().launch()
