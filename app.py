"""
ARIA-LLM — public chat demo (Gradio, CPU-friendly).

A from-scratch transformer (RoPE + RMSNorm + SwiGLU + Grouped-Query Attention,
all in model/) running one of two weight sets, switchable in the UI:

  * "ARIA + Qwen2.5 weights" — real Qwen2.5-0.5B-Instruct weights loaded into
    ARIA's own architecture. Coherent general chat; proves the hand-written
    forward pass is correct.
  * "ARIA (fine-tuned)"      — the same architecture after fine-tuning on the
    project's own chat corpus. Shown only if its checkpoint is available.

Weights are large, so they are NOT committed. At startup the app looks for
them locally under deploy/ first, then falls back to downloading from a
Hugging Face model repo (set HF_MODEL_REPO). This lets the same app.py run
both locally and on Hugging Face Spaces.

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
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer

from model.gpt import GPT


QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"
IM_END = 151645  # Qwen's <|im_end|> token id

USER_TOKEN = "<|user|>"
BOT_TOKEN = "<|bot|>"
END_TOKEN = "<|endoftext|>"

# Where to find weights. Local path wins; otherwise pull from this HF repo.
# Set HF_MODEL_REPO in the Space settings, e.g. "aashutosh/aria-llm-weights".
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")

# label -> (local path, filename in the HF repo)
MODEL_FILES = {
    "ARIA + Qwen2.5 weights": ("deploy/aria_qwen.pt", "aria_qwen.pt"),
    "ARIA (fine-tuned)": ("deploy/aria_finetuned.pt", "aria_finetuned.pt"),
}

DEFAULT_SYSTEM = (
    "You are ARIA, a helpful and friendly AI assistant running on a transformer "
    "architecture built from scratch. Answer clearly and concisely."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shared Qwen tokenizer, with the fine-tune's markers registered so both model
# flavors decode correctly. Adding them is harmless for the Qwen-weights model
# (it just never emits those ids).
tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
tokenizer.add_special_tokens(
    {"additional_special_tokens": [USER_TOKEN, BOT_TOKEN, END_TOKEN]}
)
FINETUNE_END_ID = tokenizer.convert_tokens_to_ids(END_TOKEN)

# Lazily loaded, then cached: label -> (model, config).
_loaded = {}


def _resolve_path(local_path: str, hf_filename: str):
    """Return a usable file path, downloading from HF on demand, or None."""
    if os.path.exists(local_path):
        return local_path
    if HF_MODEL_REPO:
        try:
            # pyrefly: ignore [missing-import]
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=HF_MODEL_REPO, filename=hf_filename)
        except Exception as e:  # noqa: BLE001 - surface a friendly message
            print(f"[warn] could not fetch {hf_filename} from {HF_MODEL_REPO}: {e}")
            return None
    return None


def available_models():
    """Labels whose weights can actually be found (local or on HF)."""
    labels = []
    for label, (local_path, hf_filename) in MODEL_FILES.items():
        if os.path.exists(local_path) or HF_MODEL_REPO:
            labels.append(label)
    return labels or list(MODEL_FILES.keys())


def load_model(label: str):
    """Load + cache the model for a label. Returns (model, config)."""
    if label in _loaded:
        return _loaded[label]

    local_path, hf_filename = MODEL_FILES[label]
    path = _resolve_path(local_path, hf_filename)
    if path is None:
        raise gr.Error(
            f"Weights for '{label}' not found. Place {local_path} locally, "
            f"or set HF_MODEL_REPO to a repo containing {hf_filename}."
        )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    state = ckpt.get("model_state_dict") or ckpt.get("model_state")

    model = GPT.from_config(config)
    model.load_state_dict(state, strict=False)  # Qwen has no o_proj/mlp bias
    model = model.to(device).eval()

    _loaded[label] = (model, config)
    return _loaded[label]


def build_prompt(label, config, history, user_msg, system_msg):
    """Format the prompt for the selected model's expected turn style."""
    finetuned = config.get("tokenizer_type") == "qwen_hf" and config.get("chat_format")

    if finetuned:
        # The fine-tune learned the corpus's own <|user|>/<|bot|> markers.
        text = ""
        for u, b in history:
            text += f"{USER_TOKEN} {u}\n{BOT_TOKEN} {b}\n{END_TOKEN}\n"
        text += f"{USER_TOKEN} {user_msg}\n{BOT_TOKEN} "
        stop_id = FINETUNE_END_ID
    else:
        # Qwen weights -> Qwen's own chat template.
        messages = [{"role": "system", "content": system_msg}]
        for u, b in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": b})
        messages.append({"role": "user", "content": user_msg})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        stop_id = IM_END

    return text, stop_id


@torch.no_grad()
def generate_stream(model, input_ids, stop_id, max_new_tokens,
                    temperature, top_p, repetition_penalty, max_context):
    """Yield generated token ids one at a time (top-p + repetition penalty)."""
    generated = []
    for _ in range(max_new_tokens):
        context = input_ids[:, -max_context:]
        logits, _ = model(context)
        next_logits = logits[:, -1, :].float()

        if repetition_penalty and repetition_penalty != 1.0 and generated:
            for tok_id in set(generated):
                if next_logits[0, tok_id] > 0:
                    next_logits[0, tok_id] /= repetition_penalty
                else:
                    next_logits[0, tok_id] *= repetition_penalty

        if temperature and temperature > 0:
            next_logits = next_logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_logits[remove] = float("-inf")
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        token_id = next_token.item()
        if token_id == stop_id:
            break
        generated.append(token_id)
        input_ids = torch.cat([input_ids, next_token.to(input_ids.device)], dim=-1)
        yield token_id


def respond(user_msg, history, model_label, temperature, max_new_tokens):
    """Gradio ChatInterface callback: stream a reply for the chosen model."""
    history = history or []
    model, config = load_model(model_label)

    text, stop_id = build_prompt(model_label, config, history, user_msg, DEFAULT_SYSTEM)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    piece_ids = []
    for token_id in generate_stream(
        model, input_ids, stop_id,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=0.9,
        repetition_penalty=1.3,
        max_context=1024,
    ):
        piece_ids.append(token_id)
        yield tokenizer.decode(piece_ids, skip_special_tokens=True)


ABOUT = """
### ARIA-LLM
A GPT-style transformer implemented **from scratch** — RoPE, RMSNorm, SwiGLU,
and Grouped-Query Attention are all hand-written (see `model/`). The forward
pass, attention, and normalization are original code.

Pick a weight set:
- **ARIA + Qwen2.5 weights** — real Qwen2.5-0.5B-Instruct weights loaded into
  this architecture (proves the implementation is numerically correct).
- **ARIA (fine-tuned)** — the same architecture after fine-tuning on the
  project's own conversational corpus.

Running on a free CPU tier, so replies stream a bit slowly. Built by Aashutosh.
"""


def build_ui():
    labels = available_models()
    with gr.Blocks(title="ARIA-LLM") as demo:
        gr.Markdown("# 🤖 ARIA-LLM — chat with a from-scratch transformer")
        with gr.Row():
            model_label = gr.Radio(
                choices=labels, value=labels[0], label="Model",
                info="Switch the weights driving the same hand-written architecture",
            )
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
            max_new_tokens = gr.Slider(16, 200, value=96, step=8, label="Max new tokens")
        gr.ChatInterface(
            fn=respond,
            additional_inputs=[model_label, temperature, max_new_tokens],
            examples=[["Hello, who are you?"], ["What can you do?"],
                      ["Tell me a fun fact."]],
        )
        gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    build_ui().queue().launch()
