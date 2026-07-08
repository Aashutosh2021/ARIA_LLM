"""
ARIA-LLM
Real Conversation on the From-Scratch Architecture

This is the headline demo of the project.

It loads **real pretrained Qwen2.5-0.5B-Instruct weights** into ARIA-LLM's
own hand-written transformer (RoPE + RMSNorm + SwiGLU + Grouped-Query
Attention, all implemented from scratch in `model/`). The forward pass,
attention, and normalization are 100% our code -- only the *weights* come
from Qwen. The result is a genuinely conversational assistant powered by
the architecture built in this project.

Prerequisite (one time): convert the weights into our format:
    python scripts/import_qwen.py
This downloads Qwen2.5-0.5B-Instruct and saves it as
`checkpoints/qwen2.5-0.5b-instruct.pt`.

Usage:
    python chat_qwen.py
    python chat_qwen.py --device cpu
    python chat_qwen.py --temperature 0.7 --max-new-tokens 300

In-chat commands:
    /reset             clear the conversation history
    /system <text>     set a new system prompt and reset
    /temp <value>      change sampling temperature
    /help              show this help
    /exit  /quit       leave
"""

import argparse
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer

from model.gpt import GPT
from utils.device import get_device, device_name


CKPT_PATH = "checkpoints/ARIA_model.pt"
QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"
IM_END = 151645  # Qwen's <|im_end|> token id

DEFAULT_SYSTEM = (
    "You are ARIA, a helpful and friendly AI assistant running entirely on a "
    "transformer architecture built from scratch. Answer clearly and concisely."
)

BANNER = r"""
   _   ___ ___    _
  /_\ |_ _| _ \  /_\    from-scratch architecture
 / _ \ | ||   / / _ \   running real Qwen2.5 weights
/_/ \_\___|_|_\/_/ \_\
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chat via ARIA's from-scratch architecture + Qwen2.5 weights"
    )
    parser.add_argument("--checkpoint", default=CKPT_PATH)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-context", type=int, default=1536,
                        help="Max prompt tokens kept before generating")
    return parser.parse_args()


def load_model(ckpt_path: str, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model = GPT.from_config(config)
    # Fine-tuned checkpoints go through training/checkpoint.py ("model_state"),
    # the raw import_qwen.py conversion uses "model_state_dict" -- support both.
    state_dict = checkpoint.get("model_state_dict") or checkpoint["model_state"]
    # strict=False: Qwen has no biases on o_proj / mlp, so those stay zero
    # (mathematically identical to having no bias).
    model.load_state_dict(state_dict, strict=False)

    # fp16 on GPU halves memory and works now that the attention mask uses a
    # dtype-safe negative value.
    if device.type == "cuda":
        model = model.half()

    return model.to(device).eval(), config


@torch.no_grad()
def generate(model, input_ids, device, max_new_tokens,
             temperature, top_p, repetition_penalty, max_context,
             stop_token_id=IM_END):
    """Autoregressive generation with temperature / top-p / repetition penalty.

    Yields decoded-token ids one at a time so the caller can stream them.
    """

    generated = []

    for _ in range(max_new_tokens):
        # Crop context so the prompt never exceeds the window.
        context = input_ids[:, -max_context:]

        logits, _ = model(context)
        next_logits = logits[:, -1, :].float()

        # Repetition penalty: discourage tokens already produced.
        if repetition_penalty and repetition_penalty != 1.0 and generated:
            for tok_id in set(generated):
                if next_logits[0, tok_id] > 0:
                    next_logits[0, tok_id] /= repetition_penalty
                else:
                    next_logits[0, tok_id] *= repetition_penalty

        if temperature and temperature > 0:
            next_logits = next_logits / temperature

            # Top-p (nucleus) filtering.
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
        if token_id == stop_token_id:
            break

        generated.append(token_id)
        input_ids = torch.cat([input_ids, next_token.to(input_ids.device)], dim=-1)
        yield token_id


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Create it first:  python scripts/import_qwen.py"
        )

    device = get_device(args.device)

    print(BANNER)
    print("Loading tokenizer + converted Qwen weights ...")

    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
    model, config = load_model(str(ckpt_path), device)

    # Fine-tuned checkpoints (train_qwen_finetune.py) keep the corpus's own
    # <|user|>/<|bot|>/<|endoftext|> markers instead of Qwen's chat template --
    # re-register them (same ids, since they're added in the same order) and
    # switch the prompt building + stop condition to match.
    finetuned = config.get("tokenizer_type") == "qwen_hf" and config.get("chat_format")
    if finetuned:
        user_token = config.get("user_token", "<|user|>")
        bot_token = config.get("bot_token", "<|bot|>")
        end_token = config.get("end_token", "<|endoftext|>")
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [user_token, bot_token, end_token]}
        )
        stop_token_id = tokenizer.convert_tokens_to_ids(end_token)
    else:
        stop_token_id = IM_END

    print(f"Ready on {device_name(device)}.  "
          f"Params: {model.num_params():,}  (our architecture, Qwen weights)")
    print("Type /help for commands.")
    print("-" * 64)

    messages = [{"role": "system", "content": args.system}]
    temperature = args.temperature

    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break

        if not user:
            continue

        low = user.lower()

        if low in ("/exit", "/quit"):
            print("bye!")
            break
        if low == "/help":
            print(__doc__)
            continue
        if low == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print("  (conversation reset)")
            continue
        if low.startswith("/system "):
            args.system = user[len("/system "):].strip()
            messages = [{"role": "system", "content": args.system}]
            print("  (system prompt set; conversation reset)")
            continue
        if low.startswith("/temp "):
            try:
                temperature = float(user.split()[1])
                print(f"  temperature -> {temperature}")
            except (ValueError, IndexError):
                print("  usage: /temp 0.7")
            continue

        messages.append({"role": "user", "content": user})

        if finetuned:
            # Corpus format has no system turn -- replay just the user/bot
            # history with the trained markers.
            text = ""
            for m in messages:
                if m["role"] == "user":
                    text += f"{user_token} {m['content']}\n"
                elif m["role"] == "assistant":
                    text += f"{bot_token} {m['content']}\n{end_token}\n"
            text += f"{bot_token} "
        else:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

        print("aira> ", end="", flush=True)
        piece_ids = []
        for token_id in generate(
            model, input_ids, device,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_context=args.max_context,
            stop_token_id=stop_token_id,
        ):
            piece_ids.append(token_id)
            # Stream: decode incrementally for a live typing effect.
            print(tokenizer.decode([token_id], skip_special_tokens=True),
                  end="", flush=True)
        print()

        reply = tokenizer.decode(piece_ids, skip_special_tokens=True).strip()
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
