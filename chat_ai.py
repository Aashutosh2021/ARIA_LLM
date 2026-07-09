"""
ARIA-LLM — simple terminal chat.

Defaults to the coherent model: ARIA's from-scratch architecture running real
Qwen2.5-0.5B-Instruct weights. Point --checkpoint at a fine-tuned checkpoint
to talk to your own trained model instead; the prompt format switches
automatically based on the checkpoint's config.

    python chat_ai.py                                          # Qwen-weights (coherent)
    python chat_ai.py --checkpoint checkpoints_chat_qwen/best.pt   # your fine-tuned model
"""

import argparse

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer

from chat_qwen import load_model, generate
from utils.device import get_device

QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"
IM_END = 151645  # Qwen's <|im_end|>
DEFAULT_SYSTEM = (
    "You are ARIA, a helpful and friendly AI assistant. Answer clearly and concisely."
)

p = argparse.ArgumentParser(description="Chat with ARIA")
p.add_argument("--checkpoint", default="checkpoints/qwen2.5-0.5b-instruct.pt")
p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
p.add_argument("--temperature", type=float, default=0.7)
p.add_argument("--max-new-tokens", type=int, default=200)
args = p.parse_args()

device = get_device(args.device)
model, config = load_model(args.checkpoint, device)

tok = AutoTokenizer.from_pretrained(QWEN_ID)
tok.add_special_tokens(
    {"additional_special_tokens": [config.get("user_token", "<|user|>"),
                                   config.get("bot_token", "<|bot|>"),
                                   config.get("end_token", "<|endoftext|>")]}
)

# A fine-tuned checkpoint uses the corpus's own <|user|>/<|bot|> markers; the
# raw Qwen-weights checkpoint uses Qwen's chat template.
finetuned = config.get("tokenizer_type") == "qwen_hf" and config.get("chat_format")
if finetuned:
    stop_id = tok.convert_tokens_to_ids(config.get("end_token", "<|endoftext|>"))
else:
    stop_id = IM_END

history = []  # list of (user, bot)

print("Ready. Chat with ARIA! Press Ctrl+C or Ctrl+D to exit.\n")
while True:
    try:
        q = input("you> ").strip()
        if not q:
            continue

        if finetuned:
            text = ""
            for u, b in history:
                text += f"<|user|> {u}\n<|bot|> {b}\n<|endoftext|>\n"
            text += f"<|user|> {q}\n<|bot|> "
        else:
            messages = [{"role": "system", "content": DEFAULT_SYSTEM}]
            for u, b in history:
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": b})
            messages.append({"role": "user", "content": q})
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        ids = tok.encode(text, return_tensors="pt").to(device)
        out = [t for t in generate(model, ids, device,
                                   max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, top_p=0.9,
                                   repetition_penalty=1.3, max_context=1024,
                                   stop_token_id=stop_id)]
        reply = tok.decode(out, skip_special_tokens=True).strip()
        print(f"aria> {reply}\n")
        history.append((q, reply))
    except (KeyboardInterrupt, EOFError):
        print("\nbye!")
        break
