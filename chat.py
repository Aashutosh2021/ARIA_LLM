"""
ARIA-LLM
Interactive Chat / REPL

Talks to a trained ARIA-LLM checkpoint. Two behaviours, chosen
automatically from the checkpoint's config:

  * chat_format models (trained with train_chat.py): the prompt is wrapped
    in <|user|> / <|bot|> turn markers and generation STOPS at the
    end-of-turn marker, so you get a single clean reply. Short conversation
    history is kept for context.

  * plain models (trained with train.py on e.g. TinyStories): classic text
    continuation from your prompt.

Usage:
    python chat.py --checkpoint checkpoints_chat/best.pt      # conversational
    python chat.py --checkpoint checkpoints_ts/best.pt        # story continuation
    python chat.py --temperature 0.7 --top-k 40

In-chat commands:
    /temp <value>     change sampling temperature
    /topk <value>     change top-k        (0 disables)
    /topp <value>     change top-p        (1.0 disables)
    /tokens <value>   change max new tokens per reply
    /greedy           toggle greedy decoding
    /reset            clear conversation history
    /settings         show current settings
    /help             show this help
    /exit  /quit      leave
"""

import argparse
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

from model.gpt import GPT
from training.checkpoint import load_checkpoint
from utils.device import get_device, device_name
from utils.helper import load_tokenizer


BANNER = r"""
   _   ___ ___    _        _    _    __  __
  /_\ |_ _| _ \  /_\  ___ | |  | |  |  \/  |
 / _ \ | ||   / / _ \|___|| |__| |__| |\/| |
/_/ \_\___|_|_\/_/ \_\    |____|____|_|  |_|

   ARIA-LLM interactive chat  (type /help for commands)
"""


def parse_args():
    p = argparse.ArgumentParser(description="Chat with ARIA-LLM")
    p.add_argument("--checkpoint", default="checkpoints_chat/best.pt")
    p.add_argument("--vocab", default=None)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--history-turns", type=int, default=3,
                   help="How many past turns to keep as context")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


class Settings:
    def __init__(self, args):
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.max_new_tokens = args.max_new_tokens
        self.greedy = args.greedy

    def show(self):
        mode = "greedy" if self.greedy else "sampling"
        print(f"  mode={mode}  temperature={self.temperature}  "
              f"top_k={self.top_k}  top_p={self.top_p}  "
              f"max_new_tokens={self.max_new_tokens}")


@torch.no_grad()
def generate_ids(model, ids, device, max_new_tokens, temperature,
                 top_k, top_p, greedy, stop_strings, tokenizer):
    """Generate tokens, stopping when any stop_string appears in the output."""

    max_len = getattr(model, "max_sequence_length", 256)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []

    for _ in range(max_new_tokens):
        context = tokens[:, -max_len:]
        logits, _ = model(context)
        next_logits = logits[:, -1, :].float()

        if greedy or temperature <= 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            if top_k and top_k > 0:
                k = min(top_k, next_logits.size(-1))
                thresh = torch.topk(next_logits, k, dim=-1).values[..., -1, None]
                next_logits = next_logits.masked_fill(next_logits < thresh,
                                                       float("-inf"))
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

        generated.append(next_token.item())
        tokens = torch.cat([tokens, next_token], dim=1)

        # Stop as soon as a marker appears in the decoded text.
        text = tokenizer.decode(generated)
        for s in stop_strings:
            if s in text:
                return text.split(s)[0]

    return tokenizer.decode(generated)


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train a conversational model first:\n"
            f"  python scripts/build_chat_corpus.py\n"
            f"  python train_chat.py --device cuda"
        )

    vocab_path = Path(args.vocab) if args.vocab else ckpt_path.parent / "vocab.json"
    if not vocab_path.exists():
        raise SystemExit(f"Vocab not found: {vocab_path}")

    device = get_device(args.device)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    tokenizer = load_tokenizer(config.get("tokenizer_type", "word"), vocab_path)
    config["vocab_size"] = len(tokenizer)

    model = GPT.from_config(config)
    load_checkpoint(ckpt_path, model, map_location=device)
    model.to(device).eval()

    settings = Settings(args)

    chat_format = config.get("chat_format", False)
    U = config.get("user_token", "<|user|>")
    B = config.get("bot_token", "<|bot|>")
    END = config.get("end_token", "<|endoftext|>")

    print(BANNER)
    print(f"device={device_name(device)}  |  vocab={len(tokenizer)}  "
          f"|  params={model.num_params():,}")
    settings.show()
    print("-" * 60)
    if chat_format:
        print("Conversational model (from scratch). Say hi!")
    else:
        print("Text-continuation model. Give it the start of a sentence.")
    print("-" * 60)

    history = []  # list of (user, bot)

    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break

        if not user:
            continue

        if user.startswith("/"):
            parts = user.split()
            name, value = parts[0].lower(), (parts[1] if len(parts) > 1 else None)
            if name in ("/exit", "/quit"):
                print("bye!"); break
            elif name == "/help":
                print(__doc__)
            elif name == "/settings":
                settings.show()
            elif name == "/reset":
                history = []; print("  (history cleared)")
            elif name == "/greedy":
                settings.greedy = not settings.greedy
                print(f"  greedy {'ON' if settings.greedy else 'OFF'}")
            elif name == "/temp" and value:
                settings.temperature = float(value); print(f"  temperature -> {value}")
            elif name == "/topk" and value:
                settings.top_k = int(value); print(f"  top_k -> {value}")
            elif name == "/topp" and value:
                settings.top_p = float(value); print(f"  top_p -> {value}")
            elif name == "/tokens" and value:
                settings.max_new_tokens = int(value); print(f"  max_new_tokens -> {value}")
            else:
                print(f"  unknown command: {user}")
            continue

        if chat_format:
            # Build the prompt from recent history + this turn.
            convo = ""
            for u, b in history[-args.history_turns:]:
                convo += f"{U} {u}\n{B} {b}\n{END}\n"
            convo += f"{U} {user}\n{B}"
            ids = tokenizer.encode(convo)

            reply = generate_ids(
                model, ids, device,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_k=settings.top_k,
                top_p=settings.top_p,
                greedy=settings.greedy,
                stop_strings=[END, U, B],
                tokenizer=tokenizer,
            ).strip()

            history.append((user, reply))
            print(f"aira> {reply}")
        else:
            # Plain continuation.
            ids = tokenizer.encode(user)
            out = generate_ids(
                model, ids, device,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_k=settings.top_k,
                top_p=settings.top_p,
                greedy=settings.greedy,
                stop_strings=[],
                tokenizer=tokenizer,
            )
            print(f"aira> {user}{out}")


if __name__ == "__main__":
    main()
