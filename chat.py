"""
ARIA-LLM
Interactive Chat / REPL

A simple interactive loop for talking to a trained ARIA-LLM checkpoint.
Type a prompt, the model continues it, repeat. This is a *text
continuation* model (like GPT-2), not an instruction-tuned assistant, so
it completes your text rather than answering questions.

Usage:
    python chat.py                                   # uses checkpoints/best.pt
    python chat.py --checkpoint checkpoints/last.pt
    python chat.py --temperature 0.9 --top-k 30

In-chat commands:
    /temp <value>     change sampling temperature (e.g. /temp 0.7)
    /topk <value>     change top-k        (0 disables)
    /topp <value>     change top-p        (1.0 disables)
    /tokens <value>   change max new tokens per reply
    /greedy           toggle greedy decoding on/off
    /settings         print the current sampling settings
    /help             show this help
    /exit  or  /quit  leave the chat
"""

import argparse
from pathlib import Path

import torch

from model.gpt import GPT
from training.checkpoint import load_checkpoint
from inference.generator import TextGenerator
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
    parser = argparse.ArgumentParser(description="Chat with ARIA-LLM")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--vocab", default=None,
                        help="Vocab json (defaults to vocab.json next to ckpt)")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


class ChatSettings:
    """Mutable sampling settings the user can tweak mid-session."""

    def __init__(self, args):
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.max_new_tokens = args.max_new_tokens
        self.greedy = args.greedy

    def show(self):
        mode = "greedy" if self.greedy else "sampling"
        print(
            f"  mode={mode}  temperature={self.temperature}  "
            f"top_k={self.top_k}  top_p={self.top_p}  "
            f"max_new_tokens={self.max_new_tokens}"
        )


def handle_command(cmd: str, settings: ChatSettings) -> bool:
    """
    Handle a /command. Returns True if the loop should continue,
    False if the user asked to exit.
    """

    parts = cmd.split()
    name = parts[0].lower()
    value = parts[1] if len(parts) > 1 else None

    if name in ("/exit", "/quit"):
        return False

    elif name == "/help":
        print(__doc__)

    elif name == "/settings":
        settings.show()

    elif name == "/greedy":
        settings.greedy = not settings.greedy
        print(f"  greedy decoding {'ON' if settings.greedy else 'OFF'}")

    elif name == "/temp" and value:
        settings.temperature = float(value)
        print(f"  temperature -> {settings.temperature}")

    elif name == "/topk" and value:
        settings.top_k = int(value)
        print(f"  top_k -> {settings.top_k}")

    elif name == "/topp" and value:
        settings.top_p = float(value)
        print(f"  top_p -> {settings.top_p}")

    elif name == "/tokens" and value:
        settings.max_new_tokens = int(value)
        print(f"  max_new_tokens -> {settings.max_new_tokens}")

    else:
        print(f"  unknown command: {cmd}  (try /help)")

    return True


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train one first:  python train.py"
        )

    vocab_path = (
        Path(args.vocab) if args.vocab else ckpt_path.parent / "vocab.json"
    )
    if not vocab_path.exists():
        raise SystemExit(f"Vocab not found: {vocab_path}")

    device = get_device(args.device)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    tokenizer = load_tokenizer(config.get("tokenizer_type", "word"), vocab_path)
    config["vocab_size"] = len(tokenizer)

    model = GPT.from_config(config)
    load_checkpoint(ckpt_path, model, map_location=device)

    generator = TextGenerator(model, tokenizer, device=device)
    settings = ChatSettings(args)

    print(BANNER)
    print(f"device={device_name(device)}  |  vocab={len(tokenizer)}  "
          f"|  params={model.num_params():,}")
    settings.show()
    print("-" * 60)
    print("Note: this model is trained on a tiny toy corpus, so it")
    print("continues your text rather than holding a real conversation.")
    print("-" * 60)

    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break

        if not user:
            continue

        if user.startswith("/"):
            if not handle_command(user, settings):
                print("bye!")
                break
            continue

        reply = generator.generate(
            prompt=user,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
            greedy=settings.greedy,
        )

        print(f"aira> {reply}")


if __name__ == "__main__":
    main()


