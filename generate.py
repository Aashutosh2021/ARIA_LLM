"""
ARIA-LLM
Generation Entry Point

Loads a trained checkpoint plus its saved tokenizer vocabulary and
generates text from a prompt.

Usage:
    python generate.py --checkpoint checkpoints/best.pt --prompt "the cat"
    python generate.py --checkpoint checkpoints/best.pt --prompt "the cat" \
        --temperature 0.8 --top-k 20 --max-new-tokens 40
"""

import argparse
from pathlib import Path

from model.gpt import GPT
from training.checkpoint import load_checkpoint
from inference.generator import TextGenerator
from utils.device import get_device, device_name
from utils.helper import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with ARIA-LLM")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--vocab", default=None,
                        help="Vocab json (defaults to vocab.json next to ckpt)")
    parser.add_argument("--prompt", default="the")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train one first:  python train.py"
        )

    vocab_path = Path(args.vocab) if args.vocab else ckpt_path.parent / "vocab.json"
    if not vocab_path.exists():
        raise SystemExit(f"Vocab not found: {vocab_path}")

    device = get_device(args.device)

    # Rebuild the model from the config stored in the checkpoint, then
    # load its weights. The config tells us which tokenizer to reload.
    # pyrefly: ignore [missing-import]
    import torch
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    tokenizer = load_tokenizer(config.get("tokenizer_type", "word"), vocab_path)
    config["vocab_size"] = len(tokenizer)

    model = GPT.from_config(config)
    load_checkpoint(ckpt_path, model, map_location=device)

    generator = TextGenerator(model, tokenizer, device=device)

    print(f"device={device_name(device)}  |  vocab={len(tokenizer)}")
    print(f"prompt: {args.prompt!r}")
    print("-" * 60)

    text = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
    )

    print(text)


if __name__ == "__main__":
    main()


