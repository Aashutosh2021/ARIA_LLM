"""
ARIA-LLM
Conversational Chat (pretrained instruction-tuned model)

This is the *real conversation* mode. It does NOT use the from-scratch
ARIA model -- that model (see chat.py) is a small text-continuation
network and cannot hold a real conversation on consumer hardware.

Instead this loads a small **instruction-tuned** chat model
(Qwen2.5-0.5B-Instruct by default) through HuggingFace Transformers,
applies the model's chat template, keeps the conversation history, and
generates coherent replies. It runs comfortably on a 4 GB GPU (fp16) or
on CPU (slower).

The first run downloads the model (~1 GB for the 0.5B default).

Usage:
    python chat_ai.py                       # Qwen2.5-0.5B-Instruct on GPU
    python chat_ai.py --model qwen1.5b      # better, ~3 GB
    python chat_ai.py --device cpu
    python chat_ai.py --system "You are a pirate who loves math."

In-chat commands:
    /reset             clear the conversation history
    /system <text>     set a new system prompt and reset
    /temp <value>      change sampling temperature
    /help              show this help
    /exit  /quit       leave
"""

import argparse

import torch


MODEL_IDS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

DEFAULT_SYSTEM = (
    "You are ARIA, a friendly, concise and helpful AI assistant. "
    "You answer clearly and stay on topic."
)

BANNER = r"""
   _   ___ ___    _        _    _    __  __    (chat)
  /_\ |_ _| _ \  /_\  ___ | |  | |  |  \/  |
 / _ \ | ||   / / _ \|___|| |__| |__| |\/| |
/_/ \_\___|_|_\/_/ \_\    |____|____|_|  |_|

   Conversational mode (instruction-tuned) -- type /help
"""


def parse_args():
    parser = argparse.ArgumentParser(description="ARIA conversational chat")
    parser.add_argument("--model", default="qwen0.5b",
                        choices=list(MODEL_IDS.keys()))
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--system", default=DEFAULT_SYSTEM,
                        help="System prompt / persona")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    return parser.parse_args()


def resolve_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(args.device)
    model_id = MODEL_IDS[args.model]
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(BANNER)
    print(f"Loading {model_id} on {device} ... (first run downloads it)")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = (
        AutoModelForCausalLM
        .from_pretrained(model_id, torch_dtype=dtype)
        .to(device)
        .eval()
    )

    print(f"Ready. device={device}. Say hi!  (/help for commands)")
    print("-" * 60)

    messages = [{"role": "system", "content": args.system}]
    temperature = args.temperature

    def generate_reply():
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )

        reply_ids = output[0][inputs.input_ids.shape[-1]:]
        return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()

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
            print(f"  (system prompt set; conversation reset)")
            continue

        if low.startswith("/temp "):
            try:
                temperature = float(user.split()[1])
                print(f"  temperature -> {temperature}")
            except (ValueError, IndexError):
                print("  usage: /temp 0.7")
            continue

        messages.append({"role": "user", "content": user})
        reply = generate_reply()
        messages.append({"role": "assistant", "content": reply})

        print(f"aira> {reply}")


if __name__ == "__main__":
    main()


