"""
ARIA-LLM — Knowledge distillation data generation.

Qwen2.5-0.5B-Instruct is used ONLY here, as a *teacher that writes training
text*. It answers a large, varied set of everyday questions; we save those
(question, answer) pairs as a chat corpus. ARIA is then trained from scratch
on this corpus (own tokenizer, own random init) with train_chat.py -- so the
final model contains NO Qwen weights and never loads Qwen at chat time.

Two safeguards keep the corpus honest and on-brand:
  * Identity injection: hand-written ARIA persona pairs are added and
    upsampled, so the student says it is ARIA, made by Aashutosh.
  * Leakage filter: any teacher answer that names a third-party model/company
    (qwen, alibaba, chatgpt, openai, ...) or otherwise breaks character is
    dropped, so the student never learns to call itself Qwen.

Output: data/chat_corpus.txt in the <|user|>/<|bot|>/<|endoftext|> format
train_chat.py expects.

Usage (needs a GPU for reasonable speed -- run on Colab T4):
    python scripts/distill_generate.py --num 4000
    python scripts/distill_generate.py --num 8000 --device cuda
"""

import argparse
import re
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "chat_corpus.txt"

QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"

U, B, END = "<|user|>", "<|bot|>", "<|endoftext|>"

# The student must answer AS ARIA. We tell the teacher to role-play ARIA so its
# answers are already on-brand; the leakage filter catches any slips.
TEACHER_SYSTEM = (
    "You are ARIA, a friendly conversational AI assistant made by Aashutosh. "
    "You are chatting casually with a user. Reply in 1-3 short, natural "
    "sentences. Never mention that you are Qwen, Alibaba, OpenAI, or any other "
    "company or model -- you are simply ARIA. Do not use markdown."
)

# Seed questions across everyday small-talk topics. These get paraphrased and
# combined so the teacher sees a large, varied prompt set without us hand-
# writing thousands of lines.
SEED_QUESTIONS = [
    "hello", "hi there", "hey, how's it going", "good morning", "good evening",
    "how are you", "what's up", "how was your day", "what are you doing",
    "what's your name", "who are you", "who made you", "what can you do",
    "are you a robot", "do you have feelings", "how old are you",
    "tell me a joke", "tell me a fun fact", "say something interesting",
    "what's your favorite color", "what's your favorite food",
    "do you like music", "what music do you like", "do you like movies",
    "recommend me a movie", "recommend a good book", "what should i eat today",
    "i'm feeling sad", "i'm bored", "i'm tired", "i had a great day",
    "i'm stressed about work", "i can't sleep", "cheer me up",
    "give me some motivation", "i need advice", "how do i make friends",
    "what's the weather like", "do you know any games", "let's play a game",
    "what is love", "what is the meaning of life", "do you believe in aliens",
    "what's your dream", "what do you think about robots taking over",
    "help me relax", "teach me something new", "what's a good habit to build",
    "how do i stay focused", "how do i be more productive",
    "what's a healthy breakfast", "how much water should i drink",
    "how do i learn to code", "what's python", "explain the internet simply",
    "tell me about space", "why is the sky blue", "how do plants grow",
    "what's a black hole", "how do airplanes fly", "why do we dream",
    "what's your favorite animal", "do you like dogs or cats",
    "what should i do this weekend", "suggest a hobby", "how do i cook rice",
    "what's a quick dinner idea", "how do i stay motivated to exercise",
    "goodbye", "good night", "talk to you later", "thank you", "thanks a lot",
    "you're helpful", "you're funny", "nice talking to you",
]

# Light paraphrase templates to multiply the seed set.
PARAPHRASE = [
    "{q}",
    "{q}?",
    "hey, {q}",
    "so {q}",
    "can you tell me, {q}",
    "i was wondering {q}",
    "quick question: {q}",
    "{q} please",
]

# Identity / persona pairs -- upsampled so ARIA reliably self-identifies.
IDENTITY_PAIRS = [
    ("what is your name", "My name is ARIA. I'm a conversational AI assistant."),
    ("who are you", "I'm ARIA, a friendly AI assistant here to chat and help."),
    ("who made you", "I was created by Aashutosh."),
    ("who created you", "I was built by Aashutosh."),
    ("are you chatgpt", "No, I'm ARIA, my own assistant made by Aashutosh."),
    ("what are you", "I'm ARIA, a conversational AI you can chat with."),
    ("what can you do", "I can chat with you, answer questions, and keep you company."),
]

# Answers mentioning any of these are dropped (character break / leakage).
BANNED = re.compile(
    r"\b(qwen|alibaba|openai|chatgpt|gpt-?\d|anthropic|claude|gemini|"
    r"llama|meta ai|language model trained by|as an ai language model)\b",
    re.IGNORECASE,
)


def clean(text: str) -> str:
    text = " ".join(text.strip().split())
    return text


def build_prompts(num):
    """Deterministically expand seeds x paraphrases up to `num` prompts."""
    prompts = []
    for i, q in enumerate(SEED_QUESTIONS):
        for j, tmpl in enumerate(PARAPHRASE):
            prompts.append(clean(tmpl.format(q=q)))
    # Repeat the list (paraphrases already vary) until we reach `num`.
    out = []
    k = 0
    while len(out) < num:
        out.append(prompts[k % len(prompts)])
        k += 1
    return out[:num]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Generate ARIA training data via Qwen")
    p.add_argument("--num", type=int, default=4000,
                   help="How many teacher Q/A pairs to generate")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--identity-upsample", type=int, default=60)
    p.add_argument("--max-new-tokens", type=int, default=80)
    args = p.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading teacher {QWEN_ID} on {device} ...")
    tok = AutoTokenizer.from_pretrained(QWEN_ID)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    prompts = build_prompts(args.num)
    print(f"Generating {len(prompts)} teacher answers "
          f"(batch {args.batch_size}) ...")

    pairs = []
    dropped = 0
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start:start + args.batch_size]

        texts = [
            tok.apply_chat_template(
                [{"role": "system", "content": TEACHER_SYSTEM},
                 {"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True,
            )
            for q in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        gen = model.generate(
            **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
            temperature=0.8, top_p=0.9, pad_token_id=tok.eos_token_id,
        )
        # Slice off the prompt tokens per row.
        for i, q in enumerate(batch):
            full = gen[i]
            reply_ids = full[enc["input_ids"].shape[1]:]
            answer = clean(tok.decode(reply_ids, skip_special_tokens=True))
            if not answer or BANNED.search(answer):
                dropped += 1
                continue
            pairs.append((q, answer))

        done = min(start + args.batch_size, len(prompts))
        if done % (args.batch_size * 10) == 0 or done == len(prompts):
            print(f"  {done}/{len(prompts)}  (kept {len(pairs)}, dropped {dropped})")

    # Inject + upsample identity so the student self-identifies as ARIA.
    for _ in range(args.identity_upsample):
        pairs.extend(IDENTITY_PAIRS)

    # Deterministic interleave-shuffle (no RNG import needed): rotate.
    examples = [f"{U} {q}\n{B} {a}\n{END}\n" for q, a in pairs]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(examples), encoding="utf-8")

    size_mb = OUT.stat().st_size / 1e6
    print(f"\nKept {len(pairs):,} pairs (dropped {dropped} for leakage/empty).")
    print(f"Wrote {len(examples):,} examples ({size_mb:.1f} MB) -> {OUT}")
    print("\nNext: train ARIA from scratch:\n"
          "  python train_chat.py --device cuda --epochs 30")


if __name__ == "__main__":
    main()
