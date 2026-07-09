# Deploying ARIA-LLM to Hugging Face Spaces

Puts a public chat demo of **ARIA** (your from-scratch model — no third-party
model at inference) online for free. Two pieces:

1. **A model repo** — holds the weights (`aria.pt`) + tokenizer (`aria_vocab.json`).
2. **A Space** — holds the app (`app.py`) and pulls those files at startup.

---

## 0. Produce the model (knowledge distillation → from-scratch training)

This runs on Colab (needs a GPU) — see `distill_aria_colab.ipynb`. In short:

```bash
# 1) Qwen generates a conversational dataset (Qwen used ONLY here, as a data
#    generator — its weights never end up in ARIA).
python scripts/distill_generate.py --num 6000 --device cuda

# 2) Train ARIA from scratch on that data (own tokenizer, own random init).
python train_chat.py --device cuda --epochs 30

# 3) Bundle weights + tokenizer for deployment.
python scripts/export_for_deploy.py --checkpoint checkpoints_chat/best.pt --out-dir deploy
```

Test locally before deploying:

```bash
pip install -r requirements-space.txt
python app.py   # open the printed http://127.0.0.1:7860 and chat
```

---

## 1. Hugging Face account + login

- Sign up: https://huggingface.co/join
- Write token: https://huggingface.co/settings/tokens
- Locally:

```bash
pip install huggingface_hub
huggingface-cli login   # paste the write token
```

---

## 2. Create the model repo and upload the bundle

Replace `<user>` with your HF username.

```bash
huggingface-cli repo create aria-llm-weights --type model

huggingface-cli upload <user>/aria-llm-weights deploy/aria.pt aria.pt
huggingface-cli upload <user>/aria-llm-weights deploy/aria_vocab.json aria_vocab.json
```

(The Colab notebook's last cell can do this upload for you.)

---

## 3. Create the Space

```bash
huggingface-cli repo create aria-llm --type space --space_sdk gradio

git clone https://huggingface.co/spaces/<user>/aria-llm
cd aria-llm

# copy from this repo, with the required Space filenames:
cp ../ARIA-LLM/app.py                  app.py
cp ../ARIA-LLM/requirements-space.txt  requirements.txt
cp ../ARIA-LLM/README-space.md         README.md
cp -r ../ARIA-LLM/model                model
cp -r ../ARIA-LLM/tokenizer            tokenizer
cp -r ../ARIA-LLM/utils                utils

git add .
git commit -m "ARIA-LLM Gradio demo"
git push
```

`app.py` imports from `model/`, `tokenizer/`, and `utils/`, so those folders
must be present in the Space.

---

## 4. Point the Space at your weights

Space UI: **Settings → Variables and secrets → New variable**

```
HF_MODEL_REPO = <user>/aria-llm-weights
```

The Space rebuilds, downloads `aria.pt` + `aria_vocab.json` on first load, and
goes live at:

```
https://huggingface.co/spaces/<user>/aria-llm
```

That URL is your resume link.

---

## Notes / troubleshooting

- **First response is slow**: the Space downloads weights on first request and
  CPU generation isn't fast. Later replies are quicker (model stays in RAM).
- **Quality**: ARIA is a small from-scratch model — expect short, casual
  replies, not GPT-level answers. That's the honest tradeoff of a genuinely
  own-built model. More/better distillation data (raise `--num`) and more
  training epochs improve it.
- If the model repo is private and the Space can't read it, either make the
  model repo public or add your HF token as a Space secret named `HF_TOKEN`.
