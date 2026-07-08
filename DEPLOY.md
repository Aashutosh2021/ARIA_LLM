# Deploying ARIA-LLM to Hugging Face Spaces

This puts a public chat demo online for free. Two pieces:

1. **A model repo** — holds the large weight files (`.pt`).
2. **A Space** — holds the app (`app.py`) and pulls the weights from the model
   repo at startup.

Keeping weights out of the Space repo keeps the Space small and lets you update
weights without redeploying the app.

---

## 0. One-time local prep

Produce the slim, deploy-ready weights (drops optimizer state; ~6 GB → ~1.4 GB):

```bash
# the working Qwen-weights model (ready now)
python scripts/export_for_deploy.py \
    --checkpoint checkpoints/qwen2.5-0.5b-instruct.pt \
    --out deploy/aria_qwen.pt

# the fine-tuned model (after you retrain it on Colab -- see Part E of the notebook)
python scripts/export_for_deploy.py \
    --checkpoint checkpoints_chat_qwen/best.pt \
    --out deploy/aria_finetuned.pt
```

Test the app locally before deploying (uses the local `deploy/` files):

```bash
pip install -r requirements-space.txt
python app.py
# open the printed http://127.0.0.1:7860 URL and chat
```

---

## 1. Create a Hugging Face account + login

- Sign up at https://huggingface.co/join
- Create an access token (write): https://huggingface.co/settings/tokens
- Locally:

```bash
pip install huggingface_hub
huggingface-cli login   # paste the write token
```

---

## 2. Create the model repo and upload weights

Pick a name; below uses `aria-llm-weights`. Replace `<user>` with your HF
username.

```bash
huggingface-cli repo create aria-llm-weights --type model

huggingface-cli upload <user>/aria-llm-weights deploy/aria_qwen.pt aria_qwen.pt
# after retraining:
huggingface-cli upload <user>/aria-llm-weights deploy/aria_finetuned.pt aria_finetuned.pt
```

The second argument is the local file; the third is the name it gets in the
repo (must match the filenames in `app.py`'s `MODEL_FILES`).

---

## 3. Create the Space

```bash
huggingface-cli repo create aria-llm --type space --space_sdk gradio
```

Then clone it and copy in the app files. **Note the renames** — the Space needs
the files named exactly `app.py`, `requirements.txt`, `README.md`:

```bash
git clone https://huggingface.co/spaces/<user>/aria-llm
cd aria-llm

# copy from this repo:
cp ../ARIA-LLM/app.py            app.py
cp ../ARIA-LLM/requirements-space.txt  requirements.txt
cp ../ARIA-LLM/README-space.md   README.md
cp -r ../ARIA-LLM/model          model
cp -r ../ARIA-LLM/utils          utils

git add .
git commit -m "ARIA-LLM Gradio demo"
git push
```

`app.py` imports from `model/` (and `utils/` indirectly), so those folders must
be present in the Space.

---

## 4. Point the Space at your weights

In the Space UI: **Settings → Variables and secrets → New variable**

```
HF_MODEL_REPO = <user>/aria-llm-weights
```

The Space will rebuild, download the weights on first load, and go live at:

```
https://huggingface.co/spaces/<user>/aria-llm
```

That URL is your resume link.

---

## Notes / troubleshooting

- **Only one model shows in the toggle** until you've uploaded
  `aria_finetuned.pt`. That's expected — the Qwen-weights model works
  standalone.
- **First response is slow**: the Space downloads weights on first request and
  CPU generation isn't fast. Subsequent replies are quicker (model stays cached
  in RAM).
- **Out of memory on the free tier**: the free CPU Space has 16 GB RAM; two
  fp32 358M models fit, but they load lazily (only when selected), so you won't
  hold both at once unless a visitor switches. If you ever hit limits, remove
  one model from `MODEL_FILES`.
- If the private model repo can't be read by the Space, either make the model
  repo public or add your HF token as a Space secret named `HF_TOKEN`.
