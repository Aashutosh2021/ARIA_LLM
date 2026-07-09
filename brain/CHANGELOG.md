# Changelog

All notable changes to the ARIA-LLM project.

## [1.2.0] - 2026-07-09
### Added
- Knowledge-distillation data pipeline: `scripts/distill_generate.py` uses Qwen2.5-0.5B **only as a teacher that writes training text** (Qwen weights never enter ARIA). `distill_aria_colab.ipynb` drives generate -> train-from-scratch -> export.
- `app.py` (Gradio) serves the from-scratch ARIA for a public demo; `scripts/export_for_deploy.py` bundles weights + tokenizer.
### Note
- ARIA remains 100% from-scratch **at the weight level**. Qwen is used once, offline, as a data generator only — no third-party weights are loaded at train or inference time.

## [1.1.0] - 2026-07-09
### Removed
- Removed Qwen **weight-transplant** path and its scripts (`chat_qwen.py`, `train_qwen_finetune.py`, `scripts/import_qwen.py`, `finetune_qwen_colab.ipynb`, `chat_ai.py`) — ARIA no longer loads any Qwen weights into its own model.
- Removed Qwen checkpoints (`checkpoints/qwen2.5-0.5b-instruct.pt`, `checkpoints_chat_qwen/`).

## [1.0.0] - 2026-07-07
### Added
- Created the permanent AI Brain documentation registry (`brain/` directory and registry files).
- Auto-scanning python helper scripts for codebase integrity and stats registry verification.

## [0.2.0] - 2026-07-05
### Added
- Created Qwen weight import utility (`scripts/import_qwen.py`) to convert and map standard Hugging Face model keys to custom handwritten PyTorch modules.
- Created `chat_qwen.py` supporting real Qwen2.5 weights streaming inference locally.
- Formatted `data/dialogs.txt` conversational dataset using the `scripts/format_dialogs.py` utility.

### Fixed
- Fixed BPE preprocessing bug in `dataset/cleaner.py` that removed all conversational newlines, flattening conversational inputs. Preserved proper Turn-based Dialogue structure.
- Retrained model on 80,000 conversational templates, correcting generation looping issues.
