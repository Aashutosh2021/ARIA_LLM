# Changelog

All notable changes to the ARIA-LLM project.

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
