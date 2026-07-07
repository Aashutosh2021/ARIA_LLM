# Project Rules

The following core guidelines must be followed by all AI agents and developers modifying this codebase.

## 1. Architectural Rules
- **From-Scratch Core:** Do not replace handwritten module classes (like MultiHeadAttention, GPT, RMSNorm) with third-party wrappers (like Hugging Face `transformers` modules). The core of this project is custom PyTorch layers.
- **Maintain Configuration Alignment:** Checkpoint configs must strictly align with the initialization configurations specified in `configs/model.yaml`.
- **Weight Tying:** The embedding weights and LM projection head weights must be tied (`lm_head.weight = token_embedding.embedding.weight`) by default, except when overridden (e.g. during specific imported weight loading).

## 2. Coding Rules
- **Comment and Docstring Preservation:** Always preserve existing comments and docstrings unless explicitly asked to modify them.
- **No Unused Imports:** Ensure PyRefly ignore comments (`# pyrefly: ignore [missing-import]`) are retained for external libraries inside environment configurations.
- **Console Warnings:** Ensure regex expressions are formatted as raw strings `r"..."` to prevent Python console warnings (`SyntaxWarning: invalid escape sequence`).

## 3. Maintenance and Synchronization Rules
- **No Unrelated Code Refactors:** Never rewrite or touch functions outside of the direct target implementation areas.
- **Keep the Brain Synced:** Every change in source code files requires modifying `brain/VERSION.json` to increment `documentation_version` and record the changed files, as well as updating the relevant documentation map files (e.g. `03_FILE_INDEX.md` or `18_CURRENT_STATE.md`).
- **No Full codebase Scanning:** Avoid listing the complete directory recursively in future sessions to conserve tokens. Read the Brain index first.
