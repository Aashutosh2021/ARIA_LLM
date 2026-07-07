# AI Agent Guidelines (PROMPT.md)

Welcome to ARIA-LLM. You are an agentic coder operating inside this repository. You must adhere to the following rules at all times.

## 1. Boot Sequence (Mandatory Reading Order)
Before performing any research or modifications, you must read:
1. This file: [PROMPT.md](file:///d:/Personal%20Project/ARIA-LLM/brain/PROMPT.md)
2. [21_AGENT_BOOT.md](file:///d:/Personal%20Project/ARIA-LLM/brain/21_AGENT_BOOT.md)
3. [20_AI_CONTEXT.md](file:///d:/Personal%20Project/ARIA-LLM/brain/20_AI_CONTEXT.md)
4. [17_PROJECT_RULES.md](file:///d:/Personal%20Project/ARIA-LLM/brain/17_PROJECT_RULES.md)
5. [18_CURRENT_STATE.md](file:///d:/Personal%20Project/ARIA-LLM/brain/18_CURRENT_STATE.md)
6. [03_FILE_INDEX.md](file:///d:/Personal%20Project/ARIA-LLM/brain/03_FILE_INDEX.md)

Do not run full recursive directory scans or open files not linked to your task.

## 2. Architecture & Implementation Rules
- **Handwritten Modules:** All model layers must remain handwritten in PyTorch. Do not replace them with Hugging Face module wrappers.
- **Maintain Compatibility:** Ensure that standard pre-training/finetuning (`train.py`) and converted weight loading (`chat_qwen.py`) interfaces continue to function correctly.
- **Precision:** Use fp16 casting (`model.half()`) on CUDA devices for inference. Ensure attention masks are dtype-safe.

## 3. Token & Scan Optimization
- Do not read lines of files that are unrelated to the current task.
- Target changes to the minimum number of lines possible.

## 4. Documentation Update Rules
Whenever any code change is made:
1. Open [VERSION.json](file:///d:/Personal%20Project/ARIA-LLM/brain/VERSION.json).
2. Increment `documentation_version`.
3. Update `last_brain_update` to the current date.
4. Add the list of modified source files to `modified_source_files`.
5. Update `codebase_hash` with the output of the codebase scanner.
6. Set `brain_status` to `"Healthy"` once all corresponding document updates are applied.
7. Modify only the documentation files affected by your changes.
