# External Services

ARIA-LLM operates entirely locally with minimal external connections. 

## External Dependencies

### 1. Hugging Face Hub (via `transformers` and `huggingface_hub`)
- **Purpose:** Used in `scripts/import_qwen.py` to download the pretrained weights (`pytorch_model.bin` or safetensors) and tokenization properties of the **`Qwen/Qwen2.5-0.5B-Instruct`** model.
- **Access Protocol:** standard HTTPS API connections.
- **Authentication:** Public access; does not require Hugging Face Auth Tokens (`HF_TOKEN`).

### 2. KaggleHub (via `kagglehub` package)
- **Purpose:** Used in `import_datase.py` (which download dataset `kreeshrajani/3k-conversations-dataset-for-chatbot`) to pull external training assets.
- **Access Protocol:** Standard HTTPS API connections.
