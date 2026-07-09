# Feature Dependency Matrix

This matrix maps core user features to implementation files in the repository.

| Feature | `model/` | `dataset/` | `tokenizer/` | `inference/` | `training/` | `scripts/` | Main Files |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Conversational Data Setup** | | X | X | | | X | `prepare_conversations.py`, `cleaner.py` |
| **BPE Tokenization** | | | X | | | | `bpe_tokenizer.py`, `vocab.py` |
| **GPT Model Optimization** | X | | | | X | | `train.py`, `trainer.py`, `gpt.py` |
| **Autoregressive Sampling** | X | | | X | | | `generator.py`, `sampler.py` |
| **Streaming Chat UI** | X | | | X | | | `chat_qwen.py`, `chat.py` |

## Mapping Summary
- **Model Training:** Tight coupling between `model/gpt.py`, `training/trainer.py`, `dataset/preprocess.py`, and `tokenizer/bpe_tokenizer.py`.
- **Inference Pipeline:** Autoregressive streaming loops run inside `chat_qwen.py` utilizing the `inference/` folder and `model/gpt.py` evaluations.
