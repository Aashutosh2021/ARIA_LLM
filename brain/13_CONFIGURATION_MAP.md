# Configuration Map

The configuration of ARIA-LLM is declared across three YAML files in the `configs/` directory.

## 1. Model Configuration (`configs/model.yaml`)

Contains default architecture properties for initializing `GPT` models from scratch:

- **`model_name`**: Name identifier for the run. Default: `AIRA-LLM-v0.1`.
- **`vocab_size`**: Vocabulary capacity. Default: `30000`.
- **`max_sequence_length`**: Maximum context length (window size). Default: `256`.
- **`embedding_dim`**: Vector size of embeddings. Default: `256`.
- **`hidden_size`**: Size of intermediate attention hidden dimensions. Default: `256`.
- **`num_layers`**: Number of transformer layers. Default: `6`.
- **`num_heads`**: Number of attention heads. Default: `8`.
- **`dropout`**: Dropout regularization probability. Default: `0.1`.
- **`ffn_multiplier`**: MLP expansion factor. Default: `4`.
- **`bias`**: Use bias on linear projections. Default: `false`.

---

## 2. Dataset Configuration (`configs/dataset.yaml`)

Contains configurations for loading text corpora:

- **`dataset_name`**: Data run metadata tag. Default: `TinyStories`.
- **`dataset_path`**: Storage path. Default: `datasets/`.
- **`shuffle`**: Randomize batches. Default: `true`.
- **`train_split`**: Ratio of data used for optimization. Default: `0.9`.
- **`validation_split`**: Ratio used for evaluation. Default: `0.1`.
- **`num_workers`**: Multiprocessing dataloader count. Default: `4`.

---

## 3. Training Configuration (`configs/training.yaml`)

Contains hyperparameters for the `Trainer` loops:

- **`batch_size`**: Multi-sequence batch size. Default: `16`.
- **`epochs`**: Total dataset passes. Default: `20`.
- **`learning_rate`**: Peak optimization rate. Default: `0.0003`.
- **`weight_decay`**: Regularization penalty scale. Default: `0.01`.
- **`warmup_steps`**: Steps to linearly scale lr to peak. Default: `1000`.
- **`gradient_clip`**: Normalization boundary for gradients. Default: `1.0`.
- **`checkpoint_every`**: Step frequency to save checkpoints. Default: `1000`.
- **`save_path`**: Destination checkpoints path. Default: `checkpoints/`.
