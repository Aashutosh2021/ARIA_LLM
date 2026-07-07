# Method Index

A registry of key functions and methods across the ARIA-LLM modules.

## Architecture & Model Forward (`model/`)

- **`GPT.from_config(config: dict)`** ([model/gpt.py](file:///d:/Personal%20Project/ARIA-LLM/model/gpt.py))
  Factory method that instantiates the model parameters based on key config overrides.
- **`GPT.forward(input_ids, mask=None)`** ([model/gpt.py](file:///d:/Personal%20Project/ARIA-LLM/model/gpt.py))
  Processes token ids, applies positional encoding (or precomputed RoPE), passes them through layer block modules, and outputs predictions logits.
- **`TransformerBlock.forward(x, freqs_cis=None, mask=None)`** ([model/transformer_block.py](file:///d:/Personal%20Project/ARIA-LLM/model/transformer_block.py))
  Applies attention, adds residual connection, applies FFN/SwiGLU, and adds final residual.
- **`MultiHeadAttention.forward(x, freqs_cis=None, mask=None)`** ([model/multi_head_attention.py](file:///d:/Personal%20Project/ARIA-LLM/model/multi_head_attention.py))
  Linear projects queries, keys, and values. Applies RoPE if needed. Runs scaled dot product attention.
- **`precompute_freqs_cis(dim, end, theta)`** ([model/rope.py](file:///d:/Personal%20Project/ARIA-LLM/model/rope.py))
  Calculates rotary angle matrices.
- **`apply_rotary_emb(xq, xk, freqs_cis)`** ([model/rope.py](file:///d:/Personal%20Project/ARIA-LLM/model/rope.py))
  Applies computed rotational matrices to Q and K.
- **`init_gpt_weights(module, num_layers)`** ([model/init.py](file:///d:/Personal%20Project/ARIA-LLM/model/init.py))
  Performs parameter initialization.

## Preprocessing & Data Parsing (`dataset/`)

- **`DatasetLoader.load()`** ([dataset/loader.py](file:///d:/Personal%20Project/ARIA-LLM/dataset/loader.py))
  Dispatches to loading functions based on extension name.
- **`DatasetCleaner.clean(text: str)`** ([dataset/cleaner.py](file:///d:/Personal%20Project/ARIA-LLM/dataset/cleaner.py))
  Runs pipeline: lowercase, removes URLs, emails, HTML, extra spaces (excluding newlines), and extra newlines.
- **`DatasetPreprocessor.create_sequences(token_ids, sequence_length, stride)`** ([dataset/preprocess.py](file:///d:/Personal%20Project/ARIA-LLM/dataset/preprocess.py))
  Prepares strided input windows and next-token target labels.

## Tokenizers (`tokenizer/`)

- **`BPETokenizer.train(text: str)`** ([tokenizer/bpe_tokenizer.py](file:///d:/Personal%20Project/ARIA-LLM/tokenizer/bpe_tokenizer.py))
  Trains byte-pair mergers iteratively up to vocab target.
- **`BPETokenizer.encode(text: str)`** ([tokenizer/bpe_tokenizer.py](file:///d:/Personal%20Project/ARIA-LLM/tokenizer/bpe_tokenizer.py))
  Applies learned merges to encode text to IDs.
- **`BPETokenizer.decode(ids: list)`** ([tokenizer/bpe_tokenizer.py](file:///d:/Personal%20Project/ARIA-LLM/tokenizer/bpe_tokenizer.py))
  Converts token IDs back to string.

## Inference (`inference/`)

- **`TextGenerator.generate(prompt: str, max_new_tokens: int, ...)`** ([inference/generator.py](file:///d:/Personal%20Project/ARIA-LLM/inference/generator.py))
  Takes standard prompt, encodes it, calls `_generate_ids`, and decodes.
- **`sample_next_token(logits, temperature, top_k, top_p, greedy)`** ([inference/sampler.py](file:///d:/Personal%20Project/ARIA-LLM/inference/sampler.py))
  Applies various logit adjustments and samples.
- **`beam_search(model, ids, ...)`** ([inference/beam_search.py](file:///d:/Personal%20Project/ARIA-LLM/inference/beam_search.py))
  Performs beam search decoding.

## Optimization & Loops (`training/`, `utils/`)

- **`Trainer.train(train_loader, val_loader, epochs)`** ([training/trainer.py](file:///d:/Personal%20Project/ARIA-LLM/training/trainer.py))
  Runs the training epoch loop, triggers evaluation, loss backward calculations, and checkpoints.
- **`checkpoint.save_checkpoint(...)`** ([training/checkpoint.py](file:///d:/Personal%20Project/ARIA-LLM/training/checkpoint.py))
  Serializes PyTorch model weights state dictionaries, scheduler states, step info, and configurations.
- **`device.get_device(prefer)`** ([utils/device.py](file:///d:/Personal%20Project/ARIA-LLM/utils/device.py))
  Determines device target (`cuda`, `mps`, or `cpu`).
- **`helper.load_all_configs()`** ([utils/helper.py](file:///d:/Personal%20Project/ARIA-LLM/utils/helper.py))
  Parses config files (`dataset.yaml`, `model.yaml`, `training.yaml`).
