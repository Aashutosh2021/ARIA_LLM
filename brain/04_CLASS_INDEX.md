# Class Index

The following classes are implemented in the ARIA-LLM project.

## Model Classes (`model/`)

- **`GPT`** ([model/gpt.py](file:///d:/Personal%20Project/ARIA-LLM/model/gpt.py)): The top-level PyTorch module. Combines token embeddings, rotary positional embeddings (or absolute positional embeddings), transformer blocks, layer norm, and the language modeling linear projection head.
- **`TransformerBlock`** ([model/transformer_block.py](file:///d:/Personal%20Project/ARIA-LLM/model/transformer_block.py)): Houses attention (MHA/GQA) and feedforward blocks with residual connections and normalizations.
- **`MultiHeadAttention`** ([model/multi_head_attention.py](file:///d:/Personal%20Project/ARIA-LLM/model/multi_head_attention.py)): Orchestrates self-attention, projection of Q/K/V tensors, RoPE applications, and key/value cached calculations.
- **`FeedForward`** ([model/feed_forward.py](file:///d:/Personal%20Project/ARIA-LLM/model/feed_forward.py)): Handles intermediate projections. Supports standard linear layers with activations, or SwiGLU gated linear layers.
- **`RMSNorm`** ([model/rmsnorm.py](file:///d:/Personal%20Project/ARIA-LLM/model/rmsnorm.py)): Modern RMS normalization layer.
- **`LayerNorm`** ([model/layer_norm.py](file:///d:/Personal%20Project/ARIA-LLM/model/layer_norm.py)): Custom fallback layer normalization.
- **`TokenEmbedding`** ([model/embedding.py](file:///d:/Personal%20Project/ARIA-LLM/model/embedding.py)): Wraps token-to-vector embedding.
- **`PositionalEncoding`** ([model/position.py](file:///d:/Personal%20Project/ARIA-LLM/model/position.py)): Classical learned absolute positional encoding parameters.
- **`ScaledDotProductAttention`** ([model/attention.py](file:///d:/Personal%20Project/ARIA-LLM/model/attention.py)): Computes attention query-key products, masks, softmax, and value scalings.

## Preprocessing & Data (`dataset/`)

- **`DatasetCleaner`** ([dataset/cleaner.py](file:///d:/Personal%20Project/ARIA-LLM/dataset/cleaner.py)): Main cleaning object with horizontal space preserving overrides.
- **`DatasetLoader`** ([dataset/loader.py](file:///d:/Personal%20Project/ARIA-LLM/dataset/loader.py)): Handles txt, csv, json, and jsonl parsing.
- **`DatasetPreprocessor`** ([dataset/preprocess.py](file:///d:/Personal%20Project/ARIA-LLM/dataset/preprocess.py)): Fit, tokenization sequence construction, and strided training window maker.
- **`LLMDataset`** ([dataset/dataset.py](file:///d:/Personal%20Project/ARIA-LLM/dataset/dataset.py)): Standard PyTorch Dataset wrapping inputs and targets.

## Tokenizers (`tokenizer/`)

- **`BaseTokenizer`** ([tokenizer/base_tokenizer.py](file:///d:/Personal%20Project/ARIA-LLM/tokenizer/base_tokenizer.py)): Interface definition for BPE, Char, and Word tokenizers.
- **`BPETokenizer`** ([tokenizer/bpe_tokenizer.py](file:///d:/Personal%20Project/ARIA-LLM/tokenizer/bpe_tokenizer.py)): Byte-level BPE tokenizer training and encoding engine.
- **`CharTokenizer`** ([tokenizer/char_tokenizer.py](file:///d:/Personal%20Project/ARIA-LLM/tokenizer/char_tokenizer.py)): Character level encoder.
- **`WordTokenizer`** ([tokenizer/word_tokenizer.py](file:///d:/Personal%20Project/ARIA-LLM/tokenizer/word_tokenizer.py)): Simple word level whitespace tokenization fallback.
- **`Vocabulary`** ([tokenizer/vocab.py](file:///d:/Personal%20Project/ARIA-LLM/tokenizer/vocab.py)): Dict vocabulary mapping.
- **`SpecialTokens`** ([tokenizer/special_tokens.py](file:///d:/Personal%20Project/ARIA-LLM/tokenizer/special_tokens.py)): Helper structure holds standard PAD, UNK, BOS, EOS, and MASK variables.

## Inference (`inference/`)

- **`TextGenerator`** ([inference/generator.py](file:///d:/Personal%20Project/ARIA-LLM/inference/generator.py)): Main autoregressive decoding wrapper.

## Training (`training/`)

- **`Trainer`** ([training/trainer.py](file:///d:/Personal%20Project/ARIA-LLM/training/trainer.py)): High level executor coordinating inputs feeding, optimizer backward steps, and model evaluations.
- **`LanguageModelLoss`** ([training/loss.py](file:///d:/Personal%20Project/ARIA-LLM/training/loss.py)): Custom wrapper on PyTorch CrossEntropyLoss.
- **`TrainingLogger`** ([training/logger.py](file:///d:/Personal%20Project/ARIA-LLM/training/logger.py)): Log output controller.

## Utilities (`utils/`)

- **`AverageMeter`** ([utils/metrics.py](file:///d:/Personal%20Project/ARIA-LLM/utils/metrics.py)): Simple metric tracker.
- **`ChatSettings`** ([chat.py](file:///d:/Personal%20Project/ARIA-LLM/chat.py)): Keeps track of command line options like temp, topk, topp, and max tokens.
