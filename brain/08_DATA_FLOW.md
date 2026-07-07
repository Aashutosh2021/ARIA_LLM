# Data Flow

Mapping of data inputs to tensor outputs throughout the ARIA-LLM codebase.

## Preprocessing Data Flow

```mermaid
graph LR
    RawText[Raw Text Dataset File] --> Loader[dataset/loader.py]
    Loader --> Cleaner[dataset/cleaner.py: Clean Text]
    Cleaner --> Tokenizer[tokenizer/bpe_tokenizer.py: Encode]
    Tokenizer --> TokenIDs[Flat List of Token IDs]
    TokenIDs --> StridedWindows[dataset/preprocess.py: create_sequences]
    StridedWindows --> InputsY[Inputs X: seq_len / Targets Y: seq_len shifted by 1]
    InputsY --> DataLoader[PyTorch DataLoaders: Batch size B, Sequence length S]
```

## Model Forward Pass Tensor Shape Transformations

Here is the flow of shapes during the `GPT.forward()` call:

1. **Input Batch:**
   - Tensor size: `(B, S)` where `B` is batch size and `S` is sequence length.
2. **Embedding layer (`model/embedding.py`):**
   - Maps `(B, S)` to embedding output shape `(B, S, E)` where `E` is `embedding_dim`.
3. **Positional Encoding:**
   - Absolute positional encoding adds shape `(1, S, E)` to `(B, S, E)` yielding `(B, S, E)`.
   - If RoPE is used, precomputed complex frequencies `freqs_cis` of shape `(S, head_dim // 2)` are applied inside self-attention.
4. **Transformer Layer block (`model/transformer_block.py`):**
   - Normalizes input using RMSNorm / LayerNorm: `(B, S, E)`.
   - Passes to `MultiHeadAttention` which projects Q, K, V matrices:
     - Shape of `xq`, `xk`, `xv`: `(B, S, num_heads, head_dim)` where `head_dim = E // num_heads`.
     - Key/Value Cache stores shapes `(B, max_seq_len, num_kv_heads, head_dim)`.
     - Scaled dot product attention yields shape `(B, num_heads, S, S)`.
     - Output projection maps back to shape `(B, S, E)`.
   - Adds residual connection.
   - Passes through `FeedForward` layer:
     - MLP projects `(B, S, E)` to intermediate `(B, S, hidden_dim)`, applies activation, and projects back to `(B, S, E)`.
     - SwiGLU projects to `(B, S, hidden_dim)` via `w1` and `w3` gates, multiplies outputs element-wise, then projects back to `(B, S, E)` via `w2`.
   - Adds residual connection.
5. **Final Output Head:**
   - `ln_f` Normalization maps to `(B, S, E)`.
   - `lm_head` linear projection maps to `(B, S, V)` where `V` is `vocab_size`.
6. **Loss Calculation (`training/loss.py`):**
   - Transposes prediction logits to `(B * S, V)` and targets to `(B * S)`.
   - Calculates PyTorch Cross Entropy Loss.
