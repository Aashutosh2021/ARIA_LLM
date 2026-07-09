# Call Graph

Standard execution call paths.

## 1. Model Checkpoint Loading (`chat.py` startup)

This diagram shows how checkpoints are resolved and loaded during script boot:

```mermaid
graph TD
    Main[chat.py: main()] --> LoadVocab[helper.load_tokenizer: reads BPE vocab.json]
    Main --> GPTInit[GPT.from_config: builds handwritten classes from config]
    Main --> LoadCheckpoint[checkpoint.load_checkpoint: loads weights into model]
    LoadCheckpoint --> EvalMode[model.eval: sets evaluation mode]
```

## 2. Autoregressive Generation Step Loop

This diagram maps the method execution path of a single token generation step in `chat_qwen.py`:

```mermaid
graph TD
    GenerateCall[chat.py: generate_ids()] --> CropContext[Crop input_ids to context window]
    CropContext --> ModelCall[GPT.forward()]
    ModelCall --> BlockLoop[TransformerBlock.forward loop]
    BlockLoop --> MHA[MultiHeadAttention.forward]
    MHA --> ApplyRoPE[Apply Rotary Position Embedding]
    ApplyRoPE --> ScaledDot[ScaledDotProductAttention.forward]
    ScaledDot --> FFN[FeedForward.forward SwiGLU]
    FFN --> OutputLogits[GPT.lm_head Linear Proj]
    OutputLogits --> Sampler[inference/sampler.py: sample_next_token with Temperature & Top-p]
```
