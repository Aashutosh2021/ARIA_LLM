# Call Graph

Standard execution call paths.

## 1. Converted Weights Loading (`chat_qwen.py` startup)

This diagram shows how weights are resolved, mapped, and loaded during script boot:

```mermaid
graph TD
    Main[chat_qwen.py: main()] --> LoadModel[chat_qwen.py: load_model()]
    LoadModel --> TorchLoad[torch.load: load state dict from pt]
    TorchLoad --> GPTInit[GPT.from_config: builds handwritten classes]
    GPTInit --> LoadStateDict[model.load_state_dict: sets strict=False]
    LoadStateDict --> HalfPrecision[model.half: casts to float16 if GPU]
    HalfPrecision --> EvalMode[model.eval: sets evaluation mode]
```

## 2. Autoregressive Generation Step Loop

This diagram maps the method execution path of a single token generation step in `chat_qwen.py`:

```mermaid
graph TD
    GenerateCall[chat_qwen.py: generate()] --> CropContext[Crop input_ids to context window]
    CropContext --> ModelCall[GPT.forward()]
    ModelCall --> BlockLoop[TransformerBlock.forward loop]
    BlockLoop --> MHA[MultiHeadAttention.forward]
    MHA --> ApplyRoPE[Apply Rotary Position Embedding]
    ApplyRoPE --> ScaledDot[ScaledDotProductAttention.forward]
    ScaledDot --> FFN[FeedForward.forward SwiGLU]
    FFN --> OutputLogits[GPT.lm_head Linear Proj]
    OutputLogits --> Sampler[chat_qwen.py: multinomial sampling with Temperature & Top-p]
```
