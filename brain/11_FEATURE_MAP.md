# Feature Map

The core user features provided by ARIA-LLM.

## Feature Overview

```mermaid
graph TD
    User([User Objective]) --> Prep[1. Data Generation & Preprocessing]
    User --> Train[2. Model Training]
    User --> ChatLocal[3. Chat with Local trained Model]
    User --> ChatQwen[4. Chat with Converted Qwen Model]

    %% Prep Subgraph
    subgraph Feature 1
        Prep --> GenConversations[prepare_conversations.py: generates synthetic dialogs]
        Prep --> CleanText[cleaner.py: strips HTML/URLs, preserves newlines]
        Prep --> TrainTokenizer[bpe_tokenizer.py: trains BPE vocabulary]
    end

    %% Train Subgraph
    subgraph Feature 2
        Train --> CustomTrainer[trainer.py: runs training epoch step loops]
        Train --> WarmupCosine[scheduler.py: scales learning rates]
        Train --> Checkpointing[checkpoint.py: saves checkpoints/best.pt]
    end

    %% Chat Local Subgraph
    subgraph Feature 3
        ChatLocal --> LocalREPL[chat.py: interactive dialogue loop]
        LocalREPL --> LocalInference[generator.py: autoregressive token decoding]
    end

    %% Chat Qwen Subgraph
    subgraph Feature 4
        ChatQwen --> QwenConvert[import_qwen.py: downloads & translates Qwen weights]
        ChatQwen --> QwenREPL[chat_qwen.py: streams responses via hand-written layers]
    end
```

## Features Details

### 1. Preprocessing & Custom Tokenization
- **Cleaner Pipeline:** Clean raw text with custom regulations. Specifically ignores newline removals when in conversational training mode.
- **Custom BPE Tokenizer:** Learns BPE merges from raw text corpora and translates strings to integer token lists.

### 2. PyTorch Training Loop
- **AdamW Optimization:** Standard training optimization.
- **Cosine Warmup:** Linear learning rate warmup followed by a cosine decay cycle.
- **Gradient Clipping:** Prevents exploding gradients during training.

### 3. Advanced Sampling Inference
- **Autoregressive Decoding:** Generates tokens step-by-step.
- **Nucleus (Top-p) & Top-k Sampling:** Prunes low probability outputs to balance vocabulary creativity and coherence.
- **Temperature Scaling:** Standard scaling for logits.
- **Repetition Penalty:** Penalizes already produced tokens to avoid generation loops.

### 4. Qwen-2.5 Weight Importer
- **State Dictionary Mapping:** Maps Hugging Face's Qwen2.5 weights schema to our custom handwritten transformer blocks.
- **Grouped-Query Attention Support:** Handles caching and RoPE transformations.
