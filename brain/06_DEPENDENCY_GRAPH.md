# Dependency Graph

Component and module relationship maps.

## Component Flow Overview

```mermaid
graph TD
    %% Directories/Modules
    subgraph Configs [configs/]
        ModelYAML[model.yaml]
        TrainYAML[training.yaml]
        DataYAML[dataset.yaml]
    end

    subgraph DataPrep [dataset/ & tokenizer/]
        Cleaner[cleaner.py]
        Preprocessor[preprocess.py]
        Loader[loader.py]
        Tokenizer[bpe_tokenizer.py]
    end

    subgraph CoreModel [model/]
        GPT[gpt.py]
        TransformerBlock[transformer_block.py]
        MHA[multi_head_attention.py]
        RoPE[rope.py]
        RMSNorm[rmsnorm.py]
        FFN[feed_forward.py]
    end

    subgraph TrainingLoop [training/]
        Trainer[trainer.py]
        Loss[loss.py]
        Scheduler[scheduler.py]
        Optimizer[optimizer.py]
        Checkpoint[checkpoint.py]
    end

    subgraph InferenceLayer [inference/]
        Generator[generator.py]
        Sampler[sampler.py]
    end

    %% Dependencies relationships
    Configs -. Loads Configurations .-> train.py & chat_qwen.py
    Loader --> Preprocessor
    Tokenizer --> Preprocessor
    Cleaner --> Preprocessor
    
    Preprocessor --> train.py
    
    GPT --> TransformerBlock
    TransformerBlock --> MHA & FFN & RMSNorm
    MHA --> RoPE
    
    train.py --> Trainer
    Trainer --> GPT
    Trainer --> Loss & Optimizer & Scheduler & Checkpoint
    
    chat_qwen.py --> GPT
    chat_qwen.py --> InferenceLayer
    Generator --> Sampler
```

## Description of Interactions
1. **Configuration Loader:** `utils/helper.py` reads YAML config files to set parameters for `GPT` class initialization and `Trainer` training constraints.
2. **Preprocessing Pipeline:** `dataset/preprocess.py` combines text cleaner (`dataset/cleaner.py`) and vocab tokenizers (`tokenizer/bpe_tokenizer.py`) to prepare dataset loader sequences.
3. **Training Execution:** `train.py` loads raw text datasets, configures the `GPT` model, and boots `Trainer` to run gradients minimization loops.
4. **Interactive Chat:** `chat_qwen.py` loads Qwen2.5 weights, passes user queries to `GPT.forward()` through the `inference/generator.py` autoregressive generator, streaming decoded tokens back.
