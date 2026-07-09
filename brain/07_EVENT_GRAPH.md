# Event Graph

Events and operational flows in ARIA-LLM.

## Interactive Chat Event Loop (`chat.py`)

This graph maps the event flow for a single user query execution in `chat.py`:

```mermaid
sequenceDiagram
    autonumber
    actor User as User Console
    participant Chat as chat.py
    participant Tok as BPETokenizer
    participant Model as GPT (model/gpt.py)
    participant Gen as generate_ids (chat.py)

    User->>Chat: Enters query
    Note over Chat: Formats conversation chat templates if chat_format is True
    Chat->>Tok: encode(prompt_text)
    Tok-->>Chat: returns input_ids
    Chat->>Gen: calls generate_ids(model, input_ids)
    
    loop Max New Tokens
        Gen->>Model: forward(context_ids)
        Model-->>Gen: returns next logits
        Note over Gen: Adjust logits (temperature, top-k, top-p, greedy)
        Gen->>Tok: decode(next_token_id)
        alt Stop string encountered
            Note over Gen: Breaks loop
        end
    end
    
    Gen-->>Chat: returns reply
    Chat-->>User: prints reply
    Note over Chat: Appends assistant reply to history
```

## Dataset Training Cycle (`train.py`)

This graph maps the event flow of training:

```mermaid
graph TD
    Start[train.py Boot] --> LoadData[dataset/loader.py: load raw text]
    LoadData --> PrepData[dataset/preprocess.py: clean and tokenize text]
    PrepData --> FitTok[Train Tokenizer if BPE]
    FitTok --> SaveVocab[Save vocab.json]
    SaveVocab --> SeqGen[Create inputs/targets sequences]
    SeqGen --> BuildDataloader[Create PyTorch DataLoader batches]
    BuildDataloader --> BuildModel[Instantiate GPT from configs]
    BuildModel --> SetupOptim[Build AdamW Optimizer & Cosine Scheduler]
    SetupOptim --> LoopEpochs[Epoch loop: 1 to N]
    
    subgraph Epoch Loop
        LoopEpochs --> TrainStep[Train Batch Forward / Backward]
        TrainStep --> ClipGrad[Gradient Clipping]
        ClipGrad --> StepOptim[Optimizer and Scheduler steps]
        StepOptim --> Evaluate[Evaluate Validation loss]
    end
    
    Evaluate -- val_loss < best_val_loss --> SaveBest[Save checkpoints/best.pt]
    LoopEpochs -- Completed N Epochs --> SaveLast[Save checkpoints/last.pt]
```
