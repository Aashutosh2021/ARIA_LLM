# Impact Map

A guide outlining the downstream dependencies and impact radius when specific files are modified.

## Change Impact Table

| Component File | Potential Modification | Impact Radius | Downstream Files to Check / Retrain |
|---|---|---|---|
| [model/gpt.py](file:///d:/Personal%20Project/ARIA-LLM/model/gpt.py) | Adding new layer arguments | High | [chat_qwen.py](file:///d:/Personal%20Project/ARIA-LLM/chat_qwen.py), [train.py](file:///d:/Personal%20Project/ARIA-LLM/train.py), [scripts/import_qwen.py](file:///d:/Personal%20Project/ARIA-LLM/scripts/import_qwen.py) |
| [model/rope.py](file:///d:/Personal%20Project/ARIA-LLM/model/rope.py) | Modifying theta angle calculations | Medium | [model/multi_head_attention.py](file:///d:/Personal%20Project/ARIA-LLM/model/multi_head_attention.py), [model/gpt.py](file:///d:/Personal%20Project/ARIA-LLM/model/gpt.py) |
| [model/attention.py](file:///d:/Personal%20Project/ARIA-LLM/model/attention.py) | Changing attention calculation math | High | [model/multi_head_attention.py](file:///d:/Personal%20Project/ARIA-LLM/model/multi_head_attention.py), all model checks. Requires retraining model weights. |
| [dataset/cleaner.py](file:///d:/Personal%20Project/ARIA-LLM/dataset/cleaner.py) | Adding regex cleaning rules | High | **Requires retraining.** Changes tokenization outcomes and vocab sizes. |
| [tokenizer/bpe_tokenizer.py](file:///d:/Personal%20Project/ARIA-LLM/tokenizer/bpe_tokenizer.py) | Merging modifications | High | **Requires retraining.** Modifies BPE vocab output hashes. |
| [inference/sampler.py](file:///d:/Personal%20Project/ARIA-LLM/inference/sampler.py) | Adding scaling criteria | Low | [inference/generator.py](file:///d:/Personal%20Project/ARIA-LLM/inference/generator.py), [chat.py](file:///d:/Personal%20Project/ARIA-LLM/chat.py) |
| [configs/model.yaml](file:///d:/Personal%20Project/ARIA-LLM/configs/model.yaml) | Modifying layer size configs | High | All model training loops. Invalidates older checkpoint compatibility. |
