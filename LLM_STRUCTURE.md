```
LLM/
├── app
├── configs
│   ├── dataset.yaml
│   ├── model.yaml
│   └── training.yaml
├── dataset
│   ├── cleaner.py
│   ├── dataset.py
│   ├── loader.py
│   └── preprocess.py
├── inference
│   ├── beam_search.py
│   ├── generator.py
│   ├── sampler.py
│   ├── temperature.py
│   ├── topk.py
│   └── topp.py
├── model
│   ├── attention.py
│   ├── embedding.py
│   ├── feed_forward.py
│   ├── gpt.py
│   ├── layer_norm.py
│   ├── multi_head_attention.py
│   ├── position.py
│   └── transformer_block.py
├── tests
│   ├── test_attention.py
│   ├── test_embedding.py
│   └── test_tokenizer.py
├── tokenizer
│   ├── base_tokenizer.py
│   ├── bpe_tokenizer.py
│   ├── char_tokenizer.py
│   ├── special_tokens.py
│   ├── vocab.py
│   └── word_tokenizer.py
├── training
│   ├── checkpoint.py
│   ├── logger.py
│   ├── loss.py
│   ├── optimizer.py
│   ├── scheduler.py
│   └── trainer.py
└── utils
    ├── device.py
    ├── helper.py
    ├── metrics.py
    └── seed.py
```
