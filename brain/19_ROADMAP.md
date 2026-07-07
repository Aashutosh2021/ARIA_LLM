# Project Roadmap

Future extension pathways for the ARIA-LLM workspace.

## Near-Term Objectives (Next Steps)
- **Interactive UI Chat Application:** Create a modern, high-performance local web app chat interface (using Next.js or Vite) to replace the simple console terminal REPL.
- **Improved BPE Tokenization:** Speed up BPE tokenizer training loops using multi-threading or Rust binding hooks.
- **Validation Pipeline:** Enhance unit tests coverage for Rotary positional encoding math and KV caches.

## Long-Term Vision
- **Local Conversational Finetuning:** Develop direct Supervised Fine-Tuning SFT code to teach the small handwritten 4.7M model to hold more complex, custom conversations using custom system prompts.
- **RLHF (Reinforcement Learning from Human Feedback):** Implement standard DPO (Direct Preference Optimization) training scripts for the custom model.
- **Larger Model Imports:** Optimize GQA weight loading configurations to test loading Qwen2.5-1.5B or Llama-3-8B weights into ARIA-LLM.
- **Quantization Support:** Integrate int8/int4 quantization calculations to allow running larger parameters on low-end local GPUs.
