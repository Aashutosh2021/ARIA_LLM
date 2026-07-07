import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM

def convert_qwen():
    print("Downloading/Loading Qwen2.5-0.5B-Instruct via transformers...")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32)
    state_dict = model.state_dict()
    
    new_state_dict = {}
    
    print("Converting weights...")
    for k, v in state_dict.items():
        if k == "model.embed_tokens.weight":
            new_state_dict["token_embedding.embedding.weight"] = v
        elif k == "model.norm.weight":
            new_state_dict["ln_f.weight"] = v
        elif k == "lm_head.weight":
            new_state_dict["lm_head.weight"] = v
        elif k.startswith("model.layers."):
            parts = k.split(".")
            layer_idx = parts[2]
            sub = ".".join(parts[3:])
            
            if sub == "input_layernorm.weight":
                new_state_dict[f"layers.{layer_idx}.norm1.weight"] = v
            elif sub == "post_attention_layernorm.weight":
                new_state_dict[f"layers.{layer_idx}.norm2.weight"] = v
            elif sub == "self_attn.q_proj.weight":
                new_state_dict[f"layers.{layer_idx}.attention.q_proj.weight"] = v
            elif sub == "self_attn.k_proj.weight":
                new_state_dict[f"layers.{layer_idx}.attention.k_proj.weight"] = v
            elif sub == "self_attn.v_proj.weight":
                new_state_dict[f"layers.{layer_idx}.attention.v_proj.weight"] = v
            elif sub == "self_attn.q_proj.bias":
                new_state_dict[f"layers.{layer_idx}.attention.q_proj.bias"] = v
            elif sub == "self_attn.k_proj.bias":
                new_state_dict[f"layers.{layer_idx}.attention.k_proj.bias"] = v
            elif sub == "self_attn.v_proj.bias":
                new_state_dict[f"layers.{layer_idx}.attention.v_proj.bias"] = v
            elif sub == "self_attn.o_proj.weight":
                new_state_dict[f"layers.{layer_idx}.attention.proj.weight"] = v
            elif sub == "mlp.gate_proj.weight":
                new_state_dict[f"layers.{layer_idx}.ffn.gate_proj.weight"] = v
            elif sub == "mlp.up_proj.weight":
                new_state_dict[f"layers.{layer_idx}.ffn.up_proj.weight"] = v
            elif sub == "mlp.down_proj.weight":
                new_state_dict[f"layers.{layer_idx}.ffn.down_proj.weight"] = v
            else:
                print(f"Warning: unused key {k}")
        else:
            print(f"Warning: unused key {k}")
            
    # Qwen2.5-0.5B specific config
    config = {
        "vocab_size": 151936,
        "max_sequence_length": 4096,
        "embedding_dim": 896,
        "num_layers": 24,
        "num_heads": 14,
        "num_kv_heads": 2,
        "hidden_dim": 4864,
        "dropout": 0.0,
        "bias": True,
        "tie_weights": True,
        "use_rmsnorm": True,
        "use_swiglu": True,
        "use_rope": True,
        "rope_base": 1000000.0,
        "rms_norm_eps": 1e-6,
    }
    
    os.makedirs("checkpoints", exist_ok=True)
    out_path = "checkpoints/qwen2.5-0.5b-instruct.pt"
    print(f"Saving converted weights to {out_path}...")
    torch.save({
        "config": config,
        "model_state_dict": new_state_dict
    }, out_path)
    
    print("Done!")

if __name__ == "__main__":
    convert_qwen()

