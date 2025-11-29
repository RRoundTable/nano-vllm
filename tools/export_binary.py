#!/usr/bin/env python3
"""
Export Hugging Face Llama model to raw binary format for C loading.

Usage:
    python export_binary.py --model <model_name> --output data/model.bin

Binary Format (all FP16, little-endian):
    [Header]
    - dim: int32
    - n_layers: int32
    - n_heads: int32
    - n_kv_heads: int32
    - vocab_size: int32
    - max_seq_len: int32
    - hidden_dim: int32
    - head_dim: int32 (derived: dim / n_heads)
    - rope_theta: float32
    
    [Weights] (all in FP16, flattened row-major)
    - token_embedding_table: [vocab_size, dim]
    - For each layer (0 to n_layers-1):
        - rms_att_weight: [dim]
        - wq: [dim, n_heads * head_dim]
        - wk: [dim, n_kv_heads * head_dim]
        - wv: [dim, n_kv_heads * head_dim]
        - wo: [n_heads * head_dim, dim]
        - rms_ffn_weight: [dim]
        - w_gate: [dim, hidden_dim]
        - w_up: [dim, hidden_dim]
        - w_down: [hidden_dim, dim]
    - rms_final_weight: [dim]
    - lm_head: [vocab_size, dim] (optional, can use wcls if available)
"""

import argparse
import struct
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from pathlib import Path


def export_llama_binary(model_name: str, output_path: str):
    """
    Load Hugging Face Llama model and export to binary format.
    """
    print(f"Loading model: {model_name}")
    
    # Load model and config
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model = model.cpu()  # Ensure model is on CPU for export
    
    # Extract configuration
    dim = config.hidden_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads if hasattr(config, 'num_key_value_heads') else n_heads
    vocab_size = config.vocab_size
    max_seq_len = config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else 2048
    hidden_dim = config.intermediate_size
    rope_theta = config.rope_theta if hasattr(config, 'rope_theta') else 10000.0
    head_dim = dim // n_heads
    
    # Validate constraints
    assert n_heads % n_kv_heads == 0, f"n_heads must be divisible by n_kv_heads"
    
    # Note: Originally planned for HEAD_DIM=128 only, but supporting any head_dim for flexibility
    if head_dim != 128:
        print(f"\n⚠️  Note: HEAD_DIM is {head_dim} (originally planned for 128)")
        print(f"    This is fine - the C code will read head_dim from the binary header.")
    
    print(f"\nModel Configuration:")
    print(f"  dim: {dim}")
    print(f"  n_layers: {n_layers}")
    print(f"  n_heads: {n_heads}")
    print(f"  n_kv_heads: {n_kv_heads}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  head_dim: {head_dim}")
    print(f"  rope_theta: {rope_theta}")
    
    # Open output file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        # Write header
        print("\nWriting header...")
        header = struct.pack(
            'iiiiiiiif',
            dim, n_layers, n_heads, n_kv_heads,
            vocab_size, max_seq_len, hidden_dim, head_dim, rope_theta
        )
        f.write(header)
        
        # Helper function to write tensor
        def write_tensor(tensor, name):
            # Convert to FP16 and flatten
            if tensor.dtype != torch.float16:
                tensor = tensor.to(torch.float16)
            tensor = tensor.detach().cpu().numpy().flatten()
            
            # Write as raw bytes
            f.write(tensor.tobytes())
            print(f"  Written {name}: shape {tensor.shape}, size {tensor.nbytes / (1024**2):.2f} MB")
        
        # Write token embedding
        print("\nWriting token embedding...")
        write_tensor(model.model.embed_tokens.weight, "token_embedding_table")
        
        # Write layer weights
        for i in range(n_layers):
            print(f"\nWriting layer {i}...")
            layer = model.model.layers[i]
            
            # Attention
            write_tensor(layer.input_layernorm.weight, f"layer{i}.rms_att_weight")
            write_tensor(layer.self_attn.q_proj.weight.t(), f"layer{i}.wq")  # Transpose for row-major
            write_tensor(layer.self_attn.k_proj.weight.t(), f"layer{i}.wk")
            write_tensor(layer.self_attn.v_proj.weight.t(), f"layer{i}.wv")
            write_tensor(layer.self_attn.o_proj.weight.t(), f"layer{i}.wo")
            
            # FFN
            write_tensor(layer.post_attention_layernorm.weight, f"layer{i}.rms_ffn_weight")
            write_tensor(layer.mlp.gate_proj.weight.t(), f"layer{i}.w_gate")
            write_tensor(layer.mlp.up_proj.weight.t(), f"layer{i}.w_up")
            write_tensor(layer.mlp.down_proj.weight.t(), f"layer{i}.w_down")
        
        # Final norm
        print("\nWriting final norm...")
        write_tensor(model.model.norm.weight, "rms_final_weight")
        
        # LM head
        print("\nWriting LM head...")
        if hasattr(model, 'lm_head'):
            write_tensor(model.lm_head.weight.t(), "lm_head")
        else:
            # Some models share embedding and lm_head
            print("  (Using shared embedding for lm_head)")
            write_tensor(model.model.embed_tokens.weight.t(), "lm_head")
    
    file_size = output_path.stat().st_size / (1024**2)
    print(f"\n✅ Export complete!")
    print(f"   Output: {output_path}")
    print(f"   Size: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Export Llama model to binary format')
    parser.add_argument(
        '--model',
        type=str,
        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        help='Hugging Face model name (default: TinyLlama-1.1B for testing)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/model.bin',
        help='Output binary file path'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Llama Binary Export Tool")
    print("=" * 60)
    
    export_llama_binary(args.model, args.output)


if __name__ == '__main__':
    main()

