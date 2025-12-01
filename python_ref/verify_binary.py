#!/usr/bin/env python3
"""
Verify exported binary file format and print statistics.

Usage:
    python verify_binary.py data/model.bin
"""

import argparse
import struct
import numpy as np
from pathlib import Path


def verify_binary(binary_path: str):
    """
    Read and verify the exported binary file.
    """
    path = Path(binary_path)
    if not path.exists():
        print(f"❌ File not found: {binary_path}")
        return False
    
    print(f"Verifying: {binary_path}")
    print(f"File size: {path.stat().st_size / (1024**2):.2f} MB\n")
    
    with open(path, 'rb') as f:
        # Read header
        header_bytes = f.read(36)  # Updated: 9 fields now (added head_dim)
        if len(header_bytes) != 36:
            print("❌ Invalid header size")
            return False
        
        dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len, hidden_dim, head_dim, rope_theta = struct.unpack('iiiiiiiif', header_bytes)
        
        print("=" * 60)
        print("Model Configuration:")
        print("=" * 60)
        print(f"  dim:          {dim}")
        print(f"  n_layers:     {n_layers}")
        print(f"  n_heads:      {n_heads}")
        print(f"  n_kv_heads:   {n_kv_heads}")
        print(f"  vocab_size:   {vocab_size}")
        print(f"  max_seq_len:  {max_seq_len}")
        print(f"  hidden_dim:   {hidden_dim}")
        print(f"  head_dim:     {head_dim}")
        print(f"  rope_theta:   {rope_theta}")
        
        # Validate constraints
        if dim // n_heads != head_dim:
            print(f"\n⚠️  Warning: dim/n_heads ({dim // n_heads}) != head_dim ({head_dim})")
        
        if n_heads % n_kv_heads != 0:
            print(f"\n❌ Invalid GQA config: n_heads ({n_heads}) not divisible by n_kv_heads ({n_kv_heads})")
            return False
        
        # Calculate expected size
        print("\n" + "=" * 60)
        print("Weight Statistics:")
        print("=" * 60)
        
        total_params = 0
        
        # Token embedding
        tok_emb_size = vocab_size * dim
        total_params += tok_emb_size
        print(f"Token Embedding:      {tok_emb_size:>12,} params ({tok_emb_size * 2 / (1024**2):>6.2f} MB)")
        
        # Per layer
        per_layer = 0
        per_layer += dim  # rms_att
        per_layer += dim * (n_heads * head_dim)  # wq
        per_layer += dim * (n_kv_heads * head_dim)  # wk
        per_layer += dim * (n_kv_heads * head_dim)  # wv
        per_layer += (n_heads * head_dim) * dim  # wo
        per_layer += dim  # rms_ffn
        per_layer += dim * hidden_dim  # w_gate
        per_layer += dim * hidden_dim  # w_up
        per_layer += hidden_dim * dim  # w_down
        
        layer_params = per_layer * n_layers
        total_params += layer_params
        print(f"All Layers ({n_layers:>2}):      {layer_params:>12,} params ({layer_params * 2 / (1024**2):>6.2f} MB)")
        print(f"  Per layer:          {per_layer:>12,} params ({per_layer * 2 / (1024**2):>6.2f} MB)")
        
        # Final norm + LM head
        final_norm = dim
        lm_head = vocab_size * dim
        total_params += final_norm + lm_head
        print(f"Final Norm:           {final_norm:>12,} params ({final_norm * 2 / (1024**2):>6.2f} MB)")
        print(f"LM Head:              {lm_head:>12,} params ({lm_head * 2 / (1024**2):>6.2f} MB)")
        
        print(f"\n{'Total Parameters:':20} {total_params:>12,} params ({total_params * 2 / (1024**2):>6.2f} MB)")
        
        # Expected file size (header + weights in FP16)
        expected_size = 36 + total_params * 2  # Header is now 36 bytes, FP16 = 2 bytes
        actual_size = path.stat().st_size
        
        print("\n" + "=" * 60)
        print("File Size Verification:")
        print("=" * 60)
        print(f"  Expected: {expected_size:>12,} bytes ({expected_size / (1024**2):>6.2f} MB)")
        print(f"  Actual:   {actual_size:>12,} bytes ({actual_size / (1024**2):>6.2f} MB)")
        
        if expected_size == actual_size:
            print("\n✅ File size matches expected size!")
        else:
            diff = actual_size - expected_size
            print(f"\n⚠️  Size mismatch: {diff:+,} bytes ({diff / (1024**2):+.2f} MB)")
        
        # Sample first few weights
        print("\n" + "=" * 60)
        print("Weight Samples (first 10 values of token embedding):")
        print("=" * 60)
        
        sample = np.frombuffer(f.read(20), dtype=np.float16)  # Read 10 FP16 values
        for i, val in enumerate(sample):
            print(f"  [{i}] = {val:.6f}")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Verify binary model file')
    parser.add_argument('binary_path', type=str, help='Path to binary file')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Llama Binary Verification Tool")
    print("=" * 60)
    print()
    
    success = verify_binary(args.binary_path)
    
    if success:
        print("\n✅ Verification complete!")
    else:
        print("\n❌ Verification failed!")
        exit(1)


if __name__ == '__main__':
    main()

