import struct
import argparse
import numpy as np
from pathlib import Path

def convert_karpathy_bin(input_path, output_path):
    print(f"Converting {input_path} -> {output_path}")
    
    with open(input_path, 'rb') as f_in:
        # Read Karpathy Header (7 ints)
        # dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len
        header_data = f_in.read(7 * 4)
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len = struct.unpack('iiiiiii', header_data)
        
        head_dim = dim // n_heads
        rope_theta = 10000.0 # Default for Llama 2
        
        print("Detected Config:")
        print(f"  dim: {dim}")
        print(f"  hidden_dim: {hidden_dim}")
        print(f"  n_layers: {n_layers}")
        print(f"  n_heads: {n_heads}")
        print(f"  n_kv_heads: {n_kv_heads}")
        print(f"  vocab_size: {vocab_size}")
        print(f"  max_seq_len: {max_seq_len}")
        
        with open(output_path, 'wb') as f_out:
            # Write My Header (8 ints + 1 float)
            # dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len, hidden_dim, head_dim, rope_theta
            new_header = struct.pack(
                'iiiiiiiif',
                dim, n_layers, n_heads, n_kv_heads,
                vocab_size, max_seq_len, hidden_dim, head_dim, rope_theta
            )
            f_out.write(new_header)
            
            # Copy Weights
            # Karpathy's order usually matches ours for standard layers
            # except possibly FFN or Shared Weights.
            # 
            # llama2.c structure:
            # 1. token_embedding [vocab_size * dim]
            # 2. layers:
            #    - attention_norm [dim]
            #    - wq [dim * n_heads * head_dim]
            #    - wk [dim * n_kv_heads * head_dim]
            #    - wv [dim * n_kv_heads * head_dim]
            #    - wo [n_heads * head_dim * dim]
            #    - ffn_norm [dim]
            #    - w_gate [dim * hidden_dim]
            #    - w_down [hidden_dim * dim]
            #    - w_up   [dim * hidden_dim]
            # 3. final_norm [dim]
            # 4. lm_head [vocab_size * dim] (if not shared)
            
            # My structure expects:
            # ...
            #    - w_gate
            #    - w_up     <-- Swapped in some implementations
            #    - w_down   <-- Swapped in some implementations
            
            # Let's just copy everything blindly first. 
            # The tensor sizes are:
            # token_emb: V*D
            # For each layer:
            #   att_norm: D
            #   wq: D * (H*HD)
            #   wk: D * (KVH*HD)
            #   wv: D * (KVH*HD)
            #   wo: (H*HD) * D
            #   ffn_norm: D
            #   w_gate: D * HDim
            #   w_down: HDim * D
            #   w_up:   D * HDim
            
            # Wait, llama2.c stores w_gate, w_down, w_up in that order?
            # Let's check size to be sure.
            # w_gate, w_up are [D, HDim]. w_down is [HDim, D].
            # Since D != HDim usually, we can distinguish?
            # In stories15M: D=288, HDim=768.
            # D * HDim = 221184.
            # All three matrices have same number of elements!
            
            # If I just copy the stream, I get: gate, down, up.
            # My model.c reads: gate, up, down.
            # So if llama2.c is (gate, down, up), I need to reorder.
            # 
            # Checking Karpathy's llama2.c source (export.py or run.c):
            # The export script writes:
            # layer.feed_forward.w1.weight (gate)
            # layer.feed_forward.w2.weight (down)
            # layer.feed_forward.w3.weight (up)
            #
            # My model.c reads:
            # l->w_gate = load_tensor(...)
            # l->w_up = load_tensor(...)
            # l->w_down = load_tensor(...)
            #
            # So I need to read 3 chunks (gate, down, up) and write (gate, up, down).
            
            # 1. Token Embedding
            blob = f_in.read(vocab_size * dim * 4)
            f_out.write(blob)
            
            # 2. Layers
            for i in range(n_layers):
                # att_norm
                f_out.write(f_in.read(dim * 4))
                # wq
                f_out.write(f_in.read(dim * n_heads * head_dim * 4))
                # wk
                f_out.write(f_in.read(dim * n_kv_heads * head_dim * 4))
                # wv
                f_out.write(f_in.read(dim * n_kv_heads * head_dim * 4))
                # wo
                f_out.write(f_in.read(n_heads * head_dim * dim * 4))
                # ffn_norm
                f_out.write(f_in.read(dim * 4))
                
                # FFN Weights: Read gate, down, up
                w_gate_data = f_in.read(dim * hidden_dim * 4)
                w_down_data = f_in.read(hidden_dim * dim * 4)
                w_up_data   = f_in.read(dim * hidden_dim * 4)
                
                # Write gate, up, down
                f_out.write(w_gate_data)
                f_out.write(w_up_data)
                f_out.write(w_down_data)
            
            # 3. Final Norm
            f_out.write(f_in.read(dim * 4))
            
            # 4. LM Head (Optional in llama2.c?)
            # If file ends here, it's shared.
            # Check remaining bytes
            remaining = f_in.read()
            expected_lm_head_size = vocab_size * dim * 4
            
            if len(remaining) == expected_lm_head_size:
                print(f"  Writing separate LM Head ({len(remaining)} bytes)")
                f_out.write(remaining)
            else:
                if len(remaining) > 0:
                    print(f"  Warning: Found {len(remaining)} extra bytes, but expected {expected_lm_head_size} for LM Head.")
                    print("  Assuming shared weights and ignoring extra bytes.")
                
                print("  Shared LM Head (copying token embedding)")
                # Re-write token embedding as LM Head
                f_in.seek(7 * 4) # Skip header to start of token embedding
                token_emb_data = f_in.read(vocab_size * dim * 4)
                f_out.write(token_emb_data)

    print("Conversion Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input llama2.c bin file")
    parser.add_argument("output", help="Output nano-vllm bin file")
    args = parser.parse_args()
    
    convert_karpathy_bin(args.input, args.output)

