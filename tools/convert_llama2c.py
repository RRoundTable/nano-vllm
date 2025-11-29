import struct
import argparse
import numpy as np
from pathlib import Path

def read_and_transpose(f_in, rows, cols, dtype=np.float32):
    """
    Read matrix of shape [rows, cols] from file,
    Transpose to [cols, rows],
    Flatten and return bytes.
    """
    # Read data (assuming float32 input)
    # Note: llama2.c bin uses float32.
    count = rows * cols
    data_bytes = f_in.read(count * 4)
    if len(data_bytes) != count * 4:
        raise ValueError(f"Failed to read {count} floats")
    
    # Load into numpy array
    arr = np.frombuffer(data_bytes, dtype=dtype)
    # Original shape: [rows, cols]
    arr = arr.reshape(rows, cols)
    # Transpose: [cols, rows]
    arr_T = arr.T
    # Flatten and return bytes
    return arr_T.flatten().tobytes()

def convert_karpathy_bin(input_path, output_path):
    print(f"Converting {input_path} -> {output_path}")
    
    with open(input_path, 'rb') as f_in:
        # Read Karpathy Header (7 ints)
        header_data = f_in.read(7 * 4)
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len = struct.unpack('iiiiiii', header_data)
        
        head_dim = dim // n_heads
        rope_theta = 10000.0 
        
        print("Detected Config:")
        print(f"  dim: {dim}")
        print(f"  hidden_dim: {hidden_dim}")
        print(f"  n_layers: {n_layers}")
        print(f"  n_heads: {n_heads}")
        print(f"  n_kv_heads: {n_kv_heads}")
        print(f"  vocab_size: {vocab_size}")
        print(f"  max_seq_len: {max_seq_len}")
        print(f"  head_dim: {head_dim}")
        
        with open(output_path, 'wb') as f_out:
            # Write My Header
            new_header = struct.pack(
                'iiiiiiiif',
                dim, n_layers, n_heads, n_kv_heads,
                vocab_size, max_seq_len, hidden_dim, head_dim, rope_theta
            )
            f_out.write(new_header)
            
            # 1. Token Embedding
            blob = f_in.read(vocab_size * dim * 4)
            f_out.write(blob)
            
            # 2. Layers
            for i in range(n_layers):
                print(f"Processing Layer {i}...")
                # att_norm
                f_out.write(f_in.read(dim * 4))
                
                # wq: [dim, n_heads * head_dim] -> Transpose -> [n_heads * head_dim, dim]
                # But wait, my `matmul` expects `weight[j * out_dim + i]` which assumes 
                # weight is stored as [in_dim, out_dim] row-major.
                # i.e. W[0,0], W[0,1]...
                #
                # Karpathy stores [out_dim, in_dim] row-major.
                # So W[0,0] is weight from Input[0] to Output[0].
                # Wait, PyTorch Linear(in, out) weights are [out, in].
                # W[i, j] is weight connecting Input[j] to Output[i].
                #
                # My matmul: `val += in[j] * weight[j * out_dim + i]`
                # It wants W[j, i] at `j * out_dim + i`.
                # This means it wants `[in_dim, out_dim]` layout.
                #
                # So if file has `[out_dim, in_dim]`, I need to Transpose it to `[in_dim, out_dim]`.
                # Yes.
                
                # wq: out=n_heads*head_dim, in=dim
                f_out.write(read_and_transpose(f_in, n_heads * head_dim, dim))
                
                # wk: out=n_kv_heads*head_dim, in=dim
                f_out.write(read_and_transpose(f_in, n_kv_heads * head_dim, dim))
                
                # wv: out=n_kv_heads*head_dim, in=dim
                f_out.write(read_and_transpose(f_in, n_kv_heads * head_dim, dim))
                
                # wo: out=dim, in=n_heads*head_dim
                f_out.write(read_and_transpose(f_in, dim, n_heads * head_dim))
                
                # ffn_norm
                f_out.write(f_in.read(dim * 4))
                
                # FFN Weights: Read gate, down, up
                # w_gate (w1): out=hidden, in=dim
                w_gate_T = read_and_transpose(f_in, hidden_dim, dim)
                
                # w_down (w2): out=dim, in=hidden
                w_down_T = read_and_transpose(f_in, dim, hidden_dim)
                
                # w_up (w3): out=hidden, in=dim
                w_up_T   = read_and_transpose(f_in, hidden_dim, dim)
                
                # Write gate, up, down
                f_out.write(w_gate_T)
                f_out.write(w_up_T)
                f_out.write(w_down_T)
            
            # 3. Final Norm
            f_out.write(f_in.read(dim * 4))
            
            # 4. LM Head
            remaining = f_in.read()
            expected_lm_head_size = vocab_size * dim * 4
            
            if len(remaining) == expected_lm_head_size:
                print(f"  Writing separate LM Head ({len(remaining)} bytes)")
                # LM Head: out=vocab_size, in=dim. Needs Transpose!
                # But I already read bytes. Convert to numpy.
                arr = np.frombuffer(remaining, dtype=np.float32)
                arr = arr.reshape(vocab_size, dim)
                arr_T = arr.T # [dim, vocab_size]
                f_out.write(arr_T.flatten().tobytes())
            else:
                print("  Shared LM Head (copying token embedding)")
                # Re-read token embedding
                f_in.seek(7 * 4) # Skip header
                token_emb_bytes = f_in.read(vocab_size * dim * 4)
                
                # Token Embedding is [vocab_size, dim].
                # LM Head needs [dim, vocab_size] (Transpose of embedding).
                arr = np.frombuffer(token_emb_bytes, dtype=np.float32)
                arr = arr.reshape(vocab_size, dim)
                arr_T = arr.T
                f_out.write(arr_T.flatten().tobytes())

    print("Conversion Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input llama2.c bin file")
    parser.add_argument("output", help="Output nano-vllm bin file")
    args = parser.parse_args()
    
    convert_karpathy_bin(args.input, args.output)
