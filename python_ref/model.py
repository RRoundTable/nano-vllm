"""
Llama Architecture Reference (Documentation Purpose)

This file documents the Llama architecture for reference.
Actual weights will be loaded from Hugging Face pretrained models.

Architecture Overview:
- Transformer decoder with RoPE (Rotary Position Embedding)
- RMSNorm instead of LayerNorm
- SwiGLU activation (SiLU + Gate)
- Grouped Query Attention (GQA) for Llama-2/3
- Fixed HEAD_DIM = 128, FP16 only

Model Structure:
1. Token Embedding: vocab_size -> dim
2. N Transformer Blocks:
   - RMSNorm + Attention (with RoPE)
   - RMSNorm + FFN (SwiGLU)
3. Final RMSNorm + LM Head

Weight Layout (will be flattened to binary):
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
- lm_head (optional, can share with token_embedding)

Key Constraints:
- HEAD_DIM = dim / n_heads (typically 64, 128, or 256)
- FP16 only (torch.float16 -> uint16 in binary)
- GQA: n_heads >= n_kv_heads (typically n_heads = 4 * n_kv_heads)
"""

# This is a reference file only.
# See export_binary.py for actual model loading and conversion.

class LlamaConfig:
    """Configuration for Llama model"""
    def __init__(self):
        self.dim = 4096           # Model dimension
        self.n_layers = 32        # Number of transformer layers
        self.n_heads = 32         # Number of attention heads
        self.n_kv_heads = 8       # Number of KV heads (GQA)
        self.vocab_size = 32000   # Vocabulary size
        self.hidden_dim = 11008   # FFN hidden dimension
        self.head_dim = 128       # Fixed: dim / n_heads
        self.max_seq_len = 2048   # Maximum sequence length
        self.rope_theta = 10000.0 # RoPE base frequency

# Attention computation pseudocode:
"""
def attention(q, k, v, n_heads, n_kv_heads, head_dim):
    # q: [seq_len, n_heads * head_dim]
    # k: [seq_len, n_kv_heads * head_dim]
    # v: [seq_len, n_kv_heads * head_dim]
    
    # Reshape
    q = q.view(seq_len, n_heads, head_dim)
    k = k.view(seq_len, n_kv_heads, head_dim)
    v = v.view(seq_len, n_kv_heads, head_dim)
    
    # Apply RoPE to q, k
    q = apply_rope(q, positions)
    k = apply_rope(k, positions)
    
    # Expand k, v for GQA (each kv head serves multiple q heads)
    n_rep = n_heads // n_kv_heads
    k = k.repeat_interleave(n_rep, dim=1)  # [seq_len, n_heads, head_dim]
    v = v.repeat_interleave(n_rep, dim=1)
    
    # Scaled dot-product attention
    scores = (q @ k.transpose(-2, -1)) / sqrt(head_dim)
    scores = scores.masked_fill(causal_mask, -inf)
    attn = softmax(scores, dim=-1)
    out = attn @ v
    
    return out.view(seq_len, n_heads * head_dim)
"""

# FFN computation pseudocode:
"""
def swiglu_ffn(x, w_gate, w_up, w_down):
    # SwiGLU: https://arxiv.org/abs/2002.05202
    gate = silu(x @ w_gate)  # SiLU(x) = x * sigmoid(x)
    up = x @ w_up
    hidden = gate * up
    out = hidden @ w_down
    return out
"""

# RMSNorm pseudocode:
"""
def rms_norm(x, weight, eps=1e-5):
    # x: [seq_len, dim]
    # weight: [dim]
    rms = sqrt(mean(x^2) + eps)
    normed = x / rms
    return normed * weight
"""

