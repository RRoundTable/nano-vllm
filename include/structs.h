#ifndef STRUCTS_H
#define STRUCTS_H

#include <stdint.h>

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
    // Derived or extended
    int head_dim;
    float rope_theta;
} Config;

typedef struct {
    float* rms_att_weight; // [dim]
    float* wq;             // [dim, n_heads * head_dim]
    float* wk;             // [dim, n_kv_heads * head_dim]
    float* wv;             // [dim, n_kv_heads * head_dim]
    float* wo;             // [n_heads * head_dim, dim]
    float* rms_ffn_weight; // [dim]
    float* w_gate;         // [dim, hidden_dim]
    float* w_up;           // [dim, hidden_dim]
    float* w_down;         // [hidden_dim, dim]
} LayerWeights;

typedef struct {
    float* token_embedding_table; // [vocab_size, dim]
    LayerWeights* layers;         // Array of [n_layers]
    float* rms_final_weight;      // [dim]
    float* lm_head;               // [vocab_size, dim]
    
    void* _data_buffer; 
} Weights;

typedef struct {
    // Current state
    float* x;      // [dim]
    float* xb;     // [dim]
    float* xb2;    // [dim]
    float* hb;     // [hidden_dim]
    float* hb2;    // [hidden_dim]
    float* q;      // [dim]
    float* k;      // [dim]
    float* v;      // [dim]
    float* att;    // [n_heads, seq_len]
    float* logits; // [vocab_size]
    
    // KV Cache (Naive)
    // [n_layers, max_seq_len, n_kv_heads, head_dim]
    float* key_cache;   
    float* value_cache; 
} RunState;

// ==========================================
// PagedAttention Structures (Phase 3)
// ==========================================

typedef struct {
    int block_size;
    int num_blocks;
    int free_blocks_count;
    int* free_block_indices; // Stack of free block indices
    
    // Physical Memory Pool
    // [num_blocks, block_size, n_kv_heads, head_dim]
    // We will allocate one huge pool for all layers or per layer?
    // vLLM usually has one large pool per layer or shared. 
    // For simplicity in C, let's make it: [n_layers, num_blocks, block_size, n_kv_heads, head_dim]
    float* pool_k; 
    float* pool_v;
} KVCacheManager;

typedef struct {
    int* block_indices; // [max_blocks_per_seq]
    int num_blocks;     // Current number of blocks used
} BlockTable;

typedef struct {
    int id;
    int active;       // 1 if generating, 0 if finished
    int pos;          // Current position (tokens generated so far)
    int seq_len;      // Total tokens to generate (prompt + steps)
    int current_token;// Last generated token
    int* prompt_tokens;
    int num_prompt_tokens;
    int* output_history; // Store full token history
    RunState* state;  // Dedicated activation memory for this seq
    BlockTable table; // Dedicated block table for this seq
} Sequence;

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

#endif // STRUCTS_H
