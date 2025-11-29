#ifndef STRUCTS_H
#define STRUCTS_H

#include <stdint.h>

typedef struct {
    int dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
    int hidden_dim;
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
    
    // Helper to keep track of allocated device memory to free it later
    // (Optional, depending on implementation)
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
    
    // KV Cache
    // [n_layers, max_seq_len, n_kv_heads, head_dim]
    float* key_cache;   
    float* value_cache; 
} RunState;

#endif // STRUCTS_H

