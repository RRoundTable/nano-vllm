#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
} Config;

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    FILE* f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", argv[1]);
        return 1;
    }

    Config config;
    // karpathy/llama2.c format is slightly different from our export_binary.py format
    // llama2.c header: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len
    // our format: dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len, hidden_dim, head_dim, rope_theta
    
    if (fread(&config, sizeof(Config), 1, f) != 1) {
        fprintf(stderr, "Failed to read config\n");
        return 1;
    }
    
    printf("Karpathy's llama2.c Model Config:\n");
    printf("  dim: %d\n", config.dim);
    printf("  hidden_dim: %d\n", config.hidden_dim);
    printf("  n_layers: %d\n", config.n_layers);
    printf("  n_heads: %d\n", config.n_heads);
    printf("  n_kv_heads: %d\n", config.n_kv_heads);
    printf("  vocab_size: %d\n", config.vocab_size);
    printf("  max_seq_len: %d\n", config.max_seq_len);

    fclose(f);
    return 0;
}

