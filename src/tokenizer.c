#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Simple Tokenizer for Llama 2 (SentencePiece .bin format)
// Based on karpathy/llama2.c

typedef struct {
    char** vocab;
    float* vocab_scores;
    int vocab_size;
    unsigned char byte_pieces[512]; // For byte fallback
} Tokenizer;

void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    
    printf("Opening tokenizer: %s\n", tokenizer_path);
    FILE* f = fopen(tokenizer_path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open tokenizer: %s\n", tokenizer_path);
        exit(1);
    }
    
    // Karpathy tokenizer.bin format:
    // max_token_length (int)
    // for i in vocab_size:
    //   score (float)
    //   len (int)
    //   text (char * len)
    
    int max_token_length;
    if (fread(&max_token_length, sizeof(int), 1, f) != 1) { 
        fprintf(stderr, "Failed to read max_token_length\n");
        exit(1); 
    }
    
    printf("Tokenizer header: max_len=%d. Expected vocab_size=%d\n", max_token_length, vocab_size);
    
    for (int i = 0; i < vocab_size; i++) {
        if (fread(&t->vocab_scores[i], sizeof(float), 1, f) != 1) { 
            fprintf(stderr, "Failed to read score for token %d\n", i);
            exit(1); 
        }
        
        int len;
        if (fread(&len, sizeof(int), 1, f) != 1) { 
            fprintf(stderr, "Failed to read len for token %d\n", i);
            exit(1); 
        }
        
        t->vocab[i] = (char*)malloc(len + 1);
        if (fread(t->vocab[i], 1, len, f) != len) { 
            fprintf(stderr, "Failed to read char for token %d (len %d)\n", i, len);
            exit(1); 
        }
        t->vocab[i][len] = '\0';
    }
    
    fclose(f);
    printf("Tokenizer loaded successfully.\n");
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
}

const char* decode_token(Tokenizer* t, int token) {
    if (token < 0 || token >= t->vocab_size) return "";
    // Replace space symbol (0xE2 0x96 0x81) with actual space ' ' if needed
    // For now just return raw string
    return t->vocab[token];
}
