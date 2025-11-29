#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char** vocab;
    float* vocab_scores;
    int vocab_size;
    unsigned char byte_pieces[512];
} Tokenizer;

void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    
    FILE* f = fopen(tokenizer_path, "rb");
    if (!f) { fprintf(stderr, "Failed\n"); exit(1); }
    
    int max_token_length;
    fread(&max_token_length, sizeof(int), 1, f);
    
    for (int i = 0; i < vocab_size; i++) {
        fread(&t->vocab_scores[i], sizeof(float), 1, f);
        int len;
        fread(&len, sizeof(int), 1, f);
        t->vocab[i] = (char*)malloc(len + 1);
        fread(t->vocab[i], 1, len, f);
        t->vocab[i][len] = '\0';
    }
    fclose(f);
}

int main(int argc, char** argv) {
    Tokenizer t;
    build_tokenizer(&t, "data/tokenizer.bin", 32000);
    
    printf("Token 0: '%s'\n", t.vocab[0]);
    printf("Token 1: '%s'\n", t.vocab[1]);
    printf("Token 2: '%s'\n", t.vocab[2]);
    printf("Token 3: '%s'\n", t.vocab[3]);
    
    // Search for "prom" or "NSURL"
    for(int i=0; i<32000; i++) {
        if (strstr(t.vocab[i], "prom") || strstr(t.vocab[i], "NSURL")) {
            printf("Found suspicious token %d: '%s'\n", i, t.vocab[i]);
        }
    }
    
    return 0;
}

