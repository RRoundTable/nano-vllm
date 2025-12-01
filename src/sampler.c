#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Simple Random Number Generator (xorshift)
// Using standard rand() for simplicity in Phase 1, but a custom one is more deterministic for testing.
// Let's use a simple LCG or just rand().
// Karpathy uses a custom RNG logic to be deterministic across platforms.
// We'll stick to rand() seeded with time() for "fun", or fixed seed for debugging.

typedef struct {
    unsigned long long state;
} Sampler;

void build_sampler(Sampler* s, unsigned long long seed) {
    s->state = seed;
    srand((unsigned int)seed); // if using rand()
}

float random_f32() {
    return (float)rand() / (float)RAND_MAX;
}

int sample(Sampler* s, float* logits, int vocab_size, float temperature) {
    // 1. Apply Temperature
    if (temperature == 0.0f) {
        // Greedy argmax
        int max_i = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_i = i;
            }
        }
        return max_i;
    }

    // Apply temperature to logits
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }

    // 2. Softmax (to get probabilities)
    // Find max for numerical stability
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float v = expf(logits[i] - max_val);
        logits[i] = v;
        sum += v;
    }
    
    // Normalize
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= sum;
    }

    // 3. Sample from distribution
    float coin = random_f32();
    float cdf = 0.0f;
    
    for (int i = 0; i < vocab_size; i++) {
        cdf += logits[i];
        if (coin < cdf) {
            return i;
        }
    }
    
    return vocab_size - 1; // Should not reach here
}

