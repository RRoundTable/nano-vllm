#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "structs.h"

// Simple Random Number Generator (xorshift)
// Using standard rand() for simplicity in Phase 1, but a custom one is more deterministic for testing.
// Let's use a simple LCG or just rand().
// Karpathy uses a custom RNG logic to be deterministic across platforms.
// We'll stick to rand() seeded with time() for "fun", or fixed seed for debugging.

typedef struct {
    unsigned long long state;
    float temperature;
    float topp;
    ProbIndex* probindex;
    int vocab_size;
} Sampler;

void build_sampler(Sampler* s, int vocab_size, float temperature, float topp, unsigned long long seed) {
    s->state = seed;
    s->temperature = temperature;
    s->topp = topp;
    s->vocab_size = vocab_size;
    s->probindex = malloc(vocab_size * sizeof(ProbIndex));
    srand((unsigned int)seed); // if using rand()
}

float random_f32() {
    return (float)rand() / (float)RAND_MAX;
}

int compare_prob_index(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_prob_index);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

int sample(Sampler* s, float* logits) {
    // 1. Apply Temperature
    if (s->temperature == 0.0f) {
        // Greedy argmax
        int max_i = 0;
        float max_val = logits[0];
        for (int i = 1; i < s->vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_i = i;
            }
        }
        return max_i;
    }

    // Apply temperature to logits
    for (int i = 0; i < s->vocab_size; i++) {
        logits[i] /= s->temperature;
    }

    // 2. Softmax (to get probabilities)
    // Find max for numerical stability
    float max_val = logits[0];
    for (int i = 1; i < s->vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < s->vocab_size; i++) {
        float v = expf(logits[i] - max_val);
        logits[i] = v;
        sum += v;
    }
    
    // Normalize
    for (int i = 0; i < s->vocab_size; i++) {
        logits[i] /= sum;
    }

    // 3. Sample from distribution
    float coin = random_f32();
    
    if (s->topp <= 0 || s->topp >= 1) {
        // simply sample from the predicted probability distribution
        float cdf = 0.0f;
        for (int i = 0; i < s->vocab_size; i++) {
            cdf += logits[i];
            if (coin < cdf) {
                return i;
            }
        }
        return s->vocab_size - 1; // in case of rounding errors
    } else {
        // top-p (nucleus) sampling, clamping the least likely tokens to zero
        return sample_topp(logits, s->vocab_size, s->topp, s->probindex, coin);
    }
}
