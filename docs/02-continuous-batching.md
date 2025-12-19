# Continuous Batching & Chunked Prefill: Maximizing Throughput

## 1. Introduction
In LLM serving, two major problems limit performance when using naive batching:
1.  **The "Straggler" Problem**: If we batch a short sequence (10 tokens) with a long one (1000 tokens), the GPU/CPU sits idle waiting for the long one to finish generation. This is static batching.
2.  **The "Stall" Problem**: Processing a very long prompt (prefill) takes a long time, blocking all other requests from generating tokens. This causes high latency (TTFT: Time To First Token) for other users.

**Continuous Batching** (also known as Iteration-level Batching) and **Chunked Prefill** are the industry-standard solutions to these problems. This document explains how they are implemented in nano-vllm.

## 2. Continuous Batching (Ragged Batching)

### 2.1. The Concept
Instead of waiting for an entire batch of sequences to finish, we manage batching at the **iteration level**. After every token generation step, we check:
- Has any sequence finished? -> Remove it.
- Has a new request arrived? -> Add it.

This means a single batch can contain sequences at completely different stages: some just starting (prefill), some in the middle of generation (decode), and some nearing completion.

### 2.2. Scheduler Implementation (`src/main.c`)
The scheduler runs a loop that constructs a "Ragged Batch" for every step.

```c
// Simplified Logic from src/main.c
while (!all_finished) {
    // 1. Scheduler Phase
    for (int i = 0; i < BATCH_SIZE; i++) {
        if (seqs[i].is_finished) continue;
        
        // Decide work for this sequence
        if (seqs[i].is_prefilling) {
             add_to_batch(seqs[i], chunk_size); // Add chunk
        } else {
             add_to_batch(seqs[i], 1);          // Add 1 token (decode)
        }
    }
    
    // 2. Inference Phase
    // Run transformer on the mixed (ragged) batch
    transformer_batch(batch_tokens, ...);
}
```

### 2.3. Data Structure: The Ragged Batch
Since sequences are not aligned, we cannot use a simple `[Batch, Seq_Len]` tensor. Instead, we use flat arrays (Ragged Tensor representation):

```
Example Batch:
Seq A (Prefill, 3 tokens)
Seq B (Decode, 1 token)

batch_tokens: [ A1, A2, A3, B1 ]
batch_pos:    [ 10, 11, 12, 50 ]
batch_seq_ids:[ 0,  0,  0,  1  ]
```

The Attention Kernel (`kernels/cpu/attention.c`) uses `batch_seq_ids` to ensure Token A only attends to Seq A's KV Cache (via Block Table), and Token B attends to Seq B.

## 3. Chunked Prefill

### 3.1. The Problem: Compute Spikes
A standard prefill (processing the prompt) computes attention for $N$ tokens at once. For a prompt of length 2048, this is a massive matrix multiplication.
- If we mix a 2048-token prefill with 1-token decodes, the decodes will "stutter" (huge latency spike) while waiting for the prefill.

### 3.2. The Solution: Time-Slicing
We break the prefill process into smaller **Chunks** (e.g., 256 tokens).
- Step 1: Process tokens 0-255 of Seq A. (Seq B generates 1 token).
- Step 2: Process tokens 256-511 of Seq A. (Seq B generates 1 token).
- ...

This keeps the total batch computation time consistent, ensuring smooth inter-token latency for decoding sequences.

### 3.3. Implementation Logic
In `src/main.c`, the scheduler tracks the progress of prefill:

```c
int start_pos = seqs[i].pos;
int end_pos = seqs[i].num_prompt_tokens;
int remaining = end_pos - start_pos;

// Take a chunk or the rest
int n_tokens = (remaining > chunk_size) ? chunk_size : remaining;

// Update position only by the chunk amount
seqs[i].pos += n_tokens;
```

The KV Cache is naturally built up incrementally. Paged Attention handles this perfectly because we just allocate new blocks as the prefill progresses.

## 4. Hardware Implications (CPU)

### 4.1. Cache Locality vs. Throughput
Continuous batching trades some instruction cache locality (handling different seqs) for massive throughput gains.
- **ALU Utilization**: By keeping the batch size high (filling it with chunks or new requests), we keep the CPU's vector units (AVX/NEON) busy.
- **Memory Bandwidth**: We maximize the utility of loaded weights. We load the model weights once from RAM to Cache, and apply them to as many tokens as possible (Ragged Batch).

### 4.2. Latency Hiding
Chunked prefill allows "interactive" sequences (like a chatbot responding) to continue generating tokens smoothly even while a heavy "analysis" job (long document summary) is starting up on the same server.

## 5. Summary

| Feature | Problem Solved | Mechanism |
| :--- | :--- | :--- |
| **Continuous Batching** | Idle hardware when seqs have different lengths | Dynamic insertion/removal of seqs at every step |
| **Ragged Batch** | Tensor shape mismatch | Flattened arrays with index mapping (`seq_ids`) |
| **Chunked Prefill** | Latency spikes (Stalls) during heavy prefills | Splitting prompts into small batches over time |

These techniques, combined with **Paged Attention**, form the modern high-performance LLM serving stack implemented in nano-vllm.

