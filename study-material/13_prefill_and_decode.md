# ⚡ Prefill & Decode — Two Phases of Inference

## The Big Idea

When an LLM responds to your message, it operates in two distinct phases:
1. **Prefill:** Quickly process your entire prompt at once
2. **Decode:** Generate response tokens one at a time

Understanding these phases explains why LLMs seem to "think" for a moment before suddenly streaming text — and why long prompts can slow things down. It also explains almost every performance optimisation you hear about in LLM deployment: continuous batching, speculative decoding, chunked prefill, and flash attention all exist to address pain points in one of these two phases.

---

## Real-Life Analogy: The Chef's Prep and Cooking 👨‍🍳

Imagine a restaurant chef responding to your order:

**Prefill = Mise en place (prep work):**
- The chef reads your full order, understands all the ingredients needed
- Chops all vegetables, measures all spices, gets everything ready at once
- This is batch processing — done in parallel, quickly

**Decode = Cooking and plating:**
- The chef cooks each component sequentially
- Plate goes out one course at a time
- Can't cook course 2 before course 1 is done

The prep (prefill) is fast and parallel. The cooking (decode) is sequential and takes time. You notice the pause before the food starts arriving — that's prefill.

---

## The Prefill Phase — Processing Your Prompt 📥

During prefill, the model processes your **entire input prompt** at once. This is the "thinking before speaking" moment.

### What happens step by step

1. Your prompt is tokenised — e.g. "What is the capital of France?" becomes ~8 tokens
2. Each token is converted to an embedding vector (a list of ~4096 numbers)
3. All embeddings are stacked into a matrix and fed through **all transformer layers simultaneously**
4. At each layer, every token attends to every other token (full self-attention over the prompt)
5. The Key (K) and Value (V) vectors for every token at every layer are saved to the **KV Cache**
6. At the end of all layers, the final hidden state of the last token is used to compute the first output token's probability distribution

### Why prefill is fast: parallel matrix math 🔢

GPUs are built for one thing: doing many matrix multiplications at the same time. During prefill, processing all 200 tokens happens as a single giant matrix operation. The GPU can handle this in one or two passes.

Think of it like a factory assembly line where every worker processes every widget simultaneously vs. one worker doing all the widgets in sequence. 200 workers (GPU cores) each handle one token. The whole batch finishes at once.

**Mathematically:** if your prompt has N tokens and each transformer layer takes time T_layer, then:
- Sequential processing: N × L × T_layer
- Parallel processing (prefill): L × T_layer (roughly — ignoring memory bandwidth)

For 200 tokens, parallel is ~200× faster than sequential.

### The KV Cache is born here 🏦

Every K and V vector computed during prefill is stored in the KV Cache. This is critical: the decode phase never re-computes these. Each decode step just reads the cached K, V vectors and appends its own new K, V pair. Without the KV Cache, decode would be O(n²) in the sequence length — completely impractical.

### Prefill bottleneck: GPU compute

What limits prefill speed is raw matrix multiplication throughput. GPUs measure this in FLOPS (floating point operations per second). A100 GPUs have ~312 TFLOPS. Longer prompts = more matrix work = longer TTFT.

---

## The Decode Phase — Generating Token by Token 🖨️

After prefill, the model generates your response one token at a time. This is the "streaming text" you see appearing word by word.

### What happens each step

1. Take the newly generated token (or last prefill token for step 1)
2. Embed it into a vector
3. Compute its Q, K, V vectors in each layer
4. Load all previous K, V vectors from the KV Cache (this is the expensive part!)
5. Run attention: the new token attends to all previous tokens via the cached K, V
6. Pass through the FFN (feed-forward network) in each layer
7. Get the logit distribution over the vocabulary
8. Apply temperature/top-p/top-k sampling and pick a token
9. Append that token's K, V to the cache
10. Return to step 1

### Why decode is sequential — no way around it 🔄

Each token depends on the previous token. Token #5 might be "Paris" only because token #4 was "capital", which was only chosen because tokens #1-3 set up the question. You cannot generate token #5 without knowing token #4. This is the fundamental constraint of **autoregressive generation**.

The sequentiality is baked into the architecture: the causal attention mask means each token only sees what came before it. This is what allows the model to generate coherent sequences, but it also prevents batching within a single sequence.

### Decode bottleneck: memory bandwidth 💾

Here's the surprising thing: during each decode step, the GPU does relatively little computation. It's processing just ONE new token (one row of the weight matrices). But it still has to **read the entire KV Cache from GPU memory** on every single step.

For a model with 40 layers, 40 attention heads, and a sequence length of 2000 tokens:
- Each K or V matrix per layer: 2000 × 128 floats × 2 bytes = ~512 KB
- Total KV Cache: 40 layers × 2 (K+V) × 512 KB = ~40 MB
- This 40 MB must be read from GPU HBM memory **every decode step**

The GPU's memory bandwidth is the ceiling — not its compute power. This is called the **memory bandwidth bottleneck**.

**Analogy:** Imagine you've memorised every answer but have to flip through a 1,000-page reference book before answering each question. The flipping, not the thinking, is what takes time.

---

## The Memory Bandwidth Bottleneck 🚧

This is one of the most important concepts in LLM production systems. Let's go deeper.

### Why small batches = wasted GPU

Modern GPUs like the H100 can do 3.35 TB/s of memory reads. But they can also do 989 TFLOPS of compute. During decode:
- Memory reads: ~40 MB per step → at 3.35 TB/s, this takes ~0.012ms
- Compute: just one new token → extremely low FLOP count

Result: the GPU is **compute-starved** during decode. It's reading memory fast, but there's almost nothing to compute. GPU utilisation might be only 5-10% during single-user decode.

### The arithmetic intensity problem

"Arithmetic intensity" = FLOPs / bytes read. High arithmetic intensity = GPU is working hard. Low arithmetic intensity = GPU is just a memory bus.

- Prefill: high arithmetic intensity (many tokens, big matrices) → GPU is fully utilised
- Decode: low arithmetic intensity (one token, tiny computation, big cache reads) → GPU sits idle

This is why batching decode requests from multiple users dramatically improves throughput — you get more compute per memory read.

### How quantisation helps 🗜️

If each KV cache value is stored as FP16 (16 bits) instead of FP32 (32 bits), you halve the memory reads, roughly doubling decode speed. INT8 or INT4 quantisation can push this further. This is a primary reason why 8-bit and 4-bit quantisation is so popular for inference.

### How Flash Attention helps ⚡

Flash Attention is an algorithm that rewrites the attention computation to be more memory-efficient by keeping data in faster SRAM (on-chip cache) instead of slower HBM (off-chip memory). For prefill, this is a huge win. For decode, it helps but the fundamental bandwidth bottleneck remains.

---

## Speculative Decoding 🎯

Speculative decoding is the cleverest trick in LLM inference. It exploits an asymmetry: **the big model can verify N tokens in parallel as fast as it generates 1 token**.

### The idea

1. A small **draft model** (e.g. 7B parameters) generates 4-8 candidate tokens autoregressively — this is fast because it's a small model
2. The large **target model** (e.g. 70B parameters) takes those 4-8 candidates and runs them all through in **one prefill-like parallel pass**
3. It computes what its own probability distribution would have been at each position
4. Accept each draft token if it matches what the target would have generated (within a probability threshold)
5. Reject the first mismatch, regenerate from there, and repeat

### Why it works

The target model verification pass is cheap — it's essentially a short prefill. If the draft model generates 4 tokens and all 4 are accepted, you've effectively done 4 decode steps in the time of ~1.5 decode steps. Speedup of ~2-3x is typical; 4x is achievable.

**The guarantee:** Speculative decoding is mathematically proven to produce outputs with **identical distribution** to the target model alone. It's not an approximation — it's exact.

### When draft tokens get rejected 🔴

The acceptance rate depends on how well the draft model's output matches the target. For common phrases ("the", "is", "a"), acceptance is nearly 100%. For specialised vocabulary or unusual completions, it may be lower.

Typical acceptance rates: 70-85%. With 4 draft tokens and 80% acceptance, expected accepted tokens per round = 3.2. Combined with the parallel verification, that's still ~2.5x faster.

### Real-world implementations

- **Medusa** (2023): Adds multiple "draft heads" directly to the target model instead of a separate draft model
- **Lookahead decoding**: Uses a Jacobi iteration approach without a separate draft model
- **Assisted generation** in HuggingFace Transformers: Plug-and-play speculative decoding

---

## Continuous Batching in Production ⚙️

At deployment scale, an LLM server handles hundreds of users simultaneously. Continuous batching is how throughput is maximised.

### The naïve approach and why it fails

**Static batching:** Wait until you have N requests, process all N together from start to finish. Problem: requests have very different lengths. User A might generate 20 tokens; user B might generate 500 tokens. User A's GPU slot sits **idle** while waiting for User B to finish.

This is catastrophically wasteful. GPU utilisation can drop to 20-30% with static batching.

### Continuous (iteration-level) batching

**Key insight:** At each decode step, we can choose which requests to include in the batch independently of all other steps.

How it works:
1. Every decode step, look at the pool of active requests
2. Form a batch of N requests for this step
3. When any request generates `<eos>` (end of sequence), immediately remove it from the batch
4. Immediately add a new waiting request to fill the slot
5. The GPU never waits

**Analogy:** A revolving door at a busy building. Instead of waiting for everyone inside to leave before letting the next group in, people enter and exit continuously. The building (GPU) is always full.

### Why this matters enormously

With continuous batching:
- GPU utilisation: 85-95%
- Throughput: 5-10x higher than static batching
- Latency: lower on average because requests start immediately

This is how Claude, ChatGPT, Gemini, and every production LLM handles thousands of simultaneous users. The implementation is in frameworks like vLLM, TensorRT-LLM, and TGI (Text Generation Inference).

### Chunked prefill: one more trick 🍰

Very long prompts (100k tokens) create another problem: their prefill takes so long that all other users are blocked waiting.

**Chunked prefill** splits a long prompt into chunks (e.g. 4096 tokens each) and interleaves the prefill chunks with decode steps from other users:

- Step 1: prefill tokens 1-4096 for User A (long prompt)
- Step 2: decode token for User B, User C, User D
- Step 3: prefill tokens 4097-8192 for User A
- Step 4: decode for Users B, C, D
- ...

This keeps TTFT for other users low while still processing the long prompt.

---

## Time to First Token (TTFT) vs Generation Throughput 📊

These are the two most important latency metrics in LLM serving, and they have opposite pressures.

### Time to First Token (TTFT)

TTFT = time from pressing Enter to seeing the first output token.

This is entirely determined by **prefill time**. Prefill time scales roughly linearly with prompt length:
- 100 tokens: ~20-50ms on an A100
- 1,000 tokens: ~100-300ms
- 10,000 tokens: ~1-3 seconds
- 100,000 tokens: ~10-30 seconds (you really notice this!)

TTFT matters most for **interactive use cases** — chatbots, assistants, code completion. Nobody wants to wait 10 seconds before the first word appears.

### Generation throughput (tokens/second)

Throughput = how many tokens per second the model generates during decode.

This is determined by the memory bandwidth bottleneck described above. Typical values:
- Single user, A100 80GB: ~50-100 tokens/sec for a 70B model
- Batch of 16 users, A100: ~30-50 tokens/sec per user, 500-800 total tokens/sec

Throughput matters most for **batch processing** — summarising documents, generating reports, running evaluations.

### The fundamental tradeoff

Improving TTFT usually means reducing prefill time. You can do this by:
- Using smaller models (but quality drops)
- Using hardware with more compute (expensive)
- Using chunked prefill (helps multi-user, not single-user)

Improving throughput usually means larger batches. But larger batches increase TTFT for individual users (their request waits for the batch to fill up).

**You cannot optimise for both simultaneously.** Production systems make explicit tradeoffs based on their use case.

### Practical SLA targets

Real production services often set:
- P50 TTFT < 500ms (median user sees response start in 0.5s)
- P99 TTFT < 2000ms (99% of users wait less than 2s)
- Throughput > 1000 tokens/sec total across all users

---

## Real-World Numbers 🔢

Let's anchor this with concrete measurements.

### Prefill speeds (as of 2024-2025)

| Hardware | Model | Prompt size | Prefill speed |
|---|---|---|---|
| A100 80GB | Llama 3 70B | 1k tokens | ~350ms |
| A100 80GB | Llama 3 70B | 10k tokens | ~2.8s |
| H100 80GB | Llama 3 70B | 1k tokens | ~180ms |
| H100 80GB | GPT-4 scale (~200B) | 1k tokens | ~500ms (estimated) |
| 2× A100 | Mixtral 8×7B | 4k tokens | ~1.2s |

### Decode speeds (tokens/second)

| Hardware | Model | Batch size | Tokens/sec per user |
|---|---|---|---|
| A100 80GB | Llama 3 8B | 1 | ~120 tok/s |
| A100 80GB | Llama 3 70B | 1 | ~25 tok/s |
| A100 80GB | Llama 3 70B | 16 | ~18 tok/s (288 total) |
| H100 80GB | Llama 3 70B | 1 | ~50 tok/s |
| M2 Ultra (Mac) | Llama 3 8B | 1 | ~30 tok/s |

Human reading speed is ~250 words/minute ≈ 4 words/second ≈ ~5 tokens/second. So even a single-user 70B model on one A100 is already generating ~5× faster than you can read — the streaming is artificially slowed!

### KV Cache memory consumption

| Model | Context length | Precision | KV Cache size |
|---|---|---|---|
| Llama 3 8B | 8k tokens | FP16 | ~1 GB |
| Llama 3 70B | 8k tokens | FP16 | ~8 GB |
| Llama 3 70B | 128k tokens | FP16 | ~128 GB (!!) |
| Llama 3 70B | 128k tokens | INT8 | ~64 GB |

This is why long-context models are so memory-intensive. 128k context on a 70B model requires ~128 GB just for the KV Cache — that's two A100s before you even load the model weights.

---

## Common Misconceptions 🚫

### "LLMs generate text in real-time as they think"
**Wrong.** The model doesn't "think" during decode. Decode is a mechanical process: read cache, multiply matrices, sample token. The "thinking" (if any) happened during the training process when weights were learned. The model doesn't deliberate — it just computes.

### "Longer responses take longer because the model is doing more thinking"
**Partly right, wrong reason.** Longer responses do take longer, but not because of more thinking. Each additional token takes roughly the same amount of compute as the previous one. More tokens = more sequential steps = more time. It's mechanical.

### "Prefill and decode use the same amount of GPU power"
**Wrong.** Prefill typically saturates GPU compute at 80-95% utilisation. Decode with a small batch runs at 5-20% utilisation. This is the core efficiency problem.

### "A model generating 50 tok/s is slow"
**Context-dependent.** 50 tok/s is roughly 10× faster than human reading speed. For batch jobs (e.g. summarising 10,000 documents), 50 tok/s might be too slow. For interactive chat, it's plenty fast.

### "Speculative decoding changes the model's outputs"
**Wrong.** Speculative decoding is mathematically proven to produce outputs with identical distribution to the original model. The draft model is just a shortcut to propose candidates; the target model's distribution is preserved exactly.

### "The KV Cache only needs memory equal to the model weights"
**Often wrong for long contexts.** For a 70B model with 128k context, the KV Cache (~128 GB) is larger than the model weights (~140 GB in FP16). They're roughly comparable, and together they require extreme hardware.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| Prefill | Process the entire input prompt in parallel — fast, compute-bound |
| Decode | Generate output tokens one at a time — sequential, memory-bound |
| TTFT | Time-to-First-Token — how long prefill takes |
| Token/s | How fast tokens are generated during decode |
| KV Cache | Stored K, V vectors from prefill — decode reads this on every step |
| Memory bandwidth bottleneck | Decode is slow because it reads the entire KV Cache each step, not because of compute |
| Speculative decoding | Draft model proposes tokens; large model verifies in parallel — same output quality, faster |
| Continuous batching | Serve many users simultaneously by batching decode steps — key to production efficiency |
| TTFT vs throughput | Fundamental tradeoff: optimising one usually hurts the other |

---

## Up Next
👉 **Logits & Token Selection** — once decode produces a distribution, how is the next token picked?
