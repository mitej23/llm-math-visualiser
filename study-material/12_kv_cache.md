# 💾 KV Cache — Memory for Efficiency

## The Big Idea

When an LLM generates text token by token, it would be incredibly wasteful to recompute the Key and Value vectors for every previous token at every step. The **KV Cache** stores these computed values so they can be reused, making generation dramatically faster.

---

## Real-Life Analogy: The Smart Meeting Notes 📝

Imagine you're in a long meeting where someone is answering questions one at a time. After answering question 5, they need to remember everything said in questions 1-4 to answer question 6.

**Without KV Cache:** Before answering each new question, everyone re-reads the entire transcript of the meeting from the beginning. Extremely slow.

**With KV Cache:** You keep a running notepad. After question 5, the notepad already has summaries of questions 1-4. For question 6, you just add question 6's summary and continue. You never re-read the old parts.

The KV Cache is the notepad — a running store of processed context so the model never re-derives what it already knows.

---

## Why Keys and Values? Not Queries?

Recall from attention:
- **Query (Q):** What is this current token looking for?
- **Key (K):** What does this past token advertise?
- **Value (V):** What information does this past token carry?

When generating token #100:
- Token #100's **Query** is new — it needs to be computed fresh
- Tokens #1-99's **Keys and Values** are unchanged — the past doesn't change!

So we cache the K and V vectors for all past tokens. Only the current token's Q needs to be computed from scratch.

---

## What Gets Recomputed Without KV Cache 🔁

Without a KV cache, every time the model generates a new token, it has to run the full forward pass over ALL previous tokens.

**Let's make this concrete with a story:**

Imagine the model has already generated: "The quick brown fox jumps over the lazy" (9 tokens) and needs to generate token #10.

**Without cache:**
1. Feed all 9 tokens through the embedding layer → 9 embedding vectors
2. For layer 1: compute Q, K, V for all 9 tokens → run full self-attention → FFN for all 9 tokens
3. Repeat for layers 2, 3, ..., 32 (for a 32-layer model)
4. Use the output at position 9 to predict token #10

**Then for token #11:**
1. Start over. Feed all 10 tokens through embedding
2. Compute Q, K, V for all 10 tokens in all 32 layers
3. ...

After 100 tokens, you're running 100 tokens through 32 layers per step. After 1000 tokens, you're running 1000 tokens through 32 layers per step.

**The computational cost:**
- Generating 1000 tokens requires: 1+2+3+...+1000 = ~500,000 "token-layer" forward passes
- With 32 layers: 16 million forward passes total
- Time grows as O(N²) where N is the number of generated tokens

For a 100,000-token document, this would take thousands of times longer than necessary. Completely impractical.

**What specifically is being wasted?**
The key insight: for token #5, its K and V vectors are exactly the same whether you're generating token #6 or token #1000. The past doesn't change. Recomputing K and V for token #5 on every step is pure waste.

---

## How the KV Cache Works — Step by Step 🔧

**Prefill phase (before generation starts):**
1. Feed the entire prompt (e.g., "The quick brown fox") through all layers at once
2. For every layer, save the K and V vectors for every prompt token
3. These are stored in the KV cache: a dictionary of [layer][head] → K and V tensors

**Generation phase (generating new tokens):**

Step 1 — Generating the first new token:
1. Compute Q, K, V for the *last* prompt token position only
2. K and V get appended to the cache
3. Attention: the new Q attends to ALL cached K, V (the whole prompt + any already-generated tokens)
4. Output → new token predicted → append to sequence

Step 2 — Generating each subsequent token:
1. Compute Q, K, V for the *just-generated* token only (1 token, not N tokens!)
2. Append this token's K, V to the cache
3. Attention: new Q attends to all cached K, V (cache grows by 1 each step)
4. Predict next token

**The critical efficiency gain:**
Each generation step processes exactly 1 new token instead of the entire sequence. The cache grows by 1 entry per step, but reading from the cache is much cheaper than recomputing everything.

**Analogy:** Writing a book chapter by chapter. Without cache: rewrite the whole book every time you add a chapter. With cache: just add the new chapter to what you already have.

---

## Step-by-Step: Without Cache

Generating token #N without cache:

1. Process ALL N tokens through all layers: compute Q, K, V for every token
2. Run attention for the last token against all K, V pairs
3. Generate token #N+1

Do this for every new token → O(N²) work grows quadratically. For long documents: catastrophically slow.

---

## Step-by-Step: With Cache

Generating token #N with cache:

1. Compute Q, K, V only for the new token (token #N)
2. **Append** K, V to the existing KV cache (which already has K,V for tokens #1 to #N-1)
3. Run attention: new Q attends to all cached K, V values
4. Generate token #N+1, append to cache

Now generation is O(1) per token (the cache grows linearly but each step is O(N) at most — much better than O(N²)).

---

## Memory Cost of the KV Cache 📦

The cache memory consumption is one of the biggest constraints in deploying LLMs at scale.

**The formula:**
```
KV cache size = 2 × layers × heads × tokens × head_dim × bytes_per_value
```
(The ×2 is for K and V — we store both)

**Worked example — LLaMA 3 8B:**
- 32 layers
- 8 KV heads (it uses GQA)
- head_dim = 128
- float16 = 2 bytes per value

For 4,096 tokens:
```
2 × 32 × 8 × 4096 × 128 × 2 = 536,870,912 bytes ≈ 537 MB
```

For 128,000 tokens (LLaMA 3's context limit):
```
2 × 32 × 8 × 128000 × 128 × 2 ≈ 16.8 GB
```

A model that weighs 16GB (8B params at fp16) needs another 16GB just for the KV cache at max context. This means you need 32GB+ GPU RAM to run LLaMA 3 8B at maximum context length.

**The original Transformer had it worse:**
GPT-3's full MHA (no GQA): 96 layers × 96 heads × tokens × 128 dim × 2 bytes

For 100,000 tokens:
```
2 × 96 × 96 × 100000 × 128 × 2 ≈ 471 GB
```

That's 471GB just for the KV cache — more than the model weights themselves. This is why GPT-3 couldn't practically handle very long contexts.

**Why does the cache stay in memory?**
The KV cache must remain in GPU VRAM (fast memory) to achieve fast generation speeds. If it spills to CPU RAM or disk, generation becomes orders of magnitude slower due to data transfer bottlenecks.

**The memory-quality tradeoff:**
This is the fundamental tradeoff in long-context LLMs. More context → better quality (the model can refer to more information) but exponentially more memory. The innovations below address this tradeoff.

---

## KV Cache Memory Size 📦

The cache grows with:
- **Sequence length** (more tokens = more cached K, V pairs)
- **Number of layers** (each layer has its own K, V)
- **Number of heads** (each head has its own K, V)
- **Dimension per head**
- **Data type** (float16 vs float8)

**Rough calculation** for LLaMA 3 8B generating 4k tokens:
- 32 layers × 32 heads × 2 (K and V) × 4096 tokens × 128 dimensions × 2 bytes (fp16)
≈ **32 × 32 × 2 × 4096 × 128 × 2 bytes ≈ ~1.6 GB**

This is why long contexts can fill GPU memory fast. KV cache is often the bottleneck in deploying LLMs at scale.

---

## Multi-Query Attention — Sharing Keys and Values 🤝

Multi-Query Attention (MQA) was introduced by Noam Shazeer in 2019 specifically to reduce the KV cache size.

**The key idea:**
In standard Multi-Head Attention (MHA):
- 32 query heads, each with its own K and V → 32 K heads + 32 V heads
- KV cache stores 32 pairs of K, V per token

In MQA:
- 32 query heads, but ALL share ONE K head and ONE V head
- KV cache stores just 1 pair of K, V per token
- Cache is 32× smaller!

**How the attention computation changes:**
```
Standard MHA:
  For each head h: Attention(Q_h, K_h, V_h)

MQA:
  For each head h: Attention(Q_h, K_shared, V_shared)
```
Each query head still does independent attention, but they all "look at" the same K and V.

**The quality cost:**
In MQA, all heads see the same information (same K and V). This reduces the diversity of what different heads can attend to. Empirically, MQA causes a small but measurable quality regression.

**Where MQA is used:**
- Falcon: Used MQA for its efficiency
- Some Gemini variants
- PaLM also uses a variant

The quality regression led researchers to develop a middle ground: GQA.

---

## Grouped Query Attention — The Middle Ground ⚖️

Grouped Query Attention (GQA), introduced in 2023 by Ainslie et al., is the compromise between MHA (full quality, large cache) and MQA (degraded quality, tiny cache).

**The idea:**
Instead of all heads sharing K/V (MQA) or each head having its own K/V (MHA), divide query heads into groups. Each group shares one K/V pair.

```
32 query heads, 8 KV heads:
  Group 1: Q heads 1-4 share K₁, V₁
  Group 2: Q heads 5-8 share K₂, V₂
  ...
  Group 8: Q heads 29-32 share K₈, V₈
```

**Cache size comparison:**
- MHA: 32 K + 32 V vectors per token per layer
- GQA (8 groups): 8 K + 8 V vectors per token per layer → 4× smaller cache
- MQA: 1 K + 1 V vector per token per layer → 32× smaller cache

**Quality vs memory tradeoff:**
- GQA quality is very close to MHA (within 1-2% on benchmarks)
- GQA cache is 4-8× smaller than MHA
- GQA quality is noticeably better than MQA

This makes GQA the sweet spot and it's now the standard in modern LLMs.

**GQA in production:**
- LLaMA 2 70B: 64 Q heads, 8 KV heads (8× reduction)
- LLaMA 3 8B: 32 Q heads, 8 KV heads (4× reduction)
- LLaMA 3 70B: 64 Q heads, 8 KV heads (8× reduction)
- Mistral 7B: 32 Q heads, 8 KV heads (4× reduction)
- Gemma 7B: 16 Q heads, 16 KV heads (MHA — full cache)

---

## Strategies to Shrink the KV Cache 🗜️

Since the cache can be large, several techniques reduce its size:

### Grouped-Query Attention (GQA)
Instead of a K, V pair per head, share K, V across groups of heads.
- 32 query heads, but only 8 K/V heads → 4× smaller cache
- Used in LLaMA 2/3, Mistral

### Multi-Query Attention (MQA)
All query heads share a single K, V pair.
- 32 query heads, 1 K/V pair → 32× smaller cache
- Used in Falcon, some Gemini variants

### Quantized KV Cache
Store K, V values in lower precision (int8, int4 instead of float16)
- 2-4× smaller cache, slight quality cost

### Sliding Window Attention (SWA)
Only attend to the last W tokens, discarding older K, V values.
- Fixed cache size regardless of sequence length
- Used in Mistral for long context

---

## Paged Attention — Memory-Efficient KV Cache 📄

When serving LLMs to many users simultaneously, a major problem is memory fragmentation.

**The problem with naive KV cache allocation:**

Imagine serving 100 users simultaneously. Each user's KV cache must be stored contiguously in GPU memory (traditional approach). But:
- User A's sequence is 2,000 tokens → allocate 2,000-token block
- User B's sequence is 500 tokens → allocate 500-token block
- User C might need up to 4,000 tokens but starts at 100 → allocate 4,000-token block

This wastes memory (User B's 1,500 unused slots are reserved but empty) and causes fragmentation (gaps between allocations).

**Paged Attention (used in vLLM):**

Inspired by virtual memory management in operating systems (the paging system), vLLM stores KV cache in fixed-size **pages** (blocks of e.g. 16 tokens).

- Each sequence's KV cache is broken into pages
- Pages don't need to be contiguous in GPU memory
- A page table maps logical positions (token 0, 1, 2...) to physical pages
- When a sequence needs more memory, just allocate a new page anywhere available

**Benefits:**
- Near-zero memory waste (only the last partially-filled page wastes space)
- Multiple sequences can share pages (for prompt sharing — see below)
- No need to pre-allocate maximum context length

**Prompt caching / prefix sharing:**
If 100 users send the same system prompt ("You are a helpful assistant..."), all 100 sequences share the same initial KV cache pages. Those pages are computed once and shared in memory. This can save enormous memory when serving with standard prompts.

**Real-world impact:**
vLLM with PagedAttention achieves 3-24× higher throughput than naive serving frameworks. This is why vLLM became the industry standard for LLM serving almost immediately after its release in 2023.

---

## Cache Eviction: What to Forget 🧹

For very long sequences, you can't keep everything in the cache. Different strategies for what to drop:

- **FIFO:** Evict oldest tokens first (simple but loses important early context)
- **Attention Score Based:** Evict tokens that received low attention (they're "not important")
- **"Sink" tokens:** Always keep the first few tokens (they tend to receive disproportionate attention — a quirk of transformer training)

---

## KV Cache in Production Systems 🏭

Understanding how the KV cache is managed in production shows the engineering challenges at scale.

**Prefill vs decode throughput:**

LLM inference has two phases:
1. **Prefill:** Process the entire prompt at once. Very compute-intensive, memory bandwidth is not the bottleneck. Can batch efficiently.
2. **Decode:** Generate tokens one at a time. The bottleneck is reading the KV cache from memory. Memory bandwidth limited.

At decode time, generating 1 token requires reading the entire KV cache from GPU memory. For a 100k-token context, this is gigabytes of data read per generated token. GPU memory bandwidth (e.g., 2TB/s for H100) limits how fast you can generate.

**Continuous batching:**

Traditional batching: start 100 requests together, they must all finish before the next batch starts. Requests that finish early waste their GPU time.

Continuous batching (vLLM, TGI): when one sequence in a batch finishes, immediately slot in a new request. This keeps GPU utilization high.

The KV cache for finished sequences is freed immediately, making room for new sequences.

**KV cache offloading:**

For very long contexts where the KV cache doesn't fit in GPU VRAM:
- Offload old KV entries to CPU RAM or NVMe SSD
- Fetch them back when needed for attention
- CPU RAM bandwidth (~50GB/s) is much slower than GPU bandwidth (2TB/s), so this is costly but enables longer contexts

Projects like FlexGen use CPU offloading to run large models on consumer hardware.

**Flash Attention's relationship to KV cache:**

Flash Attention (Dao et al., 2022) is a hardware-efficient attention algorithm that doesn't store the full attention matrix in memory. It's often confused with KV caching but they're different:
- Flash Attention: optimises the attention computation itself (recomputing vs storing intermediate attention matrices)
- KV cache: stores K and V across generation steps to avoid recomputation

Modern systems use both together: Flash Attention for efficient attention computation, KV cache for efficient multi-step generation.

---

## Analogy Summary

| Real Life | KV Cache Equivalent |
|---|---|
| Meeting notepad | The cache itself |
| Past meeting summaries | Cached K and V vectors |
| New question being asked | Current token's Query |
| Adding today's notes to the pad | Appending new K, V to cache |
| Running out of notepad pages | Cache eviction / memory limit |
| Sharing notes between meetings | Prefix caching / prompt caching |
| Filing cabinet pages (non-sequential) | Paged Attention pages |

---

## Common Misconceptions ❌

**Misconception 1: "KV cache stores the model's 'memory' of previous conversations"**
Reality: The KV cache is a *computational optimisation* within a single inference session. It doesn't persist between conversations. When you start a new chat, the cache is empty. The "memory" across conversations would need to be explicit retrieval-augmented generation (RAG) or fine-tuning, not the KV cache.

**Misconception 2: "Bigger context window = always better generation quality"**
Reality: Longer contexts are useful, but there are diminishing returns and costs. Models have a "lost in the middle" problem — they attend more to early and late context and less to middle context. Also, each extra token in context makes the next token slightly slower to generate (more KV entries to read). Quality isn't free.

**Misconception 3: "Flash Attention replaces the KV cache"**
Reality: These are complementary techniques. Flash Attention optimises the attention computation within one forward pass. KV cache avoids re-running forward passes across generation steps. You can (and should) use both.

**Misconception 4: "MQA/GQA hurt quality a lot"**
Reality: Empirically, GQA's quality drop compared to full MHA is very small (often < 1% on benchmarks). The memory savings are enormous. This is why GQA is now the default in LLaMA, Mistral, and others.

**Misconception 5: "The KV cache solves the whole inference speed problem"**
Reality: KV cache solves the decode-phase recomputation problem. But prefill (processing the initial prompt) still scales O(N²) with the attention computation. Long prompts are still slow to process initially. Flash Attention and related techniques address prefill efficiency separately.

**Misconception 6: "You only need KV cache for long sequences"**
Reality: KV cache helps from the very first generation step. Even generating a 50-token response benefits significantly from caching, since without it you'd recompute the prompt tokens' K and V 50 times.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| KV Cache | Stored Key and Value vectors for all past tokens |
| Why cache K and V | Past tokens don't change — recomputing them is wasteful |
| Cache size | Grows with sequence length × layers × heads × dimension |
| GQA/MQA | Reduce cache by sharing K,V across heads |
| Paged Attention | Non-contiguous memory for KV cache — more efficient serving |
| Cache eviction | Strategy for dropping old K,V when memory runs out |
| Speedup | Turns O(N²) generation into approximately O(N) |
| Production impact | Critical for serving many users simultaneously efficiently |

---

## Up Next
👉 **Prefill & Decode** — the two distinct phases of LLM inference that use the cache differently.
