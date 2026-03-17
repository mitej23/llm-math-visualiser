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

## Cache Eviction: What to Forget 🧹

For very long sequences, you can't keep everything in the cache. Different strategies for what to drop:

- **FIFO:** Evict oldest tokens first (simple but loses important early context)
- **Attention Score Based:** Evict tokens that received low attention (they're "not important")
- **"Sink" tokens:** Always keep the first few tokens (they tend to receive disproportionate attention — a quirk of transformer training)

---

## Analogy Summary

| Real Life | KV Cache Equivalent |
|---|---|
| Meeting notepad | The cache itself |
| Past meeting summaries | Cached K and V vectors |
| New question being asked | Current token's Query |
| Adding today's notes to the pad | Appending new K, V to cache |
| Running out of notepad pages | Cache eviction / memory limit |

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| KV Cache | Stored Key and Value vectors for all past tokens |
| Why cache K and V | Past tokens don't change — recomputing them is wasteful |
| Cache size | Grows with sequence length × layers × heads × dimension |
| GQA/MQA | Reduce cache by sharing K,V across heads |
| Cache eviction | Strategy for dropping old K,V when memory runs out |
| Speedup | Turns O(N²) generation into approximately O(N) |

---

## Up Next
👉 **Prefill & Decode** — the two distinct phases of LLM inference that use the cache differently.
