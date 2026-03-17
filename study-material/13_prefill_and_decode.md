# ⚡ Prefill & Decode — Two Phases of Inference

## The Big Idea

When an LLM responds to your message, it operates in two distinct phases:
1. **Prefill:** Quickly process your entire prompt at once
2. **Decode:** Generate response tokens one at a time

Understanding these phases explains why LLMs seem to "think" for a moment before suddenly streaming text — and why long prompts can slow things down.

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

## Phase 1: Prefill 📥

During prefill, the model processes your **entire input prompt** at once.

**What happens:**
- All tokens in your prompt are processed simultaneously through all layers
- All K, V vectors for every input token are computed
- These are stored in the KV Cache
- The model computes the first output token's probability

**Why it's fast:** Parallel processing! Modern GPUs excel at computing many things simultaneously. Processing 1,000 tokens takes roughly the same time as processing 100 tokens (if the GPU isn't memory-limited).

**What you see:** A brief pause before the model starts streaming output. This is prefill time.

**Prefill bottleneck:** Compute. The limiting factor is the GPU's raw matrix multiplication speed.

---

## Phase 2: Decode (Autoregressive Generation) 🖨️

During decode, the model generates tokens **one at a time**, in sequence.

**What happens each step:**
1. Take the previous token (or the last prefill token for the first decode step)
2. Compute its Q, K, V vectors
3. Retrieve all past K, V from cache
4. Run attention against the cache
5. Run through FFN
6. Get logits → sample next token
7. Append that token's K, V to cache
8. Repeat

**Why it's sequential:** Each token's generation depends on the previous token. You can't generate token #5 without knowing what token #4 is.

**Decode bottleneck:** Memory bandwidth. The GPU must read the entire KV Cache on every step. Bigger cache = slower decode.

**What you see:** The streaming text appearing word by word.

---

## Why There's a Time-to-First-Token (TTFT) Delay ⏱️

The gap between you pressing Enter and the first word appearing = prefill time.

This scales with prompt length:
- Short prompt (100 tokens): TTFT ≈ milliseconds
- Long prompt (100k tokens): TTFT ≈ several seconds

**Analogy:** The chef's mise en place takes longer when you ordered a 12-course tasting menu vs. a simple sandwich.

---

## Throughput vs. Latency 📊

These two phases have opposite optimization goals:

| Phase | Goal | Bottleneck | Optimization |
|---|---|---|---|
| Prefill | High throughput (many tokens/sec) | GPU compute | Larger batch size, better hardware |
| Decode | Low latency (fast per-token) | Memory bandwidth | Smaller KV cache, quantization |

**Analogy:**
- Prefill is like loading boxes into a truck (throughput): get as many in as fast as possible
- Decode is like unboxing one item at a time for a customer (latency): each step depends on the previous

---

## Speculative Decoding — A Cleverness 🎯

One bottleneck in decode: generating one token at a time is inherently sequential. Can we parallelize?

**Speculative Decoding** uses a **small, fast "draft" model** to propose several tokens at once, then the **large "target" model** verifies them in parallel:

1. Draft model rapidly generates 5-10 candidate tokens
2. Large model checks all 5-10 in one parallel pass (like prefill!)
3. Accept as many as pass the check, reject the rest
4. Repeat

**Why it works:** The large model can verify many tokens simultaneously (parallel), even though it can only generate one at a time (sequential). If the draft model is right most of the time, you get the large model's quality at the small model's speed.

**Analogy:** A student drafts an essay quickly. A professor reviews it in one sitting, accepting paragraphs that are good and rewriting ones that aren't. The reviewing is much faster than if the professor had written it from scratch.

---

## Continuous Batching — Serving Multiple Users ⚙️

At a deployment level, an LLM server handles many users simultaneously.

**Naïve approach:** Process one user's request fully, then start the next. Very inefficient.

**Continuous batching (iteration-level batching):**
- Each decode step, batch together multiple users' next-token computations
- When one user finishes (generates `<eos>`), immediately add a new user to the batch
- The GPU is always fully utilized

**Analogy:** A taxi dispatcher who continuously assigns cars to passengers as they finish rides, rather than waiting until all current rides are done before dispatching again.

This is how services like Claude, ChatGPT, and others handle thousands of simultaneous users.

---

## Chunked Prefill

For very long prompts, prefill can be split into chunks to balance TTFT and GPU utilization:
- Instead of processing all 100k tokens at once (long TTFT), process 10k at a time
- Interleave with decode steps from other users
- Balances latency across users sharing the server

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| Prefill | Process the entire input prompt in parallel — fast, compute-bound |
| Decode | Generate output tokens one at a time — sequential, memory-bound |
| TTFT | Time-to-First-Token — how long prefill takes |
| Token/s | How fast tokens are generated during decode |
| Speculative decoding | Draft model proposes tokens; large model verifies in parallel |
| Continuous batching | Serve many users simultaneously by batching decode steps |

---

## Up Next
👉 **Logits & Token Selection** — once decode produces a distribution, how is the next token picked?
