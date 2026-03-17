# 🔭 Multi-Head Attention — Multiple Perspectives

## The Big Idea

A single attention mechanism can only "look" at words from one perspective at a time. **Multi-Head Attention** runs attention multiple times in parallel, each time using different learned projections — allowing the model to capture many different types of relationships simultaneously.

---

## Real-Life Analogy: The Expert Panel 👥

Imagine you're hiring a new employee and you have a panel of 8 experts, each evaluating the candidate from their specialty:

- **HR expert** focuses on: cultural fit, communication skills
- **Technical lead** focuses on: coding ability, problem-solving
- **Manager** focuses on: teamwork, reliability
- **CEO** focuses on: strategic thinking, ambition

Each expert asks the same candidate (the same input) different questions, and comes away with different assessments. You combine all their feedback to make a complete hiring decision.

Multi-head attention is this expert panel. Each "head" is an expert that looks at the input from a different learned angle.

---

## How It Works

For each attention head `i` from 1 to H (H = number of heads):

1. **Project** the input into smaller Q, K, V vectors using learned weight matrices
2. **Run attention** independently to get an output
3. The outputs from all heads are **concatenated** (stacked together)
4. **Projected again** with a final weight matrix to produce the output

**Analogy:** Each expert writes their assessment on a notecard. All notecards are taped together into one long document. A coordinator then summarizes the combined document into a final recommendation.

---

## Why Smaller Projections?

If the embedding dimension is 512 and you have 8 heads, each head works with 64-dimensional queries, keys, and values (512 ÷ 8 = 64).

**Why?** To keep the total computation the same as single-head attention. You get the richness of 8 perspectives for the same cost as 1 full-dimension perspective.

**Analogy:** Instead of one photographer taking 8 photos sequentially, you hire 8 photographers who each take a smaller-format photo simultaneously. Same total work, but you get 8 different angles in parallel.

---

## What Different Heads Learn 🎓

Research has found that different attention heads specialize in different linguistic patterns:

**Example patterns found in real models:**
- Head 1: Detects the **subject** of the sentence
- Head 2: Tracks **coreference** (what pronouns refer to)
- Head 3: Notices **syntactic dependencies** (verb ↔ object)
- Head 4: Connects **related entities** (person ↔ their possessions)
- Head 5: Links **temporal markers** (yesterday ↔ past-tense verbs)
- Head 6: Finds **contrasting elements** ("but," "however")
- Head 7–8: Capture local neighborhood context

No one programs these patterns — they emerge from training on language.

**Analogy:** When you learn to read, you automatically develop different mental "scanners" — one for spelling, one for grammar, one for meaning, one for tone — without anyone teaching you to split your attention this way. Your brain does it because it works better.

---

## Multi-Head vs. Single-Head

**Single-head attention:**
- One perspective
- May "average out" multiple types of relationships
- Misses subtle distinctions

**Multi-head attention:**
- Multiple perspectives simultaneously
- Each head specializes in a different relationship type
- Richer, more nuanced representations

**Analogy:** Trying to describe a diamond with only one light source vs. a jeweler's loupe under many different lighting angles. The multi-angle view reveals facets invisible from any single direction.

---

## Grouped-Query Attention (GQA) — A Modern Optimization 🚀

In very large models, having a separate Key and Value matrix for every head becomes memory-intensive.

**Grouped-Query Attention** shares K and V matrices across groups of heads:

- 8 heads, but only 2 sets of K/V matrices (4 heads share each set)
- Each head still has its own Query matrix
- Massive memory savings, small quality cost

**Analogy:** 8 detectives share 2 evidence boards, but each detective asks their own unique questions about the evidence. The shared board is efficient; the unique questions still give varied perspectives.

Used in: LLaMA 2/3, Mistral, Gemma.

---

## Multi-Query Attention (MQA) — The Extreme Version ⚡

The extreme case: **one** K/V set shared by **all** heads.

- All heads look at the same "evidence board"
- Only the Query varies per head
- Maximum memory efficiency, moderate quality cost

Used in: Falcon, PaLM, some Gemini variants.

---

## Flash Attention — Making It Fast ⚡

Attention is computationally expensive — for a sequence of length N, standard attention is O(N²) in memory (every pair of tokens needs a score).

**Flash Attention** is a clever algorithm that computes the same result while using much less memory, by reorganizing how the computation accesses GPU memory.

It doesn't change *what* attention computes — just *how efficiently* it computes it.

**Analogy:** Two methods for sorting a huge pile of mail:
- Standard: Lay all mail on the floor, compare every pair → needs a huge floor (memory)
- Flash: Process in batches, keeping only what you need at each step → same result, smaller workspace

Flash Attention 2 and 3 are used in all modern production LLMs.

---

## Number of Heads in Practice

| Model | Heads | Dimension | Dimension per Head |
|---|---|---|---|
| GPT-2 small | 12 | 768 | 64 |
| LLaMA 3 8B | 32 | 4,096 | 128 |
| LLaMA 3 70B | 64 | 8,192 | 128 |
| GPT-4 (est.) | 128+ | ~25,600+ | ~200+ |

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| Multi-Head Attention | Run attention multiple times in parallel with different learned perspectives |
| Head | One attention mechanism with its own Q, K, V projection weights |
| Parallel perspectives | Different heads specialize in different relationship types |
| Concatenate + Project | Combine all head outputs into one representation |
| GQA | Share K/V across groups of heads to save memory |
| MQA | Share K/V across ALL heads — maximum efficiency |
| Flash Attention | Efficient algorithm for computing the same attention with less GPU memory |

---

## Why One Head Isn't Enough 🔎

This is the core motivation for multi-head attention, and it's worth really understanding.

### The Averaging Problem

Single-head attention computes one attention distribution over the sequence. This distribution is a weighted blend — it has to "decide" how to weight many different relationships all at once, using one set of scores.

Consider the sentence: **"The lawyer defended her client because she believed in justice."**

Multiple relationships are simultaneously important here:
1. "she" refers to "lawyer" (coreference)
2. "defended" takes "client" as its object (syntactic dependency)
3. "believed" takes "justice" as its prepositional object
4. "lawyer" and "defended" share a subject-verb relationship

A single attention head trying to capture all of these at once faces an impossible trade-off. If it strongly focuses on the coreference link (she → lawyer), it might miss the syntactic dependency (defended → client). If it captures both moderately, it captures neither well.

**The result:** Single-head attention tends to produce "averaged" representations that capture dominant patterns but miss the subtler, equally-important relationships.

### An Analogy: The Jack-of-All-Trades Problem 🔧

Imagine hiring one person to simultaneously be your accountant, lawyer, therapist, and personal trainer. They'd have to split their attention between all four roles, and the quality of service in each area would be diluted. You'd get a mediocre accountant, mediocre legal advice, and a so-so workout.

Hiring four specialists — one for each role — and then combining their reports gives you expert-quality service in each domain plus a rich overall picture.

Multi-head attention does exactly this: each head is a specialist, not a generalist.

### Experimental Evidence

The paper "Are Sixteen Heads Really Better Than One?" (Michel et al. 2019) ablated individual heads in BERT and found:

- A majority of heads can be removed with minimal performance loss (they were redundant or unimportant for that task)
- But a **small subset of heads** are critically important — removing them causes a large performance drop
- These critical heads often specialize in a specific, identifiable linguistic pattern

This suggests the model learns to concentrate important functions in a small number of heads, while other heads provide backup or handle rare cases.

---

## Each Head's Dimension — How the Math Splits Up 📐

The dimension arithmetic behind multi-head attention is elegant. Let's trace through it carefully.

### The Setup

Suppose:
- `d_model = 512` (the embedding dimension)
- `H = 8` (number of heads)
- `d_head = d_model / H = 512 / 8 = 64`

### Each Head's Projection Matrices

For each head `h`:
- `W_Q^h`: shape `512 × 64` — projects from full embedding space to 64-dim query space
- `W_K^h`: shape `512 × 64` — projects to 64-dim key space
- `W_V^h`: shape `512 × 64` — projects to 64-dim value space

Each head runs full attention in its 64-dimensional space and produces a 64-dimensional output per token.

### Total Weight Count (Attention Only)

8 heads × 3 matrices × (512 × 64) = 8 × 3 × 32,768 = **786,432 parameters** for Q, K, V.

This is exactly the same as one full-dimensional attention head:
1 head × 3 matrices × (512 × 512) = 3 × 262,144 = **786,432 parameters**.

The computational cost is identical — you just distribute it across 8 smaller, parallel operations.

### The Output Projection

After the 8 heads run independently, each produces a `[N × 64]` output. These are concatenated along the last dimension:

```
Concatenated: [N × (64 × 8)] = [N × 512]
```

This is then multiplied by the output projection matrix `W_O` (shape `512 × 512`) to produce the final `[N × 512]` output.

**Why the output projection?** Without it, you'd just have 8 independent attention outputs concatenated. The projection matrix lets the model learn how to *combine* the 8 perspectives into a single coherent representation. This learned combination is itself important — the model learns which heads' outputs to emphasize for which situations.

### What Happens When d_model Isn't Divisible by H?

In practice, model designers choose H to divide d_model evenly. But if needed, you can use slightly unequal head sizes or pad to the nearest multiple. Modern models like LLaMA 3 use `d_model = 4096, H = 32, d_head = 128` — a clean division.

---

## What Different Heads Specialize In 🎯

The specialization that emerges in multi-head attention is one of the most remarkable aspects of Transformers. Nobody programmed these behaviors — they emerged purely from training on text prediction.

### Evidence from BERT Interpretability Research

Researchers at Google and Hugging Face have analyzed attention heads in BERT and found consistent patterns:

**Syntactic heads:**
- Some heads strongly attend from verbs to their direct objects ("eat" → "apple")
- Other heads track the subject of the main clause
- Certain heads detect noun phrase boundaries

**Semantic heads:**
- Some heads connect pronouns to their antecedents across long distances
- Others link semantically related words (synonyms, hyponyms)
- Some capture "is-a" relationships ("poodle" → "dog" → "animal")

**Positional heads:**
- Many heads in lower layers strongly attend to nearby tokens (window of 1-3 positions)
- Some heads specifically attend to the previous token
- Others attend to the next token (even in causal models, within the allowed context)

### Evidence from GPT-Style Models

Research on GPT-2 and larger models found:

**Induction heads** (discovered by Anthropic researchers):
These heads implement a specific pattern: if the model has seen sequence [A][B] earlier, and now sees [A] again, the induction head strongly attends from the current [A] to the [B] that followed the previous [A], predicting [B] will come again.

This is thought to be the mechanistic basis for in-context learning — the model's ability to learn from examples provided in its prompt without updating its weights.

**Copy heads:**
Some heads are almost purely about copying — they attend to a token and essentially "copy" its representation into the current position. Useful for co-referent pronouns, repeated entities, and other situations where a token needs to carry forward a previous representation.

### Head Specialization Is Emergent, Not Designed

Nobody wrote code saying "head 4 should track coreference." The model discovered this split because it reduces the training loss. Tracking many different relationship types simultaneously is a better strategy for predicting the next token than having all heads do the same thing.

This is a perfect example of **emergent capability** — complex, interpretable structure arising from a simple objective (predict the next token) applied at scale.

---

## Induction Heads — The Discovery That Changed Understanding 🔬

This deserves its own section because it's one of the most significant discoveries in mechanistic interpretability.

### What Are Induction Heads?

In GPT-style models, researchers found a specific, identifiable two-head circuit:

1. **Previous token head:** This head attends from each position to the token *before* it. Output: each position gets a copy of the previous token's representation.

2. **Induction head:** This head, using the output of the previous-token head as its key, can identify "I've seen this token before" and then strongly attend to *what came after* it last time.

Together, these two heads implement a lookup table: "If I've seen this token before, what came after it?"

### Why Induction Heads Matter

Induction heads are the proposed mechanistic explanation for **in-context learning** — the mysterious ability of large language models to perform new tasks when given a few examples in the prompt:

```
Translate to French:
cat → chat
dog → chien
bird → ?
```

The model answers "oiseau" not because it was explicitly trained on this mapping, but because induction heads recognize the pattern: [English word] → [French word], and apply it to the new query.

**This was an extraordinary finding:** a specific, two-head circuit appears to be responsible for a capability (in-context learning) that many considered a form of "emergent intelligence." The behavior has a precise mechanistic explanation.

### Induction Heads and Phase Transitions

Research found that induction heads form abruptly during training — at a specific point (a "phase transition"), the loss drops suddenly and induction heads appear. Before the transition, the model has no in-context learning ability. After, it does.

This suggests that some apparently "continuous" improvements in model capability during training are actually step-function jumps corresponding to the formation of specific circuits.

---

## Grouped Query Attention (GQA) 💾

As models scale to billions of parameters and generate thousands of tokens, the **KV cache** becomes a serious bottleneck.

### The KV Cache Problem

When generating text auto-regressively, the model must re-compute attention for every token generated. To avoid redundant computation, the key and value vectors for all past tokens are cached in GPU memory (the KV cache).

For a model with 32 heads, 80 layers, 4096 d_model, and a 128K context window:
- KV cache size = 2 × 32 heads × 80 layers × 128,000 tokens × 4096 dimensions × 2 bytes (fp16)
- That's approximately **170 GB** — far more than a single GPU's VRAM

This makes serving long-context models extremely expensive.

### GQA: Share KV Across Query Groups

**Grouped Query Attention (GQA)**, introduced in Google's 2023 paper and adopted by LLaMA 2+, LLaMA 3, Mistral, and Gemma, solves this by sharing Key and Value heads across groups of Query heads:

- Full attention: 32 Query heads + 32 Key heads + 32 Value heads
- GQA (8 KV heads): 32 Query heads + 8 Key heads + 8 Value heads
  - Each group of 4 Query heads shares one set of K/V heads

**Memory reduction:** 8/32 = 25% of the original KV cache memory, while keeping 32 distinct Query heads (and thus 32 distinct attention patterns).

**Quality:** Empirically, GQA with 8 KV groups loses almost no performance compared to full multi-head attention. The diversity in attention comes mainly from diverse Query projections, and the shared KV doesn't hurt much.

**Intuition:** The Keys and Values describe *what information is available* (the "library catalogue"). The Queries describe *what each head is looking for* (the "search query"). You need diverse searches (many Query heads) but don't need redundant catalogue copies (many identical KV copies).

### GQA in Real Models

| Model | Query Heads | KV Heads | Reduction |
|---|---|---|---|
| LLaMA 2 7B | 32 | 32 | None (MHA) |
| LLaMA 2 70B | 64 | 8 | 8× |
| LLaMA 3 8B | 32 | 8 | 4× |
| LLaMA 3 70B | 64 | 8 | 8× |
| Mistral 7B | 32 | 8 | 4× |
| Gemma 7B | 16 | 1 (MQA) | 16× |

---

## Multi-Query Attention (MQA) 🏎️

Multi-Query Attention is the extreme version of GQA: a single set of Key and Value heads shared by *all* Query heads.

### The Architecture

```
Standard MHA: H query heads, H key heads, H value heads
GQA:          H query heads, G key heads (G < H), G value heads
MQA:          H query heads, 1 key head,  1 value head
```

With MQA, every Query head attends to the exact same Keys and Values. The diversity of attention patterns comes entirely from the different Query projections.

### Memory and Speed Benefits

MQA reduces the KV cache by a factor of H (number of heads). For a 32-head model, this is a 32× reduction in KV cache memory. This is enormous for batch inference — you can serve 32× more users with the same GPU memory, or allow 32× longer contexts.

### Quality Trade-off

MQA does degrade quality somewhat compared to full MHA, especially on tasks that benefit from diverse attention patterns (e.g., complex reasoning, long-range coreference). The "Shazeer 2019" paper that introduced MQA reported modest quality losses, especially on code generation and structured prediction tasks.

This is why GQA (a middle ground) became the preferred solution — it captures most of the memory benefits while preserving more of the quality.

### When to Use What?

| Variant | Memory Use | Quality | Best For |
|---|---|---|---|
| MHA (full) | High | Best | Small models, highest quality tasks |
| GQA | Medium | Near-MHA | Most production LLMs today |
| MQA | Lowest | Moderate | High-throughput serving, very long contexts |

---

## How Many Heads Do Real Models Use? 📊

The "right" number of heads is an empirical question that scaling law researchers have studied.

### The Basic Rule: d_head ≈ 64–128

Almost all models converge on a head dimension of 64 to 128. Below 64, individual heads don't have enough capacity to represent complex patterns. Above 128, you're getting diminishing returns and adding parameters without proportional benefit.

Given `d_head = d_model / n_heads`, this means:
- A model with d_model = 768 should have roughly 768/64 = 12 heads (GPT-2: exactly 12)
- A model with d_model = 4096 should have roughly 4096/64 = 64 heads, or 4096/128 = 32 heads (LLaMA 3 8B: 32)

### Does More Heads Always Help?

Not necessarily. Beyond a certain point, more heads provide diminishing returns because:
1. Each head's dimension shrinks (each head is less expressive)
2. The output projection has to combine more, lower-quality outputs
3. More heads means more parameters in the Q, K, V projection matrices (but not more parameters than you'd have with fewer larger heads at the same d_model)

The "Are Sixteen Heads Really Better Than One?" paper found that pruning 50–80% of heads in BERT caused only minor performance degradation on most tasks. Some heads were effectively redundant.

### The Trend Toward GQA

The big shift in the field (2022–2024) has been:
- Moving from MHA to GQA for Query heads while keeping full Query diversity
- Keeping d_head around 128
- Scaling depth (number of layers) more than width (d_model or n_heads)

This reflects the insight that KV cache memory is the primary bottleneck at inference time, and GQA addresses it directly.

---

## Common Misconceptions ❌

### "More heads always means better model"

Not true. More heads with the same d_model means smaller d_head per head, which may hurt quality. The empirical sweet spot is d_head ≈ 64–128. Doubling the number of heads while keeping d_model fixed (halving d_head) typically hurts performance.

### "All 32 heads are always doing different things"

Studies show many heads learn similar or redundant functions, especially in smaller models. In BERT-base (12 heads), up to 8 heads can be pruned per layer with minimal performance loss. The model learns "backup" heads that overlap in function, providing robustness.

### "GQA loses a lot of quality compared to full MHA"

Empirically, GQA loses very little quality compared to full MHA, especially with 8 or more KV groups. The KV diversity matters less than Query diversity for most tasks. LLaMA 3 70B uses GQA with 8 KV heads for its 64 Query heads (8× compression) with negligible quality loss.

### "Attention heads are like neurons — each has a specific function you can look up"

Some heads have interpretable, consistent functions (like the induction heads). But many heads have functions that vary depending on context, interact in complex ways with other heads, or perform computations that don't map cleanly to human-interpretable linguistic concepts. Interpretability is an active research area and we don't have complete answers.

### "Flash Attention changes what the model computes"

Flash Attention is a pure algorithmic optimization — it produces mathematically identical outputs to standard attention, just computed more efficiently. It's like using a faster route to the same destination. The model's behavior doesn't change at all.

### "Multi-head attention was always in the original Transformer"

Yes, the original 2017 "Attention Is All You Need" paper did introduce multi-head attention. But many of the key improvements — GQA, MQA, Flash Attention, RoPE, SwiGLU activations — came in the 2020–2023 period and are what make modern LLMs so capable and efficient.

---

## Up Next
👉 **Feed-Forward Networks** — the "thinking" layer that processes each token after attention.
