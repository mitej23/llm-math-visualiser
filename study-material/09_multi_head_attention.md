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

## Up Next
👉 **Feed-Forward Networks** — the "thinking" layer that processes each token after attention.
