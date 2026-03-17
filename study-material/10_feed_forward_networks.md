# 🧮 Feed-Forward Networks — The "Thinking" Layers

## The Big Idea

After attention gathers context from across the sentence, each token passes through a **Feed-Forward Network (FFN)** — a small, private neural network that processes that token independently. If attention is "consulting colleagues," the FFN is "going to your office to think it over."

---

## Real-Life Analogy: The Individual Expert Processing 🧠

Imagine a team meeting at work:

1. **Attention phase:** Everyone shares updates — "The project is delayed," "The client wants X," "Budget was cut." Each person *listens* and takes notes from others.

2. **FFN phase:** Each person then **goes to their own desk** to privately process the information they received and decide what to do with it. They don't consult anyone else — this is solo thinking time.

The FFN is each person's solo desk work. It takes the attention output (enriched with context) and processes it through its own learned transformations.

---

## Structure of a Feed-Forward Network

A standard FFN in a Transformer has this shape:

```
Input (dim d_model)
    ↓
Linear layer: expand to 4× dimension (d_model → 4 × d_model)
    ↓
Activation Function (ReLU, SiLU, GELU)
    ↓
Linear layer: compress back (4 × d_model → d_model)
    ↓
Output (dim d_model)
```

**Analogy:** It's like brainstorming on a whiteboard:
1. **Expand:** Write out every possible angle and idea (4× more space)
2. **Activate:** Filter — highlight only the ideas that "light up" (activation function)
3. **Compress:** Distill into a clean summary (compress back down)

---

## Why Expand to 4× (or More)?

The expansion gives the network a large intermediate space to "think in."

A wider layer means more possible intermediate representations — more room to discover complex patterns before compressing to the final output.

**Analogy:** Brainstorming with 100 sticky notes before narrowing to the 5 best ideas is more powerful than starting with only 5 notes. The expansion is the brainstorm phase.

In practice, modern LLMs use expansion factors of 4× to 8×:
- LLaMA 3 8B: 4,096 → 14,336 (3.5× expanded)
- GPT-3: 12,288 → 49,152 (4×)

---

## SwiGLU — The Modern FFN Variant 🌟

Most modern models don't use a plain FFN. They use **SwiGLU** — a gated version:

Instead of:
```
FFN(x) = activation(x × W1) × W2
```

SwiGLU does:
```
FFN(x) = (SiLU(x × W1) ⊙ (x × W3)) × W2
```

Two parallel linear projections, one gated by a SiLU activation, multiplied element-wise.

**Why?**
- The gate (`W3` path) acts like a filter — it decides how much of the W1 output to let through
- Works like a dynamic volume control: some features get amplified, others get muted
- Empirically produces better model quality

**Analogy:** Two performers on stage:
- **Performer A** (W1): Does the main act
- **Performer B** (W3): Controls the spotlight

The audience sees Performer A — but only the parts illuminated by Performer B's spotlight. The spotlight is learned, so the model figures out what to highlight.

Used in: LLaMA, PaLM, Gemma, Mistral.

---

## The FFN Is Per-Token, Not Cross-Token

This is a crucial distinction from attention:

- **Attention:** Token mixes information with ALL other tokens in the sequence
- **FFN:** Each token is processed **independently** — no cross-token communication

The FFN doesn't "see" neighboring tokens at all. It just takes the attention output for one token and transforms it.

**Analogy:** Attention is a group discussion. The FFN is every person then writing their own personal summary report, in isolation.

---

## What the FFN Learns 📚

Research has found that FFN layers act like **knowledge stores**:

- The middle (expanded) layer stores factual associations learned during training
- Example: the FFN might "know" that if the current context is "Paris is the capital of ___", the relevant completion is "France"
- This factual knowledge is distributed across the weights of the FFN matrices

Some researchers describe FFN layers as "key-value memories" — where certain patterns in the input "retrieve" stored factual knowledge.

**Analogy:** Your long-term memory. Attention is checking in with colleagues about what's relevant. The FFN is consulting your own memory — "what do I already know about this topic?"

---

## FFN vs. Attention — Complementary Roles

| Mechanism | Role | Communication |
|---|---|---|
| Attention | Context gathering — "what's relevant in this sequence?" | Cross-token |
| FFN | Knowledge retrieval — "what do I know about this?" | Per-token (isolated) |

They're designed to complement each other:
- Attention handles **context and relationships** between tokens
- FFN handles **deep transformation** of each token's representation using stored knowledge

---

## FFN Parameters Dominate Model Size

In large models, FFN layers contain the majority of parameters:

For a model with dimension 4096 and FFN expansion 4×:
- FFN matrices: 4096 × 16384 + 16384 × 4096 ≈ **134 million parameters per layer**
- For 32 layers: ~4.3 billion parameters just in FFN layers!

This is why FFN layers are the main target when compressing or quantizing models.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| FFN | Small per-token neural network — processes each token privately |
| Expand then compress | Widen the representation for rich "thinking," then narrow it back |
| 4× expansion | More intermediate space = more powerful pattern detection |
| SwiGLU | Gated FFN variant — one path gates the other, better quality |
| Per-token processing | FFN doesn't see other tokens — no cross-token communication |
| Knowledge store | FFN layers hold factual knowledge learned during training |

---

## Up Next
👉 **Layer Normalization** — keeping values stable so deep networks can train properly.
