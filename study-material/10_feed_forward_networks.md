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

## FFN Position in the Transformer Block 🏗️

The FFN always appears in the second half of each transformer block. The full block looks like:

```
Input (x)
    ↓
LayerNorm
    ↓
Multi-Head Attention
    ↓
Add residual  (x = x + attention_output)
    ↓
LayerNorm
    ↓
FFN   ← HERE
    ↓
Add residual  (x = x + ffn_output)
    ↓
Output
```

Think of the block as a two-stage processing unit:
- **Stage 1 (Attention):** Mix information across tokens — "What is everyone saying?"
- **Stage 2 (FFN):** Process that mixed information deeply — "Now let me think about what I've learned."

The residual connections (the "Add" steps) let the original signal flow through unchanged, while the FFN adds to it. This is critical for training deep networks — the gradient can flow back through the residual path.

**Real-world detail:** In a 32-layer model like LLaMA 3 8B, this pattern repeats 32 times. Each repetition gives the model another chance to gather context (attention) and then deeply reason about it (FFN). The progression from layer 1 to layer 32 is roughly: syntactic understanding → semantic understanding → world knowledge → task-specific reasoning.

---

## Why Expand to 4× (or More)?

The expansion gives the network a large intermediate space to "think in."

A wider layer means more possible intermediate representations — more room to discover complex patterns before compressing to the final output.

**Analogy:** Brainstorming with 100 sticky notes before narrowing to the 5 best ideas is more powerful than starting with only 5 notes. The expansion is the brainstorm phase.

In practice, modern LLMs use expansion factors of 4× to 8×:
- LLaMA 3 8B: 4,096 → 14,336 (3.5× expanded)
- GPT-3: 12,288 → 49,152 (4×)

---

## The 4x Expansion Ratio — Why? 🔬

The 4× ratio wasn't arbitrary — it was chosen empirically by the original Transformer authors (Vaswani et al., 2017) and has stuck because it works well.

**What the wider middle layer actually does:**

Imagine the FFN's intermediate layer as a "feature detector array." Each of the 16,384 neurons in the wide middle layer can represent a different learned feature — like "this token is part of a French phrase," "this context is about finance," "the subject of this sentence is plural," etc.

With only 4,096 neurons, you'd have fewer feature detectors. With 16,384, you can detect a much richer set of patterns.

**The superposition hypothesis:** Researchers believe neural networks can store more features than they have neurons by representing features as overlapping combinations of neuron activations. A wider layer can store more features in superposition. Anthropic's research on mechanistic interpretability has found evidence that FFNs store thousands of distinct feature detectors even in relatively small models.

**The diminishing returns tradeoff:** Going from 4× to 8× expansion adds parameters but each doubling yields smaller quality improvements. The 4× ratio hits a sweet spot of:
- Rich enough intermediate space to detect complex features
- Not so wide that the model wastes compute on redundant neurons
- Parameters dominated by FFN (more on this below)

**Why not just make the model deeper instead of wider?**
Depth (more layers) helps the model build increasingly abstract representations. Width (wider FFN) helps each layer reason with more nuance. Modern architectures use both: many layers *and* wide FFNs. But for a given parameter budget, FFN width is typically more efficient than adding more layers.

---

## The FFN as a Key-Value Memory Store 🗄️

This is one of the most fascinating findings in interpretability research (Geva et al., 2021, "Transformer Feed-Forward Layers Are Key-Value Memories").

**The core insight:** The FFN's first linear layer acts like a set of **keys**. The second linear layer stores the corresponding **values**. When an input pattern matches a key, the associated value is retrieved and added to the token representation.

**How it works mechanically:**
1. Input vector `x` is multiplied by W1 (the key matrix). Each row of W1 is a "key" — a pattern to look for in the input.
2. The activation function (ReLU/GELU/SiLU) fires only for keys that "match" the input. Keys that don't match get zeroed out.
3. The surviving activations are multiplied by W2 (the value matrix). Each row of W2 stores what to output when that key fires.

**A concrete example:** Imagine training on millions of documents. The model sees "Paris is the capital of ___" thousands of times. The FFN may learn:
- **Key neuron:** Activates strongly when the input represents "Paris" in a context asking for capital cities
- **Value neuron:** Adds "France"-like information to the representation

This is why you can sometimes "edit" factual knowledge in LLMs by directly modifying FFN weights (ROME, MEMIT techniques). You're literally editing the key-value store.

**Not just facts:** The key-value memory also stores:
- Grammar patterns ("ing → present continuous")
- Stylistic patterns ("formal → use long sentences")
- Logical patterns ("if negation → flip the conclusion")

**Real model evidence:** Researchers have found that factual associations like "capital of France" are encoded in specific FFN neurons in early-to-middle layers, while more abstract reasoning patterns appear in later layers.

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

## SwiGLU and Gated Linear Units — Deeper Dive 🔬

**What is a "gate" in neural network terms?**

A gate is a learned vector of values between 0 and 1 that is multiplied element-wise with another vector. Values near 0 suppress that feature; values near 1 pass it through. It's like a dimmer switch for each feature dimension.

**The SiLU activation:**
SiLU (also called Swish) is defined as: `SiLU(x) = x × sigmoid(x)`

Unlike ReLU, which is completely off below 0 and on above 0, SiLU:
- Is slightly negative for small negative values (a little bit of negative information can flow)
- Grows smoothly for positive values
- Is differentiable everywhere (better gradients than ReLU)

**Why SwiGLU beats plain SiLU:**
The gating mechanism in SwiGLU adds a second "dimension" of selection. Not only does the activation filter based on magnitude (like SiLU), but the gate further selects *which* features matter for *this particular input*. The gate is computed from the same input `x`, so it's input-dependent — the model can dynamically decide what to highlight based on the context.

**The parameter cost of SwiGLU:**
Standard FFN: W1 (d_model → d_ff), W2 (d_ff → d_model) = 2 matrices
SwiGLU: W1 (d_model → d_ff), W3 (d_model → d_ff), W2 (d_ff → d_model) = 3 matrices

SwiGLU uses 1.5× more parameters for the same d_ff. To keep total parameters equal, models using SwiGLU typically reduce d_ff by 2/3. So instead of 4× expansion, they use ~2.7× expansion but with the extra expressivity of the gate.

**Examples in production:**
- LLaMA 3 8B: d_model=4096, d_ff=14336 (3.5× with SwiGLU)
- PaLM 2: SwiGLU throughout
- Gemma: SwiGLU
- Mistral 7B: SwiGLU

---

## The FFN Is Per-Token, Not Cross-Token

This is a crucial distinction from attention:

- **Attention:** Token mixes information with ALL other tokens in the sequence
- **FFN:** Each token is processed **independently** — no cross-token communication

The FFN doesn't "see" neighboring tokens at all. It just takes the attention output for one token and transforms it.

**Analogy:** Attention is a group discussion. The FFN is every person then writing their own personal summary report, in isolation.

---

## FFN vs Attention — Different Jobs 🥊

| Mechanism | Role | Communication | What it learns |
|---|---|---|---|
| Attention | Context gathering — "what's relevant in this sequence?" | Cross-token | Relationships, dependencies, references |
| FFN | Knowledge retrieval + transformation | Per-token (isolated) | Facts, patterns, linguistic rules |

**A deeper look at why both are needed:**

Imagine processing the sentence: "The Eiffel Tower, which is 330 metres tall, is in Paris."

- **Attention** handles: "Paris" attends to "Eiffel Tower" (they're related), "330 metres" attends to "Eiffel Tower" (it's the height), "is in" attends to both "Tower" and "Paris" (it's the relationship)
- **FFN** handles: Once "Paris" has gathered that it's in a sentence about the Eiffel Tower, the FFN retrieves from memory: "Paris is in France," "Paris is known for the Eiffel Tower," "Paris is a major European city"

Neither can do the other's job:
- Attention can't store factual world knowledge — it only looks at the current sequence
- FFN can't share information between tokens — it only transforms one token at a time

**The division of labour across layers:**
Research suggests that in early layers, attention does more syntactic work (subject-verb agreement, dependency parsing). In later layers, attention becomes more semantic. FFN layers throughout do knowledge retrieval, but the *type* of knowledge shifts from surface patterns in early layers to deep world knowledge in later layers.

---

## What Gets Stored in the FFN? 📚

Research has found that FFN layers act like **knowledge stores**:

- The middle (expanded) layer stores factual associations learned during training
- Example: the FFN might "know" that if the current context is "Paris is the capital of ___", the relevant completion is "France"
- This factual knowledge is distributed across the weights of the FFN matrices

Some researchers describe FFN layers as "key-value memories" — where certain patterns in the input "retrieve" stored factual knowledge.

**Categories of what FFN layers store:**

1. **World facts:** "Paris → France," "Einstein → physicist," "Python → programming language"
2. **Linguistic patterns:** Plural/singular transformations, tense conjugations, common collocations
3. **Logical templates:** "If X then Y" type patterns, negation handling, conditional reasoning
4. **Stylistic features:** Formal vs informal language patterns, domain-specific vocabulary
5. **Mathematical relationships:** "3 + 4 = 7," "square root of 4 = 2" (simple arithmetic facts)

**Evidence from model editing research:**
When you use ROME (Rank-One Model Editing) to change "The Eiffel Tower is located in → Berlin" in a model, you're literally changing specific neurons in a specific FFN layer. The fact was stored in that precise weight matrix. This gives concrete proof that the key-value memory interpretation is real, not just theoretical.

**The temporal aspect:** If a model was trained up to 2024, its FFN stores facts as of 2024. The FFN acts like a frozen encyclopedia — rich with knowledge from training, but not updatable without retraining or fine-tuning.

---

## FFN Layer Size in Real Models 📏

Understanding how FFN sizing scales across different models gives intuition about the tradeoffs model designers make.

| Model | d_model | d_ff | Expansion | FFN Type | Params per FFN |
|---|---|---|---|---|---|
| GPT-2 (124M) | 768 | 3,072 | 4× | Standard | ~4.7M |
| GPT-3 (175B) | 12,288 | 49,152 | 4× | Standard | ~1.2B |
| LLaMA 3 8B | 4,096 | 14,336 | 3.5× | SwiGLU | ~176M |
| LLaMA 3 70B | 8,192 | 28,672 | 3.5× | SwiGLU | ~705M |
| Mistral 7B | 4,096 | 14,336 | 3.5× | SwiGLU | ~176M |
| Gemma 7B | 3,072 | 24,576 | 8× | SwiGLU | ~226M |

**Key observations:**
- Gemma 7B uses 8× expansion — more aggressive FFN widening
- GPT-3 uses classic 4× with no gating — older architecture
- LLaMA models consistently use ~3.5× with SwiGLU to balance parameter count vs quality
- Per-layer FFN parameter counts are enormous — this is where model size comes from

**FFN parameters vs total model parameters:**
For a model with d_model=4096, d_ff=14336, 32 layers (SwiGLU, 3 matrices):
- Per layer: 4096×14336 + 14336×4096 + 4096×14336 ≈ 176M params
- 32 layers: ~5.6B params just in FFN
- For an 8B model, FFN is ~70% of total parameters!

This is why FFN layers are the prime target when compressing or quantizing models. Reducing FFN precision from float16 to int8 halves the memory for 70% of your model.

---

## Common Misconceptions ❌

**Misconception 1: "The FFN is just a standard neural network, nothing special"**
Reality: The FFN's role as a key-value memory is a deep insight. It's not just transforming vectors — it's retrieving stored knowledge from its weights. The pattern of expand → activate → compress is specifically designed to enable this knowledge retrieval efficiently.

**Misconception 2: "Attention does all the work; FFN is just cleanup"**
Reality: FFN layers contain ~70% of model parameters and do the heavy lifting of knowledge retrieval and feature transformation. Studies that ablate (remove) FFN layers show dramatic quality degradation — often worse than removing attention layers.

**Misconception 3: "Making the FFN wider always helps"**
Reality: There are diminishing returns. Going from 4× to 8× expansion helps, but going to 16× helps much less while doubling the parameter count. The compute and memory cost grows proportionally but the quality gain flattens.

**Misconception 4: "SwiGLU is only a minor improvement"**
Reality: Gated linear units (SwiGLU, GeGLU) consistently outperform plain FFNs in quality benchmarks. The empirical gains are large enough that virtually all major new models have switched to gated variants.

**Misconception 5: "The FFN can learn new facts after training"**
Reality: FFN weights are frozen after training. The model's "factual knowledge" is fixed at training time. This is why LLMs can confidently hallucinate — the FFN retrieves the nearest stored pattern even when it doesn't exactly match the query, sometimes producing plausible-but-wrong facts.

**Misconception 6: "Each token uses a different FFN"**
Reality: All tokens at the same layer pass through the *same* FFN weights. The FFN is a shared function, not a per-token network. What differs is the *input* to the FFN (each token's attention output is different), producing different outputs from the same weights.

---

## FFN Parameters Dominate Model Size

In large models, FFN layers contain the majority of parameters:

For a model with dimension 4096 and FFN expansion 4×:
- FFN matrices: 4096 × 16384 + 16384 × 4096 ≈ **134 million parameters per layer**
- For 32 layers: ~4.3 billion parameters just in FFN layers!

This is why FFN layers are the main target when compressing or quantizing models.

---

## Connections to Other Topics 🔗

- **Residual connections** (topic 9): The FFN output is added to the input, not replacing it. This residual design is essential — without it, deep networks can't train stably.
- **Layer normalization** (topic 11): Applied before the FFN in pre-norm models to keep the FFN's input well-scaled.
- **Mixture of Experts** (MoE): Some modern models (Mixtral, GPT-4 likely) replace a single FFN with multiple FFNs and a router that selects which FFN to use. This dramatically increases capacity without proportionally increasing compute.
- **Knowledge editing:** Techniques like ROME and MEMIT directly modify FFN weights to update factual knowledge without full retraining.

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
| Key-value memory | First layer "looks up" patterns; second layer "retrieves" stored facts |
| FFN vs attention | Attention mixes tokens; FFN transforms each token's knowledge independently |

---

## Up Next
👉 **Layer Normalization** — keeping values stable so deep networks can train properly.
