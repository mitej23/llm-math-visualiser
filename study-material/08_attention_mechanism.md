# 👁️ Attention Mechanism — Relating Words to Each Other

## The Big Idea

**Attention** is the revolutionary idea at the heart of the Transformer. It lets every word in a sentence decide which other words are most relevant to understanding its meaning — and blend in information from those words.

"The bank near the river is muddy" — to understand what "bank" means here, you need to "attend" to "river."

---

## Real-Life Analogy: The Detective's Board 🕵️

Picture Sherlock Holmes standing at his evidence board, with photos, notes, and strings connecting related clues.

When Holmes is trying to understand one clue (say, a muddy boot print), he:
1. **Looks at all other clues** (the river map, the fishing rod, the suspect's alibi)
2. **Decides which clues are relevant** to interpreting the boot print
3. **Pulls information from those relevant clues** into his current reasoning
4. **Updates his understanding** of the boot print accordingly

Attention is exactly this process — every token looks at all other tokens, decides relevance, and borrows information from the most relevant ones.

---

## The Three Roles: Query, Key, Value 🗝️

Every token plays **three roles simultaneously**. Think of a library:

### 🔍 Query — "What am I looking for?"
When the word "bank" wants to understand its context, it sends out a **query**: "I'm looking for information about my surroundings — water? money? construction?"

### 🏷️ Key — "What am I advertising?"
Every word also has a **key** — a description of what information it can offer: "river" advertises "I'm a body of water." "finance" advertises "I'm about money."

### 📦 Value — "What information do I carry?"
Every word also has a **value** — the actual information to share if selected: "river" offers its full contextual meaning.

**The mechanism:**
1. Compare the query of "bank" with the keys of every other word
2. Compute similarity scores (higher = more relevant)
3. Use those scores as weights to blend the values together
4. The result is an updated representation of "bank" enriched by context

---

## The Attention Score — Computing Relevance 📊

To find how relevant word B is to word A:

1. Take word A's **query** vector
2. Take word B's **key** vector
3. Compute the **dot product** (multiply element-wise and sum)
4. A high dot product = high similarity = B is relevant to A

**Analogy:** Imagine you're looking for something in a store. Your "query" is "I want something sweet and crunchy." Each product on the shelf has a "key" (its label). You compare your query to each label — the more similar the label to your query, the higher the score.

---

## Softmax — Turning Scores into Weights 🎚️

The raw scores are converted to weights using **softmax** — a function that:
- Keeps all weights positive
- Makes them sum to 1 (like percentages)
- Amplifies high scores and shrinks low scores

Result: "bank" assigns 70% weight to "river," 5% to "the," 1% to "muddy," etc.

**Analogy:** After comparing all products in the store, you don't buy ALL of them equally — you spend most of your money on what best matches what you wanted, and a little on alternatives.

---

## Weighted Value Blending 🎨

Now blend the **values** of all words using these weights:

```
New representation of "bank" =
  0.70 × value("river") +
  0.10 × value("muddy") +
  0.05 × value("the") +
  ... (all other words at small weights)
```

The result is a new vector for "bank" that is now **semantically colored by "river"** — the model now knows this "bank" is the waterside kind, not the financial kind.

**Analogy:** You're mixing paints. The color of "bank" gets mostly influenced by the dominant "river blue," with small hints of other colors. The final color tells you "this is a riverside bank."

---

## The Scaling Factor — Why Divide by √d? 📐

In practice, the dot product scores are divided by √(dimension) before softmax.

**Why?** When vectors have many dimensions, dot products can become very large, making softmax collapse to an almost-one-hot distribution (one token gets 99.9% weight).

**Analogy:** Imagine a volume knob that goes up to 1,000,000. You turn it to 10 for a gentle sound, but the difference between 10 and 11 is lost in the noise. Scaling brings everything to a manageable range, so the differences between attention scores are meaningful.

---

## Self-Attention vs. Cross-Attention 🔄

**Self-attention:** Words in a sequence attend to each other.
- "The bank near the river" — bank attends to river, near, the
- Used in: every transformer block

**Cross-attention:** One sequence attends to another.
- Translation: English words attend to French words
- Used in: encoder-decoder models, multimodal models

Most LLMs (GPT, LLaMA) use only self-attention.

---

## What Attention Learns 🧠

Through training, attention heads learn to specialize:

- Some heads track **subject-verb agreement** ("The dogs *run*" — "dogs" and "run" strongly attend to each other)
- Some heads resolve **coreference** ("When she arrived, *she* went to..." — the second "she" attends to the first)
- Some heads capture **syntactic structure** (detect noun phrases, verb phrases)
- Some heads capture **semantic relationships** (synonyms, related concepts)

---

## A Complete Worked Example 🔬

Sentence: "The cat sat on the mat because it was soft."

What does "it" refer to? The mat.

In attention:
- Query of "it" looks for: "what nearby noun is this pronoun referring to?"
- Key of "mat" signals: "I'm a physical object that can have softness"
- Key of "cat" signals: "I'm an animal that can sit"
- Since "soft" comes right after "it," and "mat" can be soft, "mat's" key scores highly
- "it" blends in the value of "mat" → now knows "it = mat"

The model has resolved the pronoun through attention.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| Attention | Each token looks at all others and decides who's relevant |
| Query | What am I looking for? |
| Key | What do I advertise about myself? |
| Value | What information do I share if selected? |
| Attention Score | Dot product of query and key — higher = more relevant |
| Softmax | Converts scores to weights that sum to 1 |
| Weighted Blend | Mix values using attention weights to get a context-enriched representation |
| Self-Attention | Words in one sequence attending to each other |

---

## The Query-Key-Value Framework in Depth 🏛️

The QKV framework is so central to modern AI that it's worth really internalising. Let's go beyond the quick description and understand *why* this particular abstraction is so powerful.

### The Library Analogy (Extended) 📚

Imagine you walk into a vast library and you're looking for information about space exploration. You have a specific question in mind: "What were the key engineering challenges of the Apollo missions?"

- Your **Query** is the question: "What engineering challenges? Apollo? Moon landings?"
- Every book in the library has a **Key** — its catalogue card: "Engineering, Space, NASA, 1960s"
- The **Value** of each book is its actual content — all the detailed information inside it

You compare your query against every catalogue card (Key). Some cards match well:
- "Apollo 13 by Jim Lovell" — excellent key match → high score
- "Advanced Rocket Propulsion" — good key match → medium score
- "French Cooking Techniques" — terrible key match → near-zero score

You pull books in proportion to how well their keys matched your query. You mostly read "Apollo 13," skim "Advanced Rocket Propulsion," and completely ignore the cookbook.

Now here's the crucial insight: **the Query, Key, and Value are all learned projections of the same token embedding.** The model learns, through training, how to project the input vector into three different "spaces" — one that is good at asking questions, one that is good at describing content, and one that carries the actual useful information to share.

### Why Three Separate Matrices?

You might wonder: why not just compute similarity directly between token vectors? Why the extra Q, K, V projections?

**Because raw token vectors are not optimised for comparison.** The embedding space packs a lot of information (syntax, semantics, position) into a fixed-size vector. The Q and K projections transform the vectors into a subspace where dot-product similarity is *meaningful* for the current layer's purpose.

Different layers can learn different projection matrices, meaning the same token is "asked about" and "describes itself" differently at different levels of abstraction. At layer 1, the query might be "what's the part of speech near me?" At layer 20, it might be "what entities in this document are coreferent with me?"

### The Projection Matrices

Concretely, for a model with d_model dimensions:

- Q projection: `d_model × d_k` matrix (turns token vector into query vector)
- K projection: `d_model × d_k` matrix (turns token vector into key vector)
- V projection: `d_model × d_v` matrix (turns token vector into value vector)

These matrices are randomly initialized and learned entirely from the training data. Nobody writes rules about what the queries or keys should represent — the model discovers this on its own by learning to predict the next token better.

---

## Dot Product Attention — Step by Step 🔢

Let's trace through an exact numerical example with a tiny sequence of 4 tokens.

**Sentence:** "Cats chase mice."
**Tokens:** [Cats, chase, mice, .]
**Simplified dimension:** 3 (for illustration; real models use 64–128)

### Step 1: Compute Q, K, V for each token

Each token's embedding is multiplied by the learned Q, K, V matrices:

```
Q = Embedding × W_Q    → shape [4 tokens × 3 dimensions]
K = Embedding × W_K    → shape [4 tokens × 3 dimensions]
V = Embedding × W_V    → shape [4 tokens × 3 dimensions]
```

### Step 2: Compute raw scores — Q × K^T

We compute the dot product of every query with every key. The result is a 4×4 matrix:

```
Scores[i][j] = Q[i] · K[j]
```

Example scores (made up for illustration):
```
         Cats  chase  mice   .
Cats   [  3.2,  1.1,  2.8,  0.2 ]
chase  [  2.1,  3.8,  2.9,  0.1 ]
mice   [  1.5,  0.9,  3.5,  0.3 ]
.      [  0.4,  0.3,  0.5,  3.1 ]
```

Reading row 1 ("chase"): "chase" assigns high relevance to itself (3.8), strong relevance to "mice" (2.9) — it's looking for what it's chasing.

### Step 3: Scale by √d_k

With d_k = 3, we divide every score by √3 ≈ 1.73:

```
Scores / √3:
         Cats  chase  mice   .
Cats   [  1.85, 0.64, 1.62, 0.12 ]
chase  [  1.21, 2.20, 1.68, 0.06 ]
...
```

This prevents extremely peaked softmax distributions in high-dimensional cases.

### Step 4: Apply softmax row-wise

Each row is turned into a probability distribution:

```
Attention weights (after softmax, row for "chase"):
  Cats: 0.13,  chase: 0.36,  mice: 0.49,  (.): 0.02
```

"Chase" attends most strongly to "mice" (0.49) and itself (0.36). Makes semantic sense — a verb attends to its object.

### Step 5: Weighted sum of values

Multiply the attention weights by the value vectors and sum them up:

```
New_chase = 0.13 × V(Cats) + 0.36 × V(chase) + 0.49 × V(mice) + 0.02 × V(.)
```

The resulting vector for "chase" now contains a blend of information about cats and mice, weighted toward mice. It knows it's a verb connecting a subject (cats) to an object (mice).

---

## Why Scale by sqrt(d_k)? 🎚️

This section deserves more depth because it's a common question.

### The Problem Without Scaling

Imagine the key and query vectors have d_k = 64 dimensions. Each element is a random number drawn from a standard distribution with mean 0 and standard deviation 1. The dot product of two such vectors is the sum of 64 products of random numbers.

Statistically, this sum has a standard deviation of √64 = 8.

So raw scores might look like: [0.2, -1.4, 7.8, -3.2, 6.5, 1.1, ...]

When we feed these into softmax, the large values (7.8, 6.5) become *exponentially* larger after the exp() function. The softmax becomes extremely concentrated:

```
Softmax([0.2, -1.4, 7.8, -3.2, 6.5]) ≈ [0.0001, 0.0000, 0.9876, 0.0000, 0.0123]
```

One token gets almost all the weight. This is called a **peaked softmax**, and it's bad for two reasons:
1. **The model becomes brittle** — tiny changes in inputs cause big changes in which token "wins"
2. **Gradients become tiny** — when softmax is extremely peaked, the gradients flowing backward are near zero for all but the dominant token (this is the saturated softmax problem)

### The Solution

Divide by √d_k before the softmax. This rescales the dot products to have variance 1:

```
Score = Q · K^T / √d_k
```

Now the softmax receives values with standard deviation ≈ 1, producing smooth, informative attention distributions rather than spike patterns.

**Analogy:** Think of adjusting the zoom on a map. Without scaling, you're zoomed way in — you can only see a tiny area but with extreme detail. With scaling, you're zoomed to a normal level where you can see the relationships between locations meaningfully.

### What Happens in Very Long Sequences?

Modern models with very long contexts (128K tokens, 1M tokens) have explored alternatives to the scaling factor. Some models apply additional tricks like ALiBi (Attention with Linear Biases) or RoPE (Rotary Position Embedding) to handle the fact that attention scores grow with sequence length.

---

## Softmax — Turning Scores Into Attention Weights 🔢

Softmax is the "decision function" of attention. Understanding it deeply helps you understand why attention behaves the way it does.

### What Softmax Does

Given a vector of raw scores, softmax converts them to a probability distribution:

```
Softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)
```

**Properties:**
- All outputs are positive (exp is always positive)
- All outputs sum to 1
- The highest input becomes the highest output, but *relatively* much larger

### A Concrete Example

Raw scores: [-2, 1, 4, 0.5]

After softmax:
- exp(-2) = 0.135 → 0.135/total
- exp(1) = 2.718 → 2.718/total
- exp(4) = 54.6 → 54.6/total
- exp(0.5) = 1.648 → 1.648/total
- Total = 59.1

Weights: [0.002, 0.046, 0.924, 0.028]

The score of 4 dominates massively at 92.4%. The score of 1 gets only 4.6%. A difference of 3 in the raw scores translates to a 20× difference in attention weight.

### Temperature and Concentration

The "temperature" of a softmax controls how peaked or spread out the distribution is:

```
Softmax(z / T)
```

- **T = 1 (default):** Normal distribution
- **T < 1 (low temperature):** More concentrated — the highest score dominates even more
- **T > 1 (high temperature):** More spread out — more tokens get meaningful weight

This matters for text generation (the temperature sampling parameter you've probably seen) — but the same principle applies inside the attention mechanism.

### Why Not Use Max (argmax)?

A simpler alternative would be to just pick the single highest-scoring token and assign it 100% weight. But this would be:
1. **Non-differentiable:** You can't backpropagate through argmax
2. **Too brittle:** Small changes in inputs cause discontinuous jumps in which token "wins"
3. **Information-losing:** Blending *multiple* tokens enriches representations more than picking just one

Softmax is a smooth, differentiable approximation to argmax that preserves the "concentrate on the best match" behavior while allowing gradient flow.

---

## Causal Masking — Why GPT Can't Peek Ahead 🙈

Causal masking is what makes GPT-style models work as text generators. Without it, training and generation would be fundamentally broken.

### The Cheating Problem

During training, we show the model a full document (say, 1,000 tokens). We ask it to predict token 500 given tokens 1–499. If the model could attend to token 501, it would have the answer right in front of it — "token 500" is essentially already there in the context. The model would learn to copy, not to reason.

**Analogy:** Teaching a student by showing them the exam paper and the answer key simultaneously. They'd ace the exam but learn nothing. Causal masking removes the answer key.

### How It Works Mechanically

Before the softmax step, we add a mask to the attention scores:

```
Masked Scores[i][j] = Score[i][j]          if j <= i
                    = -infinity              if j > i
```

For any token at position i, future positions (j > i) get score -infinity. After softmax, exp(-infinity) = 0, so future tokens contribute zero to the attention output.

The resulting attention pattern is a lower triangular matrix:

```
         Cats  chase  mice   .
Cats   [  ✓     ✗      ✗     ✗  ]
chase  [  ✓     ✓      ✗     ✗  ]
mice   [  ✓     ✓      ✓     ✗  ]
.      [  ✓     ✓      ✓     ✓  ]
```

"Cats" can only see itself. "Chase" can see "Cats" and itself. "Mice" can see everything before it.

### Why This Works for Generation

At inference time, when you're generating token by token, you always know all previous tokens. The causal mask exactly matches the information available at generation time, so there's no discrepancy between training and inference. This is the key elegance of the decoder-only architecture.

### Causal Masking vs. MLM

BERT uses a different approach: **Masked Language Modeling (MLM)**. During training, 15% of tokens are randomly masked with a `[MASK]` token, and the model predicts the masked tokens. Because the model sees both left and right context (bidirectional), there's no causal constraint.

The trade-off: MLM training is more sample-efficient (each step trains on 15% of tokens simultaneously), but the resulting model can't generate text in the natural autoregressive way.

---

## Cross-Attention vs. Self-Attention 🌉

Self-attention is what most LLMs use — tokens in a sequence attending to other tokens in *the same* sequence. Cross-attention is a different variant where tokens in one sequence attend to tokens in a *different* sequence.

### Self-Attention (the Default)

In self-attention, the Q, K, and V all come from the same sequence:

```
Q = Input × W_Q
K = Input × W_K
V = Input × W_V
```

Every token "reads from" the same pool of tokens it belongs to.

**Used in:** Every transformer block in GPT, LLaMA, BERT, Claude.

### Cross-Attention (For Encoder-Decoder)

In cross-attention, the Query comes from one sequence (typically the decoder's current state), while the Keys and Values come from a *different* sequence (typically the encoder's output):

```
Q = Decoder_state × W_Q         (what the decoder is looking for)
K = Encoder_output × W_K        (what the encoder produced)
V = Encoder_output × W_V        (the actual encoder content)
```

**Used in:** T5, BART, the original Transformer for translation. Also used in multimodal models to "attend to" image features from a vision encoder.

### A Translation Example

Translating "Je t'aime" (French) to "I love you" (English):

1. **Encoder** reads "Je t'aime" with full bidirectional attention, building rich French-language representations
2. **Decoder** starts generating. When generating "love," it uses cross-attention:
   - Query: "what French word corresponds to what I'm generating now (love)?"
   - Keys/Values: the encoder's representations of "Je," "t'," "aime"
   - "aime" scores highest → "love" is primarily informed by the French word for love

Cross-attention is the mechanism that bridges the two sequences in translation and summarization tasks.

### Cross-Attention in Modern Multimodal Models

Models like GPT-4V (vision), Claude with vision, and LLaVA use cross-attention (or a projection layer) to incorporate image features into the language model's processing. The decoder queries "what visual information is relevant to what I'm generating?" and the encoder (a vision model like CLIP) provides the keys and values.

---

## Attention Patterns — What the Model Actually Focuses On 🔭

Researchers have been able to visualize and interpret attention patterns in real models. Here's what's been found.

### Diagonal Patterns (Local Attention)

Many heads show strong attention to nearby tokens — tokens 2–3 positions away. This captures local syntactic dependencies that don't span long distances.

**Why:** Many important relationships in language are local — adjective-noun, verb-object, determiner-noun. These are easy to capture with local attention.

### Vertical Stripes (Tokens Attending to Specific Tokens)

Some heads show strong attention from *all* tokens to a specific token — often the most "semantically central" token in the current context, or special tokens like `[SEP]` or `[BOS]`.

**Interpretation:** These heads help "collect" or "broadcast" information across the sequence.

### Induction Heads (The Most Fascinating Discovery)

In GPT-style models, some heads learn a specific pattern: "if I've seen [A][B] earlier in the sequence, and now I see [A] again, strongly attend to the token after the previous [A]."

In other words: if the sequence contains "...Paris → famous... [later] Paris →", the induction head predicts that "famous" might come next again.

This is a kind of **in-context copying** mechanism, and it's thought to be the basis for the model's ability to do few-shot learning: after seeing a few (input, output) examples in the prompt, the model can apply the same mapping to new inputs by using induction heads to copy the pattern.

### Block-Structured Patterns

In deeper layers, attention becomes more diffuse and less interpretable by human intuition. Tokens don't attend to specific other tokens but rather to broad "categories" of tokens relevant to the current computational task.

---

## Attention Complexity and Efficiency ⚡

### The Quadratic Problem

For a sequence of N tokens, self-attention computes N×N attention scores — every pair of tokens. This is O(N²) in both time and memory.

For N = 1,000: 1,000,000 scores
For N = 100,000: 10,000,000,000 scores (10 billion!)

This is why early Transformers had strict context limits (512 or 1,024 tokens). The quadratic scaling made longer contexts prohibitively expensive.

### Flash Attention

FlashAttention (Dao et al. 2022) is a clever algorithm that computes the exact same attention output as standard attention, but reorganizes the computation to be GPU-memory-friendly. Instead of materializing the full N×N attention matrix in GPU memory, it processes in tiles, keeping only what's needed at each step.

Result: **10–20× faster**, **much less memory**, exact same output. FlashAttention 2 and 3 are now used in virtually all production LLMs.

**Analogy:** Standard attention lays out all 10 billion pairs on a massive table before doing any computation. FlashAttention processes one tile of pairs at a time, never needing the whole table simultaneously.

### Sparse Attention

Another approach: don't compute attention for all N² pairs. Instead, each token only attends to:
- All previous tokens within a local window (say, 128 tokens)
- A few "global" tokens (like the start token)

This is O(N × window_size) instead of O(N²). Used in Longformer and BigBird for very long document processing.

### Linear Attention

Some architectures approximate attention with linear complexity — O(N) instead of O(N²) — by modifying the kernel function. These are theoretically attractive but in practice tend to lose some quality on reasoning tasks. An active area of research.

---

## Common Misconceptions ❌

### "Attention means the model pays more 'mental effort' to some tokens"

Not quite. Attention weights are not about computational effort — the model spends exactly the same compute on every token. Rather, attention weights determine *whose information* gets blended into a token's representation. "High attention" means "this token's value vector has a large influence on my output."

### "The model with the highest attention score is the one that's 'most important' globally"

Attention is local to a single head and a single query token. Head 3 of layer 7, when processing "it," might highly attend to "mat." This doesn't mean "mat" is globally important — other heads at other layers might focus on completely different tokens for their own purposes.

### "Attention replaces recurrence (RNNs) completely"

Attention does away with the sequential bottleneck of RNNs (which process one token at a time). But pure attention Transformers still use positional encodings to tell the model the order of tokens — without positional information, attention is permutation-invariant (order-blind). The positional encoding is the part that replaces the recurrence's implicit position tracking.

### "You can interpret what a model 'knows' by looking at attention weights"

Attention weights are one signal, but they're often misleading as explanations. Gradient-based attribution methods tend to give more faithful explanations of which input tokens caused which outputs. Attention is a useful diagnostic tool but not a reliable explanation of model reasoning.

### "Higher temperature in generation = more creative"

Temperature scales the attention scores AND the output softmax before token sampling. Higher temperature makes attention distributions flatter (spreading attention more evenly) AND makes output token distributions flatter (increasing randomness in sampling). The effect is more diverse/random outputs, not necessarily more creative in a deep sense.

---

## Up Next
👉 **Multi-Head Attention** — doing attention multiple times simultaneously for multiple perspectives.
