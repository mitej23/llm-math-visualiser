# 🎲 Logits & Token Selection — Picking the Next Word

## The Big Idea

At the end of each decode step, the model produces a probability distribution over its entire vocabulary — a score for every possible next token. **Logits** are the raw pre-softmax scores. **Token selection** is how one token is chosen from this distribution. Different selection strategies produce very different text.

Understanding logits means understanding how the model "decides" what to say next. It's not magic — it's a massive linear projection followed by a probability calculation followed by a sampling strategy.

---

## Real-Life Analogy: The Jukebox 🎵

Imagine an old jukebox where you can see every song listed, each with a "crowd enthusiasm meter" showing how excited the audience would be to hear that song right now.

- "Bohemian Rhapsody": 95% enthusiasm
- "Yesterday": 72% enthusiasm
- "Baby Shark": 3% enthusiasm
- (all other songs listed with their scores...)

The jukebox must play **one** song. The "enthusiasm scores" are the logits. Converting them to percentages is softmax. How you pick a song from those percentages is token selection.

---

## From Hidden State to 50,257 Numbers 🔢

This is arguably the most important — and most overlooked — part of the decode step. The model spends enormous effort in its transformer layers building up a rich hidden representation. Then at the very end, it has to collapse all of that knowledge into a single probability distribution over words.

### The pipeline

```
Transformer layers (all the attention, FFN, etc.)
    ↓
Final hidden state: shape [d_model]  (e.g. 768 or 4096 numbers)
    ↓
Unembedding matrix (linear layer): shape [d_model × vocab_size]
    ↓
Logits: shape [vocab_size]  (e.g. 50,257 numbers for GPT-2, 32,000 for Llama)
```

For GPT-4 scale models with d_model = 8192 and vocab = 100k, this final linear layer has **819 million parameters just on its own**. It's a massive multiplication.

### Why this layer is so powerful 🧠

The final hidden state at position N encodes everything the model "knows" about what should come next: the topic of the conversation, the grammar of the sentence so far, the style of writing, what the user is likely expecting, and everything the model learned during training. All of this gets projected into a single vector of scores.

Remarkably, this projection is learned end-to-end. The model learns which directions in embedding space correspond to "this context is about France" → boost logits for Paris, French, Eiffel.

### The scale: a massive projection 📐

Let's compute the actual matrix size:
- GPT-2 small: 768 × 50,257 = **38.6 million** parameters in the unembedding matrix
- GPT-2 large: 1280 × 50,257 = **64.3 million** parameters
- Llama 3 8B: 4096 × 32,000 = **131 million** parameters
- Llama 3 70B: 8192 × 32,000 = **262 million** parameters

This single layer rivals a small neural network in parameter count. It's multiplied at every single decode step.

---

## Understanding Logit Values 📊

Logits are unbounded real numbers — they can be any value from -∞ to +∞. What do different values mean?

### Interpreting logit magnitudes

- **Very high logit (e.g. +10):** This token is almost certain given the context. After "The Eiffel Tower is in ___", the logit for "Paris" might be extremely high.
- **Moderate logit (e.g. +2 to +4):** This token is plausible but not certain. Multiple options in this range create genuine uncertainty.
- **Low logit (e.g. -2 to 0):** This token is unlikely but not impossible.
- **Very negative logit (e.g. -10):** This token is essentially impossible given the context. After "The capital of France is ___", the logit for "banana" would be extremely negative.

### Logit gaps matter more than absolute values

Because logits go through softmax (which is a relative operation), what matters is the **difference between logits**, not their absolute values.

If the top two logits are 5.0 and 4.5 (difference = 0.5), the model is fairly uncertain between them.
If the top two logits are 10.0 and 4.5 (difference = 5.5), the model is extremely confident about the first.

### Visualising logit distributions 📈

For "The cat sat on the ___":
```
"mat":    logit ≈ +4.2  →  high probability
"floor":  logit ≈ +3.1  →  moderate probability
"chair":  logit ≈ +2.8  →  moderate probability
"roof":   logit ≈ +0.5  →  low probability
"sun":    logit ≈ -1.2  →  very low probability
"banana": logit ≈ -4.0  →  essentially impossible
```

After softmax, "mat" might have 45% probability, "floor" 28%, "chair" 22%, and everything else splitting the remaining 5%.

---

## Softmax — The Probability Normaliser 📊

Softmax is the bridge between logits (arbitrary numbers) and probabilities (positive, sum to 1).

### The formula, step by step

Given logits [z₁, z₂, ..., zₙ]:

**Step 1: Exponentiate**
```
e₁ = e^z₁,  e₂ = e^z₂,  ...,  eₙ = e^zₙ
```
Exponentiation converts negatives to small positives and makes differences more extreme.

**Step 2: Sum**
```
S = e₁ + e₂ + ... + eₙ
```

**Step 3: Divide (normalise)**
```
p₁ = e₁/S,  p₂ = e₂/S,  ...,  pₙ = eₙ/S
```

All pᵢ are positive (because eˣ > 0 always), and p₁ + p₂ + ... + pₙ = 1. Valid probability distribution. ✅

### Why exponentiation? 🤔

Why not just normalise the logits directly (divide by sum)?

Two reasons:
1. **Logits can be negative.** You can't have negative probabilities. Exponentiation maps everything to positive.
2. **Amplification of differences.** If logit A = 4 and logit B = 2 (ratio 2:1), after exponentiation e⁴/e² = e² ≈ 7.4. The difference is amplified. This sharpens the distribution.

**Analogy:** Imagine rating movies 1-10. One person gives 8 and 6; another gives 9 and 3. In raw form, the first seems more uncertain. But after softmax-style amplification, the second person's preference becomes much clearer (e⁹ vs e³ is very lopsided).

### Numerical stability trick 🛡️

In practice, you subtract the max logit before exponentiating:
```
eᵢ = e^(zᵢ - max(z))
```
This prevents overflow (e^10000 would be infinity on a computer). The result is mathematically identical but numerically stable. This is called the "log-sum-exp trick."

---

## Greedy Decoding — The Simplest Strategy 🏆

Greedy decoding: after softmax, **always pick the token with the highest probability**. Every single time.

### How it works
```
Probabilities: {"cat": 0.45, "dog": 0.30, "fish": 0.20, "sun": 0.03, ...}
Greedy pick → "cat" (always, no randomness)
```

### When greedy is perfect ✅

For tasks with one correct answer:
- **Translation:** "La maison" → "The house" (not "A dwelling" or "The home")
- **Math answers:** 2 + 2 = ? → "4" (not "four" or "IV")
- **Factual Q&A:** Capital of France → "Paris"

Greedy is deterministic, reproducible, and fast (no sampling needed).

### When greedy fails ❌

**Repetition loops:** Without randomness, the model can get trapped:
> "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat..."

Once this loop starts, greedy reinforces it because "the" → "cat" is always highest given "...the mat."

**Boring text:** Every run of a creative writing prompt gives exactly the same output. No variation, no surprise.

**Suboptimal sequences:** The token with the highest probability at position N might not lead to the best sequence overall. Greedy is locally optimal but globally suboptimal. This is where beam search comes in.

**Analogy:** Greedy is like a GPS that always takes the shortest road at each junction, without considering that the "longer" road might avoid a traffic jam 5 minutes ahead.

---

## Beam Search — Exploring Multiple Paths 🔦

Beam search is a principled improvement over greedy that keeps multiple candidate sequences ("beams") alive simultaneously.

### The algorithm

With beam width B = 3:
1. At each position, generate the top B candidate continuations for each beam
2. Score each candidate sequence by its cumulative log-probability
3. Keep only the top B sequences overall
4. Repeat until all B beams have generated `<eos>`
5. Return the sequence with the highest total score

### Example walkthrough 📝

Beam width = 2. Generating after "The sky is":

**Step 1: Top 2 sequences**
- "The sky is **blue**" (score: -0.5)
- "The sky is **dark**" (score: -0.7)

**Step 2: Expand each**
- "The sky is blue **today**" (score: -1.1)
- "The sky is blue **and**" (score: -1.3)
- "The sky is dark **tonight**" (score: -1.5)
- "The sky is dark **blue**" (score: -1.8)

**Keep top 2:**
- "The sky is blue today" (winner)
- "The sky is blue and" (second)

**Step 3: Continue...**

### Where beam search is used 🗺️

Beam search is the standard algorithm for:
- **Machine translation** (Google Translate, DeepL)
- **Text summarisation**
- **Speech recognition**
- **Any task where quality matters more than creativity**

ChatGPT and most chat models do NOT use beam search by default — they use temperature sampling. Beam search is too slow for real-time conversation and too deterministic for creative tasks.

### Length normalisation 📏

A problem with raw log-probability: longer sequences always have lower (more negative) scores because you're multiplying many probabilities. Beam search typically divides by sequence length to avoid always preferring short sequences:

```
score = log_prob / length^α  (α ≈ 0.6-0.7 is common)
```

---

## Weight Tying — The Embedding-Unembedding Connection 🔗

Here's a beautiful piece of engineering economy that almost nobody talks about.

### Two matrices that do opposite things

The **embedding matrix** (at the start of the model) converts token IDs to vectors:
```
Token ID → [0.2, -0.5, 1.3, ..., 0.8]  (d_model numbers)
```
Shape: [vocab_size × d_model]

The **unembedding matrix** (at the end of the model) converts vectors back to logits:
```
[0.2, -0.5, 1.3, ..., 0.8] → [score_token_1, score_token_2, ..., score_token_50000]
```
Shape: [d_model × vocab_size]

Notice: these are transposes of each other!

### Weight tying: use one matrix for both 💡

Most modern LLMs (GPT-2, Llama, Mistral, Falcon, etc.) use the **same weight matrix** for both the embedding and unembedding operations. The embedding matrix is literally transposed and used as the unembedding matrix.

This is called **weight tying** (or tied embeddings).

### Why it works

If the model learns that the direction in embedding space for "Paris" points toward "city" and "France", then the unembedding matrix should also produce high scores for "Paris" when the hidden state points in the "city-in-France" direction. The learned geometry should be consistent in both directions.

### The savings 🏦

For GPT-2 small: embedding matrix is 50,257 × 768 = **38.6M parameters**.
Without tying: you'd need 38.6M × 2 = 77.2M parameters for both matrices.
With tying: just 38.6M — **a 38.6M parameter saving**.

For Llama 3 70B: 32,000 × 8192 = 262M parameters saved.

That's 262 million fewer parameters with no meaningful quality loss. Weight tying is essentially free lunch.

### When weight tying is NOT used

Some very large models (like GPT-3 and GPT-4 reportedly) do NOT tie weights. The argument: at scale, the model benefits from having separate learned representations for input encoding vs. output generation. The embedding matrix sees each token exactly once (its input representation), while the unembedding matrix must score all possible next tokens simultaneously. These might benefit from different parameterisations.

---

## Logit Bias and Logit Processors 🎛️

Beyond temperature/top-k/top-p, there are other ways to manipulate logits.

### Logit bias (direct adjustment)

The OpenAI API and similar services allow you to pass a `logit_bias` dictionary that directly adds or subtracts from specific token logits:

```python
logit_bias = {
    "50256": -100,  # Token ID 50256 (e.g. "<|endoftext|>") — never generate this
    "4091":  +5,    # Token ID 4091 (e.g. "Paris") — strongly prefer this
}
```

This is powerful for:
- **Preventing specific tokens** (e.g. preventing the model from saying certain words)
- **Forcing specific tokens** (e.g. making the model respond in a specific language)
- **Constrained generation** (e.g. always generating valid JSON by boosting `{`, `}`, `:` tokens)

### Structured output generation 🏗️

A sophisticated use of logit processors: **force the model to generate valid JSON, SQL, or any grammar**.

Tools like **outlines**, **guidance**, and **lm-format-enforcer** work by:
1. Tracking which tokens are currently valid given the partial output and the target grammar
2. Setting all invalid tokens' logits to -∞ before sampling
3. Guaranteeing the output matches the required structure

Example: Generating JSON `{"name": "..."}`:
- After `{`, only `"` is valid → all other logits set to -∞
- After `{"`, only valid JSON key characters are allowed
- After `{"name": "`, free text is allowed (the value)
- After the string, only `"` to close, then `}`, then end

This lets you use a conversational model to reliably produce structured data.

### Classifier-free guidance (CFG) 🧭

Originally from image generation (Stable Diffusion), CFG can also be applied to text:

1. Run the model twice: once with your prompt, once with a null/empty prompt
2. Final logits = (conditional logits) + guidance_scale × (conditional − unconditional)

This amplifies the "signal" from your specific prompt and reduces generic outputs. Used occasionally for very controlled text generation.

---

## Common Misconceptions 🚫

### "The model always picks the highest probability token"
**Wrong for most modern LLMs.** Pure greedy (always highest) is used for some tasks, but the default for chat models is temperature sampling (temp ≈ 0.7, top-p ≈ 0.9). ChatGPT, Claude, Gemini all sample by default, which is why you get slightly different answers each time.

### "Higher probability token = better token"
**Not always.** The model might be very confident about a mediocre choice. Token probability reflects training data patterns, not absolute quality. Sometimes a slightly less probable token leads to a much better continuation.

### "Logits are the model's 'thoughts'"
**Misleading.** Logits are just the output of a matrix multiplication. There's no "thought" — it's a computation. The rich representation is in the hidden state; logits are just the readout.

### "Beam search always produces better outputs than greedy"
**Task-dependent.** For translation, yes. For open-ended generation, beam search often produces dull, generic text because it systematically avoids low-probability tokens — which happen to be the creative, surprising ones.

### "Softmax is done once per response"
**Wrong.** Softmax is done at every single decode step. A 500-token response runs softmax 500 times over the full vocabulary each time.

### "The vocabulary size doesn't matter much"
**It matters a lot.** A larger vocabulary means a larger embedding and unembedding matrix, more parameters, and slower logit computation. GPT-2 has 50,257 tokens; Llama 3 has 128,000. The unembedding computation for Llama 3 is 2.5× more expensive than for GPT-2 per token.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| Logits | Raw unnormalized scores for every possible next token — output of the unembedding matrix |
| Softmax | Converts logits to probabilities that sum to 1 via exponentiation and normalisation |
| Unembedding matrix | Massive linear projection from hidden state dimension to vocabulary size |
| Greedy | Always pick the highest-probability token — fast, deterministic, but repetitive |
| Beam search | Keep top B candidate sequences, choose the highest scoring complete sequence |
| Weight tying | Embedding and unembedding matrices are the same — saves ~200-300M parameters |
| Logit bias | Directly adjust specific token scores before sampling — enables constrained generation |
| Structured generation | Set invalid tokens to -∞ to guarantee output matches a target grammar |

---

## Up Next
👉 **Temperature, Top-k, Top-p** — a deeper dive into controlling creativity and coherence.
