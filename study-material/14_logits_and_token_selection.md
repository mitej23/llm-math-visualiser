# 🎲 Logits & Token Selection — Picking the Next Word

## The Big Idea

At the end of each decode step, the model produces a probability distribution over its entire vocabulary — a score for every possible next token. **Logits** are the raw pre-softmax scores. **Token selection** is how one token is chosen from this distribution. Different selection strategies produce very different text.

---

## Real-Life Analogy: The Jukebox 🎵

Imagine an old jukebox where you can see every song listed, each with a "crowd enthusiasm meter" showing how excited the audience would be to hear that song right now.

- "Bohemian Rhapsody": 95% enthusiasm
- "Yesterday": 72% enthusiasm
- "Baby Shark": 3% enthusiasm
- (all other songs listed with their scores...)

The jukebox must play **one** song. The "enthusiasm scores" are the logits. Converting them to percentages is softmax. How you pick a song from those percentages is token selection.

---

## What Are Logits?

After all transformer layers, the final hidden state for the last token is projected through a linear layer to produce **logits** — one raw score per vocabulary token:

```
Hidden state (4096 dim)
    ↓
Linear projection (4096 → 32,000)
    ↓
Logits: [3.2, -1.5, 0.8, 7.1, 2.3, ..., -0.9]  (32,000 numbers)
```

These are unbounded — they can be any value, positive or negative.

**Analogy:** Before judges score Olympic diving, each judge has a raw impression of the dive — some judges are generous (high scores) and some strict (low scores). The logits are these raw impressions before any normalization.

---

## Softmax — Turning Logits into Probabilities 📊

**Softmax** converts raw logits into a probability distribution:

1. Exponentiate every logit (e^score)
2. Sum all the exponentiated values
3. Divide each by the sum

Result: all values are positive and sum to exactly 1.0 — a valid probability distribution.

**Properties:**
- The highest logit gets the highest probability
- The gap between logits gets amplified (high scores become very high probability)
- Low scores get pushed toward zero

**Analogy:** Converting the judges' raw impressions into actual scores out of 10. The most-impressed judges' ratings dominate. The least-impressed get proportionally less weight.

---

## Greedy Decoding — Always Pick the Highest 🏆

The simplest strategy: **always pick the token with the highest probability**.

```
Probabilities: {"cat": 0.45, "dog": 0.30, "fish": 0.20, "sun": 0.03, ...}
Greedy pick: "cat" every time
```

**Pros:** Deterministic, consistent, usually produces grammatically correct text.

**Cons:** Can be repetitive, may get stuck in loops, misses creative alternatives.

**Analogy:** A customer at a restaurant who always orders the most popular item. Safe, but they never discover hidden gems.

---

## Temperature — The Creativity Dial 🌡️

Before softmax, you can divide all logits by a **temperature** T:

- **T = 1.0:** Normal. Probabilities unchanged.
- **T < 1.0 (e.g., 0.5):** Divide logits by 0.5 → scores become more extreme → top token dominates even more → **less random, more focused**
- **T > 1.0 (e.g., 2.0):** Divide by 2 → scores become flatter → probabilities more equal → **more random, more creative**

**Analogy:** Temperature is the thermostat of creativity:
- **Cold (T < 1):** The model "freezes" into predictable, conservative outputs. Like a robot writing a legal document.
- **Room temperature (T ≈ 1):** Natural, human-like text.
- **Hot (T > 1):** The model "melts" into wild, unpredictable outputs. Like a poet having a fever dream.

At T → 0: Pure greedy (always pick highest). At T → ∞: Uniform random (any token equally likely).

---

## Top-k Sampling — Only Consider the Best K Options 🎯

**Top-k** restricts sampling to only the k highest-probability tokens:

1. Sort tokens by probability
2. Keep only top k tokens
3. Renormalize to sum to 1
4. Sample from these k options

**k = 1:** Same as greedy decoding (always pick the best).
**k = 50:** Pick one of the 50 most likely tokens.
**k = vocabulary size:** No restriction (pure sampling).

**Analogy:** You're at a buffet. Instead of choosing from 200 dishes, the chef only shows you the top 10 dishes that are fresh and popular today. You pick from those 10. Less overwhelmed, better quality selection.

**Problem with top-k:** The "right" k is context-dependent. Sometimes 10 tokens are reasonable continuations; sometimes 10,000 are (e.g., after "The city of" — hundreds of valid city names). A fixed k is arbitrary.

---

## Top-p (Nucleus) Sampling — Dynamic Cutoff 🎱

**Top-p** (nucleus sampling) takes the smallest set of tokens whose cumulative probability exceeds p:

1. Sort tokens by probability (highest first)
2. Add them up until the sum exceeds p (e.g., 0.9)
3. Sample from this "nucleus"

**p = 0.9:** Include tokens until you've covered 90% of the probability mass.
**p = 1.0:** Include all tokens (no restriction).

**Analogy:** Instead of "show me the top 10 dishes" (fixed k), say "show me dishes until we've covered 90% of what diners usually order." In a restaurant with one wildly popular item, that might be just 1-2 dishes. In a diverse menu, it might be 20.

Top-p dynamically adjusts the number of options based on confidence:
- Model is very confident? Nucleus is small (few options, focused)
- Model is uncertain? Nucleus is large (many options, creative)

---

## Min-p Sampling — A Newer Alternative 🔬

**Min-p** keeps tokens whose probability is at least p × (max probability):

- If the top token has probability 0.8 and min-p = 0.1, keep all tokens with probability ≥ 0.08
- Naturally adapts: when model is confident, only a few tokens qualify; when uncertain, more qualify

Similar goals to top-p but with different mathematical properties. Gaining popularity in 2024-2025.

---

## Combining Strategies in Practice 🎛️

Real-world inference often combines:
- **Temperature** (0.7): Slightly focused
- **Top-p** (0.9): Nucleus sampling
- **Top-k** (50): Hard cap

Applied in order: top-k first, then top-p, then temperature, then sample.

Different use cases want different settings:

| Use Case | Temperature | Top-p | Character |
|---|---|---|---|
| Legal document | 0.1 | 0.9 | Precise, formal |
| Code completion | 0.2 | 0.95 | Accurate, consistent |
| Normal chat | 0.7 | 0.9 | Natural, varied |
| Creative writing | 1.0 | 0.95 | Expressive, diverse |
| Brainstorming | 1.3 | 0.99 | Wild, exploratory |

---

## Repetition Penalties ⛔

LLMs tend to repeat themselves. To counter this, a **repetition penalty** reduces the logit scores of tokens that have already appeared:

- Token "the" appeared 10 times → its logit gets penalized → less likely to appear again
- Strength controls how aggressively to penalize repetition

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| Logits | Raw unnormalized scores for every possible next token |
| Softmax | Converts logits to probabilities that sum to 1 |
| Greedy | Always pick the highest-probability token |
| Temperature | Scale that controls how focused or random the distribution is |
| Top-k | Only sample from the k most likely tokens |
| Top-p | Only sample from the smallest set of tokens covering p% of probability |
| Min-p | Only sample tokens above a fraction of the peak probability |

---

## Up Next
👉 **Temperature, Top-k, Top-p** — a deeper dive into controlling creativity and coherence.
