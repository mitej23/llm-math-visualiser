# 🎨 Temperature, Top-k, Top-p — Controlling Creativity

## The Big Idea

These three parameters are the "creative knobs" you turn to make an LLM more predictable or more creative, more focused or more exploratory. Understanding them helps you get the output you actually want.

---

## The Core Trade-off: Coherence vs. Diversity 🎭

Every generation choice involves a trade-off:

- **High coherence / low diversity:** The model repeats what it's most confident about — accurate but boring and repetitive
- **Low coherence / high diversity:** The model explores unlikely options — creative but sometimes nonsensical

Temperature, top-k, and top-p are all different tools for navigating this trade-off.

---

## Temperature — The Master Dial 🌡️

We introduced temperature briefly before. Let's go deeper.

Temperature T reshapes the probability distribution before sampling:

**At T = 1.0 (baseline):**
```
"cat": 45%, "dog": 30%, "fish": 20%, "sun": 5%
```

**At T = 0.5 (cold / focused):**
The logits are doubled in effect. High probabilities dominate more:
```
"cat": 72%, "dog": 20%, "fish": 7%, "sun": ~0%
```

**At T = 2.0 (hot / random):**
The differences are halved. Distribution flattens:
```
"cat": 30%, "dog": 27%, "fish": 25%, "sun": 18%
```

**Visual analogy:** Think of the probability distribution as a landscape:
- **Cold temperature** = Drains the water → only the tallest mountain peaks stay above water (only top choices are accessible)
- **Hot temperature** = Floods the landscape → even valleys fill up (all tokens become possible)

---

## Deep Dive: Temperature in Practice 🔬

**Temperature ≈ 0.0 to 0.3 — "Robot mode"**
- Nearly deterministic
- Very consistent output across runs
- Good for: factual questions, code, math, summarization
- Risk: repetitive loops if not careful

**Temperature ≈ 0.4 to 0.7 — "Professional mode"**
- Some variation, still grounded
- Good for: business writing, technical explanations, most practical tasks
- The sweet spot for most use cases

**Temperature ≈ 0.8 to 1.2 — "Natural mode"**
- Feels human and varied
- Good for: conversational chat, creative writing within constraints
- Most chat models default near here

**Temperature > 1.5 — "Chaos mode"**
- High surprise, sometimes incoherent
- Good for: brainstorming unexpected ideas, exploring creative extremes
- Risk: grammatical errors, logical jumps

**Temperature > 2.0 — Almost unusable for coherent text**
- Flat distribution → essentially random token picking
- Everything becomes equally likely including nonsense

---

## Top-k — The Shortlist 📋

Top-k sampling: before sampling, discard all but the k highest-probability tokens.

**Why:** Prevents the model from picking very low-probability "crazy" tokens even with high temperature.

**k = 1:** Greedy — always pick the winner.
**k = 10:** Pick from 10 candidates.
**k = 50:** Pick from 50 candidates.
**k = ∞ (no top-k):** All tokens eligible.

**Analogy:** Imagine a talent show. Temperature decides how scores are spread. Top-k says "only send the top 50 contestants to the final round — everyone else goes home." Even if temperature flattens scores and the 51st contestant is close, they're still out.

**When top-k works well:** Text with clear syntactic constraints — e.g., after "The president of the United __" there are only a few sensible countries. k=10 is plenty.

**When top-k fails:** After "She walked into the ___" there are hundreds of valid continuations (kitchen, room, store, park, forest...). k=50 might cut off perfectly valid options.

---

## Top-p — The Dynamic Shortlist 🎯

Top-p (nucleus sampling) was invented specifically to fix top-k's weakness.

Instead of "keep the top k tokens," it says "keep the fewest tokens that together account for p% of the probability."

**Formal definition:**
Sort tokens by probability (highest first). Keep adding tokens until their cumulative probability exceeds p. Sample from this set (renormalized to sum to 1).

**Example 1: Model is very confident**
```
"Paris": 89%, "London": 8%, "Rome": 2%, ...
```
With p=0.9: "Paris" + "London" already = 97% > 90%. Stop at 2 tokens.
Nucleus size = 2 (very focused)

**Example 2: Model is uncertain**
```
"kitchen": 5%, "room": 5%, "forest": 4%, "park": 4%, "store": 4%, ...
(dozens of options each with ~5%)
```
With p=0.9: Need to include ~18+ tokens to reach 90%.
Nucleus size = 18+ (very diverse)

**The magic:** Top-p automatically adapts. When the model is confident → small nucleus → focused. When uncertain → large nucleus → creative.

**Analogy:** A playlist curated to "cover 90% of what you'd enjoy." If you love one genre, that's 5 songs. If you're eclectic, that's 200 songs. The playlist adapts to your taste, not an arbitrary count.

---

## Combining Temperature + Top-k + Top-p 🎛️

They're usually applied together. Order matters:

1. **Compute logits**
2. **Apply temperature** (scale logits by 1/T)
3. **Apply top-k** (keep only top k tokens)
4. **Apply top-p** (within those k, keep nucleus covering p%)
5. **Sample** from the remaining tokens

Each filter progressively narrows the options.

**Analogy:** You're choosing a movie:
1. Temperature: Assign "desire scores" — sometimes you're in the mood for anything (high T), sometimes very specific (low T)
2. Top-k: "Only consider the 50 highest-rated movies"
3. Top-p: "Of those, only consider ones that together make up 90% of my likely satisfaction"
4. Sample: Randomly pick from the remaining (weighted by desire scores)

---

## Common Real-World Presets

**Code assistant (precise):**
```
temperature: 0.2
top_k: 40
top_p: 0.95
```

**General chat assistant:**
```
temperature: 0.7
top_k: 50
top_p: 0.9
```

**Creative writing:**
```
temperature: 0.9
top_k: 0 (disabled)
top_p: 0.95
```

**Brainstorming:**
```
temperature: 1.2
top_k: 0 (disabled)
top_p: 0.99
```

---

## Repetition Penalty — Avoiding Loops ♻️

LLMs, especially at higher temperature, can get stuck in repetitive patterns:

> "The cat sat on the mat. The cat sat on the mat. The cat sat on..."

The **repetition penalty** multiplies the logit of any token that appeared recently by a penalty factor < 1:

```
Modified logit = logit × penalty^(times_appeared)
```

A penalty of 1.2 means a token that appeared once has its score reduced by 1/1.2 ≈ 83%. A token that appeared 3 times is at (0.83)³ ≈ 57% of original.

**Analogy:** Like a conversation where you get slightly bored every time someone repeats themselves. After they've said the same thing 5 times, you actively want to hear something new.

---

## Frequency vs. Presence Penalty

Two variants used in OpenAI/Anthropic APIs:

**Frequency penalty:** Penalty proportional to how many times the token has appeared (more appearances = bigger penalty). Discourages repetition of commonly used tokens.

**Presence penalty:** Fixed penalty for any token that appeared at all, regardless of count. Encourages the model to talk about new topics.

---

## Key Takeaways

| Parameter | Effect | Default | Use when |
|---|---|---|---|
| Temperature | Controls probability distribution sharpness | 1.0 | Always — tune first |
| Top-k | Cuts off all but k highest tokens | 50 | When you want a hard limit |
| Top-p | Cuts to nucleus covering p% of probability | 0.9 | Most use cases — adapts better than top-k |
| Repetition penalty | Reduces likelihood of repeating tokens | 1.0 (none) | Generation loops |
| Frequency penalty | Penalizes by frequency of appearance | 0 | Verbose/repetitive outputs |
| Presence penalty | Flat penalty for any used token | 0 | Topic diversity |

---

## Up Next
👉 **Mixture of Experts (MoE)** — how LLMs scale to hundreds of billions of parameters without proportional slowdown.
