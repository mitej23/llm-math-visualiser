# 🎨 Temperature, Top-k, Top-p — Controlling Creativity

## The Big Idea

These three parameters are the "creative knobs" you turn to make an LLM more predictable or more creative, more focused or more exploratory. Understanding them helps you get the output you actually want.

They all operate on **logits** (or the probabilities derived from them) **before** the final sampling step. They're filters that shape the probability distribution. Used wisely, they are extraordinarily powerful. Used poorly, they produce repetitive mush or incoherent chaos.

---

## The Core Trade-off: Coherence vs. Diversity 🎭

Every generation choice involves a trade-off:

- **High coherence / low diversity:** The model repeats what it's most confident about — accurate but boring and repetitive
- **Low coherence / high diversity:** The model explores unlikely options — creative but sometimes nonsensical

Temperature, top-k, and top-p are all different tools for navigating this trade-off.

---

## Temperature — The Creativity Dial 🌡️

Temperature T reshapes the probability distribution **before** softmax by dividing all logits by T.

### The math

```
adjusted_logit[i] = logit[i] / T
```

Then softmax is applied to the adjusted logits as normal.

**At T = 1.0 (baseline):**
```
"cat": 45%, "dog": 30%, "fish": 20%, "sun": 5%
```

**At T = 0.5 (cold / focused):**
Logits are divided by 0.5 (i.e. doubled). High logits get amplified more:
```
"cat": 72%, "dog": 20%, "fish": 7%, "sun": ~1%
```

**At T = 2.0 (hot / random):**
Logits are divided by 2 (halved). Differences shrink:
```
"cat": 30%, "dog": 27%, "fish": 25%, "sun": 18%
```

### Why dividing works

Dividing logits by T < 1 **makes them bigger** (sharper differences → peaked distribution).
Dividing logits by T > 1 **makes them smaller** (smaller differences → flatter distribution).

**Visual analogy:** Think of the probability distribution as a landscape of mountains:
- **Cold temperature** (T < 1) = drains the water → only the tallest mountain peaks stay above water. Only the most likely tokens are accessible.
- **Room temperature** (T = 1) = the natural water level. Normal distribution.
- **Hot temperature** (T > 1) = floods the landscape → even the valleys fill up. All tokens become reachable.

### Temperature in practice — the four zones 🌡️

**Zone 1: T = 0 to 0.3 — "Robot mode"**
- Nearly deterministic (T → 0 approaches greedy decoding)
- Very consistent output across runs
- Good for: factual Q&A, math, code, summarisation where one answer is clearly right
- Risk: repetitive loops, overly formal tone
- Use when: you want the model's most confident answer

**Zone 2: T = 0.4 to 0.7 — "Professional mode"**
- Some variation, still well-grounded
- Output feels purposeful, not random
- Good for: business writing, technical explanations, structured tasks
- The sweet spot for most practical applications
- Use when: quality and consistency both matter

**Zone 3: T = 0.8 to 1.2 — "Natural mode"**
- Feels human and varied — conversations have natural rhythm
- Good for: conversational chat, creative writing within constraints, storytelling
- Most chat models (Claude, ChatGPT) default somewhere here
- Use when: you want natural, engaging text

**Zone 4: T > 1.5 — "Chaos mode"**
- High surprise, sometimes incoherent sentences
- Good for: brainstorming extreme ideas, exploring the model's creative range
- Risk: grammatical errors, topic drift, logical non-sequiturs
- T > 2.0 is almost always unusable for coherent prose

### Why not just use T = 0? 🤔

At T = 0 (greedy), the model gets stuck in repetition loops on long generations. It's also overconfident — the model doesn't actually know the "one right answer" for most open-ended tasks. A bit of randomness (T = 0.2-0.4) prevents loops while staying focused.

---

## Top-k Sampling — Restricting the Choice Set 📋

Top-k sampling: before sampling, discard all tokens except the k highest-probability ones. Then renormalise the remaining k to sum to 1, and sample.

### The algorithm

1. Compute probabilities via softmax
2. Sort tokens by probability, highest first
3. Keep only the top k tokens, discard the rest
4. Renormalise: divide each kept probability by the sum of kept probabilities
5. Sample from these k options

### Why top-k exists

Without any filtering, even at moderate temperature, there's a small but non-zero chance of sampling truly bizarre tokens. "The president of the United ___" — the model might sample "🦆" or "bananafrog" if they appear in the vocabulary and temperature is high enough.

Top-k creates a hard floor: no token ranked below k can ever be selected, regardless of temperature.

### k = 1 vs k = large

- **k = 1:** Identical to greedy decoding. No randomness.
- **k = 10:** Pick from the 10 most likely tokens. Focused but some variety.
- **k = 50:** Standard setting. Good balance for most tasks.
- **k = vocabulary_size:** No filtering. Pure sampling.

**Analogy:** A talent show audition. Temperature decides how scores are spread. Top-k says "only the top 50 contestants make it to the final round." The 51st-ranked contestant is eliminated, even if their score is close to 50th.

### The weakness of top-k 🔴

The "right" k is highly context-dependent and no single k works everywhere:

**After "She walked into the ___":**
Hundreds of valid continuations (kitchen, room, garden, forest, building, bar, hospital...). k = 50 might cut off perfectly valid options. k should be large here.

**After "The president of the United ___":**
Only a handful of valid continuations (States, Kingdom, Arab Emirates, Nations...). k = 50 would include many improbable tokens that shouldn't be there. k should be small here.

A fixed k is inherently a blunt instrument. This is why top-p was invented.

---

## Top-p (Nucleus) Sampling — Adaptive Filtering 🎯

Top-p (nucleus sampling) dynamically adjusts the number of candidate tokens based on the model's confidence. Instead of "always keep k tokens," it says "keep the fewest tokens whose combined probability reaches p."

### The algorithm

1. Sort tokens by probability, highest first
2. Accumulate probabilities until the running total reaches p (e.g. 0.9)
3. Keep only those tokens (the "nucleus")
4. Renormalise to sum to 1 and sample

### Example: confident model

After "The capital of France is ___":
```
"Paris":  89%
"Lyon":    5%
"Nice":    3%
"Marseille": 2%
...
```
Cumulative: "Paris" alone = 89%. "Paris" + "Lyon" = 94% > 90%.
Nucleus with p=0.9: just 1-2 tokens. Very focused. ✅

### Example: uncertain model

After "She walked into the ___":
```
"kitchen": 6%, "room": 5%, "forest": 5%, "garden": 4%,
"hospital": 4%, "bar": 4%, "store": 3%, "park": 3% ...
(many options, each ~3-6%)
```
Cumulative: need ~15-20 tokens to reach 90%.
Nucleus with p=0.9: ~15-20 tokens. Diverse. ✅

**The magic:** top-p automatically adapts to the model's confidence. When confident → small nucleus → focused. When uncertain → large nucleus → creative. You don't need to manually tune k for each context.

### Why p = 0.9 and not p = 1.0?

p = 1.0 includes every token (no filtering). The tail of the distribution contains improbable tokens that add noise without adding quality. Setting p = 0.9 cuts the bottom 10% of probability mass — the truly implausible tokens — while keeping all meaningful candidates.

**Analogy:** A restaurant menu. Instead of "show me the 10 most popular dishes" (fixed k), you say "show me dishes until we've covered 90% of what most diners order." If one dish is ordered by 95% of customers, you only see that one dish. If the menu is diverse with nothing dominating, you see many options. The selection adapts to the situation.

---

## Combining Temperature + Top-k + Top-p 🎛️

In practice, all three are usually applied together. The order matters:

### The full pipeline

```
Raw logits from model
    ↓  [Step 1] Divide logits by temperature T
Scaled logits
    ↓  [Step 2] Softmax → probabilities
Probabilities
    ↓  [Step 3] Apply top-k: keep only top k tokens, discard rest
Filtered probabilities (k tokens)
    ↓  [Step 4] Apply top-p: within those k, further reduce to nucleus covering p%
Nucleus probabilities
    ↓  [Step 5] Renormalise to sum to 1
    ↓  [Step 6] Sample one token
Final token
```

Each stage progressively narrows the candidate set.

### Practical guidance for combinations

**For creative writing:**
- Temperature is the primary lever — push it up (0.8-1.0)
- Set top-k = 0 (disabled) or very high — let top-p do the work
- top-p = 0.9-0.95 provides natural filtering

**For code generation:**
- Temperature is the primary lever — pull it down (0.1-0.3)
- top-k = 40 prevents weird syntax tokens
- top-p = 0.95 keeps the nucleus broad enough for diverse code patterns

**For factual Q&A:**
- Temperature ≈ 0 (greedy or near-greedy)
- top-k and top-p barely matter because temperature is so low

**Analogy:** Choosing a movie:
1. Temperature: "How adventurous am I feeling?" (high = willing to watch anything, low = only want what I know I like)
2. Top-k: "Only consider the 50 highest-rated movies" (hard floor on quality)
3. Top-p: "Of those, keep only the movies that together represent 90% of my likely satisfaction"
4. Sample: Pick randomly from the remaining options (weighted by adjusted scores)

---

## Repetition Penalty 🔄

LLMs have a structural tendency to repeat themselves. Here's why and how to fix it.

### Why repetition happens

At high temperature, after generating "the cat sat on the mat", the phrase "the cat" has appeared, and the model's weights give high probability to continuing familiar patterns. The phrase reinforces itself.

At low temperature (greedy), repetition is even worse — the model always picks the same highest-probability continuation, creating infinite loops.

### How repetition penalty works

Before sampling, any token that has appeared in the recent context has its logit **divided** by a penalty factor > 1.0:

```
If penalty = 1.3 and token "the" has appeared:
  new_logit["the"] = original_logit["the"] / 1.3
```

This makes previously seen tokens less likely to be chosen again.

**Multiple appearances:**
Some implementations apply the penalty proportionally to frequency:
```
new_logit[token] = original_logit[token] / (penalty ^ times_appeared)
```
The more times a token appeared, the more its logit is suppressed.

### Finding the right penalty value ⚖️

- **1.0:** No penalty (default). May produce loops.
- **1.1-1.2:** Mild penalty. Reduces repetition without strongly forcing novelty.
- **1.3-1.5:** Moderate penalty. Clearly discourages repetition.
- **1.5+:** Strong penalty. Forces the model to use new vocabulary aggressively. Can sound unnatural.

Too high a penalty and the model stops using common words like "the", "is", "a" — producing bizarre text.

### Frequency penalty vs. presence penalty (OpenAI API)

The OpenAI API (and Anthropic's API) expose two variants:

**Frequency penalty** (range: -2 to +2):
- Scales with how many times the token appeared
- "the" appearing 10 times is penalised 10× more than a token appearing once
- Good for: preventing verbose repetition of specific phrases

**Presence penalty** (range: -2 to +2):
- Fixed penalty for any token that appeared at all, regardless of frequency
- Once a token has appeared once, it's equally penalised whether it appeared 1 or 100 times
- Good for: encouraging the model to introduce new topics and vocabulary

---

## Min-p Sampling — A New Alternative 🔬

Min-p sampling was proposed in 2023 as an alternative to top-p with different mathematical properties. As of 2024-2025, it's gaining traction in open-source communities (Llama.cpp, KoboldCPP).

### The idea

Instead of "keep tokens whose cumulative probability reaches p," min-p says "keep tokens whose probability is at least (min_p × max_probability)."

```
threshold = min_p × max(probabilities)
keep all tokens where probability ≥ threshold
```

### Example

Max probability = 0.8 (model is very confident about one token).
With min-p = 0.1:
```
threshold = 0.1 × 0.8 = 0.08
Keep all tokens with probability ≥ 0.08
```

Max probability = 0.05 (model is uncertain, many equally likely tokens).
With min-p = 0.1:
```
threshold = 0.1 × 0.05 = 0.005
Keep all tokens with probability ≥ 0.005
→ Very many tokens kept (high diversity)
```

### Min-p vs. top-p

Both adapt to model confidence. The difference is in how they scale:
- **Top-p** looks at cumulative probability mass
- **Min-p** looks at relative probability compared to the top token

Min-p proponents argue it handles very high-confidence situations better (when the top token is near 1.0, top-p might still include many tokens that are just barely above 0 cumulative).

Typical setting: min-p = 0.05 to 0.1 combined with temperature = 0.7-1.0.

---

## Practical Settings for Different Use Cases 🎛️

Here's a concise guide to real-world configuration:

### By task type

| Task | Temperature | Top-k | Top-p | Notes |
|---|---|---|---|---|
| Factual Q&A | 0.0-0.2 | 10-20 | 0.9 | Near-greedy; one right answer |
| Code generation | 0.1-0.3 | 40 | 0.95 | Low temp; syntax must be correct |
| Code explanation | 0.3-0.5 | 50 | 0.9 | Slightly warmer for natural language |
| Summarisation | 0.3-0.5 | 50 | 0.9 | Factual but varied phrasing |
| General chat | 0.6-0.8 | 50 | 0.9 | Natural, engaging conversation |
| Story writing | 0.8-1.0 | 0 (off) | 0.95 | Let creativity flow |
| Poetry | 0.9-1.2 | 0 (off) | 0.97 | High diversity for artistic choices |
| Brainstorming | 1.0-1.3 | 0 (off) | 0.99 | Wild ideas welcome |

### By model type

Different models are trained with different "implicit temperatures" in their RLHF/RLAIF fine-tuning. A model fine-tuned for precise coding might behave like T=0.3 even at T=1.0. Always experiment with your specific model.

Instruction-tuned models (ChatGPT, Claude, Gemini) are calibrated for conversational use at default temperatures. Base models (untuned) often need lower temperature to produce coherent text.

### By response length target

For short, precise outputs (1-3 sentences): lower temperature.
For long-form content (essays, stories, reports): slightly higher temperature to avoid repetition in the extended text.

### The "if in doubt" preset

When you don't know what to use:
```
temperature: 0.7
top_p: 0.9
top_k: 50
repetition_penalty: 1.1
```
This is a solid all-purpose starting point for most chat tasks.

---

## Common Misconceptions 🚫

### "Higher temperature = smarter / more thoughtful responses"
**Wrong.** Higher temperature = more random. The model doesn't think more carefully; it samples from a flatter distribution. Sometimes this produces surprising insights (by accident), but it also produces more errors. Temperature controls randomness, not intelligence.

### "You need all three (temperature + top-k + top-p) for good results"
**Not necessarily.** Many models produce excellent results with just temperature + top-p. Top-k is sometimes redundant when top-p is already filtering aggressively. The OpenAI documentation even recommends not using both top-k and top-p simultaneously in most cases.

### "Temperature and top-p do the same thing"
**No.** Temperature reshapes the probability distribution (makes it more or less peaked) before filtering. Top-p then filters based on cumulative probability. Temperature changes the distribution; top-p determines what fraction of that distribution you sample from. They're complementary, not redundant.

### "Low temperature always produces better outputs"
**Depends on the task.** For factual Q&A: yes, lower is often better. For creative writing: too low and the output is generic and boring. For long generations: too low and you get repetitive loops.

### "Top-p = 0.9 means only 90% of vocabulary is available"
**Completely wrong.** Top-p = 0.9 means "keep the fewest tokens that together account for 90% of probability mass." When the model is confident, this might be just 2-3 tokens (far fewer than 90% of vocabulary). When uncertain, it might be 100+ tokens. 90% refers to probability mass, not token count.

### "Repetition penalty is always helpful"
**Not always.** In tasks like writing code or structured data where repetition is intentional (e.g. `for i in range(10): print(i)` repeats `i` and `print`), repetition penalty actively hurts quality. Turn it off for code generation.

---

## Key Takeaways

| Parameter | Effect | Default | Use when |
|---|---|---|---|
| Temperature | Controls probability distribution sharpness | 1.0 | Always — tune first |
| Top-k | Cuts off all but k highest tokens | 50 | When you want a hard upper limit |
| Top-p | Cuts to nucleus covering p% of probability | 0.9 | Most use cases — adapts better than top-k |
| Repetition penalty | Reduces likelihood of repeating tokens | 1.0 (none) | Generation loops, long outputs |
| Frequency penalty | Penalises by frequency of appearance | 0 | Verbose/repetitive outputs |
| Presence penalty | Flat penalty for any used token | 0 | Topic diversity, meandering responses |
| Min-p | Keep tokens above fraction of peak probability | 0.05 | Alternative to top-p, especially at high T |

---

## Up Next
👉 **Mixture of Experts (MoE)** — how LLMs scale to hundreds of billions of parameters without proportional slowdown.
