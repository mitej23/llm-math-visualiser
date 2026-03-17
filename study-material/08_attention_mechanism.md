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

## Up Next
👉 **Multi-Head Attention** — doing attention multiple times simultaneously for multiple perspectives.
