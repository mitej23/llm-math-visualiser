# 🌌 Embeddings — How Numbers Carry Meaning

## The Big Idea

Tokenization gives each word a number (like a library ID). But the number 1,547 doesn't "mean" anything — it's just a label. **Embeddings** convert these IDs into rich vectors where similar meanings end up close together in a mathematical space.

---

## Real-Life Analogy: The City Map 🗺️

Imagine you have a map of restaurants in your city. You plot each restaurant as a point:

- All Italian restaurants cluster in one area of the map
- All sushi places cluster in another area
- Nearby restaurants share characteristics (similar price range, vibe, cuisine type)

Now someone says "I'm near the Italian cluster, slightly toward the seafood section." Without naming a specific restaurant, the location **itself** tells you a lot.

Embeddings do exactly this for words — they place each word as a point in a high-dimensional space, where nearby points have similar meanings.

---

## Why Not Just Use the Token ID?

Token IDs are arbitrary labels. Token #1547 = "king" and Token #1548 = "queen" happen to be neighbors in the vocabulary only by accident.

But semantically, "king" and "queen" are *very* related. And "king" and "pizza" are *not*.

We need a representation where the **distance between vectors** encodes **semantic similarity**.

---

## The Embedding Table 📊

Before any processing, the model has an **embedding table** — a learned lookup table:

```
Token #1547 ("king")   → [0.2, -0.5, 0.8, 0.1, ..., 0.3]   (512 numbers)
Token #1548 ("queen")  → [0.2, -0.4, 0.8, 0.1, ..., 0.4]   (512 numbers)
Token #7823 ("pizza")  → [-0.8, 0.9, -0.3, 0.7, ..., 0.1]  (512 numbers)
```

Each token ID maps to a vector of numbers (usually 512–4096 numbers in LLMs).

"King" and "queen" have very similar vectors. "Pizza" is far away.

**Analogy:** The embedding table is like a phonebook where the phone number (embedding vector) somehow captures everything about the person — their personality, interests, relationships — not just their address.

---

## The Magic of Embedding Arithmetic ✨

This is where embeddings get beautiful. Because meaning is encoded in direction and distance, you can do **math with meaning**:

**Famous example:**
> King − Man + Woman ≈ Queen

Translated: Start at "king," subtract the "maleness" direction, add the "femaleness" direction, and you land near "queen."

**Analogy:** Think of it like GPS coordinates:
- Paris is at coordinates (48.8, 2.3)
- France is at coordinates (46.2, 2.2)
- Germany is at coordinates (51.2, 10.4)
- Berlin is at...?

Paris − France + Germany ≈ Berlin (capital of France → capital of Germany)

The relationship between a capital and its country is encoded in the *direction* of the vector difference.

---

## What Dimensions Encode 🔭

Modern embeddings have hundreds or thousands of dimensions. Each dimension *loosely* corresponds to some abstract feature:

- Dimension 47 might respond to "royalty-related words"
- Dimension 112 might respond to "food-related words"
- Dimension 203 might respond to "past tense verbs"

But here's the catch: **no single dimension is interpretable on its own.** The meaning emerges from the combination of all dimensions together.

**Analogy:** Like describing a person's personality with 512 different scales (introversion/extroversion, risk-taking, creativity, etc.). No single scale tells the whole story, but together they create a complete picture.

---

## How Embeddings Are Learned 🎓

Embeddings aren't hand-crafted — they're **learned during training**:

1. Start with random vectors for every token
2. Train the model on massive amounts of text
3. The model learns that "bank" and "river" often appear near each other
4. It also learns "bank" and "finance" appear together
5. The word "bank" ends up with a vector that somehow captures both meanings

The embeddings adjust so that words appearing in similar **contexts** end up with similar **vectors**.

**Analogy:** Children learn what words mean not from dictionary definitions, but by hearing them used in many different situations. "Dog" is said when a furry animal is present, when someone says "puppy," when discussing walks, etc. Embeddings learn the same way — from context.

---

## Contextual vs. Static Embeddings 🔄

Early embeddings (like Word2Vec) gave each word **one fixed vector** — "bank" would always have the same embedding, whether used in "river bank" or "bank account."

Modern LLMs use **contextual embeddings** — the vector for a word changes depending on its surrounding context. The word "bank" next to "river" gets a different internal representation than "bank" next to "finance."

**Analogy:** The word "cool" means different things in:
- "It's cool outside" (temperature)
- "That's cool!" (approval)
- "She played it cool" (composure)

Modern LLMs distinguish these — their internal representation of "cool" adapts based on context.

---

## Embedding Dimensions in Practice

| Model | Embedding Dimension |
|---|---|
| GPT-2 small | 768 |
| GPT-3 | 12,288 |
| LLaMA 3 8B | 4,096 |
| LLaMA 3 70B | 8,192 |

Larger models use bigger embeddings — more dimensions = more capacity to represent subtle distinctions.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| Embedding | A vector of numbers that encodes the meaning of a token |
| Embedding Table | A learned lookup table: token ID → vector |
| Semantic Similarity | Similar meanings → nearby vectors in embedding space |
| Embedding Arithmetic | You can do math on meanings: king - man + woman = queen |
| Contextual Embedding | The same word gets different vectors in different contexts |
| Embedding Dimension | How many numbers describe each token (higher = richer) |

---

## Up Next
👉 **Positional Encoding** — embeddings don't know *where* in the sentence a word is. We need to fix that.
