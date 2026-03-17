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

## The Embedding Table — A Giant Lookup Dictionary 🗄️

Think of the embedding table as an enormous Excel spreadsheet:
- **Rows:** one for each token in the vocabulary (e.g., 128,256 rows for LLaMA 3)
- **Columns:** one for each dimension (e.g., 4,096 columns)
- **Each cell:** a single floating-point number learned during training

When the model receives token ID 1547, it doesn't compute anything — it simply looks up row 1547 and retrieves the entire row as a vector. This is a pure lookup, like using a dictionary.

**How much memory does this take?**
For LLaMA 3 8B:
- 128,256 tokens × 4,096 dimensions × 2 bytes (fp16) = ~1 GB just for the embedding table!

That's a significant fraction of the total model size, which is why vocabulary size is carefully balanced.

**Weight tying:** Some models (like GPT-2) use the same matrix for both the input embedding table AND the output prediction layer. The idea: if the embedding of "cat" represents "cat-ness," then the output layer should also use this same "cat-ness" vector to decide when to output "cat." This trick reduces parameters and often improves quality.

---

## What Do the Dimensions Actually Represent? 🔍

This is one of the most fascinating questions in AI research. The honest answer: **we don't fully know.**

Researchers have identified some interesting patterns:
- Some dimensions correlate with **sentiment** (positive vs negative words)
- Some correlate with **animacy** (living vs non-living things)
- Some seem to track **grammatical role** (noun-ness, verb-ness)
- Some capture **domain** (medical terms, legal terms, sports)

But here's the twist: embeddings use **superposition** — meaning many concepts can be encoded in the same dimensions at once, compressed together. This is why individual dimensions aren't cleanly interpretable.

**Mechanistic Interpretability** is a growing field trying to reverse-engineer what each neuron and dimension actually encodes. The field has found "monosemantic neurons" (neurons that fire for one concept) and "polysemantic neurons" (neurons that fire for completely unrelated concepts, like "the name Michael" AND "a specific NBA team" AND "the word 'base'").

---

## Word2Vec and the History of Embeddings 📜

Before modern LLMs, the breakthrough was **Word2Vec** (Google, 2013). It was the first method to show that meaningful geometry emerges in word vectors:

**How Word2Vec works:**
1. Take a word, look at its surrounding words (its "context window")
2. Train a simple neural network to predict: given this word, what words appear nearby?
3. The internal representation the network learns = the word embedding

Two training variants:
- **CBOW** (Continuous Bag of Words): predict the center word from surrounding words
- **Skip-gram**: predict surrounding words from the center word

**Why it was revolutionary:** Before Word2Vec, words were just IDs. After Word2Vec, "king" and "queen" were geometrically similar. The king-man+woman=queen result made the world take notice.

**GloVe** (Stanford, 2014) followed with a different approach — instead of predicting contexts, it analyzed word co-occurrence statistics across the entire corpus. GloVe often performed similarly to Word2Vec but was faster to train.

These "static" embeddings (one vector per word, regardless of context) were state-of-the-art until BERT arrived in 2018.

---

## Contextual Embeddings — How BERT Changed Everything 🏆

**BERT** (Google, 2018) introduced truly contextual embeddings. Unlike Word2Vec's single static vector per word, BERT produces a **different vector depending on the surrounding sentence**.

The same word "bank" in:
- "I went to the **bank** to deposit money" → one vector
- "We sat by the river **bank**" → a completely different vector

**How BERT achieves this:** By running the full sentence through multiple layers of a Transformer. Each layer refines the representation by considering all other words. By the final layer, every word's representation has absorbed information from the entire context.

**What changed for NLP:**
- Pre-BERT: train a model from scratch for each task (sentiment analysis, question answering, etc.)
- Post-BERT: pre-train on massive text once, then fine-tune on specific tasks. This "transfer learning" revolution made NLP tasks dramatically easier.

Modern LLMs like GPT and Claude use the same contextual principle, but instead of bidirectional context (BERT can look both left and right), they use **unidirectional** context (they can only look left — at previous tokens). This is necessary for text generation, where future tokens haven't been written yet.

---

## Embedding Size in Real Models 📐

The embedding dimension is one of the most important hyperparameters in LLM design:

| Model | Params | Embed Dim | Layers | Heads |
|---|---|---|---|---|
| GPT-2 Small | 117M | 768 | 12 | 12 |
| GPT-2 XL | 1.5B | 1600 | 48 | 25 |
| GPT-3 | 175B | 12,288 | 96 | 96 |
| LLaMA 3 8B | 8B | 4,096 | 32 | 32 |
| LLaMA 3 70B | 70B | 8,192 | 80 | 64 |
| LLaMA 3 405B | 405B | 16,384 | 126 | 128 |

Notice the pattern: bigger models have larger embedding dimensions. This is because a larger embedding space can represent more subtle semantic distinctions. With only 768 dimensions, there are fewer "directions" available, so concepts have to share space. With 12,288 dimensions, each concept can have its own clean direction.

---

## Common Misconceptions About Embeddings ⚠️

**Misconception 1: "Embeddings are the model's knowledge."**
Not quite. Embeddings are just the *input representation*. The actual "knowledge" is distributed across all the model's weights — the attention heads, feed-forward layers, etc. The embedding is just how text gets turned into numbers at the entrance.

**Misconception 2: "Similar token IDs = similar meanings."**
Completely false. Token ID 500 and token ID 501 are neighbors by accident. The embedding table is what creates meaningful proximity — not the ID numbers themselves.

**Misconception 3: "The model uses embeddings all the way through."**
No! The initial embedding is just the starting representation. Every transformer layer *modifies* this representation. By the time you reach the final layer, the vectors bear little resemblance to the original embeddings. These modified vectors are called **hidden states** or **activations**, not embeddings.

**Misconception 4: "You can directly compare embeddings from different models."**
False. Each model trains its own embedding space from random initialization. There's no universal coordinate system. "King" in GPT-4's embedding space points in a completely different direction than "king" in LLaMA's embedding space.

**Misconception 5: "Larger embedding dimensions always = better."**
More dimensions help up to a point, but they also increase computational cost quadratically in attention. The right size depends on the model's overall parameter budget and the tasks it needs to handle.

---

## The Semantic Space — Why Geometry Matters 📐

The fact that meanings can be represented as **locations in space** is profoundly powerful. Here's why geometry matters:

**Clustering:** Words with similar meanings cluster together. "Happy," "joyful," "elated" are all nearby. This means when the model encounters an unfamiliar word, if its embedding is near the "happy" cluster, the model can make reasonable inferences.

**Directions encode relationships:** The vector from "man" to "woman" encodes the "gender" direction. The vector from "king" to "queen" is nearly parallel. This means the gender relationship generalizes — "uncle" to "aunt," "brother" to "sister," "actor" to "actress" all point in similar directions.

**Cosine similarity:** Rather than Euclidean distance, embeddings typically use cosine similarity (angle between vectors). This is because the magnitude of a vector doesn't carry meaning — only its direction does. A word that appears frequently might have a larger-magnitude vector just due to having more training examples.

**Applications beyond LLMs:** Embeddings are the backbone of:
- Semantic search (find documents similar in meaning, not just keyword matches)
- Recommendation systems (find products similar to ones you liked)
- Translation (map sentences in different languages to nearby points)
- Anomaly detection (find texts that are semantically unusual)

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
| Static Embedding | One fixed vector per word regardless of context (Word2Vec, GloVe) |
| Hidden State | The modified representation after transformer layers process the embedding |

---

## Up Next
👉 **Positional Encoding** — embeddings don't know *where* in the sentence a word is. We need to fix that.
