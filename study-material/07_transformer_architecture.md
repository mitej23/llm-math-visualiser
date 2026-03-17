# 🏗️ Transformer Architecture — The Big Picture

## The Big Idea

The Transformer is the core architecture behind all modern LLMs. It's a specific way of stacking the pieces we've learned — embeddings, attention, feed-forward layers — into a machine that can understand and generate text. Understanding the overall blueprint helps every other piece make sense.

---

## Real-Life Analogy: The Translation Bureau 🌍

Imagine a professional translation bureau that handles complex, nuanced documents:

1. **Reception desk (Tokenization + Embedding):** Receives a document, converts it into a structured form the translators can work with
2. **Context room (Attention):** All translators read the full document simultaneously and discuss which parts relate to which — "This pronoun in paragraph 7 refers back to the name in paragraph 2"
3. **Thinking room (Feed-Forward):** Each translator privately processes their section in light of all the context
4. **Quality control (Layer Norm):** A supervisor checks that no single translator is going wildly off-track
5. **Output desk (Linear + Softmax):** Combines everything into the final translated document

A Transformer does all of this, many times over, stacking multiple "floors" of context rooms and thinking rooms.

---

## The Transformer Stack

Here's the data flow from start to finish:

```
Input text
    ↓
[Tokenization] → token IDs (sequence of numbers)
    ↓
[Embedding Lookup] → vectors for each token
    ↓
[Positional Encoding] → add position information
    ↓
┌─────────────────────────────────┐
│        TRANSFORMER BLOCK        │ ← This repeats N times
│                                 │
│  ┌─ Layer Norm                  │
│  ├─ Multi-Head Attention        │
│  ├─ Residual Connection (+)     │
│  ├─ Layer Norm                  │
│  ├─ Feed-Forward Network        │
│  └─ Residual Connection (+)     │
└─────────────────────────────────┘
    ↓ (repeated for each layer)
[Final Layer Norm]
    ↓
[Linear Projection → vocabulary size]
    ↓
[Softmax → probabilities]
    ↓
Predicted next token
```

This is the **decoder-only** architecture used by GPT, LLaMA, Claude, and most modern chat models.

---

## The Transformer Block — The Core Unit 🧱

The transformer block is the repeating unit. Stacking 32, 64, or 96 of these gives you a large language model.

Each block does two things:
1. **Attention:** Let each token "look at" all other tokens and gather relevant information
2. **Feed-Forward:** Process each token's updated representation through a small neural network

Between and after each of these, there's normalization and a residual connection.

---

## Residual Connections — The "Skip Highway" 🛣️

One of the most important tricks in the Transformer:

```
Output = Layer(Input) + Input
```

Instead of just passing the output of each layer to the next, you **add the original input back**.

**Why?**

**Analogy:** Imagine you're editing a document. Rather than handing the editor a blank page and saying "rewrite this," you give them the original draft and say "mark what to change." They return the original with only the changes highlighted. You then apply the changes and keep the rest.

The residual connection means each layer only needs to learn **the difference** (the correction), not the whole representation from scratch. This makes very deep networks trainable.

**Practical effect:** Gradients can flow directly from the output back to early layers without vanishing. Deep learning works.

---

## The Encoder vs. Decoder

**Encoder (used in BERT, T5's encoder):**
- Reads the entire input at once
- Each token can attend to all other tokens (both before and after)
- Builds a deep understanding of the input
- Good for: classification, question answering from a given passage

**Decoder (used in GPT, LLaMA, Claude):**
- Generates text token by token
- Each token can only attend to tokens that came **before** it (causal masking)
- This is important: when predicting the 5th word, the model can't "cheat" by looking at the 6th word
- Good for: text generation, completion

**Encoder-Decoder (used in original Transformer, T5):**
- Encoder reads input (e.g., French sentence)
- Decoder generates output (e.g., English translation)
- Used for translation and summarization

Most modern chat models use decoder-only architecture.

---

## Causal Masking — No Cheating! 🙈

In a decoder model, when training the model to predict word #5, we must ensure it can't see words #6, #7, etc. Otherwise it would just copy the answer.

This is enforced by **masking** — setting future positions to negative infinity before the softmax, so they contribute zero to the attention.

**Analogy:** An open-book exam where you can see all previous pages but the book is folded shut at the current question. You can use past context but can't look ahead.

---

## How Many Layers?

| Model | Layers | Dimension |
|---|---|---|
| GPT-2 small | 12 | 768 |
| GPT-3 175B | 96 | 12,288 |
| LLaMA 3 8B | 32 | 4,096 |
| LLaMA 3 70B | 80 | 8,192 |

More layers = deeper understanding, more compute.

Each layer refines the representation. Early layers tend to capture syntactic patterns (grammar, word structure). Later layers tend to capture semantic and factual knowledge.

---

## The Output Head — Turning Vectors into Words 🎯

After all transformer blocks, we need to turn the final vector into a probability over the vocabulary:

1. **Linear projection:** Map the embedding dimension to vocabulary size (e.g., 4096 → 32,000)
2. **Softmax:** Convert raw scores to probabilities that sum to 1
3. **Sampling:** Pick the next token based on these probabilities

This final step produces the **logits** (raw scores) and then the predicted token. We'll cover this in depth in the "Logits & Token Selection" chapter.

---

## The Whole Pipeline in One Sentence

> Input text → tokens → embeddings + positions → N transformer blocks (each doing attention then feed-forward) → final projection → next token probability.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| Transformer Block | The repeating unit: attention + feed-forward + normalizations |
| Residual Connection | Add input to output — helps gradients flow and layers learn corrections |
| Causal Masking | Prevent tokens from seeing future tokens during training/generation |
| Encoder | Reads full context in both directions — good for understanding |
| Decoder | Generates text left-to-right — good for generation |
| Output Head | Final layer that converts vectors to vocabulary probabilities |

---

## Up Next
👉 **Attention Mechanism** — the most important and novel part of the Transformer.
