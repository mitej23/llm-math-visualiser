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

## Encoder-Only, Decoder-Only, and Encoder-Decoder 🗺️

These three architectures are often confused. They are genuinely different, and choosing the wrong one for a task will hurt you.

### Encoder-Only — The "Reader" 📖

The archetypal example is **BERT** (Bidirectional Encoder Representations from Transformers), released by Google in 2018. An encoder-only model reads the entire input sequence at once, and every token can attend to every other token — including tokens that come *after* it. This is called **bidirectional attention**.

Think of it like reading a whole book before answering a question. You have the full context available from the start.

**What it's great at:**
- Sentence classification ("Is this review positive or negative?")
- Named entity recognition ("Which words in this sentence are place names?")
- Question answering from a passage ("Given this paragraph, what year was the treaty signed?")
- Sentence similarity and retrieval

**What it cannot do:** Generate new text. Because it processes the whole sequence at once, there is no natural "next token" to predict — it's a reader, not a writer.

**Real models:** BERT, RoBERTa, DeBERTa, ELECTRA, all sentence embedding models like `text-embedding-ada-002`.

### Decoder-Only — The "Writer" ✍️

The archetypal example is **GPT** (Generative Pre-trained Transformer). A decoder generates text one token at a time, and each token can only attend to tokens before it (causal masking). The training objective is simply: predict the next token given all previous tokens.

This turns out to be extraordinarily powerful. By training on billions of documents with this simple objective, the model is forced to learn grammar, facts, reasoning, and style — all to predict the next word better.

**What it's great at:**
- Text generation, story writing, code generation
- Instruction following (with fine-tuning)
- Few-shot in-context learning
- Reasoning through problems step by step

**Real models:** GPT-2, GPT-3, GPT-4, Claude, LLaMA, Mistral, Falcon, Gemma.

### Encoder-Decoder — The "Translator" 🔄

The original 2017 Transformer paper ("Attention Is All You Need") used this design for machine translation. The encoder reads the full source sequence, building rich representations. The decoder then generates the output sequence token by token, attending to its own previous outputs AND the encoder's representations via **cross-attention**.

**What it's great at:**
- Machine translation ("Translate this French sentence to English")
- Abstractive summarization ("Write a 3-sentence summary of this article")
- Any task where you have a defined input and a generated output

**Real models:** T5, BART, mT5, MarianMT.

### When to Use Which?

| Task | Architecture | Why |
|---|---|---|
| Text classification | Encoder-only | Needs full bidirectional context |
| Open-ended text generation | Decoder-only | Natural auto-regressive generation |
| Translation/summarization | Encoder-decoder | Separate encoding and decoding stages |
| Embeddings/retrieval | Encoder-only | Fixed-size representation of input |
| Instruction-following chat | Decoder-only (fine-tuned) | State-of-the-art performance at scale |

**The trend:** The industry has largely converged on decoder-only models. GPT-4, Claude, and LLaMA 3 are all decoder-only. Even tasks that feel "encoder-like" (e.g., classification) can be done with large decoder-only models via prompting, often matching or beating BERT-style models.

---

## A Deep Dive into One Transformer Block 🔬

Let's walk through a single transformer block step by step. This is the unit that repeats 12, 32, 96, or even 120 times in large models.

Assume the input is a matrix of shape `[sequence_length × d_model]` — one vector per token.

### Step 1: Pre-LayerNorm

The first thing that happens is **normalization**. In modern LLMs (pre-norm design), the input is normalized *before* being fed into the attention sublayer.

**What normalization does:** It rescales each vector so it has a mean near 0 and a standard deviation near 1. This prevents the values from growing wildly large as they pass through many layers.

**Analogy:** Like a musician tuning their instrument before playing. The actual notes haven't changed, but everything is calibrated to a common standard before the real work begins.

### Step 2: Multi-Head Self-Attention

Each token's vector asks: "Who should I be gathering information from right now?" It computes Query, Key, and Value vectors, runs dot-product attention across the entire sequence, and produces an output that blends information from relevant positions.

The output has the same shape as the input: `[sequence_length × d_model]`. Nothing about the sequence length or dimensionality changes — but the *content* of each vector is now enriched by context.

### Step 3: First Residual Addition

The output of the attention sublayer is added back to the *original* input (before the LayerNorm). This is the residual connection.

```
x = x + Attention(LayerNorm(x))
```

If the attention layer learned nothing useful, this addition just keeps the original signal intact. The network defaults to "no change" rather than "random noise." This is incredibly valuable for training stability.

### Step 4: Pre-LayerNorm (again)

Before the feed-forward sublayer, the vector is normalized again. Same reasoning: keeps values in a manageable range before a potentially large computation.

### Step 5: Feed-Forward Network

Each token's vector is processed independently by a two-layer MLP (multi-layer perceptron). This is a "thinking" step: after gathering context via attention, each token processes its enriched representation through a learned function.

The FFN typically expands the dimension by 4×, applies a non-linearity (GELU or SiLU in modern models), then projects back down.

- Input: vector of size `d_model` (e.g., 4096)
- Hidden layer: `4 × d_model` (e.g., 16,384 neurons)
- Output: back to `d_model` (e.g., 4096)

This is where much of the "knowledge storage" happens. Research suggests FFN layers act like key-value memories, storing factual associations.

### Step 6: Second Residual Addition

Again, the FFN output is added back to the running vector:

```
x = x + FFN(LayerNorm(x))
```

The block is done. The output goes to the next block, which repeats the whole sequence.

### Putting It All Together:

```
Input x
  → LayerNorm → MultiHeadAttention → + x → (new x)
  → LayerNorm → FeedForward        → + x → (new x)
Output x
```

Both the attention and FFN sublayers are "wrapped" with a residual connection and a normalization. This pattern is consistent across virtually all modern LLMs.

---

## Why Residual Connections Are Critical 🏗️

Residual connections (also called skip connections) were introduced in ResNets for image processing in 2015, and the Transformer adopted them wholesale. They solve one of the deepest problems in training neural networks.

### The Vanishing Gradient Problem

When you train a neural network, you update weights by computing gradients — signals that flow backward from the loss through every layer. In a very deep network (say, 96 layers), these gradient signals tend to shrink as they travel backward through layer after layer. By the time the gradient reaches the first few layers, it's essentially zero. Those early layers learn nothing.

This is called the **vanishing gradient problem**, and it made deep networks nearly untrainable before residual connections.

**Analogy:** Imagine sending a message through 96 intermediaries, each of whom slightly mishears and garbles the message. By the time it reaches the origin, the original message is indecipherable. The signal has degraded to noise.

### How Residual Connections Solve It

A residual connection creates a "shortcut highway" that lets gradients flow directly from the output all the way back to early layers without passing through every intermediate layer.

```
Output = Layer(Input) + Input
Gradient of Output = Gradient from Layer + Gradient of 1 (the identity)
```

The "gradient of 1" means there's always a direct path for gradients to flow backward. Even if the `Layer` part contributes a near-zero gradient, the 1 keeps the highway open.

**Analogy:** Instead of a message passing through 96 intermediaries, you have a direct phone line that also rings the origin simultaneously. The original message always arrives intact, and you also get whatever the intermediaries made of it.

### What Each Layer Actually Learns

Because of residual connections, each layer doesn't need to reconstruct the whole representation from scratch. It only needs to learn **what to add** — the residual, the correction.

This is a much easier learning problem. Early layers might add nothing at all (the identity residual). Later layers make targeted refinements. The network can be very deep without any layer having to do all the heavy lifting.

### Worked Example

Say a token's current vector represents "bank" with no context. After one attention layer, the residual-wrapped output might be:

- Original "bank" signal: 95% preserved
- Added "river context" correction: 5% new information

After 32 layers of these small corrections, "bank" has been enriched with river context, syntactic role, discourse position, and much more — but the original signal was never lost. The residual connections form a information superhighway that carries the original token signal through the entire network while layers refine it along the way.

---

## What Each Layer Learns 🧠

One of the most fascinating discoveries in LLM interpretability research is that different layers in a Transformer learn qualitatively different things — and there's a rough hierarchy from surface features to deep reasoning.

### Lower Layers (1–10%): Surface and Syntax

Early layers capture the most basic properties of language:

- **Spelling and morphology:** Does this word end in "-ing"? Is it plural?
- **Part of speech:** Is this a noun, verb, adjective?
- **Local syntax:** Simple subject-verb-object structure in short spans
- **Token frequency:** How common is this word overall?

**Evidence:** If you take the hidden states from layer 1 of GPT-2 and train a simple linear classifier on them, you can predict part-of-speech tags with ~90% accuracy. By layer 3, the model already has strong syntactic representations.

**Analogy:** Like learning the alphabet and basic grammar rules first. Before you can understand literature, you need to know what words are and how sentences are structured.

### Middle Layers (40–70%): Semantics and Relationships

Middle layers build on the syntactic foundation to encode meaning and relationships:

- **Word sense disambiguation:** "bank" is now clearly "riverbank" vs. "financial institution"
- **Coreference:** "John met Mary. He liked her." — "he" is now linked to "John"
- **Semantic similarity:** "car" and "vehicle" are brought closer in the representation space
- **Entity tracking:** The model knows that a "she" mentioned three sentences ago refers to the same person as the named entity earlier

**Evidence:** Probing studies on BERT and GPT-2 show that semantic similarity tasks peak in accuracy at middle layers (6–9 out of 12 for BERT-base).

**Analogy:** This is where you move from knowing words to understanding what they mean and how they relate. Like going from "these are the words" to "this is what the sentence actually means."

### Upper Layers (80–100%): Reasoning and Task Completion

The topmost layers handle higher-order cognition:

- **Logical reasoning:** "If A implies B and B implies C, then..."
- **World knowledge retrieval:** Factual associations stored in the feed-forward layers
- **Task-specific behavior:** Following instructions, maintaining a persona, completing specific formats
- **Long-range coherence:** Keeping track of the overall topic across a long document

**Evidence:** When researchers "ablate" (remove or freeze) upper layers of GPT-2, the model degrades at logical reasoning tasks but still produces syntactically correct text. The surface form survives; the reasoning collapses.

**Analogy:** Upper layers are the "executive function" — not just understanding what is said, but figuring out what to do with it and producing a response that is appropriately task-directed.

### Why This Matters for Practitioners

- **Fine-tuning:** You typically fine-tune all layers for instruction-following. But for narrow tasks (e.g., sentiment classification), you might freeze early layers and only fine-tune upper layers, which saves compute.
- **Embeddings:** For semantic search, embeddings from middle layers often work better than the final layer.
- **Interpretability:** If a model "knows" a fact incorrectly, the error is often localized to specific middle-upper layers' FFN sublayers.

---

## Pre-norm vs Post-norm ⚖️

This is a subtle but important architectural choice. The original 2017 Transformer paper used **post-norm**. Every modern LLM uses **pre-norm**. Here's why.

### Post-norm (Original Transformer)

In post-norm, the LayerNorm is applied *after* the residual addition:

```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

The residual connection and the layer output are added together *first*, then normalized. This was the original design in the "Attention Is All You Need" paper and was used in early BERT and GPT-2 variants.

**Problem:** Post-norm creates a subtle but serious training instability issue. During the early stages of training, the attention and FFN outputs can have very different scales from the residual. The normalization only happens *after* they've been combined, which means the early training dynamics can be chaotic. Training post-norm Transformers required very careful learning rate warmup schedules and often would "explode" (gradients go to infinity) if not carefully calibrated.

**Analogy:** Imagine two people trying to mix different-temperature liquids. They pour them together first, then try to measure the temperature. The brief mixing period might cause unpredictable reactions before you can measure anything.

### Pre-norm (Modern LLMs)

In pre-norm, the LayerNorm is applied *before* the attention or FFN:

```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

The input is normalized before being fed into the sublayer. The residual addition happens after with the *unnormalized* x.

**Benefits:**
1. **Training stability:** The gradients at early training steps are much better behaved. You can use larger learning rates without warmup or with minimal warmup.
2. **Depth:** Models with more layers benefit more from pre-norm. GPT-3's 96 layers and GPT-4's ~120 layers would be extremely difficult to train with post-norm.
3. **Simpler tuning:** Fewer hyperparameters to worry about for keeping training stable.

**Used by:** GPT-3, GPT-4, LLaMA 1/2/3, Mistral, Claude, Gemma — essentially every modern LLM.

**The trade-off:** Some research suggests pre-norm models have slightly weaker "representation diversity" in their layers — the representations at different depths are more similar than in post-norm models. But this is far outweighed by the training stability benefits at large scale.

### RMSNorm — A Modern Twist

Many modern models (LLaMA, Mistral, Gemma) replace LayerNorm with **RMSNorm** (Root Mean Square Normalization). It's a simpler version that only normalizes by the root mean square of the values, without centering by the mean. It's slightly faster and empirically works just as well.

---

## The Feed-Forward Sublayer in Context 🔧

The feed-forward network (FFN) inside each transformer block is often overshadowed by the flashier attention mechanism, but it makes up a significant portion of the model's parameters and is thought to store most of the factual knowledge.

### The Architecture

Each FFN has two linear transformations with a non-linearity in between:

```
FFN(x) = NonLinearity(x · W1 + b1) · W2 + b2
```

Where:
- `W1`: projects from `d_model` up to `4 × d_model` (or `8/3 × d_model` in some variants)
- NonLinearity: originally ReLU, now usually GELU or SiLU
- `W2`: projects back down to `d_model`

In LLaMA and Mistral, a **gated variant** (SwiGLU) is used:
```
FFN(x) = SiLU(x · W_gate) × (x · W1) · W2
```

The gating mechanism gives the FFN more expressive power.

### Why the 4× Expansion?

Expanding to a larger hidden dimension gives the FFN more "room" to perform complex non-linear transformations. It's like a scratchpad — the larger the scratchpad, the more complex the computation the layer can do.

### The FFN as a Memory Bank

Research by Geva et al. (2021) demonstrated that FFN layers in Transformers act as **key-value memories**. Each neuron in the expanded hidden layer can be thought of as:
- **Key:** A pattern in the input (e.g., "the input mentions Paris and a famous tower")
- **Value:** Information to add (e.g., "Eiffel Tower, iron, built 1889, Gustave Eiffel")

When the right pattern is activated in the input, the corresponding neuron fires and injects the associated information into the output.

**Analogy:** Think of FFN layers as the model's "fact lookup" system. Attention says "which tokens are relevant?" and FFN says "what do I know about this combination of tokens?" They are complementary systems.

### FFN vs. Attention — Parameter Split

In a typical Transformer block:
- Attention parameters: roughly 4 × d_model² (Q, K, V, O projection matrices)
- FFN parameters: roughly 8 × d_model² (two large matrices with 4× expansion)

The FFN has roughly **twice as many parameters** as the attention mechanism. Most of the model's knowledge is stored here.

---

## Hyperparameters: Layers, Heads, d_model 📐

Designing a Transformer requires choosing several key hyperparameters. Here's what they mean and how they interact.

### d_model — The Vector Width

`d_model` is the size of the vector that represents each token at every position in the network. A token's d_model-dimensional vector is its current "state of understanding" as it passes through the layers.

- GPT-2 small: d_model = 768
- GPT-3: d_model = 12,288
- LLaMA 3 8B: d_model = 4,096
- LLaMA 3 70B: d_model = 8,192

Larger d_model = richer representations but more compute and memory per token.

### Number of Layers (depth)

More layers = more rounds of refinement. Each layer can build on what the previous layers learned, enabling increasingly abstract representations.

The relationship between performance and depth is roughly logarithmic — doubling the layers doesn't double the capability, but it does improve it meaningfully.

### Number of Attention Heads

More heads = more relationship types the model can track simultaneously. In practice, models use 8–128 heads, with `d_head = d_model / n_heads` typically around 64–128.

### The Scaling Laws

Researchers at Anthropic and OpenAI discovered that there are predictable "scaling laws" relating model size (parameters), training data, and compute to model performance. The key insight:

> Given a fixed compute budget, the optimal model is not the biggest possible model trained for minimal steps. It's a model that is somewhat smaller but trained on much more data.

This "Chinchilla scaling law" (Hoffman et al. 2022) changed how modern LLMs are built. LLaMA models are intentionally designed to be smaller but trained on far more tokens than earlier models of the same parameter count, making them more capable per unit of inference cost.

---

## Common Misconceptions ❌

### "More layers always means smarter"

Not exactly. More layers need more training data and compute to train well. An undertrained deep model can perform worse than a well-trained shallow one. The Chinchilla scaling laws show that model size and training tokens need to be scaled together.

### "Attention is the most important part"

The FFN sublayers actually have more parameters than the attention sublayers and store most of the factual knowledge. Attention and FFN are complementary — attention gathers context, FFN processes it.

### "BERT and GPT are just different sizes of the same thing"

They have fundamentally different architectures and different training objectives:
- BERT: encoder-only, bidirectional, trained with masked language modeling
- GPT: decoder-only, causal, trained with next-token prediction

They are good at different tasks. You wouldn't use GPT for building a sentence embedding model, and BERT can't generate text.

### "The model reads the whole input then generates"

In a decoder-only model, generation is auto-regressive — the model generates one token at a time, and each new token is appended to the input for the next step. There is no separate "reading phase" and "writing phase." The same forward pass both reads context and produces the next token.

### "Residual connections are optional for small models"

Even for small models (e.g., GPT-2 12 layers), removing residual connections would make training significantly harder and likely fail or produce a much worse model. Residual connections are non-negotiable in Transformer architectures.

---

## Up Next
👉 **Attention Mechanism** — the most important and novel part of the Transformer.
