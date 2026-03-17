# ➗ Linear Layers & Matrix Multiplication — The Core Operation

## The Big Idea

At the heart of every neural network is one operation that runs millions of times per second: **multiply numbers and add them up**. That's it. Matrix multiplication is just a fancy, efficient way to do this for many numbers at once.

---

## Real-Life Analogy: The Restaurant Bill 🍽️

Imagine three friends at a restaurant. They each ordered different things:
- Alice: 2 pizzas, 1 salad, 3 drinks
- Bob: 0 pizzas, 2 salads, 1 drink
- Carol: 1 pizza, 0 salads, 2 drinks

To calculate each person's bill, you multiply their quantities by the prices:
- Pizza: $10
- Salad: $8
- Drink: $4

**Alice's bill:** (2×10) + (1×8) + (3×4) = 20 + 8 + 12 = **$40**

You just did matrix multiplication! You took a vector of quantities and multiplied it by a vector of prices to get one number.

Now imagine doing this for thousands of people with thousands of items — that's why we use matrices (grids of numbers) instead of writing it all out.

---

## What Is a Linear Layer?

A **linear layer** (also called a **fully connected layer** or **dense layer**) does this:

> Take an input vector → multiply by a weight matrix → add a bias → get an output vector

**Analogy:** A linear layer is like a currency exchange booth:
- You bring in euros (input vector)
- The exchange rates are the weights (weight matrix)
- A small fee is added (bias)
- You get out dollars (output vector)

The "exchange rates" are the weights the network has learned.

---

## Vectors — Just a List of Numbers 📋

A **vector** is just a list of numbers. Nothing scary about it.

- `[3, 1, 4, 1, 5]` is a vector
- A word embedding might be a vector of 512 numbers
- The output of a layer might be a vector of 256 numbers

**Analogy:** Your Netflix profile is a vector. Imagine Netflix represents you as: `[loves_action: 0.9, loves_romance: 0.2, loves_comedy: 0.7, ...]`. Each number captures one aspect of your taste. That's an embedding vector.

---

## Matrices — A Grid of Numbers 🔢

A **matrix** is just a 2D grid of numbers, like a spreadsheet.

```
Weight Matrix W:
[ 0.5  -0.3   0.8 ]
[ 0.1   0.9  -0.2 ]
[ 0.4  -0.1   0.6 ]
[ 0.7   0.3   0.1 ]
```

Each row says: "To compute this output neuron, use these multipliers on the inputs."

---

## Matrix Multiplication — The Core Operation 🎯

When you multiply a vector by a matrix, you're computing multiple dot products at once.

**Dot product** = multiply each pair of numbers, then add them all up.

**Example:**
- Input vector: `[a, b, c]`
- First row of W: `[w1, w2, w3]`
- First output = `(a × w1) + (b × w2) + (c × w3)`

Do this for every row of W and you get the output vector.

**Analogy:** Imagine you're a judge scoring a gymnastics routine:
- Each judge (row in W) has different priorities: strength, flexibility, style
- The gymnast (input vector) has scores on each dimension
- Each judge computes their own total score
- The output is a vector of all judges' scores

One gymnast → multiple perspectives → multiple output scores. That's matrix multiplication.

---

## Bias — The Adjustment Knob ⚙️

After multiplying, we add a **bias** — a small constant that shifts the output.

**Analogy:** Think of it like a thermostat offset. If your thermostat always reads 2° too cold, you add +2 to every reading. The bias lets the network shift its output up or down regardless of input.

Without bias, a neuron that gets zero input always outputs zero. With bias, it can output something useful even when inputs are all zero.

---

## Why "Linear"?

A linear transformation has a specific mathematical property: if you scale the input, the output scales the same way. If you add two inputs, the output is the sum of the two outputs.

**Why this matters:** This means a stack of linear layers can *always* be collapsed into just one linear layer. They're not actually learning complex patterns — just rotations and scalings.

This is why we need **activation functions** (next topic) — to break linearity and let the network learn truly complex relationships.

---

## The Dimensions Game 📐

Linear layers transform vectors from one size to another:

| Operation | What Changes |
|---|---|
| `512 → 2048` | Expand (more dimensions to think in) |
| `2048 → 512` | Compress (distill down to essentials) |
| `512 → 32000` | Project to vocabulary size (for next-word prediction) |

**Analogy:** It's like translating from a compact shorthand to full English and back. You expand the representation to "think," then compress it for a decision.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| Vector | A list of numbers |
| Matrix | A 2D grid of numbers |
| Weight Matrix | The learned "recipe" for transforming one vector to another |
| Bias | A constant adjustment added after the multiplication |
| Linear Layer | Multiply input by weights, add bias, get output |
| Dot Product | Multiply pairs of numbers and sum them — the basic unit of matrix math |

---

## Up Next
👉 **Activation Functions** — what breaks the linearity so networks can learn complex things.

---

---

## Visualising Matrix Multiplication Geometrically 🔭

Here's an intuition that makes linear layers click on a deeper level: **every linear layer is a geometric transformation of space**.

When you multiply a vector by a matrix, you're not just crunching numbers — you're physically moving points in space. The transformation can:

- **Rotate** vectors (imagine spinning a compass needle)
- **Scale** vectors (stretch or shrink along any direction)
- **Reflect** vectors (flip like a mirror)
- **Shear** vectors (slant like italic text)

### The 2D example

Imagine you have a flat grid of points, like graph paper. Each point is described by two numbers (x, y). A 2×2 weight matrix transforms every point on the grid simultaneously.

- A weight matrix with 1s on the diagonal and 0s elsewhere = no change (identity)
- A weight matrix that swaps values = reflection across the diagonal
- A weight matrix with large values stretches all points outward

### Why this matters for LLMs

In a language model, an embedding vector for "king" lives somewhere in a 512-dimensional space (imagine 512-dimensional graph paper). A linear layer moves that point to a new location in space. Multiple linear layers applied in sequence gradually move the point through different "regions" of concept space.

This is why researchers talk about "moving through representation space" — the geometric metaphor is literally accurate. When you ask a language model "What's the capital of France?", each layer moves the representation through different regions until it lands in the "Paris" neighbourhood of concept space.

### The beautiful constraint

Linear transformations always keep straight lines straight and parallel lines parallel. They can't curve lines or create corners. This is why we need activation functions — to introduce the ability to create curves and corners in representation space.

---

## Dimensions in Real Transformers — GPT-2 vs GPT-4 📊

Understanding the actual numbers in real models helps you grasp the scale of what's happening.

### GPT-2 (the small, fully open model from 2019)

- **Embedding dimension (d_model):** 1,024
- **Hidden dimension (FFN):** 4,096
- **Number of layers:** 48
- **Vocabulary size:** 50,257

Every time the model processes a single token, it runs 48 sequential linear transformations — each one taking a 1,024-dimensional vector and transforming it. The feed-forward network inside each layer expands to 4,096 dimensions and then compresses back down. That expand-then-compress pattern gives the model more "room to think" temporarily.

### GPT-3 (175 billion parameters)

- **Embedding dimension:** 12,288
- **Hidden dimension:** 49,152
- **Number of layers:** 96
- **Vocabulary size:** 50,257

The weight matrix for just one feed-forward layer would be 12,288 × 49,152 — roughly 600 million numbers in a single matrix. There are hundreds of such matrices across all layers.

### GPT-4 (estimated, not officially confirmed)

Believed to be a mixture-of-experts model with dimensions in the 16,000+ range per expert and over 100 transformer layers. The vocabulary might be around 100,000 tokens to handle more languages.

### The key pattern: always 4×

Notice that in GPT-2, the hidden dimension (4,096) is exactly 4× the embedding dimension (1,024). Same ratio in GPT-3: 49,152 / 12,288 = 4. This "4×" expand-then-compress pattern is a design choice that has become standard. It gives the FFN layer enough width to be expressive without being wasteful.

---

## Why Can't We Just Use One Big Linear Layer? 🤔

This is a great question. If a linear layer transforms dimensions, why not just use one enormous layer that goes directly from input to output?

### The mathematical reason

As explained earlier — stacking linear layers without activations in between is mathematically identical to a single linear layer. You can always collapse them. But more importantly, even a single huge linear layer can't learn non-linear relationships. No matter how big it is, it can only represent straight-line relationships in the data.

### The practical reason

Imagine input dimension 512 and output dimension 512. A direct layer would have 512 × 512 = 262,144 weights. If you insert an intermediate layer of 64 dimensions: (512 × 64) + (64 × 512) = 65,536 weights — 75% fewer parameters for the same input/output shape.

More importantly, the intermediate smaller dimension forces the model to **compress and re-expand**, which acts like a bottleneck that encourages learning efficient representations. It's like requiring students to summarise a textbook chapter in one paragraph before writing an essay — the compression forces deeper understanding.

### The depth advantage

With multiple layers (and activations between them), each layer builds on the abstractions of the previous one. Layer 1 might learn simple patterns, layer 2 combines those into more complex patterns, and so on. A single huge layer has no opportunity for this hierarchical composition.

**Analogy:** You could build a skyscraper using one very thick slab of concrete, or you could use many floors. The many-floors design allows different activities at different heights, makes better use of materials, and can be much taller. Same principle for network depth.

---

## The Projection Matrices in Self-Attention 🎯

Linear layers don't just appear in the feed-forward parts of transformers — they play a critical role in the attention mechanism.

In self-attention, every token's embedding is projected into three separate spaces:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I offer?"
- **Value (V):** "What information do I carry?"

Each of these projections is a linear layer (a matrix multiplication). If the model has 8 attention heads and an embedding dimension of 512, each head uses projection matrices of shape 512 × 64.

After attention computes the weighted combination of Values, there's another linear layer — the **output projection** — that combines the results from all heads back into the main 512-dimensional embedding space.

### Why project at all?

Without these projections, every attention head would see the same representation. The projections let different heads specialise — one head might project to a space where grammatical relationships are prominent, while another head projects to a space that highlights semantic similarity.

**Analogy:** Imagine reading the same newspaper through different coloured lenses. Each lens (projection matrix) emphasises different aspects — one makes financial news stand out, another highlights political news. The linear projections in attention heads work similarly, letting each head "focus" on different aspects of the token embeddings.

---

## Common Misconceptions About Linear Layers 🚫

### "Linear layers are simple and unimportant"

Linear layers do the majority of the heavy computational lifting in transformers. The feed-forward network (which is mostly linear layers) accounts for roughly 2/3 of a transformer's parameters. They're doing enormous amounts of work — compressing, expanding, rotating, and projecting representations.

### "The weight matrix is fixed after training"

Correct — but only during inference. During fine-tuning, all weight matrices can be updated again. Techniques like LoRA (Low-Rank Adaptation) specifically target which weight matrices to fine-tune, making adaptation efficient by adding small correction matrices to the existing ones.

### "More dimensions = always better"

Wider layers (larger dimension) give more expressive power, but also require more computation and more training data to fit well. Extremely wide layers can overfit or become hard to train. Architecture research is about finding the right balance.

### "Linear means the whole model is linear"

The combination of linear layers WITH activation functions makes the whole network non-linear. Each individual linear layer is linear, but the interleaving of activations creates a powerful non-linear function overall. The "linear" in "linear layer" describes just that one operation, not the network as a whole.

### "Linear layers can't learn anything about context"

Partially true — a single linear layer applied to one token independently can't see other tokens. But the attention mechanism (which uses linear projections) combines information across tokens. So linear layers work in concert with attention to enable context-sensitive representations.

---

## Connections to Other Topics 🔗

- **Neural Networks** (Topic 01): Linear layers are the core computation inside each node
- **Activation Functions** (Topic 03): What makes the stack of linear layers non-linear and powerful
- **Attention** (Topic 06): Uses linear projections (Q, K, V matrices) to focus on relevant context
- **Layer Normalisation**: Applied before or after linear layers to keep activations well-scaled
- **Residual Connections**: Allow the output of a linear layer to be added back to its input, enabling very deep networks to train
