# ⚖️ Layer Normalization — Keeping Values Stable

## The Big Idea

As data flows through dozens of transformer layers, numbers can grow explosively large or collapse to near-zero. Either extreme makes training unstable. **Layer Normalization** rescales the values at each layer to keep them in a healthy range — like a volume control that prevents both deafening feedback and complete silence.

---

## Real-Life Analogy: The Audio Engineer 🎚️

Imagine you're recording a live concert:

- Some instruments (drums) are extremely loud
- Some instruments (flute) are very quiet
- If you record without adjusting levels, the drums overwhelm everything and the flute is inaudible

A good audio engineer constantly adjusts levels — normalizing the mix so every instrument is heard at an appropriate volume.

Layer normalization does this for the numbers flowing through a neural network. After each layer, it "normalizes" the activations so they have consistent scale and center.

---

## The Problem Without Normalization

Imagine running numbers through 96 transformer layers with multiplications and additions at each step:

- Numbers could spiral upward: 1 → 10 → 100 → 1,000 → ... → ∞
- Or collapse to near-zero: 1 → 0.5 → 0.25 → ... → ~0

**Why is this a problem?**
- Activation functions like ReLU at very large values become straight lines (they just pass everything through) — no non-linearity
- Gradients during training also explode or vanish, making learning impossible
- The model becomes numerically unstable

**Analogy:** Imagine a game of telephone where each person either whispers (signal dying) or shouts (signal distorting). You need a "reset" at each step to keep the message clear.

---

## What Normalization Does

For each vector of activations in a layer:

1. **Compute the mean** (average value across the vector)
2. **Compute the variance** (how spread out the values are)
3. **Subtract the mean** — centers the values around 0
4. **Divide by standard deviation** — makes the spread consistent
5. **Apply learned scale and shift** — let the model tune the final range

The result: every token's activation vector has mean ≈ 0 and variance ≈ 1 (before the learned scale/shift).

**Analogy:** Every runner in a race has a different pace. Normalization converts each runner's time to "how many standard deviations above/below the average" they ran. A 5-minute mile means different things at a school race vs. the Olympics — normalization removes this context-dependency.

---

## Batch Norm vs. Layer Norm 🔄

**Batch Normalization** (used in CNNs):
- Normalizes across the **batch dimension** — compares one token's activation to the same activation in other training examples
- Works well for fixed-size inputs (images)
- Struggles with variable-length sequences (text)

**Layer Normalization** (used in Transformers):
- Normalizes across the **feature dimension** — normalizes each token's activation vector independently
- Works for variable-length sequences
- No dependency on batch size

**Analogy:**
- Batch Norm: Compare your test score to everyone else who took the test today
- Layer Norm: Compare each answer you gave to the other answers you gave on the same test

Layer Norm is self-contained per token — exactly what you need for sequences.

---

## Pre-Norm vs. Post-Norm ⏱️

**Original Transformer (Post-Norm):**
```
Output = LayerNorm(Input + Sublayer(Input))
```
Normalization happens *after* the sublayer and residual connection.

**Modern Transformers (Pre-Norm):**
```
Output = Input + Sublayer(LayerNorm(Input))
```
Normalization happens *before* the sublayer.

Modern models predominantly use Pre-Norm because:
- Training is more stable (gradients flow better through the residual path)
- Deeper models are easier to train
- Converges faster at large scale

**Analogy:**
- Post-Norm: You eat dinner, then weigh yourself → the meal affects the measurement
- Pre-Norm: You weigh yourself before dinner → measurement is of your "base" state, then you eat

Pre-Norm normalizes the input before processing, so the sublayer always sees well-scaled inputs.

---

## RMSNorm — A Simpler Variant 📐

**RMSNorm (Root Mean Square Normalization)** is used in modern LLMs (LLaMA, Mistral, Gemma).

It simplifies Layer Norm by skipping the mean subtraction:

1. Compute the RMS (root mean square) of the vector
2. Divide each element by the RMS
3. Apply learned scale

**Why simpler?**
- Re-centering (subtracting mean) turns out to be less important than rescaling
- RMSNorm is faster to compute (about 20% speedup)
- Empirically matches or exceeds Layer Norm quality in modern architectures

**Analogy:** A "good enough" audio normalizer that only adjusts volume (amplitude) without also shifting the baseline. Simpler, faster, works just as well in practice.

---

## The Learned Scale and Shift (γ and β)

After normalization, two learned parameters restore the model's ability to choose its own scale:

- **γ (gamma):** Scale factor — multiplied after normalization
- **β (beta):** Shift — added after normalization (not used in RMSNorm)

Initially γ=1 and β=0 (no change). The model learns to adjust these during training.

**Why?** Pure normalization to mean=0, variance=1 might actually be *too* constraining. The learned scale/shift lets the model say "actually, I want this layer's activations to be a bit larger (or shifted)"—giving back some flexibility while keeping extreme values in check.

---

## Where Normalization Is Applied

In a typical transformer block:
```
x → LayerNorm → Attention → + (residual) → x
x → LayerNorm → FFN → + (residual) → x
```

Also added at the very end of the model before the output head:
```
x → Final LayerNorm → Linear (vocabulary projection) → logits
```

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| Layer Norm | Rescale each token's activation vector to mean≈0, std≈1 |
| Batch Norm | Normalize across training examples (used in CNNs, not Transformers) |
| Pre-Norm | Normalize before the sublayer — more stable training |
| Post-Norm | Normalize after the sublayer — original Transformer design |
| RMSNorm | Simplified Layer Norm using only RMS scaling, no mean subtraction |
| γ, β | Learned scale and shift applied after normalization |

---

## Up Next
👉 **KV Cache** — a performance optimization that makes inference dramatically faster.
