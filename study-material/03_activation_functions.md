# ⚡ Activation Functions — Breaking Linearity

## The Big Idea

A neural network made of only linear layers is actually very limited. No matter how many layers you stack, the result is still equivalent to just one layer. **Activation functions break this limitation** by introducing non-linearity — the ability to model complex, curved relationships.

---

## Real-Life Analogy: The Light Dimmer 💡

Imagine a light switch vs. a light dimmer:

- **Linear function** = a regular light switch. Output is always proportional to input. More voltage in → more light out. Perfectly predictable.
- **Activation function** = a smart dimmer that has a threshold. Below 30% power → it stays completely off. Above 30% → it gradually turns on. Above 80% → it maxes out and stays bright.

The smart dimmer makes **decisions based on thresholds** — it behaves differently in different regions. This is what activation functions do for neural networks.

---

## Why Linearity Alone Fails

**Analogy:** Imagine predicting if a student will pass an exam based on:
- Hours studied
- Hours slept

A linear model can only draw a straight line through the data. But reality is curved:
- 0 hours studied → fail (regardless of sleep)
- 8 hours studied + 0 hours sleep → might still fail
- 8 hours studied + 8 hours sleep → probably passes

The relationship is not a straight line — it has corners, thresholds, interactions. You need a non-linear function to capture this.

---

## ReLU — The Most Common Activation 🔋

**ReLU** stands for **Rectified Linear Unit**. Despite the fancy name, it's extremely simple:

> If the input is negative → output 0. If positive → output the same number.

```
input: -5   →  output: 0
input: -0.1 →  output: 0
input:  0   →  output: 0
input:  2   →  output: 2
input:  7   →  output: 7
```

**Analogy:** ReLU is like a **one-way valve** in plumbing:
- Water flowing backward (negative signal)? Block it. Output is 0.
- Water flowing forward (positive signal)? Let it through unchanged.

**Why it works:**
- Simple and fast to compute
- Lets gradients flow through positive neurons during training
- Creates "dead zones" — neurons that output zero don't contribute, making the network sparse and efficient

---

## SiLU — The Smoother Version 🌊

**SiLU** (Sigmoid Linear Unit, sometimes called **Swish**) is a smoother variant:

> It's like ReLU but with a gentle curve instead of a sharp corner.

**Analogy:** Instead of a one-way valve that slams shut, SiLU is like a **pressure-sensitive valve** that:
- Fully closes for very negative inputs
- Has a slight opening even for slightly negative inputs
- Opens fully for positive inputs

This smoothness helps gradients flow better during training, especially in very deep networks. SiLU is used in modern LLMs like LLaMA.

---

## Sigmoid — The Probability Squisher 📊

**Sigmoid** squashes any number into the range 0 to 1.

> Very large positive numbers → close to 1
> Very large negative numbers → close to 0
> Zero → exactly 0.5

**Analogy:** A mood-o-meter on a scale of 0-100%:
- Terrible day → 5%
- Average day → 50%
- Best day ever → 95%

No matter how extreme the input, the output stays between 0 and 1. This makes it useful for **probabilities** and **binary decisions** (yes/no, true/false).

Used in older networks and for output layers that need probability-like values.

---

## Tanh — The Centered Version 〰️

**Tanh** (hyperbolic tangent) is like sigmoid but centered at 0, ranging from -1 to +1.

**Analogy:** A sentiment scale:
- -1 = extremely negative
- 0 = neutral
- +1 = extremely positive

Useful when you need outputs that can go negative, and want them centered around zero.

---

## GELU — The Modern Favorite 🏆

**GELU** (Gaussian Error Linear Unit) is used in GPT, BERT, and most modern transformers.

It's similar to ReLU but with a smooth, probabilistic twist:
- Near-zero inputs are partially gated (sometimes passed, sometimes blocked)
- Large positive inputs pass through cleanly
- Large negative inputs are suppressed

**Analogy:** GELU is like a thoughtful editor reviewing drafts:
- Strong, clear ideas (high positive) → approved every time
- Weak, uncertain ideas (near zero) → sometimes approved, sometimes cut
- Bad ideas (negative) → usually rejected

This stochastic quality helps the network make better use of uncertain signals.

---

## Why Different Activations?

| Activation | Shape | Best For | Used In |
|---|---|---|---|
| ReLU | Sharp corner at 0 | General deep learning | CNNs, older models |
| SiLU/Swish | Smooth curve | Deep transformers | LLaMA, modern LLMs |
| Sigmoid | S-shaped, 0→1 | Binary output, gates | LSTM, binary classifiers |
| Tanh | S-shaped, -1→1 | Centered outputs | RNNs, LSTM |
| GELU | Smooth, similar to ReLU | Language models | GPT, BERT |

---

## The Role in a Network

A full network layer looks like:

```
Input → Linear Layer (multiply + add) → Activation Function → Output
```

The linear layer transforms the shape. The activation function introduces the curves and thresholds that allow complex pattern recognition.

**Without activation:** Stack 100 linear layers → still just one linear layer in disguise.

**With activation:** Each layer can learn genuinely new, non-linear relationships.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| Non-linearity | The ability to model curved, threshold-based relationships |
| ReLU | Keep positives, zero out negatives — like a one-way valve |
| SiLU | Smooth ReLU — gentle instead of sharp |
| Sigmoid | Squeeze any number into 0–1 range |
| Tanh | Squeeze any number into -1–1 range |
| GELU | Probabilistic smooth gate, popular in modern LLMs |

---

## Up Next
👉 **Tokenization** — before any of this math can happen, text must become numbers.

---

---

## The Vanishing Gradient Problem — Why Old Activations Failed 📉

To understand why modern activations like GELU and SiLU replaced older ones like sigmoid and tanh, you need to understand one of the most important problems in deep learning: the vanishing gradient.

### What is a gradient?

During backpropagation, every weight receives a "gradient" — a number that tells it how much to change. A large gradient means "change a lot." A small gradient near zero means "barely change." Gradients are what drive learning.

### What goes wrong with sigmoid in deep networks

Sigmoid squashes all inputs to the range 0–1. Specifically, large positive numbers map to values very close to 1, and large negative numbers map to values very close to 0.

Here's the problem: when you compute the gradient of sigmoid, the maximum value it can return is 0.25 (it happens right at the centre, at input=0). At the extremes, the gradient approaches 0.

Now imagine a network with 20 layers, all using sigmoid. When backpropagation multiplies these gradients together layer by layer:

0.25 × 0.25 × 0.25 × ... × 0.25 (20 times) ≈ 0.000000000001

The gradient has essentially vanished by the time it reaches the early layers. Those layers receive an update signal so tiny it might as well be zero. They stop learning.

**Analogy:** Imagine a game of telephone across 20 people. Each person only passes on 25% of the message they heard. By the 20th person, the original message is completely lost. That's what happens to gradients in deep sigmoid networks.

### Why ReLU was a breakthrough

ReLU's gradient is either exactly 0 (for negative inputs) or exactly 1 (for positive inputs). There's no compression near 1 — the gradient for positive neurons is always preserved at full strength. This allowed researchers to train much deeper networks successfully.

The catch: neurons that receive negative inputs get a gradient of exactly zero — they never update. If enough neurons get stuck outputting zero, you've lost much of the network's capacity. This is the dead ReLU problem (see below).

---

## GELU — The Gaussian Error Linear Unit 🔬

GELU is the activation function used in the original BERT model and all versions of GPT. Understanding it helps you understand why modern models perform so well.

### The intuition

GELU is based on the idea of **stochastic regularisation**. During training, some neurons are randomly "dropped out" (set to zero) to prevent the model from over-relying on any single neuron — a technique called Dropout.

GELU mathematically approximates this behaviour in a continuous, smooth way. Instead of asking "is this input positive or negative?" (like ReLU), GELU asks: "If I were randomly deciding whether to activate this neuron, how likely would I be to activate it given this input?"

For large positive inputs, the answer is "very likely" — the neuron activates strongly. For large negative inputs, "very unlikely" — the neuron is suppressed. For values near zero, it's uncertain — the output is somewhere in between, and the exact value depends smoothly on the input.

### What this looks like

- Input = 2.0 → output ≈ 1.95 (almost unchanged — strong activation)
- Input = 0.5 → output ≈ 0.35 (partial activation)
- Input = 0.0 → output = 0.0
- Input = -0.5 → output ≈ -0.15 (small negative pass-through)
- Input = -2.0 → output ≈ -0.05 (nearly suppressed)

Notice that unlike ReLU, GELU allows small negative values to pass through for inputs just below zero. This richer behaviour is part of why it works better for language tasks.

### Practical note

GELU is slightly more expensive to compute than ReLU (it uses the error function or an approximation). But the improved training quality is worth the computational cost for language models, which is why it became the standard.

---

## SiLU / Swish — Smooth Self-Gating 🌀

SiLU (Sigmoid Linear Unit) and Swish are the same function. It was discovered by neural architecture search — an automated process where a computer tries thousands of activation functions and finds the ones that train best. Swish won.

### What "self-gating" means

SiLU multiplies the input by its own sigmoid value:

output = input × sigmoid(input)

The sigmoid(input) part acts as a gate that's controlled by the input itself. When input is large and positive, sigmoid ≈ 1, so the gate is fully open (output ≈ input). When input is large and negative, sigmoid ≈ 0, so the gate is nearly closed (output ≈ 0). When input is near zero, the gate is partially open.

**Analogy:** Imagine a water pipe where the valve is controlled by the water pressure itself. High pressure → valve opens wide. Low pressure → valve stays mostly shut. The pipe regulates itself based on its own flow. That's self-gating.

### Why SiLU outperforms ReLU on language tasks

The smooth curve of SiLU means gradients flow more consistently during training. There are no sharp discontinuities (unlike ReLU's sharp corner at zero) that can create unstable training dynamics in deep networks.

Additionally, SiLU allows small negative values to pass through (just like GELU), which preserves more information from the previous layer than ReLU's hard zero cutoff.

### Where it's used

- **LLaMA** (all versions): Uses SiLU in the feed-forward network — specifically in the "SwiGLU" variant
- **Mistral, Gemma, Falcon**: Also use SiLU-based activations
- **PaLM**: Uses the SwiGLU variant

---

## Dead ReLU Problem — When Neurons Go Dark 💀

The dead ReLU problem is one of the most important practical challenges in training neural networks with ReLU activations.

### What happens

During training, if a neuron receives a large negative gradient update, all its weights shift in a way that makes its input always negative. Once that happens:

1. The neuron always outputs zero (because ReLU(negative) = 0)
2. The gradient for that neuron is zero (because the gradient of ReLU below zero is zero)
3. The weights never update again (because gradient × learning rate = 0)
4. The neuron is permanently dead — it contributes nothing to the network

**Analogy:** A light bulb that blows out. Once it's gone, it produces no light and you can't fix it by plugging in more electricity. The circuit is open. The neuron is stuck.

### How common is it?

In models with millions of neurons and aggressive learning rates, it's not unusual to see 10-40% of ReLU neurons die during training. This is wasted capacity — those neurons could have been learning useful representations but instead sit doing nothing.

### Solutions

**Leaky ReLU:** Instead of outputting exactly 0 for negative inputs, output a tiny fraction (like 0.01 × input). The neuron can still receive gradient updates even when suppressed, so it can "recover" if the inputs shift back to positive.

**Parametric ReLU (PReLU):** Like Leaky ReLU, but the leak amount is a learned parameter — the network decides how much to let through.

**SiLU and GELU:** Both naturally avoid the dead neuron problem because they never output exactly zero for any input. The gradient is always non-zero, so every neuron can always update.

**Proper weight initialisation:** Techniques like He initialisation set initial weights so that neurons aren't immediately pushed into the negative dead zone before training can begin.

---

## Activation Functions in Real LLMs 🤖

Here's a practical guide to which activation functions appear in which real models, and why:

### GPT-2 and GPT-3 — GELU

OpenAI's GPT series uses GELU in the feed-forward network inside each transformer block. GELU was chosen based on BERT's success (which also used GELU). The smooth, probabilistic gating behaviour works well for language modelling tasks.

Architecture: each transformer block contains `Linear → GELU → Linear`

### LLaMA 1 and 2 — SiLU (via SwiGLU)

Meta's LLaMA uses a variant called SwiGLU, which combines SiLU with a gating mechanism:

The feed-forward network has three matrices instead of two. One branch applies SiLU activation, then both branches are element-wise multiplied. This gated structure gives the network more flexibility to selectively pass information.

Research showed SwiGLU outperformed plain GELU and SiLU on most benchmarks, so it became popular in open-source models.

### LLaMA 3 — SwiGLU (same as LLaMA 2)

LLaMA 3 kept the same activation architecture, just scaled up.

### BERT — GELU

BERT (Bidirectional Encoder Representations from Transformers) from Google was one of the first models to use GELU, introducing it to the mainstream.

### Older models (pre-2017) — Sigmoid and Tanh

Early recurrent neural networks (LSTMs, GRUs) used sigmoid and tanh internally for their gating mechanisms. These are specifically chosen because their 0–1 and -1–1 outputs make them natural "gates" (controlling how much information to pass through).

### Summary table

| Model | Activation | Why |
|---|---|---|
| GPT-2, GPT-3 | GELU | Smooth, probabilistic, works well for language |
| GPT-4 (estimated) | GELU or SwiGLU | Not officially confirmed |
| LLaMA 1/2/3 | SwiGLU (SiLU-based) | Better benchmark performance |
| Mistral | SwiGLU | Based on LLaMA architecture |
| BERT | GELU | Original GELU adopter |
| Falcon | GELU | Conservative, well-proven choice |
| LSTM / GRU | Sigmoid + Tanh | Required for gating mechanisms |

---

## Why Non-Linearity is the Key to Intelligence 🧠

This deserves a deeper look because it's genuinely profound.

### What non-linearity enables

A linear function can only describe proportional relationships: double the input, double the output. But the real world is full of non-linear relationships:

- Language: the meaning of "not bad" is not linear in "not" + "bad"
- Vision: recognising a face requires detecting curved edges, not just straight lines
- Reasoning: "if A then B, and if B then C, therefore if A then C" is a non-linear logical chain

Without non-linearity, a neural network is fundamentally limited to learning weighted averages. With non-linearity, it can learn arbitrary decision boundaries, complex patterns, and hierarchical abstractions.

### The universal approximation theorem

There's a beautiful mathematical result: a neural network with a single hidden layer using a non-linear activation function can approximate ANY continuous function to arbitrary precision — given enough neurons.

This means, in principle, a neural network is powerful enough to model any pattern that exists in the physical world. The activation function is what unlocks this power.

**Analogy:** Think of building shapes out of LEGO. Linear layers give you flat rectangular pieces. Activation functions give you curved pieces, corner pieces, and special connectors. With only flat rectangles, you can make cubes and walls. With all the pieces, you can build anything.

### Depth multiplies expressiveness

The universal approximation theorem says ONE wide hidden layer is enough — in theory. But in practice, depth (many layers with activations) is dramatically more efficient.

A function that requires a billion neurons to represent with one hidden layer might be representable with only a thousand neurons spread across ten layers. Deep + activation-function networks find a much richer set of abstractions than shallow ones.

This is ultimately why deep learning works: stacking many linear + activation blocks creates a hierarchy of representations that efficiently captures the structure of complex real-world data.

---

## Common Misconceptions 🚫

### "Sigmoid is outdated and should never be used"

Sigmoid is still used in two important contexts:

1. **Binary classification output**: When a model outputs a single probability (is this email spam? yes/no), sigmoid is the right choice for the final output layer.

2. **LSTM gates**: Long Short-Term Memory networks specifically need 0–1 outputs for their forget/input/output gates. Sigmoid is correct there.

The misconception comes from using sigmoid in hidden layers of deep networks, which causes vanishing gradients. In those contexts, yes, use GELU or SiLU instead.

### "ReLU is broken because of dead neurons"

Dead neurons are a real issue, but well-tuned networks with ReLU work excellently in practice. Proper learning rates, weight initialisation, and batch normalisation largely mitigate the problem. ReLU remains widely used in vision models (CNNs) where dead neuron issues are less problematic than in very deep language models.

### "The activation function doesn't matter much"

For simple tasks, this might be true. But for large language models, the choice of activation function measurably affects performance on benchmarks. The LLaMA research specifically showed that SwiGLU outperformed both ReLU and GELU activations when keeping all other factors constant.

### "Activation functions only go after linear layers"

In most architectures, yes. But some architectures use activations in other creative ways:
- **Gated Linear Units**: Apply activation to only half the output, multiplied by the other half
- **Attention mechanisms**: Sometimes use softmax (a multi-output activation) to produce attention weights
- **Output layers**: Use activation functions appropriate for the task (sigmoid for binary, softmax for multi-class)

---

## Connections to Other Topics 🔗

- **Linear Layers** (Topic 02): Activation functions always appear right after a linear layer — the two are always paired
- **Backpropagation** (Topic 01 deep dive): The gradient of the activation function is what determines how well gradients flow during training
- **Feed-Forward Network in Transformers**: The FFN is: Linear → Activation → Linear. Understanding activations means understanding the FFN
- **LSTM/GRU**: Use sigmoid and tanh specifically for their mathematical gating properties — cannot be replaced with ReLU
- **Layer Normalisation**: Applied before the activation function in many modern architectures, affects what range of values arrives at the activation
