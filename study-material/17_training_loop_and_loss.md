# 🔁 Training Loop & Loss Functions — How a Model Actually Learns

> **Sources used:**
> - Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), OpenAI 2022 — [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
> - Lilian Weng, *How to Train Really Large Neural Networks*, OpenAI Blog 2021 — [lilianweng.github.io](https://lilianweng.github.io/posts/2021-09-25-train-large/)
> - OpenAI Spinning Up, *Key Concepts in RL* — [spinningup.openai.com](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

---

## The Big Idea

A neural network starts as a random mess of numbers. Training is the process of **adjusting those numbers** — billions of them — until the model reliably predicts the next word. This requires three things: a way to measure "how wrong" the model is (the loss), a way to trace that wrongness back to every weight (backpropagation), and a rule for nudging the weights in the right direction (gradient descent).

---

## Real-Life Analogy: The Archery Coach 🏹

Imagine you're learning archery:

1. You shoot an arrow — it lands 30cm to the right of the bullseye **(prediction)**
2. The coach measures the miss **(loss)**
3. The coach analyses your stance — "left shoulder too high, grip too tight" **(backpropagation — tracing the error to its cause)**
4. You adjust your stance slightly **(gradient descent — updating weights)**
5. You shoot again **(next training step)**
6. Repeat thousands of times until you hit bullseye consistently **(convergence)**

Every single training step in an LLM follows this exact loop.

---

## The Training Objective: Next-Token Prediction 📖

LLMs are trained on a deceptively simple task: **predict the next token**.

Given the text `"The cat sat on the"`, predict `"mat"`.

The model is shown billions of such examples from the internet, books, and code. For each example, it guesses a probability for every token in the vocabulary:

```
"mat":    42%
"floor":  18%
"chair":  11%
"roof":    3%
...
```

If the true next token is "mat," the model is penalised for how low its probability for "mat" was. This penalty is the **loss**.

---

## Cross-Entropy Loss — The Penalty for Wrong Guesses 📏

**Cross-entropy loss** is the standard loss function for language modelling. It measures: *how much probability did the model assign to the correct answer?*

**The rule:**
- If the model said `"mat": 90%` and "mat" was correct → very small loss
- If the model said `"mat": 10%` and "mat" was correct → very large loss
- If the model said `"mat": 1%` and "mat" was correct → enormous loss

**Analogy:** Imagine a weather forecaster who said there was a 1% chance of rain, but it poured all day. The "loss" (embarrassment, wrongness) is proportional to how far off the confident prediction was.

The key property: the loss is **much higher** for confidently wrong predictions than for uncertain wrong ones. This drives the model to be both accurate and calibrated.

**Summed over the sequence:** The total loss for a training example is the average cross-entropy over all token positions.

---

## Backpropagation — Tracing the Error Backwards ↩️

After computing the loss, we need to know: **which weights caused this error, and by how much?**

Backpropagation is the algorithm that answers this. It works by applying the **chain rule** from calculus — tracing how each weight contributed to the final loss, layer by layer, backwards through the network.

**Analogy:** You baked a cake that tasted terrible. Backpropagation is like reverse-engineering the recipe:
- The cake was too bitter
- The bitterness came from too much cocoa
- The cocoa amount was decided by step 3 of the recipe
- Step 3 was influenced by how you read step 1

Each ingredient's contribution to the failure is assigned a "blame score" (gradient). The bigger the blame, the more you change that ingredient next time.

**What backprop computes:** A **gradient** for every single weight — a number that says "increase this weight → loss goes up" (positive gradient) or "increase this weight → loss goes down" (negative gradient).

---

## Gradient Descent — Nudging Weights in the Right Direction ⛷️

Once we have gradients for all weights, we update each weight using this rule:

```
new_weight = old_weight − (learning_rate × gradient)
```

- If the gradient is positive (increasing weight → higher loss) → decrease the weight
- If the gradient is negative (increasing weight → lower loss) → increase the weight
- The **learning rate** controls how big each step is

**Analogy:** Imagine rolling a ball down a hilly landscape trying to find the lowest valley. The gradient tells you which direction is downhill. The learning rate tells you how far to step each time. Too large a step → you overshoot and bounce around. Too small → you barely move. Finding the right learning rate is one of the most important hyperparameter choices in training.

---

## Mini-batch Training — Efficiency in Practice ⚡

Computing gradients over the entire training dataset at once would be impossibly slow and memory-intensive. Instead, modern training uses **mini-batches**:

1. Sample a small batch (e.g., 256 or 1024 examples)
2. Compute loss and gradients on this batch
3. Update weights
4. Sample next batch
5. Repeat through the full dataset (one **epoch**)
6. Repeat for multiple epochs

**Why this works:** The gradients computed on a random mini-batch are a noisy but statistically valid approximation of the true gradient over all data. The noise is actually helpful — it prevents the model from getting stuck in bad local minima.

---

## The Full Training Loop

```
Initialize all weights randomly
For each training step:
  1. Sample a mini-batch of text examples
  2. Tokenize and embed the text
  3. Forward pass: run through all transformer layers → get logit predictions
  4. Compute cross-entropy loss vs. true next tokens
  5. Backward pass (backpropagation): compute gradient for every weight
  6. Update every weight using gradient descent
  7. Log the loss (should decrease over time)
Repeat until convergence
```

A full LLM pretraining run repeats this loop **trillions** of times — across trillions of tokens of text.

---

## The Training Loss Curve 📉

During training, you track the loss over time. A healthy training run looks like:

- **Early training:** Loss is high (the model is making random guesses)
- **Mid training:** Loss drops sharply (the model is learning fast)
- **Late training:** Loss continues to fall slowly (the model is squeezing out the last improvements)
- **Overfitting:** If the training loss drops but validation loss rises, the model is memorising rather than generalising

**Perplexity:** Often reported instead of loss directly. Perplexity = e^(loss). A perplexity of 10 means the model is "as confused as if choosing from 10 equally likely options." Lower perplexity = better model.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| Training objective | Predict the next token — simple but powerful |
| Cross-entropy loss | Penalty for assigning low probability to the correct token |
| Backpropagation | Trace the error backwards to assign "blame" to each weight |
| Gradient | A number per weight: which direction to move to reduce loss |
| Gradient descent | Nudge every weight slightly in the direction that reduces loss |
| Learning rate | How big each weight update step is |
| Mini-batch | Process a small random subset of data per step for efficiency |
| Epoch | One full pass through the training dataset |

---

## Up Next
👉 **Supervised Fine-Tuning (SFT)** — how a pretrained base model is shaped into a helpful assistant.
