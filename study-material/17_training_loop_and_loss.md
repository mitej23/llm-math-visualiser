# 🔁 Training Loop & Loss Functions — How a Model Actually Learns

> **Sources used:**
> - Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), OpenAI 2022 — [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
> - Lilian Weng, *How to Train Really Large Neural Networks*, OpenAI Blog 2021 — [lilianweng.github.io](https://lilianweng.github.io/posts/2021-09-25-train-large/)
> - OpenAI Spinning Up, *Key Concepts in RL* — [spinningup.openai.com](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
> - Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla), DeepMind 2022 — [arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)

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

## 📐 What Is Perplexity?

Perplexity is one of the most commonly reported metrics in NLP, but it's often misunderstood. Let's make it concrete.

**Definition:** Perplexity = e^(average cross-entropy loss)

So if your model's loss is 2.3, perplexity = e^2.3 ≈ 10. This means your model is "as surprised as if it had to pick from 10 equally likely options at every step."

**Intuitive interpretation:**
- Perplexity = 1: Perfect model. Always assigns 100% probability to the correct token. (Impossible in practice.)
- Perplexity = 10: On average, the correct token is in the model's "top 10 likely choices"
- Perplexity = 100: Model is quite confused — correct token could be anywhere in a list of 100
- Perplexity = 1000: Model is very confused — essentially random over a 1000-item vocabulary slice
- Perplexity = vocabulary size (e.g., 50,000): Completely random model

**Real-world numbers:**
| Model | Approximate Perplexity (WikiText-103) |
|---|---|
| Random baseline | ~50,000 |
| Early GPT-2 (2019) | ~20-25 |
| GPT-3 (2020) | ~10-12 |
| Modern frontier models | ~4-6 |

**Why not just use loss?** Perplexity is more interpretable — it maps directly to "how many options the model is choosing between." A loss drop from 3.0 to 2.0 sounds small, but that's perplexity dropping from 20 to 7.4 — a huge quality jump.

**Analogy:** 🎲 Think of perplexity as how many sides your "confusion dice" have. A perfect model is a coin (2 sides — heads or tails, always picks heads). A bad model is a 1000-sided dice — completely unsure. Training reduces the dice from 1000 sides → 100 → 10 → a fair coin.

**Important caveat:** Perplexity is computed on a specific test set. A model fine-tuned on news articles will have low perplexity on news, but possibly high perplexity on code. It measures in-distribution performance only.

---

## 🏔️ Gradient Descent — Finding the Bottom of the Bowl

Let's go deeper into gradient descent, because it's the engine of all of deep learning.

**The loss landscape:** Imagine plotting loss (y-axis) as a function of two weights (x and z axes). You get a 3D surface — hills, valleys, ridges. Training is about finding the lowest valley on this surface. In reality, we have billions of weights, so this surface is a billion-dimensional hypersurface — but the intuition holds.

**Gradient = direction of steepest ascent.** We move in the *opposite* direction (steepest descent).

**Three flavors of gradient descent:**
1. **Batch gradient descent:** Compute gradient over the full dataset. Very accurate direction but impossibly slow. Almost never used for deep learning.
2. **Stochastic gradient descent (SGD):** Compute gradient on a single example. Very fast but very noisy — the direction bounces around a lot.
3. **Mini-batch gradient descent:** Compute gradient on a small batch (32-1024 examples). Best of both worlds. This is what all modern LLM training uses.

**Challenges in the loss landscape:**
- **Saddle points:** Regions that are flat in one direction but curved in another. Gradient = 0 here, so naive gradient descent gets stuck. Momentum (see Adam) helps escape these.
- **Sharp vs flat minima:** Research suggests "flat minima" (where the bottom of the valley is wide and shallow) generalize better than "sharp minima" (where the bottom is a narrow spike). This is related to batch size — larger batches tend to find sharper minima.
- **Local minima vs global minimum:** Deep networks have many local minima. Surprisingly, research shows many local minima have similar loss to the global minimum, so getting stuck in one isn't usually catastrophic.

**Gradient clipping:** If gradients get very large (a "gradient explosion"), weight updates become enormous and destabilize training. A common fix: clip the gradient norm to a maximum value (e.g., 1.0). This is standard practice in LLM training.

---

## 📈 Learning Rate — The Most Important Hyperparameter

The learning rate (lr) is a single number that controls how big each weight update step is. It has an outsized effect on training success.

**Too high:** Weights update too aggressively. The model overshoots the minimum and bounces around, or diverges entirely (loss goes to infinity). Symptom: loss spikes, "NaN" in training logs.

**Too low:** Weights update too slowly. Training converges but takes much longer than necessary. You might waste compute budget.

**The Goldilocks zone:** Typically 1e-4 to 3e-4 for large LLM pretraining with Adam optimizer.

**Why can't you just use a very small learning rate and be safe?** Because training runs are expensive — you're paying for GPU time. A 2× slower learning rate means 2× the GPU cost. At the scale of frontier models, this difference is tens of millions of dollars.

**Real numbers (from published training runs):**
- GPT-3 175B: peak learning rate 6×10^-5
- LLaMA 2 70B: peak learning rate 1.5×10^-4
- GPT-4 (estimated): much smaller due to scale

**Analogy:** 🚗 Think of the learning rate as your foot on the accelerator while navigating to a destination. On a highway (early training, far from minimum), press hard — go fast. On winding mountain roads near the destination, ease off — go carefully. Fixed learning rate = driving at the same speed everywhere = either dangerously fast on mountain roads or frustratingly slow on highways.

---

## 📊 Learning Rate Schedules

Modern training doesn't use a fixed learning rate. Instead, it follows a **schedule** that changes the learning rate over time.

**Warmup phase (first 1-5% of training):**
The learning rate starts very small (near zero) and gradually increases to its target value over the first few thousand steps.

**Why warmup?** At the very start of training, the weights are random and gradients are large and unstable. If you start with a large learning rate, the first few updates can be catastrophically bad. Warmup gives the model time to settle into a more stable regime before taking big steps.

**Cosine decay (rest of training):**
After warmup, the learning rate gradually decreases following a cosine curve — fast decrease at first, then slower as it approaches the final value (typically 10× smaller than the peak).

**The cosine schedule formula:**
```
lr(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))
```
Where t is the current step and T is total training steps.

**Why cosine, not linear?** Cosine decay spends more time at intermediate learning rates. Linear decay would rush through the high-quality region too quickly. Cosine was empirically found to produce better final models.

**Cooldown / final phase:** Some training runs end with a very short phase of very low learning rate for fine-grained final adjustments.

**Learning rate and batch size:** They're linked. If you double the batch size, you're computing gradients over twice as many examples per step — effectively a more accurate gradient. So you can often increase the learning rate proportionally (linear scaling rule) and maintain the same training dynamics.

**Real training curves:** A typical 100B-token LLM training run at 10,000 steps might look like:
- Steps 0-500: warmup (lr goes from 1e-7 → 2e-4)
- Steps 500-9000: cosine decay (lr goes from 2e-4 → 2e-5)
- Steps 9000-10000: cooldown (lr goes from 2e-5 → 1e-6)

---

## 🧠 The Adam Optimizer

Plain gradient descent is simple but inefficient. Modern LLM training uses **Adam** (Adaptive Moment Estimation), an optimizer that's far more sophisticated.

**The problem with plain SGD:**
1. It treats all parameters equally — every weight gets the same learning rate
2. It forgets history — each step only cares about the current gradient, ignoring past direction
3. It's sensitive to gradient scale — large gradients overwhelm small ones

**Adam's two key innovations:**

**1. Momentum (first moment):**
Instead of using the raw gradient each step, Adam maintains an exponential moving average of past gradients — a "running average" of the direction we've been moving.

```
m = 0.9 × m_prev + 0.1 × gradient  (momentum)
```

This is like adding inertia to a ball rolling down a hill. The ball keeps moving in the direction it was already going, rather than changing direction randomly on every step. This helps overcome small bumps and saddle points.

**2. Adaptive learning rate (second moment):**
Adam also tracks the squared gradient — a measure of how much each weight's gradient typically varies:

```
v = 0.999 × v_prev + 0.001 × gradient²  (variance estimate)
```

Then the update step is:
```
weight_update = -lr × m / sqrt(v + epsilon)
```

This means: weights with large, consistent gradients get smaller updates (already moving fast), while weights with small, erratic gradients get relatively larger updates (need more nudging). Each weight gets its own effective learning rate.

**Analogy:** 🧭 Think of SGD as a blindfolded hiker who only knows which direction is downhill right now. Adam is a smart hiker who: (a) remembers which direction they've been walking recently (momentum), and (b) knows which paths are normally gentle vs. normally steep (adaptive rate). They make much better progress.

**Adam hyperparameters:**
- β₁ = 0.9 (momentum decay — most common default)
- β₂ = 0.999 (variance decay — most common default)
- ε = 1e-8 (tiny constant preventing division by zero)
- These defaults work well for almost all LLM training — rarely tuned

**Adam vs AdamW:** AdamW is Adam with "weight decay" — a regularization term that gradually shrinks all weights toward zero, preventing any single weight from growing too large. Almost all modern LLMs use AdamW, not plain Adam.

---

## 📊 Overfitting, Underfitting, and Generalization

One of the most important concepts in machine learning is the distinction between **training set performance** and **generalization to new data**.

**The three regimes:**

**Underfitting:** The model hasn't learned enough. Both training loss and validation loss are high. The model is too simple or hasn't been trained long enough. Fix: train longer, use a bigger model.

**Good fit:** Training loss and validation loss are both low and roughly similar. The model has learned the underlying patterns without memorizing specific examples.

**Overfitting:** Training loss is low but validation loss is high — and growing. The model has started to memorize specific training examples rather than learning generalizable patterns. On the training data: perfect. On new data: poor.

**Analogy:** 📚 Imagine studying for an exam:
- Underfitting = You barely studied — don't know the basics, fail both practice tests and the real exam
- Good fit = You understood the concepts — can apply them to new questions on the real exam
- Overfitting = You memorized exactly the practice exam answers — great on the practice test, but the real exam has different questions

**Why language models rarely overfit severely:** LLMs are trained on datasets with trillions of tokens. It's very hard to memorize trillions of unique sequences. However, if you fine-tune on a small dataset (thousands of examples), overfitting becomes a real risk.

**Signs of overfitting during fine-tuning:**
- Training loss keeps dropping, but validation loss starts rising
- Model generates training examples verbatim
- Model performs poorly on slightly rephrased prompts

**Fixes:**
- Reduce number of training epochs (stop earlier)
- Increase dataset size
- Add dropout (randomly disable some neurons during training as regularization)
- Use weight decay (AdamW already does this)
- Reduce model size

**The double descent phenomenon:** At very large scale, the traditional "bias-variance tradeoff" breaks down. Models that are large enough to interpolate (memorize) all training data can actually generalize *better*, not worse. This is one reason why simply scaling up models continues to work.

---

## 🗂️ Data Preprocessing for Language Model Training

The quality of training data is arguably more important than the model architecture. Here's how raw text becomes training signal.

**Step 1: Data collection**
Common sources:
- Common Crawl: web crawl of ~petabytes of text
- Books: Project Gutenberg, Books3
- Code: GitHub, Stack Overflow
- Wikipedia
- Academic papers: ArXiv, PubMed
- Curated high-quality text: news articles, books

**Step 2: Deduplication**
Web-crawled data has massive duplication — the same news article appears on 500 different sites, the same Stack Overflow answer is copy-pasted everywhere. Deduplication removes near-duplicates, which:
- Reduces training time (less redundant data)
- Prevents memorization of duplicated content
- Improves diversity of training signal

**Step 3: Quality filtering**
Not all text is useful. Common filters:
- Remove HTML/boilerplate
- Filter by language (usually keep mostly English for English-centric models)
- Perplexity filtering: use a small reference model to score text quality — remove text that scores as very low quality
- Remove personal information (phone numbers, email addresses)
- Remove obviously toxic or spam content

**Step 4: Tokenization**
Raw text → sequence of integer token IDs, using a pretrained tokenizer (BPE or similar).

**Step 5: Packing into fixed-length chunks**
LLMs have a fixed context length (e.g., 2048 or 4096 tokens). Training examples are:
1. Tokenized text streams are concatenated
2. Split into fixed-length chunks
3. Chunks are padded if needed (though "packing" — putting multiple short documents end-to-end — is more efficient)

**Step 6: Shuffling**
Chunks are randomly shuffled so the model doesn't see the same types of content in the same order every epoch. This prevents the model from learning spurious ordering patterns.

**Step 7: Mini-batch assembly**
Chunks are grouped into mini-batches (e.g., 512 or 1024 chunks per batch). This is what's fed to the GPU for each training step.

**Why packing matters:** If you train on 2048-token chunks but your average document is 200 tokens, you have 90% of each chunk filled with padding (empty). This wastes ~90% of compute. Packing — putting doc1+doc2+doc3+... into a single 2048-token chunk — makes training ~10× more efficient.

---

## 🔬 Training Compute — Chinchilla Scaling Laws

One of the most important results in modern LLM research is the **Chinchilla paper** (Hoffmann et al., DeepMind, 2022). It answered: *for a given compute budget, what's the optimal model size and dataset size?*

**The finding:** To train optimally, the number of training tokens should be roughly **20× the number of model parameters**.

- 1B parameter model → train on ~20B tokens
- 7B parameter model → train on ~140B tokens
- 70B parameter model → train on ~1.4T tokens

**Before Chinchilla (the old wisdom):** GPT-3 was 175B parameters trained on 300B tokens — roughly 1.7 tokens per parameter. This is massively *under-trained* by Chinchilla standards!

**After Chinchilla:** LLaMA models proved the point — LLaMA 7B trained on 1T+ tokens outperforms GPT-3 175B on many benchmarks, despite being 25× smaller.

**The compute budget formula:**
```
Optimal tokens ≈ 20 × N_parameters
Optimal model size ≈ (C / 20)^0.5  (where C is total FLOPs budget)
```

**Practical implication:** For most use cases, it's better to train a *smaller* model on *more* data than a larger model on less data. A 7B model trained on 1T tokens (Chinchilla-optimal) is more capable and cheaper to run than a 70B model trained on 100B tokens.

**However:** Chinchilla optimizes for *training cost*. For inference, you want the smallest possible model. This is why LLaMA and Mistral models are "over-trained" beyond Chinchilla-optimal — they're small models trained on huge amounts of data to be cheap at inference time.

---

## 🚫 Common Misconceptions

**"Lower loss always means a better model"**
In pretraining, yes. But after fine-tuning, low loss on the fine-tuning dataset can mean overfitting. A model that answers "yes" to every question might have lower loss on a yes-heavy dataset but be terrible in practice.

**"More training always helps"**
Not beyond a certain point. Once loss plateaus, further training gives diminishing returns. And if you overtrain a fine-tuned model, you degrade the base pretraining knowledge.

**"Gradient descent finds the global minimum"**
It doesn't. It finds a local minimum (or saddle point). For deep learning, this usually doesn't matter — local minima are often nearly as good as global minima. But the exact minimum found depends heavily on initialization and the training trajectory.

**"Bigger batch size is always better"**
Bigger batches give more accurate gradient estimates per step, but you take fewer steps per token processed. There's a sweet spot. Beyond a certain batch size ("critical batch size"), you get diminishing returns. Also, too-large batches tend to find sharper minima that generalize worse.

**"Perplexity measures how smart the model is"**
Perplexity measures *next-token prediction quality on a specific test set*. A model could have very low perplexity on news articles but fail at reasoning or instruction-following. It's one metric, not a holistic intelligence measure.

**"You need to tune learning rate for every model"**
Mostly no. The Adam optimizer with β₁=0.9, β₂=0.999 and a cosine schedule works surprisingly well across a huge range of model sizes. The main thing to tune is the peak learning rate value, and published recipes give good starting points.

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
| Perplexity | e^(loss) — how many options the model is "choosing between" |
| Adam | Optimizer with momentum + adaptive per-parameter learning rates |
| Cosine schedule | Learning rate warmup then gradual decay — standard for LLMs |
| Overfitting | Train loss low, val loss high — model memorized instead of learned |
| Chinchilla scaling | Optimal: ~20 tokens per parameter for compute-optimal training |
| Data packing | Concatenate short documents to fill fixed-length training chunks |

---

## Up Next
👉 **Supervised Fine-Tuning (SFT)** — how a pretrained base model is shaped into a helpful assistant.
