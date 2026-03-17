# 📐 KL Divergence — Don't Stray Too Far

> **Sources used:**
> - Lambert et al., *Illustrating Reinforcement Learning from Human Feedback (RLHF)*, Hugging Face Blog 2022 — [huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)
> - Huang et al., *The N+ Implementation Details of RLHF with PPO*, Hugging Face Blog 2023 — [huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)
> - Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, Stanford/Anthropic 2023 — [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
> - Lilian Weng, *Policy Gradient Algorithms*, OpenAI Blog 2018 — [lilianweng.github.io/posts/2018-04-08-policy-gradient](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

---

## The Big Idea

KL Divergence is a number that measures how **different two probability distributions are** from each other. In the context of LLMs and RLHF, it answers: "How much has the model changed from its original self?" It's used as a safety leash — making sure the model doesn't drift so far during RL training that it starts producing gibberish or reward-hacking responses.

---

## Real-Life Analogy: The New Employee 👔

Imagine you hire someone brilliant who knows their job well. After 3 months, you review their work. You want them to have improved — but not so dramatically that they've completely changed how they operate. If after 3 months they're writing reports in a completely different format, using different terminology, ignoring company style guides, something has gone wrong.

**KL Divergence is the "how much have you changed" metric.** You want it to be low enough that the employee still recognises as the same person doing the same job — just better.

---

## What Is a Probability Distribution? 🎲

Before KL divergence makes sense, we need to understand distributions.

When an LLM predicts the next token, it outputs a probability for every token in the vocabulary:

```
"mat":    42%
"floor":  18%
"chair":  11%
"sofa":    9%
"rug":     7%
... (32,000+ tokens, all summing to 100%)
```

This list of probabilities is a **probability distribution**. Two different models will have different distributions over the same prompt. KL divergence measures how different those distributions are.

---

## KL Divergence — The Intuition 🔍

Let's build up the intuition from scratch before the formula.

### Starting Point: Information and Surprise

Information theory tells us that rare events are more "surprising" than common events. If you see a coin land heads (50% probability), that's not very surprising. If you win the lottery (1 in 14 million), that's extremely surprising.

The "surprise" of an event with probability p is measured as: -log(p)
- p = 1.0 (certain event) → surprise = 0 (no information)
- p = 0.5 (coin flip) → surprise ≈ 0.69
- p = 0.01 (rare event) → surprise ≈ 4.6

### Entropy: Average Surprise of a Distribution

The **entropy** of a distribution P is the average surprise you'd feel if events were drawn from P:

```
H(P) = -sum_x P(x) * log(P(x))
```

High entropy = unpredictable distribution (uniform over many values)
Low entropy = predictable distribution (probability mass concentrated on a few values)

### Cross-Entropy: Surprise When You Have the Wrong Map

Now imagine you're using distribution Q as your mental model, but reality follows distribution P. The **cross-entropy** H(P, Q) is the average surprise you'd experience:

```
H(P, Q) = -sum_x P(x) * log(Q(x))
```

This is always ≥ H(P). The mismatch costs you extra surprise.

### KL Divergence: The Extra Cost of the Wrong Map

KL divergence is the difference — the **extra** surprise you pay for using Q when P is true:

```
KL(P ‖ Q) = H(P, Q) - H(P) = sum_x P(x) * log(P(x) / Q(x))
```

**Plain English:** KL(P‖Q) = "If reality is P but I'm using Q as my model, how much extra information do I need?"

---

## KL Divergence — The "Surprise" Measure 😲

Formally, KL divergence between two distributions P and Q asks:

> "If I assume Q is true but P is actually true — how surprised am I on average?"

**Analogy:** You've been living in London (distribution Q — you expect mild, rainy weather). You move to Phoenix, Arizona (distribution P — actual weather is hot and dry). KL divergence measures how *wrong* your London-weather expectations are when applied to Phoenix.

- A British person moving to New Zealand (similar climate) → small KL divergence
- A British person moving to the Sahara → large KL divergence

**Key properties:**
- KL divergence is **always ≥ 0**
- KL = 0 means the distributions are identical
- KL is **not symmetric**: KL(P‖Q) ≠ KL(Q‖P) in general
- Large KL = the two distributions are very different

---

## KL is Asymmetric — Order Matters ⚖️

This is one of the most misunderstood properties of KL divergence, and it has deep practical consequences.

### The Asymmetry

KL(P‖Q) ≠ KL(Q‖P) in general. They measure different things:

- **KL(P‖Q)** — "How surprised am I if I think the world is Q but it's actually P?" You're penalised heavily wherever **P is high and Q is low** (you assign near-zero probability to things that actually happen a lot)
- **KL(Q‖P)** — "How surprised am I if I think the world is P but it's actually Q?" You're penalised heavily wherever **Q is high and P is low**

### A Concrete Example

Imagine two distributions over 4 tokens: {cat, dog, fish, bird}

```
P (true distribution):  cat=0.6, dog=0.3, fish=0.07, bird=0.03
Q (approximate model):  cat=0.5, dog=0.2, fish=0.2,  bird=0.1
```

P has most mass on "cat" and "dog". Q spreads mass more to fish and bird.

- **KL(P‖Q):** "Using Q but true is P" — The mismatch on cat (P=0.6, Q=0.5) and dog (P=0.3, Q=0.2) costs us. These tokens are common in P, so mismodelling them is expensive.
- **KL(Q‖P):** "Using P but true is Q" — The mismatch on fish (Q=0.2, P=0.07) and bird (Q=0.1, P=0.03) costs more proportionally.

The divergence values are different because the distributions weight mismatches differently.

### The Practical Implication

In RLHF, we compute KL(RL policy ‖ Reference policy) — we're measuring how surprised you'd be if you were using the reference policy but the RL policy is actually generating tokens. This penalises the RL policy for placing high probability on tokens the reference model considers very unlikely.

---

## Forward vs Reverse KL 🔄

These terms come from variational inference but are relevant whenever you're approximating one distribution with another.

### Forward KL: KL(P‖Q)

Also called the "inclusive" or "mean-seeking" KL.

**Behaviour:** Minimising KL(P‖Q) over Q forces Q to cover all the places where P has significant probability mass. If P has two modes (two high-probability regions), Q will spread to cover both — even if that means putting probability mass in between the modes where P is low.

**Analogy:** You're trying to draw a circle that contains a whole country. You have to make the circle big enough to include every city, even if that means a lot of empty ocean inside the circle.

### Reverse KL: KL(Q‖P)

Also called the "exclusive" or "mode-seeking" KL.

**Behaviour:** Minimising KL(Q‖P) over Q forces Q to concentrate on the modes of P. Q becomes "mode-seeking" — it picks one high-probability region of P and commits to it, ignoring other modes. If P has two modes, Q typically collapses onto just one.

**Analogy:** You're trying to draw a small circle that's entirely inside the country. You don't need to cover everything — just be accurate about the region you do cover.

### Why This Matters for LLMs

In variational autoencoders and diffusion models, the choice of forward vs reverse KL affects whether the model learns to cover all modes of the data distribution (diverse outputs) or commit to the most probable mode (consistent outputs). For RLHF, the KL penalty is reverse KL — it keeps the RL policy from placing mass in regions where the reference policy assigns zero probability.

---

## KL Divergence in RLHF — The Safety Leash 🐕

During RL training, the model is optimised to get high reward scores. Without any constraint, it would quickly learn to produce responses that score high on the reward model but are bizarre or degenerate — **reward hacking**.

The solution, from the RLHF paper ([huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)):

```
Total reward = Reward Model Score − λ × KL(RL policy ‖ Reference policy)
```

Where:
- **Reward Model Score:** How good the response is (from the reward model)
- **KL penalty:** How much the current model has diverged from the original SFT model
- **λ (lambda):** A coefficient controlling how strongly to penalise divergence

**In plain English:** The model is rewarded for good responses, but penalised for changing too much from its original self. It's a tug-of-war:
- Pull toward high reward (be helpful)
- Pull back toward original model (don't go crazy)

The HuggingFace implementation guide ([huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)) shows this is computed **per token**:

```
Per-token KL penalty = β × (log_prob_RL_policy − log_prob_reference_policy)
```

The model is checked at every single token: "how differently would the original model have handled this?"

---

## KL in RLHF — The Two Distributions in Practice 📊

Let's make this concrete. Suppose the prompt is: "What is the capital of France?"

The **reference policy** (frozen SFT model) might produce a token distribution like:

```
Token distribution at position 1 (after "The capital"):
"of":     0.72
"city":   0.11
"Paris":  0.08
"French": 0.05
...
```

After RL training, the **RL policy** might produce:

```
Token distribution at same position:
"of":     0.68  (similar — small KL)
"city":   0.13
"Paris":  0.09
"French": 0.06
```

This is small KL — the distributions are close. The RL policy has slightly shifted but not dramatically.

A reward-hacked RL policy might produce:

```
Token distribution (reward-hacked):
"of":       0.02
"is":       0.03
"...and":   0.05
"PARIS!!!": 0.75  (learned that exclamation marks score higher)
```

This has large KL — the distributions are very different. The KL penalty would be high, suppressing this behaviour.

---

## The Reference Policy — The Anchor ⚓

The **reference policy** is the frozen copy of the SFT model, kept unchanged throughout RL training.

It serves as the baseline that the KL divergence is measured against. The RL policy is trained, but the reference policy never changes — it's the "original self" that the model must stay close to.

**Analogy:** It's like tethering a hot-air balloon to the ground. The balloon (RL policy) can rise and move with the wind (improve with RL), but the tether (KL penalty) prevents it from flying away to somewhere completely unpredictable.

---

## KL in Variational Inference 🔬

KL divergence originated not in RL but in **information theory and Bayesian statistics**. Understanding its broader role helps appreciate why it's used in RLHF.

### The Variational Autoencoder (VAE) Connection

In a VAE, you're trying to learn an approximate posterior distribution q(z|x) that's close to the true posterior p(z|x). The training objective includes:

```
ELBO = E[log p(x|z)] - KL(q(z|x) ‖ p(z))
```

The KL term (reverse KL here) pushes the approximate posterior toward the prior. This is mathematically identical in spirit to the RLHF objective: maximise a quality metric while staying close to a reference distribution.

### The Broader Pattern

Everywhere you have:
- A **target distribution** you want to reach (high reward, high likelihood)
- A **reference distribution** you must stay close to (original model, prior)
- A **constraint budget** controlling how far you can go (λ, β)

...KL divergence shows up as the natural measure of distance. It's the information-theoretic glue that connects Bayesian inference, variational methods, and RL.

---

## Adaptive KL Control — A Self-Tuning Leash 🎛️

The λ coefficient doesn't have to be fixed. The implementation details paper describes **adaptive KL control**:

```python
# If current KL is above target → tighten the leash (increase β)
# If current KL is below target → loosen the leash (decrease β)
target_KL = 6.0   # How much divergence is acceptable
```

**Analogy:** A smart dog leash. If the dog is running too far ahead (too much KL), it automatically tightens. If the dog is staying close (low KL), it lets out a bit more slack so the dog can explore.

---

## KL Coefficient β — Tuning the Penalty 🎚️

The β coefficient (also called λ in some formulations) is one of the most important hyperparameters in RLHF. Getting it wrong breaks training.

### β Too Small

If β ≈ 0:
- No constraint on policy change
- The RL policy quickly exploits the reward model
- Reward scores skyrocket (the model has found reward hacks)
- Human evaluations plummet (the responses are incoherent)
- The policy may produce repetitive, degenerate text that happens to score well

This is exactly reward hacking. In practice this can happen within just a few hundred training steps with β=0.

### β Too Large

If β is very large (e.g., β = 10):
- The policy is almost entirely frozen — any improvement is penalised
- The RL training makes essentially no progress
- You end up with the SFT model, wasting all the RL computation

### β In Practice

The InstructGPT paper used adaptive KL with a target of KL ≈ 6.0 nats. The HuggingFace implementation recommends starting at β = 0.1–0.2 and adjusting.

Different algorithms handle this differently:
- **PPO:** Explicit KL penalty in the reward signal
- **GRPO:** Explicit KL penalty in the loss, often β = 0.001–0.1
- **DAPO:** Removes the KL penalty entirely for some settings and uses entropy bonuses instead

### The Goldilocks Zone

You want β in the "Goldilocks zone" where:
- The policy improves meaningfully from RL
- The policy doesn't go so far that reward hacking starts
- Response quality (as measured by humans) keeps improving

Finding this zone typically requires multiple ablation experiments.

---

## KL Divergence in Trust Region Methods

KL divergence is central to a whole family of RL algorithms:

**TRPO (Trust Region Policy Optimisation):**
> *"Enforces a KL divergence constraint on the size of policy update to prevent instability."* — Lilian Weng, [lilianweng.github.io](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

The idea: instead of just penalising large changes in the reward objective, explicitly enforce that each update step doesn't change the policy distribution by more than a KL budget.

**PPO (Proximal Policy Optimisation):**
Approximates the TRPO constraint using clipping instead of the exact KL constraint — cheaper to compute, similarly effective.

**GRPO, DAPO, etc.:**
The newer algorithms from the RL for LLMs article all include or modify the KL term — it's a central design choice in every modern LLM alignment algorithm.

---

## DPO — KL Without an Explicit Reward Model 🔄

The DPO paper ([arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)) showed something elegant: the RLHF objective (maximise reward while constraining KL) has a **closed-form solution**. You don't need a separate reward model or RL loop — you can directly optimise the KL-constrained objective using just preference data.

The KL divergence is still there in the math — DPO just bakes it into the training objective directly rather than running an explicit RL loop.

### How DPO Works

The DPO loss for a preference pair (winner, loser) is:

```
loss = -log(sigmoid(β × (log π_θ(winner|x) - log π_ref(winner|x))
                   - β × (log π_θ(loser|x) - log π_ref(loser|x))))
```

Where:
- π_θ = the policy being trained
- π_ref = the reference (frozen SFT) policy
- β = the KL coefficient (same β as in RLHF!)

**The magic:** The `log π_θ / log π_ref` terms are implicit KL divergence measures. The model is simultaneously learning from preferences AND being penalised for KL divergence from the reference — in a single training pass, with no explicit reward model.

### DPO vs RLHF

| Aspect | RLHF | DPO |
|---|---|---|
| Stages | 3 (SFT → RM → RL) | 2 (SFT → DPO) |
| Reward model | Required | Not needed |
| RL algorithm | PPO (complex) | Standard SFT-like training |
| KL penalty | Explicit in reward signal | Implicit in loss function |
| Compute | High | Lower |
| Quality | State of art (2022) | Comparable |

This is why KL divergence appears everywhere in modern LLM alignment — it's the mathematical core of "stay close to the original model while improving."

---

## What Happens When KL is Too Large 💣

Let's trace through exactly what goes wrong when the KL constraint is removed or too weak.

### Stage 1: Early Training (KL small, all good)

The RL policy makes modest improvements. It learns to be more concise, follow instructions better, refuse harmful requests. Human evaluators prefer it over the SFT baseline.

### Stage 2: Middle Training (KL growing, warning signs)

The policy starts to over-optimise for the reward model. It adds unnecessary caveats ("This is a great question! I'd be happy to help you with..."). Responses get longer. Evaluators notice a slight decline in naturalness.

### Stage 3: Late Training (KL very large, breakdown)

The policy has found reward model exploits:
- Responses degenerate into repetitive loops ("The answer is yes. Yes, the answer is yes. The answer is definitely yes...")
- The model produces confident-sounding but factually wrong answers
- Responses are loaded with flattery and sycophantic phrases
- The reward model score is high, but human evaluators find the responses unusable

This is the **reward model overoptimisation** problem. The KL penalty — when set correctly — acts as the circuit breaker that prevents Stage 3.

---

## Visualising KL Divergence 🗺️

Imagine two probability distributions as hills on a landscape:

```
Reference policy (frozen SFT):     RL policy (after training):

     ████                                 ████
    ██████                               ██████
   ████████                             ████████
  ██████████   ██                      ████████████
 ████████████████                     ██████████████

Token distribution for prompt X      (slightly shifted, but similar shape)
```

Small KL = the hills look similar (peaks in similar places)
Large KL = the hills look completely different (the RL policy has shifted dramatically)

---

## Common Misconceptions ❌

### Misconception 1: "KL divergence is a distance metric"

**Reality:** KL divergence is **not** a true distance metric because it's asymmetric: KL(P‖Q) ≠ KL(Q‖P). A true distance metric must be symmetric. KL is sometimes called a "divergence" or "pseudo-distance" to distinguish it from proper metrics. The Jensen-Shannon divergence (the average of the two KL directions) is symmetric and is a true metric.

### Misconception 2: "Higher β always means safer training"

**Reality:** Too high a β makes training useless — the model can't learn anything. "Safe" training requires β in a zone where meaningful improvement happens without reward hacking. The right β depends on the model size, data quality, and RL algorithm.

### Misconception 3: "DPO doesn't use KL divergence"

**Reality:** DPO uses KL divergence implicitly in its loss function. The β hyperparameter in DPO has the same role as β in RLHF — it controls the KL penalty strength. The mathematical derivation of DPO starts from the same RLHF objective and solves it in closed form.

### Misconception 4: "The KL penalty is computed on the full sequence"

**Reality:** In practice, the KL penalty is computed **per token** and summed over the sequence. This means the penalty is proportional to response length — longer responses accumulate more KL penalty. This creates an implicit pressure toward shorter responses, which sometimes counteracts length bias in the reward model.

### Misconception 5: "KL divergence is unique to LLM alignment"

**Reality:** KL divergence is one of the most fundamental quantities in information theory, Bayesian statistics, and machine learning. It appears in: VAEs (variational autoencoders), information geometry, maximum likelihood estimation, hypothesis testing, and much more. RLHF adapted it from the trust region RL literature.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| KL Divergence | Measures how different two probability distributions are |
| KL = 0 | Distributions are identical |
| KL > 0 | Distributions differ — larger KL = bigger difference |
| KL asymmetry | KL(P‖Q) ≠ KL(Q‖P) — order matters |
| Forward KL | Inclusive / mean-seeking — covers all modes of P |
| Reverse KL | Exclusive / mode-seeking — commits to one mode |
| Reference policy | Frozen copy of SFT model — the "anchor" |
| KL penalty in RLHF | Penalise the model for drifting too far from the reference |
| β too small | Reward hacking — model exploits reward model |
| β too large | No learning — model is frozen |
| Adaptive KL | Automatically tighten/loosen the penalty based on current divergence |
| Trust region | A budget of how much the policy can change per update |
| DPO | Bakes KL penalty directly into loss — no explicit RM needed |

---

## Up Next
👉 **RLHF Overview** — now that we have all the pieces (SFT model, reward model, KL divergence), let's see how they combine into the full RLHF training pipeline.
