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

## The Reference Policy — The Anchor ⚓

The **reference policy** is the frozen copy of the SFT model, kept unchanged throughout RL training.

It serves as the baseline that the KL divergence is measured against. The RL policy is trained, but the reference policy never changes — it's the "original self" that the model must stay close to.

**Analogy:** It's like tethering a hot-air balloon to the ground. The balloon (RL policy) can rise and move with the wind (improve with RL), but the tether (KL penalty) prevents it from flying away to somewhere completely unpredictable.

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

## DPO — Getting Rid of the RM but Keeping the KL 🔄

The DPO paper ([arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)) showed something elegant: the RLHF objective (maximise reward while constraining KL) has a **closed-form solution**. You don't need a separate reward model or RL loop — you can directly optimise the KL-constrained objective using just preference data.

The KL divergence is still there in the math — DPO just bakes it into the training objective directly rather than running an explicit RL loop.

This is why KL divergence appears everywhere in modern LLM alignment — it's the mathematical core of "stay close to the original model while improving."

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

## Key Takeaways

| Concept | Plain English |
|---|---|
| KL Divergence | Measures how different two probability distributions are |
| KL = 0 | Distributions are identical |
| KL > 0 | Distributions differ — larger KL = bigger difference |
| Reference policy | Frozen copy of SFT model — the "anchor" |
| KL penalty in RLHF | Penalise the model for drifting too far from the reference |
| Adaptive KL | Automatically tighten/loosen the penalty based on current divergence |
| Trust region | A budget of how much the policy can change per update |

---

## Up Next
👉 **RLHF Overview** — now that we have all the pieces (SFT model, reward model, KL divergence), let's see how they combine into the full RLHF training pipeline.
