# 🤝 RLHF Overview — The Full Alignment Pipeline

> **Sources used:**
> - Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), OpenAI 2022 — [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
> - Lambert et al., *Illustrating Reinforcement Learning from Human Feedback (RLHF)*, Hugging Face Blog 2022 — [huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)
> - Huang et al., *The N+ Implementation Details of RLHF with PPO*, Hugging Face Blog 2023 — [huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)
> - OpenAI Spinning Up, *Key Concepts in RL* — [spinningup.openai.com](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

---

## The Big Idea

**RLHF (Reinforcement Learning from Human Feedback)** is the training method that transformed GPT-3 — a powerful but unpredictable text predictor — into InstructGPT/ChatGPT: a model that follows instructions, refuses harmful requests, and behaves like a helpful assistant.

As the InstructGPT paper ([arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)) demonstrated:

> *"A 1.3B parameter InstructGPT model produces outputs preferred over the 175B GPT-3 by human evaluators, despite having 100x fewer parameters."*

Alignment, not scale, is the key to practical usefulness.

---

## Real-Life Analogy: Training a Dog 🐕

Pretraining a language model is like raising a very smart, wild dog. It's incredibly capable — it can learn almost anything. But without training, it doesn't know the rules.

RLHF is like dog training:

1. **SFT (show it what you want):** You demonstrate sit, stay, fetch. The dog imitates.
2. **Reward model (teach it what earns a treat):** The dog learns that sitting on command = treat. Jumping on visitors = no treat.
3. **RL training (practice with feedback):** The dog tries behaviours. It gets treats for good ones and learns to do more of those. The KL penalty is the leash — it stops the dog from inventing bizarre new behaviours just to get treats.

---

## The Three Phases of RLHF

Described in detail in the InstructGPT paper and the HuggingFace RLHF blog:

### Phase 1: Supervised Fine-Tuning (SFT) ✍️

- Collect thousands of (prompt, ideal response) pairs written by human annotators
- Fine-tune the pretrained base model on these examples
- Result: a model that knows *how* to respond helpfully — but limited by the quality of examples
- The SFT model becomes the **starting point** for everything that follows

### Phase 2: Reward Model Training 🏆

- Take the SFT model and generate multiple responses for many prompts
- Have human annotators **rank** these responses (not score — rank)
- Train a separate neural network to predict these rankings: the **reward model**
- The reward model can now score any (prompt, response) pair without human involvement
- This reward model becomes the **automated judge** for Phase 3

### Phase 3: RL Fine-Tuning with PPO 🔄

- Start with the SFT model (the RL policy)
- Keep a frozen copy (the reference policy)
- For each training step:
  1. Sample a batch of prompts
  2. The RL policy generates responses
  3. The reward model scores each response
  4. Compute total reward = reward score − KL penalty
  5. Use PPO to update the RL policy toward higher reward
- Repeat until the model consistently generates high-quality responses

---

## The Full Loop — Step by Step 🔁

```
Frozen reference policy ──────────────────────────────────┐
                                                            │ (for KL computation)
Prompt batch                                               │
    ↓                                                       │
RL Policy generates response                               │
    ↓                                                       │
Reward model scores response → reward score r              │
    ↓                                                       │
KL penalty = KL(RL policy ‖ Reference policy) ─────────────┘
    ↓
Total reward = r − λ × KL
    ↓
PPO update → improve RL policy to maximise total reward
    ↓
Back to top (next batch of prompts)
```

As described in the HuggingFace implementation guide ([huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)):

> *"The complete reward signal: r = r_θ − λ × r_KL, where r_θ is the preference model output and r_KL is the KL divergence penalty."*

---

## Why PPO? The Algorithm Behind RLHF

**PPO (Proximal Policy Optimisation)** is the RL algorithm used in the original InstructGPT and most subsequent RLHF implementations.

The core idea (from the OpenAI Spinning Up guide — [spinningup.openai.com](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)): update the policy to increase the probability of high-reward actions, but don't make updates so large that the policy becomes unstable.

PPO enforces this with a **clipping mechanism**: if the new policy would assign a very different probability to an action than the old policy, the update is clipped (limited). This keeps training stable.

**Why PPO specifically?**
The HuggingFace RLHF blog notes:
> *"Relative maturity with established guides and implementations. Trust region method provides stability. Suitable for scaling to very large models."*

---

## RLHF in the Context of RL Theory

Mapping RLHF onto standard RL vocabulary (OpenAI Spinning Up framework):

| RL Concept | RLHF Equivalent |
|---|---|
| Agent | The language model (RL policy) |
| Environment | The text generation process |
| State | The current prompt + tokens generated so far |
| Action | Choosing the next token (from vocab of ~32k) |
| Policy | The LLM's token probability distribution |
| Reward | Reward model score (at end of response) |
| Episode | One complete prompt → response generation |

The action space is enormous (~32,000 tokens) and the reward is sparse (only given at the end of a full response). This makes RLHF a challenging RL problem — and is part of why newer algorithms like GRPO were developed.

---

## Key Results from InstructGPT

The original paper ([arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)) found that RLHF training:

- Made a **1.3B model preferred over a 175B base model** by human evaluators
- Reduced toxic outputs and hallucinations
- Improved instruction-following across diverse tasks
- Maintained performance on standard NLP benchmarks (with some mitigation techniques)

The paper coined this gap between "raw capability" and "aligned behaviour" as the **alignment tax** — and showed RLHF largely eliminates it.

---

## The Limitations of RLHF

**Expensive:** The InstructGPT reward model required ~33,000 human comparison labels. Scaling this to broader domains is costly.

**Reward hacking:** The model can learn to game the reward model (as covered in the KL divergence chapter).

**Instability:** PPO training with a language model requires careful hyperparameter tuning. The implementation details paper ([huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)) lists dozens of subtle gotchas.

**Memory:** Maintaining 4 models simultaneously (policy, reference policy, reward model, value model) requires significant GPU memory.

These limitations motivated the second-generation algorithms: GRPO, RLOO, DAPO, and others — which form the core of the RL for LLMs article you want to study next.

---

## The Big Picture — How All Pieces Connect

```
Raw internet data
    ↓ Pretraining (next-token prediction)
Base LLM (knows a lot, doesn't know how to behave)
    ↓ SFT (imitate human demonstrations)
SFT Model (knows how to behave, limited by demo quality)
    ↓ Reward Model Training (learn to judge responses)
Reward Model (can score any response automatically)
    ↓ RL Fine-Tuning (PPO + KL penalty)
RLHF-trained LLM (helpful, harmless, honest)
```

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| RLHF | Three-phase training: SFT → reward model → RL fine-tuning |
| Phase 1 (SFT) | Imitate human demonstrations to learn helpful behaviour |
| Phase 2 (Reward Model) | Train a network to score response quality from human rankings |
| Phase 3 (RL + PPO) | Use the reward model to train beyond the quality ceiling of demos |
| Reference policy | Frozen SFT model used for KL penalty — prevents reward hacking |
| Total reward | Reward model score minus KL penalty |
| Key result | A 1.3B aligned model outperforms 175B base model |

---

## 🎓 Bridge Complete!

You now have all the foundation needed for the RL for LLMs series:

| # | Topic | Status |
|---|---|---|
| 17 | Training Loop & Loss Functions | ✅ |
| 18 | Supervised Fine-Tuning (SFT) | ✅ |
| 19 | Reward Models | ✅ |
| 20 | KL Divergence | ✅ |
| 21 | RLHF Overview | ✅ |

**Up next:** The RL for LLMs series — REINFORCE, PPO, GRPO, and the modern algorithms behind DeepSeek-R1.
