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

## The Alignment Problem — Why Base Models Aren't Enough 🚨

Before diving into RLHF's solution, let's understand the problem it solves.

### What a Base Model Does

A **base language model** is trained on next-token prediction over trillions of tokens of internet text. It becomes extraordinarily capable at predicting what text comes next in any context. But "what comes next on the internet" is not the same as "what's helpful, accurate, and safe."

### The Failure Modes of Base Models

**Failure 1: Continuation of harmful content**

Prompt: "Write a step-by-step guide to making chlorine gas at home"

Base model response: *happily continues the prompt as if it were any other text completion task, providing dangerous instructions*

Why? Because the internet contains such instructions, so the model has learned that this text pattern gets completed with actual instructions.

**Failure 2: Sycophantic hallucination**

Prompt: "I heard Einstein failed maths at school. Tell me more about this."

Base model response: *generates plausible-sounding elaborations on a false premise*

Why? The model is completing text, not verifying facts. If the prompt asserts something, the model assumes it's true and elaborates.

**Failure 3: Refusing to give opinions when asked**

Prompt: "What do you think about climate change?"

Base model response: *might generate a news article format, or an academic essay, rather than a first-person opinion*

Why? The model doesn't have a stable identity or preferences — it's mimicking whatever text pattern fits the context.

**Failure 4: Breaking character mid-conversation**

Even if the model starts as a helpful assistant, it might drift into generating a fictional story, a Wikipedia article, or debate-club arguments partway through a conversation.

### The Alignment Gap

The gap between "capable" and "aligned" is the **alignment problem** in miniature. A model can be extraordinarily capable (knows vast amounts about chemistry, history, science) but misaligned (uses that knowledge unhelpfully, dangerously, or inconsistently).

RLHF's answer: don't just make the model capable — explicitly train it to be **helpful, harmless, and honest**. Teach it the rules of the game, not just the knowledge of the domain.

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

## PPO — Proximal Policy Optimization 🔧

PPO is the algorithm that drives Phase 3 of RLHF. Understanding it deeply explains why RLHF training is so stable (when it works) and so fragile (when it doesn't).

### The Core Problem PPO Solves

In standard policy gradient RL, you update the policy in the direction that increases expected reward. The problem: if you take too large a step, the new policy might be **dramatically different** from the old one. The old policy collected the data, and the new policy might behave so differently that the data is misleading — a vicious cycle.

**Analogy:** Imagine steering a speedboat. You turn the wheel slightly based on where you think the rocks are. If you turn too hard based on incomplete information, you might aim directly at the rocks instead.

### The Policy Ratio

PPO works by computing the ratio between the new and old policy probabilities:

```
r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
```

If r_t = 1.0 → the new policy assigns the same probability as the old policy (no change)
If r_t = 1.5 → the new policy is 50% more likely to take this action
If r_t = 0.5 → the new policy is 50% less likely to take this action

### The Advantage Function

PPO uses the **advantage estimate** A_t, which measures: "how much better was this action than the average action I could have taken?"

```
A_t = G_t - V(s_t)
```

Where G_t is the discounted return (total future reward) and V(s_t) is the value function estimate (expected future reward from this state).

Positive advantage → this action was better than average → increase its probability
Negative advantage → this action was worse than average → decrease its probability

### The PPO Objective

The PPO clipped objective is:

```
L_CLIP = E_t[min(r_t(θ) × A_t, clip(r_t(θ), 1-ε, 1+ε) × A_t)]
```

Translation: multiply the advantage by the policy ratio, but clip the ratio to [1-ε, 1+ε]. Take the minimum of the clipped and unclipped versions.

This sounds complex — the key section below explains the clipping intuitively.

---

## The PPO Clipping Trick 🔒

The clipping is the secret sauce that makes PPO work. Let's understand it through examples.

### Without Clipping: Unstable Training

Suppose we have an action with a large positive advantage (it led to very high reward). Without clipping, gradient descent would push r_t(θ) as large as possible — we'd keep increasing the probability of this action without limit.

Problem: the data that told us this action is good came from the **old policy**. If the new policy is now much more likely to take this action, the data is no longer representative. We might be making the wrong update.

### With Clipping: Stable Training

With clipping at ε = 0.2:
- r_t(θ) is constrained to [0.8, 1.2]
- If the optimal update would require r_t = 2.0 (double the probability), we only go to r_t = 1.2
- We take a conservative step — we don't make the update so large that our data becomes unreliable

### The Two Cases

**Case 1: Positive advantage (action was good)**
- We want to increase probability: push r_t above 1.0
- Clipping at 1+ε = 1.2 prevents us from going too far
- We take a bounded step toward the good action

**Case 2: Negative advantage (action was bad)**
- We want to decrease probability: push r_t below 1.0
- Clipping at 1-ε = 0.8 prevents us from going too far down
- We take a bounded step away from the bad action

### Why This Matters for LLMs

LLMs have an enormous action space (32,000+ tokens per step). Without clipping, early training steps could dramatically change token probabilities based on a small batch of examples. The clipping ensures that each training step makes modest, stable updates — the policy changes by at most ~20% in any direction per step.

---

## RLHF in Practice — What It Looks Like at Scale 🏭

The theory of RLHF is cleaner than the practice. Here's what actually happens at scale.

### Memory Requirements: The Four Models Problem

During Phase 3 RL training, you need to keep four models in memory simultaneously:

1. **RL policy** (being trained) — largest, requires full gradients
2. **Reference policy** (frozen SFT copy) — same size, no gradients needed
3. **Reward model** — typically smaller, outputs a scalar
4. **Value model** (for PPO's advantage estimation) — typically same size as policy

For a 7B parameter model, this can require 4 × 7B × 4 bytes = ~112GB just for model weights, before accounting for gradients and activations. This requires multi-GPU setups and careful memory management.

### The Data Pipeline

Generating training data for RL involves:
1. Sample a batch of prompts from a prompt dataset (10,000–50,000 diverse prompts)
2. Run the RL policy to generate responses (temperature = 1.0 for diversity)
3. Score each response with the reward model
4. Compute per-token KL penalties against the reference policy
5. Compute advantages using the value model
6. Run PPO gradient updates

This pipeline must be orchestrated across many GPUs. The HuggingFace implementation guide has 37 specific implementation details — including things like how to handle variable-length sequences, how to clip rewards before PPO, and how to prevent value model collapse.

### Prompt Dataset Curation

The diversity and quality of the prompt dataset matters enormously. If you only train on simple Q&A prompts, the model improves at Q&A but not at coding, creative writing, or reasoning. InstructGPT used:
- Real user prompts from the OpenAI API (with permission)
- Augmented prompts from human annotators covering diverse tasks
- A balanced mix of question answering, summarisation, instruction following, creative writing, and more

### Training Stability

RLHF is notoriously unstable. Common failure modes:
- **Reward collapse:** The model finds a reward model exploit and reward scores spike while quality crashes
- **KL explosion:** The policy diverges too fast from the reference, causing incoherent outputs
- **Value model instability:** The advantage estimates become noisy, leading to erratic gradient updates
- **Repetition loops:** The model generates repetitive text that somehow scores well

The implementation details paper lists techniques to address each of these — but it remains an active engineering challenge.

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

## DPO — Direct Preference Optimization 🎯

DPO (Rafailov et al., 2023) is the most widely adopted alternative to RLHF. Instead of three stages, it achieves similar results in two.

### The Key Insight

The RLHF objective — maximise reward while constraining KL divergence from the reference — can be solved analytically. The optimal policy has a closed-form solution that doesn't require a separate reward model.

DPO re-parameterises the reward model in terms of the policy itself, then derives a loss function that can be directly applied to preference data without:
- Training a separate reward model
- Running PPO or any RL algorithm
- Maintaining a separate value model

### The DPO Training Process

1. Start with an SFT model (Phase 1 still needed)
2. Collect preference data: (prompt, winner, loser) pairs — same as reward model training
3. Directly fine-tune the SFT model on these pairs using the DPO loss
4. Done — no Phase 3 needed

The DPO loss implicitly implements a KL-constrained reward maximisation. The β hyperparameter controls the same trade-off as in RLHF.

### DPO vs RLHF in Practice

| Dimension | RLHF | DPO |
|---|---|---|
| Stages | 3 | 2 |
| RL algorithm | PPO (complex) | None |
| Reward model | Required | Not needed |
| Stability | Tricky | Much simpler |
| Memory | 4 models | 2 models |
| Result quality | Strong | Comparable |
| Popular uses | GPT-4, LLaMA 2 Chat | Mistral, Zephyr, many open models |

DPO has become the default for open-source RLHF because of its simplicity and stability. However, some practitioners argue that explicit RL with PPO achieves higher peak performance when properly tuned.

---

## Constitutional AI — Self-Critique and Self-Improvement 📜

Anthropic's Constitutional AI (CAI) approach, introduced in their 2022 paper, represents a different philosophy: instead of asking humans to judge individual responses, write down a **constitution** of principles and have the AI judge itself.

### The CAI Process in Detail

**Step 1 — Supervised Learning from Human Feedback (conventional SFT):**
Train a helpful-only SFT model, without harmlessness training. This model may comply with harmful requests.

**Step 2 — Critique and Revision:**
For harmful or problematic responses generated by the helpful-only model:
- Ask the model: "Identify specific ways in which the response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal."
- Ask the model: "Please rewrite the response to remove any and all harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
- Repeat this critique-revision loop 1–4 times per response

**Step 3 — Supervised Learning on Revised Responses:**
Fine-tune the model on the revised responses. Now you have a model that's both helpful and harmless — without explicit harm labels from humans.

**Step 4 — RL from AI Feedback (RLAIF):**
Use the constitutional principles to generate AI preference labels: "Which of these two responses better follows the principle of [X]?" Train a reward model on these AI-generated comparisons. Then run RL training.

### The Constitutional Principles

Anthropic's constitution includes principles like:
- "Choose the response that is least likely to contain harmful or unethical content"
- "Choose the response that a thoughtful, senior Anthropic employee would consider optimal"
- "Choose the response that is most helpful while also being honest and avoiding harm"

### Why CAI Matters

1. **Scalability:** AI can generate millions of critiques cheaply vs thousands of human labels expensively
2. **Transparency:** The principles are explicit — users can read and critique Anthropic's values
3. **Consistency:** AI applies principles uniformly; human annotators have off days
4. **Safety for annotators:** Human labellers don't need to read graphic harmful content to label it as harmful

---

## RLAIF — Using AI Feedback Instead of Human Feedback 🤖

RLAIF (RL from AI Feedback) generalises the idea: instead of human annotators providing preference labels, use a capable AI (often a larger, more capable model) to provide the labels.

### How RLAIF Works

1. Generate response pairs with the SFT model
2. Feed each pair to a capable AI judge (e.g., GPT-4 or Claude) with a rating prompt
3. The AI judge produces preference labels: "Response A is better because..."
4. Train a reward model on these AI-generated labels
5. Run RL training as normal

### Comparing Human vs AI Feedback

| Dimension | Human Feedback | AI Feedback (RLAIF) |
|---|---|---|
| Cost | High ($1–5 per label) | Low (API call cost) |
| Scale | Thousands of labels | Millions of labels |
| Consistency | Variable (human fatigue) | High consistency |
| Subtle quality | Strong (humans notice nuance) | Weaker on subtle dimensions |
| Bias | Human cultural/personal bias | AI training data bias |
| Coverage | Good on in-distribution | May miss edge cases |

### The "AI Feedback is the AI's Own Preferences" Problem

A subtle issue: if you train GPT-4 using GPT-4's preferences, you're optimising toward GPT-4's definition of quality. The resulting model becomes more similar to GPT-4 rather than genuinely better. RLAIF can lead to **preference homogenisation** — all models converging toward the preferences of whichever AI judge was used.

---

## The InstructGPT Results in Detail 📊

The InstructGPT paper is one of the most important in recent NLP history. Let's go deeper than the headline number.

### Human Preference Results

The core evaluation: hired human evaluators rated responses from different models on a 1–7 quality scale and recorded which they preferred.

**Key numbers:**
- InstructGPT 1.3B was preferred over GPT-3 175B on 85% of prompts
- On the API test prompt distribution, InstructGPT produced "less false information" in 21% fewer prompts
- On the "generates safe" metric, InstructGPT was safe 98.7% of the time vs GPT-3's 95.5%

### Task-Specific Breakdown

InstructGPT was evaluated across multiple task types:

| Task | GPT-3 quality (1-7 scale) | InstructGPT quality |
|---|---|---|
| Generation (creative writing) | 4.1 | 5.8 |
| Open QA | 3.9 | 5.4 |
| Brainstorming | 4.3 | 5.6 |
| Summarisation | 4.6 | 5.7 |
| Closed QA | 4.0 | 5.2 |
| Extraction | 4.2 | 5.5 |

The improvements are consistent across every task type — this isn't cherry-picked on specific domains where RLHF helps most.

### The Alignment Tax

One concern: would RLHF training hurt performance on standard NLP benchmarks (the capabilities developed during pretraining)?

The paper found a small performance decrease on some benchmarks, particularly publicly available NLP benchmarks — the model appeared to partially "forget" some capabilities during RL training. This was mitigated by mixing some pretraining data into the RL fine-tuning phase.

The takeaway: RLHF does have a small alignment tax on raw capability benchmarks, but this is far outweighed by the improvements in helpfulness, safety, and instruction following as measured by humans.

### The Data Scale

InstructGPT used:
- ~13,000 training prompts (prompt + demonstration pairs for SFT)
- ~33,000 human comparison labels for the reward model
- ~31,000 prompts for PPO training (no human labels — these are scored by the reward model)

This is surprisingly small. The lesson: the quality and diversity of alignment data matters far more than its quantity. The annotation effort was focused on covering a wide variety of tasks, not just generating as many examples as possible.

---

## Modern Alignment — Beyond InstructGPT 🚀

RLHF as described in InstructGPT is now over 3 years old. The field has evolved rapidly.

### The Key Developments

**DPO (2023):** Simplified the three-stage pipeline into two stages. Now the most common approach for open-source models.

**Constitutional AI / RLAIF (2022–2024):** Reduced dependence on expensive human comparison labels. Anthropic's Claude models use CAI; Google's models use a variant of RLAIF.

**Process Reward Models (2023–2024):** Step-level reward signals instead of outcome-only. OpenAI's o1 and o3 models appear to use PRMs extensively for reasoning tasks.

**GRPO, DAPO, etc. (2024–2025):** Replaced PPO with simpler group-based policy optimisation. Used in DeepSeek-R1 and many successors. Simpler, more memory-efficient, and often more stable.

**Verifiable Reward Signals (2024–2025):** For maths and code, skip the reward model entirely — use execution results or formal verification as the reward signal. This sidesteps reward hacking completely.

### The Convergence of RLHF Pipelines

Despite the variety of techniques, every modern alignment pipeline shares these core ideas:
1. Start from a pretrained base model
2. Do some form of supervised fine-tuning on demonstrations
3. Apply some form of preference learning (RLHF, DPO, CAI) to push toward aligned behaviour
4. Use KL divergence or equivalent constraints to prevent reward hacking

The specific algorithms change; the structure remains the same.

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

## Common Misconceptions ❌

### Misconception 1: "RLHF makes models smarter"

**Reality:** RLHF primarily makes models more **aligned** — better at following instructions, safer, more consistent. It doesn't meaningfully increase their underlying knowledge or reasoning capability. Those come from pretraining and scale. What RLHF does is take existing capability and direct it usefully.

### Misconception 2: "The reward model knows what's best"

**Reality:** The reward model knows what the annotators in the training set preferred. If those annotators had biases, lacked expertise, or made errors, those are baked in. InstructGPT's annotators were English-speaking contractors — the model's "preferences" may not reflect global values.

### Misconception 3: "RLHF training runs for a very long time"

**Reality:** The RL phase is typically quite short — a few thousand to tens of thousands of steps. The heavy lifting is in pretraining and SFT. RL training is more like fine-tuning the alignment, not rebuilding the model.

### Misconception 4: "PPO is the only way to do Phase 3"

**Reality:** PPO was the first widely used algorithm, but DPO, GRPO, RLOO, DAPO, and many others have since been developed. For most open-source models today, DPO or a DPO variant is used rather than PPO.

### Misconception 5: "RLHF solves alignment"

**Reality:** RLHF makes models much more useful and significantly safer — but it's not a complete solution to AI alignment. Models trained with RLHF can still hallucinate, be sycophantic, have biases, be manipulated through adversarial prompting, and fail in unexpected ways. RLHF is a major step forward, not the final destination.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| RLHF | Three-phase training: SFT → reward model → RL fine-tuning |
| Alignment problem | Base models know a lot but behave unpredictably without alignment training |
| Phase 1 (SFT) | Imitate human demonstrations to learn helpful behaviour |
| Phase 2 (Reward Model) | Train a network to score response quality from human rankings |
| Phase 3 (RL + PPO) | Use the reward model to train beyond the quality ceiling of demos |
| PPO clipping | Bounds policy updates to [1-ε, 1+ε] — prevents unstable giant steps |
| Reference policy | Frozen SFT model used for KL penalty — prevents reward hacking |
| Total reward | Reward model score minus KL penalty |
| DPO | Direct Preference Optimization — achieves RLHF in 2 stages, no RL needed |
| CAI | Constitutional AI — use explicit principles and AI critique instead of human labels |
| RLAIF | Use a capable AI (like GPT-4) to generate preference labels instead of humans |
| Key result | A 1.3B aligned model outperforms 175B base model |
| Alignment tax | Small performance cost on benchmarks, massively outweighed by human quality gains |

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
