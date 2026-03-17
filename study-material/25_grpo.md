# GRPO — Group Relative Policy Optimization

> **Sources used:**
> - Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, DeepSeek AI 2024 — [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
> - DeepSeek AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, 2025 — [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
> - Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*, 2024 — [arxiv.org/abs/2402.14740](https://arxiv.org/abs/2402.14740)
> - Schulman et al., *Proximal Policy Optimization Algorithms*, OpenAI 2017 — [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
> - Liu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, 2025 — [arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)

---

## The Big Idea

GRPO stands for Group Relative Policy Optimization. It is a reinforcement learning algorithm designed specifically for training large language models, and it addresses one of the most expensive and technically difficult parts of standard PPO-based RLHF: the critic network, also called the value model.

In standard PPO, estimating the advantage of a response — that is, figuring out whether a response was better or worse than average — requires a separate neural network called the critic or value function. This network has roughly the same number of parameters as the policy model itself. During RLHF training, you therefore need to maintain two enormous models simultaneously: the policy and the critic. Both require memory, both require gradient computation, and the critic is notoriously difficult to keep stable.

GRPO eliminates the critic entirely. Instead of asking a separate network to estimate how good a response is relative to some abstract baseline, GRPO asks a much simpler question: for a given prompt, if you generate a group of responses and score each one, which responses in the group were better than average? The group itself provides the baseline. No separate network is needed. The average reward within the group takes the place of the value function.

This is a conceptually simple but practically powerful change. GRPO was introduced in the DeepSeekMath paper in February 2024 and later used extensively in DeepSeek-R1, one of the most capable open reasoning models released as of early 2025. The results showed that GRPO can match or exceed PPO in training quality while requiring significantly less memory and being substantially easier to implement stably.

The core mechanism, stated plainly: for each prompt in a training batch, sample G responses from the current policy (typically G=8 to G=16). Score all G responses with a reward signal. Normalize the rewards within the group by subtracting the group mean and dividing by the group standard deviation. These normalized scores become the advantages. Apply PPO-style clipped policy gradient updates using these advantages. Repeat for thousands of training steps.

Everything else in GRPO is familiar from PPO: the clipping mechanism, the KL divergence penalty against the reference policy, the policy ratio. The only genuine departure is how the advantage is computed — and that one change removes an entire model from the training stack.

The full training loop at a glance:

```
For each training step:
  Sample N prompts from the prompt dataset
  For each prompt:
    Generate G responses (e.g. G=8) via temperature sampling
    Score each response with reward signal
    Compute group mean and group std of rewards
    Compute advantage_i = (reward_i - mean) / std   for each response i
  Apply PPO clipped gradient update using group advantages
  Penalize KL divergence from reference policy
```

---

## Real-Life Analogy

Imagine a university professor grading a midterm exam. There are two philosophically different approaches to grading.

The first approach uses an absolute standard. The professor decides in advance what a perfect answer looks like and deducts points from that ideal. A student who gets 72 out of 100 has failed relative to the imagined ideal, regardless of how everyone else performed. The imagined ideal is the critic: a fixed, learned estimate of what perfect performance looks like. The professor is playing the role of the value function — using prior experience to estimate what a good answer is worth in absolute terms.

The second approach is grading on a curve. The professor collects all the exams, calculates the class average, and grades each student relative to their classmates. If the class average was 68 and you scored 80, you are twelve points above average — a strong positive advantage. If you scored 55, you are thirteen points below average — a negative advantage. Your grade is not compared against some external ideal; it is compared against the actual performance of people who sat the same exam under the same conditions on the same day.

GRPO is grading on a curve for language model responses. For each prompt, the model generates a group of responses — think of these as a cohort of students all answering the same question simultaneously. Each response gets a reward score. The group average becomes the baseline. Individual advantages are computed as deviations from that group average. No separately trained network is needed to estimate the ideal, because the group itself reveals what reasonable performance looks like for this particular prompt under this particular model's current capabilities.

The analogy extends further in a useful direction. When a professor grades on a curve, grades automatically adapt to question difficulty. A notoriously hard question might produce a class average of 45. A student who scored 60 on that question did very well, even though 60 would be a poor score on an easy question. An easy question might produce a class average of 85 — that same student's 60 would represent weak performance. The baseline adapts automatically to the difficulty of each question. GRPO's group mean does the same: it adapts automatically to each prompt's difficulty without any additional mechanism.

This adaptation property is particularly valuable when training on a mixed dataset of prompts with varying difficulty. In PPO, the critic's baseline is a global learned function that may not perfectly adapt to the local difficulty of each prompt. In GRPO, the baseline is always the local group mean for that specific prompt, which is by definition perfectly calibrated to the current difficulty level.

The analogy has one limit worth noting: a professor grading on a curve knows the absolute quality of each student's response even if they score relative to peers. The reward signal in GRPO provides this absolute quality signal — it is the reward model or verifiable correctness checker. The group normalization only affects how the absolute reward signal is translated into a gradient weight, not how quality is assessed.

---

## The Problem with PPO's Critic — Deep Dive

To understand why GRPO matters, you need to understand what it replaces: the critic network in PPO-based RLHF, and why training that critic is both expensive and destabilizing.

### What the Critic Is Supposed to Do

In reinforcement learning, the advantage function measures how much better a particular action was compared to the expected outcome from that state. For language model training, an action is choosing the next token, and the state is the prompt plus all tokens generated so far. The advantage is therefore: how much better was this specific token choice than the average token choice we could have made at this point in the generation?

PPO computes advantages using the critic. The critic is a neural network that takes the current state — the full context of tokens seen so far — and predicts the expected total future reward from this state. This expected reward estimate is called the value function or V(state). The advantage for a token is: actual reward received minus the value function's prediction at that token position. Positive advantage means this token choice led to a better outcome than the critic expected; negative advantage means it led to a worse outcome.

The critic therefore needs to be a good approximator of expected reward across the entire policy's state space. That state space is enormous: for a language model with a context window of 2,048 tokens and a vocabulary of 32,000, the number of possible states is effectively infinite. The critic must somehow generalize across this space, predicting expected reward for token sequences it has never seen.

### Why Training the Critic Is Expensive

The critic in PPO-based RLHF is typically initialized from the same pretrained model as the policy. The reason: the policy's representations of text — learned through pretraining on hundreds of billions of tokens — are already useful for predicting response quality. Starting the critic from a random initialization would waste the pretraining investment and require far more data to train.

This initialization choice means the critic has roughly the same number of parameters as the policy model. For a 7 billion parameter model, the critic is also approximately 7 billion parameters. For a 70 billion parameter model, the critic is 70 billion parameters.

The memory implications compound. During training, you need the following simultaneously in GPU memory:

```
PPO training stack:
  RL Policy        — ~7B params, full gradient states
  Reference Policy — ~7B params, frozen (no gradients)
  Reward Model     — ~3B params (often smaller model)
  Critic           — ~7B params, full gradient states
```

For 16-bit precision, each parameter requires 2 bytes for the weight. The 7B policy alone requires 14 GB just for weights. Add Adam optimizer states (first and second moment estimates, which double the memory again), activations during the forward pass, and gradient tensors, and a single 7B PPO training run commonly exceeds 200 GB of GPU memory. At 2025 hardware prices, this requires either 4-8 high-end GPUs or a purpose-built multi-node cluster.

The Huang et al. implementation guide, known informally as the "N+ Implementation Details" paper, documents these memory requirements in detail and describes the engineering work required to make them manageable in practice.

### Why Training the Critic Is Unstable

Memory cost is only half the problem. The deeper issue is that the critic is fundamentally difficult to train stably alongside the policy.

The critic tries to predict the expected reward from any given state. But as the policy changes — because PPO is updating it at every training step — the distribution of states the critic encounters also changes. At step 0, the policy generates responses of type A; the critic learns to predict rewards for type A responses. By step 100, the policy has shifted toward responses of type B; the critic's predictions for type B are now based on extrapolation from type A, which may be quite inaccurate.

This is the moving target problem. The critic is perpetually chasing a distribution that shifts under its feet. The feedback loop is destabilizing: stale critic predictions produce incorrect advantage estimates, which lead to bad policy gradient updates, which cause the policy to change in unexpected ways, which makes the critic's predictions even more stale.

The implementation details required to manage this instability include:

- **Value function clipping**: limit how much the critic's predictions can change in a single step, analogous to the policy ratio clipping
- **Separate learning rates**: the critic often needs a smaller or larger learning rate than the policy, tuned separately
- **Reward normalization and clipping**: normalize rewards before passing them to the critic to prevent large-scale shifts in the critic's target values
- **Whitening of advantages**: an additional normalization step applied after the critic produces advantage estimates
- **Value loss coefficient**: tune the relative weight of the critic loss versus the policy loss in the combined objective

Each of these is a hyperparameter that requires tuning. The Huang et al. paper counts over three dozen such details. Many practitioners have observed that RLHF with PPO requires weeks of debugging to stabilize on a new task or model, largely because of critic-related issues.

### The Four-Model Problem in Context

The four-model memory requirement is sometimes described as a "VRAM wall" — a hardware barrier that prevented many researchers and organisations from running PPO training on large models. A single 70B parameter policy requires approximately 140 GB of VRAM just for weights at 16-bit precision, before accounting for gradients and optimizer states. Adding a 70B critic doubles this to 280 GB of weight memory alone, equivalent to 14 or more A100 80 GB GPUs just for model weights.

This is why PPO-based RLHF was primarily the domain of well-resourced labs until 2024. GRPO's removal of the critic brought the hardware requirements down substantially — not to the level of fine-tuning, but closer — and opened RL training to a broader range of practitioners.

### The Core Problem Statement

To summarize the critic problem in a single paragraph: the critic provides a useful service (variance reduction in advantage estimation) but at enormous cost (equal model memory, equal computational cost, and introduction of a second moving-target optimization problem that can destabilize training). GRPO's central contribution is demonstrating that for language model RL training, the useful service can be provided by a much cheaper mechanism — the group average — without meaningful loss in training quality.

---

## Group Sampling — Deep Dive

The heart of GRPO is group sampling: for each prompt in the training batch, rather than generating a single response, you generate a group of G responses from the current policy.

### How Group Sampling Works in Practice

For each prompt, the policy model performs G separate forward passes with temperature sampling, producing G distinct responses. Temperature is set above zero (typically between 0.6 and 1.0) to ensure diversity — if temperature were zero, all G responses would be identical and the group would provide no useful comparison.

In the DeepSeekMath paper, G is set to 8. In DeepSeek-R1, G ranges from 8 to 16 depending on the training phase. The responses in the group are generated independently — each response sees only the prompt, not the other responses. There is no cross-response communication during generation.

Each of the G responses is then scored by the reward signal. For DeepSeek-R1's mathematics and reasoning tasks, this reward signal is a verifiable correctness signal: the final answer is extracted from the response and compared to the known correct answer using symbolic or numeric comparison. For more general tasks, the reward signal comes from a reward model trained on human preferences, just as in standard RLHF.

The result of group sampling for one prompt is a collection of G (response, reward) pairs. This collection is the raw material from which advantage estimates are computed.

### The Data Flow

The data transformation looks like this:

```
Input:    one prompt P

Step 1 — Generate:
          R1 = policy(P, temperature=0.8)   reward_1 = score(P, R1)
          R2 = policy(P, temperature=0.8)   reward_2 = score(P, R2)
          ...
          R8 = policy(P, temperature=0.8)   reward_8 = score(P, R8)

Step 2 — Compute group statistics:
          mean   = average(reward_1, ..., reward_8)
          std    = stdev(reward_1, ..., reward_8)

Step 3 — Compute advantages:
          advantage_i = (reward_i - mean) / std   for each i

Step 4 — Gradient update:
          For each token in each response, weight gradient by advantage
          Apply PPO clipping to policy ratio
```

### Why Group Size Matters

The choice of G involves a genuine tradeoff between variance and computation.

Larger G produces more accurate estimates of the within-group mean and standard deviation. If G=2, the group mean is the average of just two samples — extremely noisy, potentially far from the true expected reward. If G=16, the group mean is the average of sixteen samples — considerably more stable, closer to the true expected reward. The advantage estimates are therefore more reliable with larger G.

However, larger G also means more computation per prompt. For a large language model, each response generation requires a full autoregressive forward pass over potentially hundreds or thousands of tokens. If G=16 and you have a batch of 64 prompts, you are generating 1,024 complete responses per training step before computing a single gradient update.

The relationship between G and the expected gradient signal quality matters particularly when the reward signal is sparse. For a binary reward (correct or incorrect), a group of G=8 responses to a moderately hard prompt might contain 2 correct and 6 incorrect. The group mean would be 0.25, and the correct responses would each have a positive advantage while the incorrect ones have negative advantages — a clean, informative signal. If G=2 and both samples happened to be incorrect, the group mean would be 0.0 (or both correct, with the same effect), and the advantages would be near zero — no gradient signal despite there being a meaningful learning opportunity.

The DeepSeekMath paper found that G=8 is a reasonable balance for mathematical reasoning tasks: stable enough advantage estimates, manageable computational cost. DeepSeek-R1 used larger groups (up to G=16) in some training stages specifically because the reasoning chains were long and the reward signal was sparse — more samples per prompt reduced the probability of empty-signal batches.

### The Group as an Implicit Exploration Mechanism

Group sampling has a secondary benefit beyond providing a baseline: it explicitly encourages exploration. Because each response in the group is sampled independently with temperature, the responses naturally vary in their content and reasoning strategy. Some will succeed; others will fail. Some will use algebraic approaches; others will use estimation or decomposition.

This diversity is essential for the training signal to be informative. If all G responses are identical — which would happen at temperature zero — all G advantages would be zero after normalization. There would be no gradient signal because there is no variation to learn from. Temperature sampling ensures that responses differ, which ensures that some responses will score higher than others, which ensures that there is a meaningful gradient signal even without a critic.

The exploration encoded in group sampling is implicit: you never explicitly ask the model to explore different strategies. The temperature sampling does this automatically. This is one of the reasons GRPO works well for tasks with rich solution spaces, like mathematical reasoning, where many different solution paths exist for the same problem.

---

## Advantage Normalization — Deep Dive

Once you have the G reward scores for a group, GRPO computes advantages through a two-step normalization process: subtract the group mean, then divide by the group standard deviation.

### Why Subtract the Mean

Subtracting the group mean centers the advantages around zero. Before subtraction, the rewards might all be positive — for example, ranging from 0.3 to 0.9 for a group where some responses are partially correct and some are fully correct. After subtraction, some responses have positive advantages (they scored above average) and some have negative advantages (they scored below average).

This centering is the key insight: the sign of the advantage tells the model which responses to reinforce and which to discourage, relative to the group's actual performance on this prompt. Without mean subtraction, all rewards might be positive, and the gradient update would push the model to increase the probability of all responses — even mediocre ones. That is not the right signal. You want to reinforce responses that were better than average and discourage responses that were worse than average.

Mean subtraction achieves this centering efficiently. It is essentially the same operation that PPO's critic performs — the critic estimates an expected reward and subtracts it from the actual reward — but computed from the current batch of G responses rather than from a separately trained neural network.

### Why Divide by the Standard Deviation

Dividing by the group standard deviation normalizes the scale of the advantages across different prompts. This is essential for training stability.

Consider two prompts in the same batch. Prompt A is easy: all eight responses are correct, with rewards ranging from 0.95 to 1.00. The standard deviation is approximately 0.02. After mean subtraction, the advantages range from roughly -0.03 to +0.02 — tiny values. Prompt B is hard: rewards range from 0.0 to 0.9, with a standard deviation of approximately 0.35. After mean subtraction, the advantages range from about -0.45 to +0.45 — much larger values.

If we stopped at mean subtraction, the gradient update from Prompt B would be roughly fifteen times larger in magnitude than the update from Prompt A. Prompt B would dominate training, regardless of whether it provides more useful signal than Prompt A. This scale mismatch would cause training to be sensitive to the mix of difficulty levels in each batch.

After dividing by the standard deviation, both prompts' advantages are in units of standard deviations within their group. An advantage of 1.0 means "this response was one standard deviation above the group mean for this prompt," whether the underlying rewards ranged from 0.95 to 1.00 or from 0.0 to 0.9. The gradient magnitudes from different prompts become comparable.

In statistical terms, GRPO is computing z-scores within each group. This is a well-understood normalization in statistics: z-scores tell you how many standard deviations above or below the mean an observation falls, regardless of the original scale of the measurements.

### A Worked Example

To make this concrete, consider a group of eight responses to a mathematical problem.

```
Rewards:  [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
          (5 correct, 3 incorrect — binary reward)

Group mean:  5/8 = 0.625
Group std:   approximately 0.484  (standard deviation of a 5/8 success rate)

Advantages after normalization:
  Correct responses:  (1.0 - 0.625) / 0.484 = +0.78
  Incorrect responses: (0.0 - 0.625) / 0.484 = -1.29
```

Every token in the five correct responses gets advantage = +0.78, which will increase those tokens' probability. Every token in the three incorrect responses gets advantage = -1.29, which will decrease those tokens' probability. The gradient update is a small positive push for tokens in correct responses and a larger negative push for tokens in incorrect responses — asymmetric, because the incorrect minority "needs" more correction than the correct majority "needs" reinforcement.

### A Note on the Z-Score Scale

After normalization, GRPO advantages are z-scores — they have mean approximately zero and standard deviation approximately one within each group. This scale has a specific implication for the interaction with PPO clipping. The clipping bound of epsilon=0.2 limits the policy ratio to [0.8, 1.2]. For the gradient to saturate the clip bound, the advantage times the step size must be large enough to move the ratio outside [0.8, 1.2]. With z-score advantages typically in the range of -2 to +2 and a standard learning rate, most training steps do not trigger the clip — the policy changes by less than 20% even before clipping is applied. The clip acts as a safety net rather than an active constraint during most of training.

This is different from what would happen with raw, unnormalized rewards. If reward values were in the range 0 to 100 (common in some game environments), a single high-reward response could saturate the clip bound on nearly every token. Normalization effectively makes the clipping regime stable and predictable.

### Edge Cases: All Correct or All Incorrect

A numerical edge case arises when all G responses have the same reward. If all eight responses are correct (all rewards = 1.0), the group mean is 1.0 and the group standard deviation is 0. Dividing by zero is undefined.

In practice, implementations handle this by adding a small epsilon to the denominator (for example, using max(std, 1e-8) or max(std, 0.001)) or by skipping the update for groups with zero variance. This case is relatively rare because temperature sampling introduces some variation, but it can occur on easy prompts at late training stages when the model is highly accurate. The DAPO paper proposes filtering out these zero-variance groups entirely (dynamic sampling), arguing that they carry no useful gradient signal and should simply be replaced with harder prompts.

### What Normalization Means for Training Dynamics

From a training dynamics perspective, advantage normalization has a regularizing effect. Prompts where the model performs consistently (all responses score similarly) produce small advantages and small gradient updates. Prompts where performance varies widely (some responses much better than others) produce larger advantages and larger gradient updates.

This is an implicit curriculum: the model spends its gradient budget on prompts where performance is inconsistent — where there is genuine uncertainty about the correct approach. Easy prompts (where the model always succeeds) and uniformly hard prompts (where the model always fails and all responses score similarly) contribute less to the gradient. The training naturally focuses on the frontier of the model's current capabilities.

This implicit curriculum is one of the reasons GRPO works well for reasoning tasks: as the model improves, easy problems stop providing gradient signal (all responses correct, zero variance) and the effective training data automatically shifts toward harder problems. The training self-adjusts to the model's current level without any explicit difficulty scheduling. This is a valuable emergent property that PPO's critic does not inherently provide — the critic's baseline does not zero out gradients on easy prompts in the same automatic way.

---

## Critic-Free Training — Deep Dive

Removing the critic is the defining architectural choice of GRPO, and it has cascading consequences for memory, stability, and implementation complexity.

### The Memory Savings in Concrete Terms

The standard PPO training stack for LLMs requires four models in memory simultaneously. With GRPO, the critic is eliminated. The training stack becomes three models:

```
PPO stack (4 models):               GRPO stack (3 models):
  RL Policy     — 7B params           RL Policy       — 7B params
  Reference     — 7B params           Reference       — 7B params
  Reward Model  — 3B params           Reward Model    — 3B params
  Critic        — 7B params           [removed]
  ─────────────────────────           ─────────────────────────
  Weight memory: ~24B params          Weight memory: ~17B params
```

For a 7B policy at 16-bit precision, the savings in raw weight memory are approximately 14 GB. But the real savings are larger. The Adam optimizer stores two additional tensors per parameter (first and second moment estimates), so eliminating the critic saves approximately 3x the raw weight memory in optimizer states as well. Total savings are in the range of 40 to 50 GB for a 7B training run — enough to use significantly fewer GPUs or to fit a larger policy at the same memory budget.

For the larger models used in DeepSeek-R1, these savings scale proportionally. Eliminating a 70B parameter critic saves roughly 420 GB of total training memory (weights plus optimizer states), which is the equivalent of approximately five high-end A100 GPUs.

### The Stability Benefits in Practice

The critic's instability comes from the moving target problem: the critic must learn to predict the policy's expected rewards, but the policy is changing with every gradient step. The faster the policy improves, the more stale the critic's predictions become, potentially creating a destabilizing feedback loop.

GRPO sidesteps this entirely. There is no critic to become stale. The group mean and standard deviation are computed fresh for each batch from the current policy's actual outputs. The baseline is always perfectly up-to-date because it is computed directly from current data, not estimated by a historical approximator.

This means one of the primary failure modes of PPO training — critic collapse, where the critic's predictions diverge enough to produce large incorrect advantages and destabilize the policy update — is simply absent in GRPO. Training monitoring becomes simpler: you track reward trends, KL divergence, and gradient norms, but not the additional diagnostic of critic prediction error.

### What Is Lost When the Critic Is Removed

Removing the critic is not purely advantageous. The critic provides two things that the group mean cannot: per-token advantage estimates and generalization across the full training history.

Per-token advantages allow PPO to assign different credit to different positions within a response. A long reasoning chain might have excellent setup in the first half and a critical error in the second half. Per-token advantages would assign positive credit to the tokens in the excellent setup and negative credit to the tokens at the point of the error. GRPO, using a single per-response advantage, assigns the same credit (negative, because the response was wrong) to all tokens, including the tokens in the excellent setup. This is a less precise training signal.

Generalization across training history means the critic, having seen many prompts over many training steps, can estimate expected reward for a new prompt based on similarity to past prompts. The group mean can only use the G samples from the current batch. For tasks where strong generalization would help (perhaps very diverse or unusual prompts), the critic might provide a better baseline.

Whether these limitations matter in practice is the empirical question. For the mathematical reasoning tasks where GRPO was tested in DeepSeekMath and DeepSeek-R1, they clearly did not matter — GRPO matched or exceeded PPO performance. For tasks requiring fine-grained stylistic credit assignment, the picture may be different.

---

## The Clipping Mechanism in GRPO — Deep Dive

GRPO uses the same clipping mechanism as PPO. This is deliberate: the clipping mechanism is independent of the critic. It addresses a different problem entirely — the instability of large policy updates.

### Why Clipping Is Still Needed Without a Critic

The intuition: even with perfectly normalized group advantages, a single training step could make catastrophically large updates to the policy if left unconstrained. If one response in the group was dramatically better than the others (advantage = +5.0), the gradient update would push very hard to increase the probability of every token in that response. This might be appropriate if the response genuinely represents a discovery that the policy should learn. But it might also be an artifact — a lucky response that won't generalize.

The clipping mechanism imposes a trust region: the policy can change, but not by more than epsilon (typically 0.2) in any single step. This forces the training to proceed cautiously, taking many small steps rather than one large one, which reduces the risk of overfitting to individual lucky samples.

### The Policy Ratio and What It Measures

The policy ratio r for a token is defined as: the probability the current (post-update) policy assigns to this token, divided by the probability the old (pre-update) policy assigned when the response was generated. This ratio measures how much the update has changed the model's behavior at this specific token position.

- r = 1.0: no change (the new and old policies assign the same probability)
- r = 1.5: the new policy is 50% more likely to pick this token
- r = 0.7: the new policy is 30% less likely to pick this token

The PPO clipping constrains this ratio to the interval [1 - epsilon, 1 + epsilon]. For epsilon = 0.2, this is [0.8, 1.2]. The objective uses the minimum of the clipped and unclipped gradients, which effectively means:

- If the gradient would push r above 1.2, stop at 1.2
- If the gradient would push r below 0.8, stop at 0.8

### The Two Cases: Positive and Negative Advantage

The clipping interacts with the advantage sign in two distinct ways:

**Positive advantage (response was above group average):**
The gradient pushes r above 1.0 — increase the probability of this token. Clipping at 1 + epsilon = 1.2 prevents the ratio from exceeding 1.2. We take a bounded step toward making this token more likely.

**Negative advantage (response was below group average):**
The gradient pushes r below 1.0 — decrease the probability of this token. Clipping at 1 - epsilon = 0.8 prevents the ratio from going below 0.8. We take a bounded step toward making this token less likely.

In both cases, the clipping prevents large changes based on limited sample evidence. After the update, additional rollouts will collect new data, and if the update was in the right direction, the new data will reinforce it.

### Clipping and Group Normalization Together

The combination of PPO clipping and group advantage normalization creates two layers of protection against training instability.

Group normalization ensures that advantages are on a consistent scale across prompts — no single prompt dominates training due to having unusually high-variance rewards. Clipping ensures that even high-advantage responses cannot push the policy ratio outside the trust region in a single step. Together, they produce updates that are:

1. Proportional to relative performance within the group (from normalization)
2. Bounded in maximum magnitude per step (from clipping)
3. Consistent in scale across different prompts in the batch (from standardization)

This is substantially more stable than REINFORCE, which has neither normalization nor clipping, and can produce huge variance in gradient magnitudes from batch to batch.

---

## DeepSeek and GRPO — Deep Dive

GRPO was introduced in the DeepSeekMath paper in February 2024 and subsequently became the central training algorithm for DeepSeek-R1, released in January 2025. These two papers provide the most detailed public documentation of how GRPO performs at scale.

### DeepSeekMath: The Origin

DeepSeekMath was focused on training language models for mathematical reasoning. Mathematical reasoning is an attractive domain for RL training because the reward signal is verifiable: either the final answer is correct, or it is not. There is no need for a subjective reward model. You can define the reward function directly from first principles.

The paper framed the problem this way: given a mathematical question with a known correct answer, score a model's response 1.0 if the extracted final answer matches the correct answer, and 0.0 otherwise. This binary reward is simple, objective, and impossible to game through stylistic manipulation — there is no way to write a confidently wrong answer that happens to get a high score.

The paper compared GRPO against four alternatives on the MATH benchmark: supervised fine-tuning alone, RFT (rejection sampling fine-tuning), DPO, PPO, and REINFORCE. GRPO achieved the best performance, with the key advantage over PPO being memory efficiency and training stability. The key advantage over REINFORCE was more stable gradients (from normalization and clipping) and faster convergence.

The hyperparameters established in DeepSeekMath became defaults for subsequent GRPO implementations: G=8 responses per prompt, epsilon=0.2 (clipping), beta=0.04 (KL penalty coefficient), temperature=0.8 for response generation, and learning rate in the range of 1e-5 to 1e-6.

### DeepSeek-R1: GRPO at 671B Scale

DeepSeek-R1 applied GRPO to a 671B parameter mixture-of-experts model — at the time of publication, one of the largest language models trained with RL that had full technical documentation available. This was a decisive demonstration of GRPO's scalability.

The training process proceeded in multiple stages:

**Stage 1 — Cold-start fine-tuning:** A small amount of chain-of-thought reasoning examples were used to fine-tune the base model, establishing a format where the model learns to reason in an extended internal monologue before giving a final answer. Without this stage, the model sometimes did not produce structured reasoning chains even when it would benefit from doing so.

**Stage 2 — GRPO reasoning training:** The primary RL training phase, using GRPO with verifiable rewards on mathematical and coding problems. Group sizes of G=8 to G=16. The model learned to extend its reasoning chains when uncertain, revisit its work when it detected errors, and produce structured outputs that were easy to evaluate.

**Stage 3 — Rejection sampling fine-tuning:** High-quality responses generated by the Stage 2 model were used to fine-tune further via supervised learning, reinforcing the reasoning patterns discovered during RL.

**Stage 4 — General alignment:** GRPO with both rule-based rewards (for math and code) and reward model scores (for general tasks) to extend helpful behavior beyond reasoning tasks.

### Emergent Chain-of-Thought

One of the most striking findings in DeepSeek-R1 is that extended chain-of-thought reasoning emerged from GRPO training without being explicitly programmed. The model learned that for hard problems, generating a longer internal monologue before committing to a final answer increased the probability of getting the reward. This is exactly what you would expect from a well-functioning RL system: the model discovers behaviors that correlate with reward and increases their probability.

This emergence is possible because GRPO's group-relative advantages capture the correlation between response quality and reasoning behavior across many examples. Responses with longer, more careful reasoning chains tended to score higher in the groups. After many training steps, the policy learned to produce extended reasoning when needed.

### Reward Design at Scale

The DeepSeek-R1 reward function had two components:

- **Accuracy reward**: 1.0 if the final answer is correct, 0.0 if wrong. For math, correctness is determined by numeric or symbolic comparison. For code, correctness is determined by running the code against test cases.
- **Format reward**: a small bonus (approximately 0.1 to 0.2) for following the expected output format — reasoning chain enclosed in one section, final answer in another. This prevented the model from learning to produce correct answers without the reasoning chain, which would degrade its ability to generalize to harder problems.

The format reward illustrates an important principle: reward shaping matters. Without the format reward, the model might learn to produce compact answers that score well on easy problems but fail on hard ones. The format reward encouraged the structural behaviors that the model would need to succeed on difficult tasks, even at a cost on simpler tasks where the reasoning chain is unnecessary.

---

## GRPO vs PPO Comparison — Deep Dive

GRPO and PPO share significant structural similarity — both use clipped policy gradients, both maintain a reference policy and apply a KL penalty, both update toward higher-reward responses. The differences are narrow but practically significant.

### Shared Components

Both algorithms share the following:

- The policy ratio r = p_new(token) / p_old(token) for each token
- The clipped objective: min(r * A, clip(r, 1-eps, 1+eps) * A)
- The KL penalty against the reference (SFT) policy
- The overall training loop: generate responses, score, update, repeat

In terms of code, the main difference is approximately twenty lines: replacing the critic forward pass and value loss with the group mean/std computation and advantage assignment.

### The Critic: Present vs Absent

The most consequential difference is the critic. PPO requires a critic — a neural network of similar scale to the policy — to estimate V(state) for each token position in each response. GRPO replaces this with group statistics computed directly from the batch.

The practical implications were described in the memory and stability sections above. From a conceptual perspective, the difference is between a learned, generalizing approximator (critic) and a local, data-driven estimator (group mean). Both serve as baselines for variance reduction in the policy gradient. The critic can in principle be better; the group mean is sufficient in practice for the tasks where GRPO has been tested.

### The Advantage Estimate: Granularity

PPO produces a per-token advantage estimate. The critic processes the full context at each token position, producing a V(state) at position t. The advantage at position t is: future reward from t minus V(t). This varies across positions within a single response — some tokens were better choices than others, and the critic tries to capture this.

GRPO produces a per-response advantage estimate. All tokens in a response share the same advantage score — the normalized group reward for that entire response. This is coarser. A response that had excellent reasoning in steps 1-8 but made an error in step 9 would get a uniformly low advantage (because the final answer is wrong), even though steps 1-8 were high quality. The model does not get credit for the good early steps.

Whether this matters is an empirical question. The evidence from DeepSeek suggests it does not matter much for mathematical reasoning with binary rewards, where the overall correctness of a response is a reasonable summary of its quality. For tasks with richer partial credit structures, the per-token advantage might be more informative.

### Implementation Complexity

One underappreciated dimension of the PPO vs GRPO comparison is implementation complexity. The Huang et al. paper lists 37 specific implementation details required to make PPO stable for LLM training. These include specific choices about reward normalization, value loss clipping, advantage whitening, sequence padding, and many more. Missing any of these can cause training instability that takes days of debugging to diagnose.

GRPO's implementation is substantially simpler. The core algorithm is clean: generate responses, compute group statistics, assign advantages, apply clipped gradient update. The main additional consideration compared to vanilla policy gradient is the group sampling mechanism — generating G responses per prompt rather than one — which requires modest changes to the data loading and generation pipeline.

### Summary Comparison Table

| Dimension | PPO | GRPO |
|---|---|---|
| Critic required | Yes — same scale as policy | No |
| Number of models in memory | 4 | 3 |
| Advantage granularity | Per-token, learned | Per-response, from group stats |
| Baseline computation | Critic forward pass | Group mean arithmetic |
| Primary stability threats | Critic collapse, reward hacking | Reward hacking, small-G variance |
| Clipping mechanism | Yes, epsilon = 0.2 | Yes, epsilon = 0.2 |
| KL penalty | Yes, against reference policy | Yes, against reference policy |
| Implementation complexity | High (37+ details) | Moderate |
| Memory footprint (7B policy) | ~200+ GB training memory | ~140-160 GB training memory |
| Used in | InstructGPT, LLaMA 2 Chat | DeepSeekMath, DeepSeek-R1, many 2025 models |

---

## How it Works in Practice

Understanding GRPO theoretically is useful, but understanding what the training loop looks like in practice — on real hardware, with real hyperparameters — is essential for anyone implementing or evaluating GRPO-based systems.

### The Training Step in Plain Language

Each training step in GRPO proceeds as follows.

**Step 1 — Rollout phase:** A batch of N prompts is drawn from the training dataset. For each prompt, the current policy generates G responses via temperature sampling (temperature typically set to 0.7 to 1.0 for diversity). Each response is generated autoregressively, token by token, until a stopping criterion is reached (end-of-sequence token or maximum length). The old policy probabilities for each generated token are recorded during this phase — they will be needed for the policy ratio computation.

**Step 2 — Scoring phase:** Each of the N * G responses is scored by the reward signal. For verifiable tasks, this involves extracting the final answer and comparing it to the known correct answer. For preference tasks, this involves passing each (prompt, response) pair through the reward model to obtain a scalar score.

**Step 3 — Advantage computation:** For each group of G responses, compute the group mean reward and group standard deviation. Compute the advantage for each response. Assign this per-response advantage to every token in the response.

**Step 4 — Policy update:** For each token in each response, compute the new policy's probability for that token (requires a forward pass through the current policy). Compute the policy ratio. Apply clipping. Compute the KL divergence penalty. Sum up the per-token objectives, compute gradients, and perform an optimizer step.

**Step 5 — Repeat:** Discard all G responses. Generate G new responses from the updated policy. Continue.

### Real Group Sizes and When to Choose Them

The choice of G interacts strongly with the reward signal and task type:

**Binary rewards (correct or incorrect):** G should be large enough that most groups contain both correct and incorrect responses. If the model is 90% accurate on a task, groups of G=8 will contain on average 7.2 correct and 0.8 incorrect responses. Groups with all-correct or all-incorrect responses have zero variance and produce no gradient signal. In this situation, increase the task difficulty rather than G.

**Continuous rewards from a reward model:** G can be smaller because even small reward differences between responses produce non-zero advantages. G=4 to G=8 is usually sufficient.

**Very hard tasks (early training):** G should be large enough that at least one response in each group is partially correct. If all G responses are completely wrong, the standard deviation is zero and there is no gradient signal. You might use G=16 combined with harder curriculum scheduling.

### Memory Savings in Concrete Terms

```
7B parameter model at bf16 precision:

PPO:
  Policy weights:      14 GB
  Policy optimizer:    28 GB  (Adam first + second moments)
  Reference weights:   14 GB
  Critic weights:      14 GB
  Critic optimizer:    28 GB
  Reward model:         6 GB  (assume 3B model)
  Activations, misc:   ~30 GB
  ──────────────────────────
  Total:               ~134 GB

GRPO:
  Policy weights:      14 GB
  Policy optimizer:    28 GB
  Reference weights:   14 GB
  Reward model:         6 GB
  Activations, misc:   ~20 GB
  ──────────────────────────
  Total:               ~82 GB

Savings: ~52 GB — equivalent to about 1 full A100 80GB GPU
```

These numbers are estimates; actual memory usage depends on sequence length, batch size, and implementation choices. The key point is that the savings are substantial and enable training on fewer GPUs or scaling to a larger policy at the same hardware budget.

### Typical Hyperparameter Settings

Based on the DeepSeekMath and DeepSeek-R1 papers and subsequent community implementations:

| Hyperparameter | Typical Value | Notes |
|---|---|---|
| Group size G | 8–16 | Larger for sparser rewards |
| Clip epsilon | 0.2 | Same as PPO default |
| KL coefficient beta | 0.01–0.04 | Lower = less constraint on policy drift |
| Learning rate | 1e-6 to 1e-5 | Smaller than SFT |
| Temperature | 0.7–1.0 | Higher = more diverse responses |
| Batch size | 64–256 prompts | Scales with hardware |
| Training steps | 1000–10000 | Depends on data and task |

---

## A Complete Worked Example

Walking through one full training step of GRPO with concrete numbers makes the algorithm concrete in a way that prose descriptions cannot.

### Setup

Prompt: "What is 15% of 240?"
Correct answer: 36
Group size G = 4 (reduced for illustration)
Reward: 1.0 if final answer is 36, else 0.0

### Step 1: Generate G=4 Responses

```
Response 1: "15% of 240. 10% of 240 is 24. 5% is 12. So 24+12=36."
            Final answer: 36  →  Reward = 1.0

Response 2: "15/100 × 240 = 3600/100 = 36."
            Final answer: 36  →  Reward = 1.0

Response 3: "15% means 15 out of 100. 240 × 15 = 3600. Answer: 3600."
            Final answer: 3600  →  Reward = 0.0

Response 4: "0.15 × 240 = 0.15 × 200 + 0.15 × 40 = 30 + 6 = 36."
            Final answer: 36  →  Reward = 1.0
```

### Step 2: Compute Group Statistics

```
Rewards:     [1.0, 1.0, 0.0, 1.0]
Group mean:  (1.0 + 1.0 + 0.0 + 1.0) / 4 = 0.75
Deviations:  [0.25, 0.25, -0.75, 0.25]
Group std:   sqrt(mean of squared deviations)
             = sqrt((0.0625 + 0.0625 + 0.5625 + 0.0625) / 4)
             = sqrt(0.1875) ≈ 0.433
```

### Step 3: Compute Advantages

```
Response 1: (1.0 - 0.75) / 0.433 = +0.577
Response 2: (1.0 - 0.75) / 0.433 = +0.577
Response 3: (0.0 - 0.75) / 0.433 = -1.732
Response 4: (1.0 - 0.75) / 0.433 = +0.577
```

Every token in Responses 1, 2, and 4 gets advantage = +0.577.
Every token in Response 3 gets advantage = -1.732.

### Step 4: Policy Update

For each token in each response, the policy ratio r = p_new / p_old is computed. The gradient objective is min(r * A, clip(r, 0.8, 1.2) * A). For Response 3, the large negative advantage pushes toward decreasing the probability of every token — but the clip prevents the ratio from going below 0.8. For Response 3, the especially bad error ("3600/100 = 3600" is nonsensical) might get the sharpest probability decrease, but all tokens including the good reasoning steps get the same -1.732 advantage weight.

### What the Model Learns

After this update, the policy is slightly more likely to produce step-by-step percentage calculations like Responses 1 and 4, and slightly less likely to produce responses like Response 3 that incorrectly apply the division operation. The magnitude of the update is governed by the learning rate and the clipping bound.

This is one prompt in one batch. A real training run processes tens of thousands of such prompts across thousands of training steps, gradually shifting the policy's distribution toward correct reasoning patterns.

---

## Common Misconceptions

### Misconception 1: "GRPO does not use any reward model"

This is partially true and frequently misunderstood. GRPO removes the need for a critic (the value function network). But GRPO still requires a reward signal — either a reward model trained on human preferences, or a verifiable reward signal like code execution or math correctness checking.

In DeepSeek-R1, the reward signal for mathematical and coding tasks was rule-based (verifiable correctness), which meant no separately trained reward model was needed for those specific tasks. But this is a property of the task domain — math and code have objectively verifiable answers — not a property of GRPO itself. GRPO is fully compatible with a reward model as the scoring function, and several published implementations use exactly this setup for general instruction-following tasks.

The confusion arises because DeepSeek-R1 was remarkable specifically because it demonstrated that you can train high-quality reasoning without a reward model, using verifiable rewards instead. But attributing this property to GRPO rather than to the task type is an error.

### Misconception 2: "Group normalization eliminates all variance in advantage estimates"

Normalizing advantages within a group reduces variance from prompt-to-prompt reward scale differences, but it does not eliminate variance in the advantage estimates themselves. The group mean and standard deviation are themselves estimated from G samples. If G is small (G=2 or G=4), the estimated mean can be far from the true expected reward for that prompt. The advantage estimates are therefore noisy, not precise.

More subtly: even with large G, if the reward function itself has high variance (different samples of the same prompt give very different rewards), the group mean will still fluctuate significantly from batch to batch. Normalization controls the scale of advantages; it does not reduce the fundamental noise in the reward signal.

### Misconception 3: "GRPO is strictly better than PPO"

GRPO is simpler, more memory-efficient than PPO, and matches or exceeds PPO performance on mathematical reasoning and coding tasks. But it has real limitations compared to PPO.

The per-response advantage estimate is coarser than per-token estimates. If a task requires rewarding specific tokens or penalizing specific error patterns, GRPO provides less targeted credit assignment. The group baseline has higher variance than a well-trained critic for tasks where the critic has had time to learn the reward structure. And for tasks with highly diverse reward scales across prompts (where the critic's global generalization is valuable), the local group mean baseline may be noisier.

"GRPO is better for the tasks it has been tested on" is accurate. "GRPO is strictly better in all scenarios" is not supported by the evidence.

### Misconception 4: "The group mean is equivalent to having a critic"

Both serve as baselines, but they are fundamentally different in nature. The critic is a learned neural network that generalizes across the full training distribution — it has seen thousands of prompts and responses and learned which types of states tend to lead to high rewards. The group mean is a local statistic computed from the current batch — it knows only about the G responses for this specific prompt in this specific batch.

The critic can, in principle, capture the fact that certain types of prompts (e.g., easy algebraic problems) reliably produce higher rewards than other types (e.g., hard combinatorics), and adjust its baseline accordingly even at the sub-response level. The group mean captures this automatically at the prompt level (because the group is per-prompt) but cannot distinguish quality differences at the position level within a response.

### Misconception 5: "GRPO only works with verifiable reward signals"

GRPO was designed with verifiable rewards in mind and has been most prominently demonstrated on math and code. But the algorithm itself is agnostic to the reward source. Several papers following DeepSeek-R1 have applied GRPO to general instruction following, dialogue, and creative writing tasks using reward models trained on human preferences. The algorithm works identically — the only difference is that the reward signal comes from a neural network rather than a verifier.

Whether GRPO matches PPO in quality for these non-verifiable tasks is a separate empirical question from whether GRPO works at all.

### Misconception 6: "Removing the critic means GRPO ignores token-level information"

GRPO's policy gradient update still operates at the token level: every token in a response gets a gradient update based on the policy ratio and the advantage. What GRPO ignores is the within-response variation in advantage — all tokens in a response share the same advantage. This is different from ignoring token-level information entirely. The gradient still flows through every token; it just flows with equal weighting rather than position-specific weighting.

---

## Connections to Other Topics

### Connection to PPO

GRPO is a direct variant of PPO. The clipping mechanism, the policy ratio computation, the KL penalty, and the basic policy gradient formulation are all inherited from PPO. The only change is the advantage computation. Any study of GRPO that has not first covered PPO is missing essential context.

The key PPO papers to understand alongside GRPO: Schulman et al. (2017) for the core algorithm, and Huang et al. (2023) for the implementation details in the LLM context.

### Connection to REINFORCE

The classic REINFORCE algorithm (Williams, 1992) is the ancestor of both PPO and GRPO. REINFORCE uses the raw cumulative reward as the gradient weight — no baseline, no clipping. GRPO can be derived from REINFORCE by adding group normalization (which functions as baseline subtraction plus scale normalization) and PPO-style clipping (which functions as a trust-region constraint).

The DeepSeekMath paper includes a comparison of GRPO against REINFORCE and shows that both additions matter: normalization alone is insufficient (training is unstable without clipping), and clipping alone is insufficient (training has high variance without normalization). The combination of both stabilizers is what makes GRPO work reliably.

### Connection to RLOO

RLOO (REINFORCE Leave-One-Out, Ahmadian et al. 2024) is the algorithm most similar to GRPO. The only difference: in RLOO, the baseline for response i is the mean of the other G-1 responses (not including response i), while in GRPO the baseline is the mean of all G responses (including response i). The leave-one-out estimator is statistically unbiased; GRPO's group mean is biased because the response's own reward influences its baseline.

For G=8 or larger, this difference is negligible in practice. RLOO and GRPO produce nearly identical training trajectories at typical group sizes. RLOO has cleaner theoretical properties; GRPO is slightly simpler to implement.

### Connection to KL Divergence

The KL penalty in GRPO is identical in purpose and mechanics to the KL penalty in RLHF: it penalizes the policy for drifting too far from the reference policy (the frozen SFT model). This prevents reward hacking — the policy finding degenerate strategies that score high on the reward signal while being qualitatively poor.

The DeepSeek-R1 paper used a relatively small KL coefficient (beta=0.04), indicating that the training was allowed to drift somewhat from the SFT policy. This is appropriate for reasoning tasks, where the desired behavior (extended chain-of-thought reasoning) may not be well-represented in the SFT training data and requires the policy to substantially change its behavior.

### Connection to Reward Models

For non-verifiable tasks, GRPO uses a reward model to score the G responses. The reward model is the same type of model as in RLHF: a fine-tuned LLM with a scalar output head, trained on human preference comparisons. All limitations of reward models — length bias, reward hacking, distributional shift — are still relevant when GRPO uses a reward model for scoring.

GRPO does not solve the reward hacking problem; it only removes the critic from the training stack. The KL penalty remains the primary defense against reward hacking in GRPO, as in all RL-for-LLM approaches.

### Connection to Dr. GRPO

Dr. GRPO (Decoupled Reward GRPO) is a direct extension of GRPO that addresses the length bias problem. In GRPO, longer responses generate more token-level gradient updates than shorter responses with the same advantage score, because each token contributes independently to the loss. This creates an implicit incentive for the model to generate longer responses.

Dr. GRPO normalizes the per-response gradient contribution by the number of tokens in the response, ensuring that a 50-token response and a 500-token response with the same advantage score make equal total contributions to the gradient update. This is a one-line change to the loss computation but significantly reduces length inflation during training.

### Connection to DAPO

DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization, Liu et al. 2025) makes two additional improvements on top of GRPO: dynamic sampling that filters out groups where all responses have the same score (avoiding zero-gradient updates), and a modified clipping scheme that treats the upper and lower bounds of the policy ratio clip asymmetrically. DAPO represents the direction GRPO-style training is evolving as of 2025.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| GRPO | Critic-free variant of PPO; uses group statistics as the advantage baseline |
| The critic problem | PPO needs a neural network the same size as the policy to estimate advantages — expensive and unstable |
| Group sampling | For each prompt, generate G responses (G=8 to G=16) from the current policy |
| Group baseline | Mean reward of the G responses replaces the PPO critic as the advantage baseline |
| Advantage normalization | Subtract group mean, divide by group std — produces z-scores; same idea as grading on a curve |
| Positive advantage | Response scored above group average — increase probability via gradient update |
| Negative advantage | Response scored below group average — decrease probability via gradient update |
| Clipping (epsilon=0.2) | Policy ratio bounded to [0.8, 1.2] — prevents catastrophically large policy updates |
| KL penalty | Same as PPO — penalizes drift from the reference (SFT) policy |
| Memory saving | Removes the critic (~25–35% total training memory reduction for a 7B policy) |
| Stability benefit | Critic collapse failure mode is absent — no learned value function to destabilize |
| Per-response advantage | All tokens in a response share the same advantage — coarser but sufficient for reasoning tasks |
| Exploration via temperature | Temperature sampling during rollout ensures G responses are diverse; diversity drives learning signal |
| DeepSeekMath | Introduced GRPO; matched PPO on math reasoning with less memory |
| DeepSeek-R1 | Applied GRPO at 671B scale; emergent chain-of-thought reasoning arose naturally |
| Verifiable rewards | Binary correctness as reward signal eliminates the reward model for math and code |
| Format reward | Small bonus for following output structure; shapes model to maintain reasoning format |
| Length bias | Longer responses dominate gradient because more tokens contribute; addressed by Dr. GRPO |
| RLOO relationship | RLOO uses leave-one-out mean (unbiased); GRPO uses full group mean; nearly equivalent at G=8+ |
| DAPO relationship | DAPO adds dynamic sampling and asymmetric clipping on top of GRPO's group relative approach |

---

## Up Next

The next topic in this series is **RLOO — REINFORCE Leave-One-Out**.

RLOO (Ahmadian et al., 2024) is the algorithm most closely related to GRPO. Like GRPO, it uses group sampling to eliminate the PPO critic and compute advantages from the current batch rather than from a separately trained value network. Unlike GRPO, it computes each response's baseline as the average reward of the other responses in the group — leaving one out. This leave-one-out estimator is statistically unbiased: each response's baseline is computed from data that does not include that response's own reward, which eliminates the self-inclusion bias present in GRPO's full group mean.

The practical significance of this bias is modest for the group sizes used in practice. When G=8, including or excluding one response changes the mean by at most 12.5% of the response's reward. For G=16, the maximum shift is 6.25%. In both cases, the effect on training dynamics is small, and the two algorithms produce nearly equivalent results in the experiments where they have been compared directly.

However, RLOO's theoretical cleanliness has made it a popular reference algorithm in the literature. Several papers analyzing the properties of group-based policy gradient algorithms use RLOO as the theoretically clean baseline and GRPO as the practically convenient approximation. Understanding the relationship between them gives you a clearer mental model of what the group mean is doing — it is an approximately unbiased estimator of the advantage, with the approximation error shrinking as G grows.

Studying RLOO after GRPO will complete your understanding of the critic-free policy gradient family, and will prepare you to engage with DAPO, Dr. GRPO, and the ongoing research into more stable and efficient RL training algorithms for language models. Each subsequent algorithm in this family addresses one specific weakness identified in GRPO or RLOO, building cumulatively on the group-relative policy gradient foundation established in DeepSeekMath.

The progression from GRPO to RLOO to DAPO mirrors a broader pattern in machine learning research: a practical algorithm is proposed (GRPO), a theoretically cleaner variant is formalized (RLOO), and then practitioners identify specific weaknesses in both (length bias, zero-gradient batches, asymmetric clipping requirements) and address them systematically (Dr. GRPO, DAPO). Understanding the full lineage — from REINFORCE through PPO through GRPO through RLOO through DAPO — gives you the conceptual vocabulary to read new papers in this space and reason about what problems they are trying to solve.

| Algorithm | Year | Key improvement over predecessor |
|---|---|---|
| REINFORCE | 1992 | The original policy gradient |
| PPO | 2017 | Clipping for stability; critic for variance reduction |
| GRPO | 2024 | Removes critic; uses group mean as baseline |
| RLOO | 2024 | Unbiased leave-one-out baseline (vs GRPO's biased group mean) |
| Dr. GRPO | 2025 | Normalizes by response length to remove length bias |
| DAPO | 2025 | Dynamic sampling + asymmetric clipping |
