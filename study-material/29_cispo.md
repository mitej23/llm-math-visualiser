# ✂️ CISPO — Clipped IS Policy Optimization

> **Sources used:**
> - Yu et al., *CISPO: Clipped IS Policy Optimization for Efficient Reinforcement Learning from Human Feedback*, 2024 — [arxiv.org/abs/2409.00672](https://arxiv.org/abs/2409.00672)
> - Schulman et al., *Proximal Policy Optimization Algorithms*, OpenAI 2017 — [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
> - Lilian Weng, *Policy Gradient Algorithms*, OpenAI Blog 2018 — [lilianweng.github.io/posts/2018-04-08-policy-gradient](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
> - Huang et al., *The N+ Implementation Details of RLHF with PPO*, HuggingFace Blog 2023 — [huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)

---

## The Big Idea

Reinforcement learning for language models is expensive. The biggest cost comes from needing to collect fresh samples from the policy after every single update. CISPO — Clipped Importance Sampling Policy Optimization — attacks that cost directly by making it safe to reuse old samples, and by making the mathematical machinery of reuse far cheaper than it was before.

The core insight of CISPO is deceptively simple: when you borrow data collected from an older version of the policy, you normally need to compute an importance sampling weight to correct the statistical bias. That weight, if not handled carefully, creates enormous variance in your gradient estimates — the gradient can suddenly jump to extreme values that destabilize training. CISPO's answer is to apply a **stop-gradient** operation to the importance sampling weight. It still uses the weight to decide whether to clip (to prevent runaway updates), but it blocks the backpropagation from flowing through the weight itself. This eliminates a full backward pass through the old policy network, roughly halving the compute needed for each training step.

The practical consequence is that CISPO can train language models at roughly twice the throughput of standard PPO, without measurably hurting final performance. For large models where a single training step might already take many seconds, cutting that step in half is not a minor improvement — it is the difference between a training run that finishes in a day versus one that takes two days.

---

## Real-Life Analogy

Imagine you are reviewing your expenses for the month. You made several purchases, and you have the receipts. Now you need to answer the question: "Would I make those same purchases today, given my current understanding of what I need?"

The receipt is the record of a decision made under an older set of preferences — the old policy. Importance sampling is the correction you apply: you check the receipt to figure out how much to trust this old data. If your current preferences are very different from when you made the purchase, the receipt is less relevant. If your preferences are almost the same, the receipt is highly trustworthy.

Standard importance sampling says: "Check the receipt AND use the weight to adjust how strongly each purchase influences your current budget decisions." Every purchase gets scaled up or down depending on how relevant it is to today's preferences.

CISPO says something subtly different: "Check the receipt, but don't let the correction weight actually influence the strength of the update — just use the weight to decide whether to throw out the purchase data entirely (clip it) or keep it." The receipt is still read. The decision of what to include or exclude still happens. But once that decision is made, the weight is discarded from the gradient calculation. You don't re-scale purchases based on how old they are; you simply decide whether to include them or exclude them.

The consequence is that the gradient calculation becomes cleaner, faster, and dramatically more stable.

---

## What Is Importance Sampling? — Deep Dive

Policy gradient methods require computing expectations under the current policy — you need to know: "On average, given what my current policy would do, how good are the actions being taken?" To compute that expectation accurately, you ideally need to sample actions from the current policy.

But collecting new samples after every gradient step is expensive. For a language model, sampling means running a full forward pass through the network to generate a response — possibly hundreds of tokens — then scoring that response with the reward model, and then computing the gradients. If your model has billions of parameters, this process is slow, and doing it after every single parameter update wastes an enormous amount of compute.

**Importance sampling (IS)** is the classical statistical technique for reusing data from one distribution to estimate expectations under a different distribution. The key formula involves the ratio of probabilities: how likely is this action under the current policy divided by how likely it was under the old policy that generated the data? This ratio is called the importance weight.

If the current policy assigns roughly the same probability to an action as the old policy did, the importance weight is close to one — the old data is directly usable. If the current policy assigns much higher probability to that action, the weight is greater than one — the old data underestimates how important that action is under the current policy. If the current policy assigns much lower probability, the weight is less than one — the old data is overweighting that action.

By multiplying each data point's contribution to the gradient estimate by its importance weight, you correct for the fact that you collected data under a different policy. The resulting estimator is statistically unbiased: on average, it gives you the correct gradient, even though the data came from the old policy.

**Off-policy learning** is the regime where you systematically collect data under one policy (the behavior policy) and train a different policy (the target policy). Importance sampling is what makes this statistically valid. In the RLHF context, this means you can generate responses using the policy from the start of a training epoch, update the policy several times using those same responses, and correct for the growing gap between the generation policy and the current policy using IS weights.

The bigger the gap between old and current policy — measured by how much their probability distributions diverge — the larger (or smaller) the IS weights become, and the less trustworthy the correction becomes. This is the variance problem that CISPO addresses.

---

## The Variance Problem With IS Weights — Deep Dive

Importance sampling is statistically valid, but it carries a serious practical problem: **variance explosion**.

When the current policy has drifted far from the policy that generated the data, importance weights can become very large. Suppose an action that had a 1% probability under the old policy now has a 20% probability under the current policy. The IS weight for that action is 20 (twenty times amplification). Any data point that happens to have this action will contribute twenty times as much to the gradient as it would have without the correction. If there are only a handful of such data points in your batch, the gradient estimate will be dominated by their scaled contributions — and the gradient will be extremely noisy.

The statistical term for this is **high variance**. High variance means your gradient estimates are all over the place. One batch gives you a large positive gradient. The next gives you a large negative gradient. The training process zigzags rather than converging smoothly. In the worst case, a single outlier data point with a very large IS weight can completely dominate a gradient step, pushing the policy parameters dramatically in a direction that is not representative of the overall data distribution.

This is not hypothetical. In practice, large IS ratios cause training instability that is difficult to diagnose. The loss curve may look fine on average while individual gradient steps are wildly inconsistent. The model may appear to be learning but then suddenly degrade because a few high-weight data points pushed the policy into a bad region of parameter space.

The variance also increases with the number of reuse steps. If you reuse a batch of data for one gradient update, the gap between old and current policy is small, IS weights are close to one, and variance is manageable. If you reuse the same batch for five gradient updates, the current policy may have drifted substantially, IS weights can span a wide range, and the gradient estimates become unreliable.

PPO addresses this problem using **ratio clipping**: it clips the product of the IS weight and the advantage to a region around one, preventing any single data point from contributing an outsized gradient. But PPO still computes the full IS weight and backpropagates gradients through it — meaning the IS weight participates in the gradient calculation and imposes its own computational cost.

CISPO takes a different approach to the same problem: rather than clipping the contribution after computing it, it removes the IS weight from the gradient calculation entirely using a stop-gradient operation, while still using the weight as a gating signal to decide whether to keep or discard each data point.

---

## The Stop-Gradient Trick — Deep Dive

The stop-gradient operation is a technique from automatic differentiation. When you compute something in a neural network forward pass and you want to use that value without letting gradients flow back through it, you wrap it in a stop-gradient. The computation still happens — the value is still used — but the backward pass treats that value as if it were a constant, not a function of the model parameters.

In CISPO, the stop-gradient is applied specifically to the importance sampling ratio. Here is the logic in sequence:

First, CISPO computes the IS ratio normally: it divides the probability of the action under the current policy by its probability under the old policy. This ratio is a real number that reflects how much the policy has changed for this particular action.

Second, CISPO uses this ratio to decide whether to clip. If the ratio is outside a small window around one — say, less than 0.8 or greater than 1.2 — the data point's gradient contribution is clipped. This is the same kind of clipping that PPO does, and it serves the same purpose: preventing any single data point from pushing the policy parameters too far in one direction.

Third — and this is the CISPO innovation — the IS ratio is wrapped in a stop-gradient before it enters the actual gradient computation. The ratio is used to determine clipping, but it does not participate in the backward pass. When gradients are propagated back through the network, the IS ratio appears as a constant in the computation graph, not as a function of the policy parameters.

The consequence is that there is no need to backpropagate through the IS ratio. In a standard PPO implementation, computing the IS ratio requires storing activations from the old policy network throughout the backward pass — the memory needed to hold the computation graph for the ratio. With stop-gradient, none of that memory is needed, because the backward pass stops at the constant-valued ratio.

Furthermore, in some implementations, the old policy forward pass can be done independently of the backward pass. The forward pass through the old policy (to compute probabilities used in the IS ratio) does not need to track gradients at all, because the ratio will be treated as a constant. This means the old policy forward pass uses significantly less GPU memory and can be run in inference mode, which is faster and cheaper.

The stop-gradient trick is not unique to CISPO — it is used widely in machine learning whenever you want to use a value for a decision without letting gradients flow through the decision-making computation. What is novel in CISPO is applying it specifically to the IS ratio in the policy gradient objective, recognizing that the gradient-of-the-IS-ratio provides so little useful signal (relative to the variance it introduces) that discarding it is a net win.

---

## How CISPO Achieves 2x Speed — Deep Dive

The speed improvement from CISPO comes from two interconnected mechanisms: reduced backward pass cost and the practical feasibility of asynchronous sampling.

**Reduced backward pass cost.** In standard PPO, the training step requires computing gradients through both the objective function and the IS weights. The IS weights themselves are ratios of probabilities output by the current policy network, which means gradients must flow back through the policy network twice: once for the main policy gradient objective, and once for the IS weight ratio computation. With stop-gradient, the IS weight is treated as a constant, and gradients only flow through the network once. This saves one backward pass through the policy network per training step. For large language models where the backward pass is the most memory-intensive and compute-intensive part of training, this is a substantial saving.

**Asynchronous sampling becomes practical.** In synchronous PPO training, the pipeline looks like this: generate samples, compute IS weights, update policy, repeat. The generation phase must complete before training can begin, and training must complete before the next generation phase begins. This serialization leaves the GPU alternating between two very different workloads — generation (forward pass heavy) and training (backward pass heavy) — neither of which fully utilizes the hardware.

With CISPO, because IS corrections are applied very conservatively (via clipping) and the gradient penalty for stale data is low, it becomes practical to let sample generation and model training happen simultaneously. One process continuously generates new samples using the current (or slightly outdated) policy. Another process continuously pulls from the sample buffer and runs gradient updates. The IS weights applied during training correct for the staleness of the data. Because CISPO's stop-gradient removes the sensitivity of the gradient to large IS weights, the training is robust to the moderate staleness that asynchronous pipelines inevitably introduce.

This asynchronous pipeline is the larger share of the 2x throughput gain. The GPU spends much less time waiting between phases. Generation and training overlap in time. The overall wall-clock time for a training run is approximately halved because the two bottleneck operations — data collection and gradient computation — no longer take turns; they run in parallel.

The combination of these two effects — one backward pass saved per step, plus asynchronous parallelism — compounds into roughly a 2x improvement in training throughput on large language model workloads as reported in the CISPO paper.

---

## CISPO vs PPO Clipping — Deep Dive

PPO and CISPO both use clipping as a mechanism to limit the effect of stale data. But the way clipping works in each algorithm is different in a way that matters for both compute and training dynamics.

**PPO clipping.** In PPO, the objective function is defined as the minimum of two terms: the unclipped IS ratio times the advantage, and the clipped IS ratio times the advantage. The clip function limits the IS ratio to a band around one — if the ratio strays too far above or below one, the objective switches to using the boundary value instead. This clipped objective is then used as the thing gradients are computed through. The IS ratio appears inside the objective, and gradients flow back through the ratio computation during the backward pass. PPO uses clipping to limit the size of policy updates while still using the IS weight to scale gradient contributions within the unclipped region.

**CISPO clipping.** In CISPO, the IS ratio is computed but immediately wrapped in a stop-gradient. The ratio is used only as a binary-like gate: if the ratio is within the acceptable range, the data point's gradient contribution passes through unchanged (the ratio effectively equals one in the gradient); if it is outside the range, the data point is excluded from the gradient update. The advantage signal flows through uncorrected. There is no scaling of gradient contributions by the IS weight within the gradient calculation — the weight is purely a selector.

In PPO, every data point in the unclipped region contributes to the gradient in proportion to how much the policy has changed for that action. Actions that the current policy rates much higher than the old policy are weighted more heavily; actions rated lower are weighted less. This is the IS correction working within the objective.

In CISPO, within-range data points all contribute equally regardless of how much the policy has shifted from the old policy (as long as the shift is within bounds). Only out-of-range data points are rejected. The gradient does not see the IS weight at all; it only sees whether the data point was included or excluded.

The practical effect on training stability is similar — both algorithms prevent runaway updates from stale or anomalous data. But CISPO's approach is cheaper to compute and tends to produce lower variance gradients because the IS weight is not introducing multiplicative scaling into the gradient signal.

One tradeoff is that CISPO's gradient estimator is technically biased: by ignoring the IS weight in the gradient calculation, it does not produce the statistically correct gradient for the current policy. PPO's gradient estimator is less biased within the clipped region. In practice, for the moderate levels of staleness that arise in reuse scenarios, the bias in CISPO's estimator is small and is more than compensated for by the reduction in variance and compute cost.

---

## Memory and Compute Analysis — Deep Dive

Understanding the memory and compute differences between PPO and CISPO requires thinking carefully about what is stored during the forward pass and what is needed during the backward pass.

**Activations and the computation graph.** When you run a forward pass through a neural network and plan to backpropagate through it, the framework (PyTorch, JAX, etc.) must store intermediate activations — the values computed at each layer — so that they can be used to compute gradients during the backward pass. For a large language model, these activations can use an enormous amount of GPU memory, often comparable to or larger than the model parameters themselves.

**PPO memory footprint.** In PPO, computing the IS ratio for a given action requires access to the old policy's log probability for that action. The old policy log probability is computed during a forward pass through the old policy. If gradients are flowing through the IS ratio, the computation graph must track through the old policy forward pass — meaning those activations are stored in memory throughout the backward pass. For a model with many billions of parameters, this is a significant additional memory cost on top of the memory already needed for the current policy forward and backward passes.

**CISPO memory footprint.** With stop-gradient applied to the IS ratio, the old policy forward pass does not need to retain activations for gradient computation. It can be run in inference mode (no gradient tracking), which typically uses significantly less memory. The IS ratio value is stored, but the entire computation graph that produced it is discarded. This means the old policy forward pass in CISPO uses approximately the same memory as simply running the model for generation — much less than a training-mode forward pass.

**Compute cost breakdown.** A rough accounting of operations per training step:

For PPO: old policy forward pass (with gradient tracking) + current policy forward pass + current policy backward pass + IS weight gradient backward pass.

For CISPO: old policy forward pass (inference mode, no gradient tracking) + current policy forward pass + current policy backward pass only.

The IS weight gradient backward pass eliminated by CISPO is not trivial. It requires propagating gradients back through all the layers of the old policy network, which has the same cost as any other backward pass through the model. Eliminating it saves roughly the same amount of compute as one full backward pass through the model — which is typically the most expensive operation in the training step.

In practice, training throughput improvements depend on hardware configuration, batch size, model size, and the degree of asynchrony achievable in the pipeline. The CISPO paper reports approximately 2x throughput on their benchmark configurations, with the asynchronous sampling pipeline contributing the larger share of the improvement.

---

## How it Works in Practice

A CISPO training loop looks very similar to a PPO training loop at a high level, with the key differences buried in the objective function and the sampling schedule.

**Data collection phase.** The policy (or a slightly older frozen snapshot of it) generates responses for a batch of prompts. Each response is scored by the reward model and the value function to produce advantages. The log probabilities of each action (token) under the generating policy are recorded and stored alongside the data.

**Reuse schedule.** Unlike standard PPO where you might run exactly one or two gradient updates per batch of data before discarding it, CISPO allows more aggressive reuse — the same batch can be used for more gradient updates because the stop-gradient mechanism reduces sensitivity to the growing staleness of the IS weights.

**Gradient step.** For each data point in the batch, the current policy computes log probabilities for the same action that was taken. The IS ratio is computed as the exponential of the difference in log probabilities (current minus old). This ratio is wrapped in stop-gradient. The ratio is checked against the clip bounds. If it falls within bounds, the advantage flows through to the gradient normally. If it falls outside bounds, the contribution is clipped or zeroed. The gradient is computed through the advantage signal only — not through the IS ratio itself.

**Asynchronous variant.** In the asynchronous implementation, a separate worker continuously generates new samples while the main training process runs gradient updates. The sample buffer accumulates data from slightly different policy snapshots. When the training process pulls a batch, it checks IS ratios and applies clipping. Because the stop-gradient means large IS ratios don't destabilize the gradient, the training is robust to the moderate staleness inherent in the asynchronous pipeline.

**Monitoring.** In practice, practitioners monitor the distribution of IS ratios as a proxy for how stale the data is. If IS ratios consistently stray far from one, the asynchronous gap (how far behind the generation worker is from the training worker) should be reduced. If IS ratios are all very close to one, the pipeline could potentially allow the generation worker to fall further behind, increasing the degree of overlap.

---

## Common Misconceptions

**Misconception 1: CISPO ignores importance sampling entirely.**

This is incorrect. CISPO does compute the IS ratio — it uses the ratio to determine whether to clip each data point's contribution. What it does not do is include the IS ratio in the gradient calculation. The IS ratio is a gating signal, not a scaling factor in the gradient. Saying CISPO ignores IS entirely confuses "not backpropagating through" with "not computing at all."

**Misconception 2: The stop-gradient makes CISPO biased in a way that PPO is not.**

Both PPO and CISPO produce biased gradient estimates when data is reused from an old policy. PPO's bias comes from the clipping operation (the gradient is zero outside the clip range, which is a form of bias). CISPO's bias comes from ignoring the IS scaling within the clip range. Neither algorithm is an unbiased gradient estimator in the off-policy setting. The key comparison is variance: CISPO trades a modest increase in bias for a substantial reduction in variance, which is typically the right tradeoff for practical training.

**Misconception 3: The 2x speed comes from a simpler objective function.**

The simplified objective is part of it, but the larger gain comes from two compounding effects: one fewer backward pass through the policy network per step, and the practical feasibility of running data collection asynchronously with gradient updates. If you only applied the stop-gradient without restructuring the sampling pipeline, you would get a meaningful but smaller speedup — perhaps 20-30% rather than 2x.

**Misconception 4: CISPO only works for small models where variance is already low.**

CISPO's benefits are, if anything, more pronounced for large models. Large models are slower to run, so the cost of the extra backward pass is higher in absolute terms. Large models also benefit more from asynchronous sampling because the per-sample cost is higher, making it more valuable to overlap generation and training. The variance problem with IS weights does not systematically scale with model size in a way that would make CISPO less applicable to large models.

**Misconception 5: CISPO is equivalent to simply not using importance sampling.**

If you removed IS weights entirely without clipping, you would be computing an on-policy gradient estimate using off-policy data — which can be significantly biased when the policies have diverged. CISPO keeps the clipping mechanism, which prevents the most harmful stale data points from contributing to the gradient at all. The difference between "no IS" and "CISPO" is that CISPO still gates out data that is too far from the current policy. This gating is essential for training stability when policies diverge significantly during multi-step reuse.

**Misconception 6: CISPO produces worse final performance than PPO.**

Empirically, CISPO and PPO converge to similar final performance on the benchmarks where they have been compared. The stop-gradient introduces a small bias, but this bias does not accumulate into systematically worse outcomes. Both algorithms are finding approximately the same optima via different computational paths, with CISPO taking roughly half the wall-clock time to get there.

---

## Connections to Other Topics

**PPO (Proximal Policy Optimization).** CISPO is a direct modification of PPO. It keeps the clipping mechanism but removes the gradient through the IS ratio. Understanding PPO's objective function — particularly why clipping was introduced and what problem it solves — is essential context for understanding why CISPO's simplification works.

**GRPO (Group Relative Policy Optimization).** GRPO is another alternative to PPO that reduces memory cost by removing the value function critic network. GRPO and CISPO attack different bottlenecks: GRPO eliminates the critic, CISPO reduces the cost of the IS weight computation. In principle, both modifications could be combined.

**Importance sampling in statistics.** IS is a classical variance reduction technique from Monte Carlo statistics. Understanding why IS estimators can have high variance — particularly when the ratio of target to proposal probability is large — provides the mathematical foundation for why CISPO's stop-gradient is a reasonable approximation.

**Reward models and RLHF pipelines.** CISPO operates within the standard RLHF pipeline: generate responses, score with reward model, compute advantages, update policy. Its contribution is specifically to the policy update step. Understanding the overall RLHF pipeline from SFT through reward modeling through RL is necessary context.

**KL divergence.** KL divergence between the current and old policy is closely related to the magnitude of IS weights. When IS ratios are close to one, KL is small. When IS ratios are large, KL is large. CISPO's clipping of IS ratios implicitly bounds the KL divergence per step, playing the same role that the KL penalty plays in other RL algorithms — keeping updates small.

**Asynchronous SGD and distributed training.** The asynchronous sampling pipeline in CISPO has parallels to asynchronous SGD, where different workers compute gradients on slightly stale parameters. The theoretical and practical literature on asynchronous SGD — particularly on how much staleness is tolerable before gradients become counterproductive — is relevant background for designing CISPO's asynchronous pipeline.

**Stop-gradient operations.** The stop-gradient trick appears in many other machine learning contexts: target networks in DQN, the prior in variational autoencoders, momentum encoders in contrastive learning. CISPO's use of stop-gradient on IS weights is a new application of a well-established technique.

**MaxRL.** MaxRL is a framework that extends these ideas further, exploring how to maximize throughput in RL-based LLM training by rethinking the full training pipeline from sampling architecture to objective design. CISPO can be seen as one step along the path toward the more comprehensive efficiency gains that MaxRL targets.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| Importance sampling | A technique to reuse data from an old policy by weighting each sample by how likely it is under the current policy |
| IS ratio | Current policy probability divided by old policy probability for the same action; close to 1 means data is fresh, far from 1 means data is stale |
| IS variance problem | Large IS ratios cause gradient estimates to be wildly noisy, destabilizing training |
| Stop-gradient | A trick that lets you use a computed value without propagating gradients back through it; the value acts as a constant from the gradient's perspective |
| CISPO core idea | Compute the IS ratio to decide whether to clip, but apply stop-gradient so the ratio does not enter the gradient calculation |
| PPO vs CISPO clipping | PPO clips the IS ratio times the advantage and backpropagates through the ratio; CISPO clips using the ratio as a gate but treats the ratio as a constant during backprop |
| One backward pass saved | Without gradients flowing through the IS ratio, one full backward pass through the old policy network is eliminated per training step |
| Asynchronous sampling | Data generation and gradient updates run simultaneously rather than taking turns; possible because CISPO's gradient is robust to moderate data staleness |
| 2x throughput | Combining the saved backward pass and asynchronous parallelism roughly doubles training throughput without reducing final performance |
| Bias-variance tradeoff | CISPO's stop-gradient introduces a small bias (gradient ignores IS scaling) in exchange for dramatically lower variance and lower compute cost |
| When clipping fires | If IS ratio falls outside a window around 1 (e.g., 0.8 to 1.2), the data point is gated out of the gradient — this prevents the worst stale data from influencing the policy update |
| No performance loss | Despite the approximation, CISPO achieves similar final benchmark performance to PPO; the efficiency gain is essentially free in terms of output quality |

---

## Up Next

→ **MaxRL** — a comprehensive framework for maximizing reinforcement learning throughput in LLM training, building on the async sampling and objective simplification ideas that CISPO demonstrates. MaxRL takes these efficiency principles further and explores how the full training pipeline — from hardware placement to reward computation to gradient synchronization — can be redesigned around throughput as the primary constraint.
