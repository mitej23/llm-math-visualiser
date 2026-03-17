# RLOO — REINFORCE Leave-One-Out

> **Sources used:**
> - Kool et al., *Buy 4 REINFORCE Samples, Get a Baseline for Free!*, ICLR 2019 — [arxiv.org/abs/1905.05765](https://arxiv.org/abs/1905.05765)
> - Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*, 2024 — [arxiv.org/abs/2402.14740](https://arxiv.org/abs/2402.14740)
> - HuggingFace TRL, *RLOO Trainer documentation* — [huggingface.co/docs/trl](https://huggingface.co/docs/trl)
> - Schulman et al., *Proximal Policy Optimization Algorithms*, 2017 — [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

---

## The Big Idea

RLOO stands for **REINFORCE Leave-One-Out**.
It is an algorithm for fine-tuning language models with reinforcement learning
that sits in a sweet spot between the complexity of PPO and the simplicity
of vanilla REINFORCE.

The core idea sounds almost obvious once you hear it:
when you want to know how good your response is relative to others,
compare it to all the other responses you generated for the same prompt —
but **leave your own response out of that comparison**.

This single change — the leave-one-out baseline — is what gives RLOO
its name and most of its advantages. It produces an advantage estimate
that is **mathematically unbiased**, does not require a separately trained
value model, and does not require the importance sampling ratio clipping
that makes PPO so complex.

The result is an algorithm that is:

- Simpler to implement than PPO or GRPO
- Cheaper to run — no value model, no critic network
- More principled — zero bias in the advantage estimator
- Equally competitive — matches PPO on instruction-following benchmarks

RLOO was formally introduced in the ICLR 2019 paper
"Buy 4 REINFORCE Samples, Get a Baseline for Free!" by Kool, van Hoof,
and Welling in the context of combinatorial optimisation.
It was later adapted for large language model fine-tuning by Ahmadian et al.
in 2024, who demonstrated that it matches or exceeds the performance of PPO
on instruction-following benchmarks while being significantly cheaper to run.

The key insight of the paper is stated in its title:
if you sample multiple responses for the same prompt, you can use the others
to construct a baseline for each one — for free,
without any extra training or any extra parameters.
This is not a trick or an approximation.
It is a statistically principled estimator with excellent theoretical
properties, well-understood bias and variance characteristics,
and a straightforward proof of unbiasedness.

Understanding RLOO means understanding a few interlocking ideas:
what a baseline is and why it matters, why the leave-one-out baseline
is specifically better than alternatives, why you can drop importance
sampling clipping when using RLOO, and how all of this compares
to GRPO and PPO. Each of these ideas is addressed in depth below.

---

## Real-Life Analogy

Imagine a creative writing class with thirty students.
The teacher gives everyone the same prompt —
"Write a short story about a lighthouse keeper" —
and everyone writes their own story.
At the end of class, the teacher reads out each story and the class votes
on how good it is compared to the rest.

Now consider two different ways the class could judge Student Seven's story.

### Method A — GRPO-Like

The teacher asks: "Is this story better than average?"
and computes the average score **including Student Seven's own rating**
in the average.

This is problematic because Student Seven's story is part of the group —
its quality directly influences the average it is being measured against.

- If Student Seven wrote the **best** story in the class, the average is
  pulled upward by their own work, making their advantage appear **smaller**
  than it really is.
- If they wrote the **worst** story, the average is pulled down, making
  their disadvantage appear **smaller** (less negative) than it really is.

Either way, the baseline is contaminated by the very thing
it is supposed to evaluate.

### Method B — RLOO

The teacher asks: "Is this story better than the average of all the
**other** twenty-nine stories?"
Student Seven's own story is left out of the average.

This means the baseline is entirely independent of Student Seven's work.
It truly measures what the rest of the class produced under the same
conditions.

- If Student Seven's story is genuinely better than the others,
  their advantage is clearly positive.
- If it is worse, the advantage is clearly negative.
- The measurement is clean and uncontaminated.

This is the leave-one-out idea.
It is not a complicated mathematical trick.
It is the obvious right way to judge a student's work against their peers —
exclude their own work from the peer average.

The sophistication lies in recognising that this produces a provably unbiased
estimator of the advantage, and in implementing it efficiently
so you do not need to make extra model calls to compute it.

Every time you generate a batch of responses to the same prompt
and use each response's siblings to form its baseline, you are doing RLOO.
The "leave-one-out" part ensures that each baseline is statistically
independent of the response being evaluated —
which is what makes it unbiased in the formal statistical sense.

---

## The Leave-One-Out Baseline — Deep Dive

A baseline in reinforcement learning is a reference value
that you subtract from the raw reward before using it to update the model.

Without a baseline, a raw REINFORCE update would try to increase the
probability of every response that received any positive reward,
even if that reward was below average.
The baseline allows you to ask the more meaningful question:
"Was this response better or worse than expected?"
rather than merely "Did this response receive a positive reward?"

### Three Properties a Good Baseline Must Have

1. It should be a good estimate of what reward the model would expect
   on average for this specific prompt.
2. It should not be correlated with the specific response being evaluated —
   otherwise you are contaminating your measurement.
3. It should have low variance so that the advantage estimates are stable
   and training does not oscillate wildly.

The leave-one-out baseline satisfies all three properties
when you generate multiple responses for the same prompt.

### How It Is Computed

Suppose you generate four responses for a given prompt —
call them R1, R2, R3, and R4.
Each receives a reward score from the reward model —
call them r1, r2, r3, and r4.

- Baseline for R1 = average of (r2, r3, r4)
- Baseline for R2 = average of (r1, r3, r4)
- Baseline for R3 = average of (r1, r2, r4)
- Baseline for R4 = average of (r1, r2, r3)

Each response's baseline is computed from responses it did not generate —
they are statistically independent samples from the same model
given the same prompt.

### Why This Is Unbiased

The crucial property is that the baseline for R1 is computed using only
R2, R3, and R4 — responses that are statistically independent of R1
given the prompt.

They were generated by the same model from the same prompt,
but once generated, each response is an independent sample.
The baseline therefore does not depend on the reward of the response
it is being compared to.
This is the definition of an unbiased baseline in the RL sense.

### Why the Number of Samples Matters

- With k=2 samples: each baseline is computed from just one other — noisy
- With k=4 samples: each baseline is the average of three — reasonable
- With k=8 samples: each baseline is the average of seven — much more stable

In practice, four to eight samples per prompt is a good range.
This is similar to GRPO, which also generates multiple samples per prompt,
but RLOO uses the leave-one-out structure rather than the full-group average.

### The Unbiasedness Proof Sketch

The key mathematical fact:
for any response Ri, the expected value of the leave-one-out baseline
equals the expected reward of any single response from the model
for that prompt.

In other words, the leave-one-out baseline is an unbiased estimator
of the true expected reward — the baseline the model would use
if it could average over infinitely many samples.
This means the advantage estimate has zero bias in expectation.

GRPO's baseline, which includes the response's own reward,
does not have this property.

---

## Why This Is Better Than GRPO's Baseline — Deep Dive

GRPO (Group Relative Policy Optimisation) computes the baseline for each
response as the average reward of the entire group,
**including the response itself**.

At first glance this seems reasonable — the group average is a natural
reference point. But it introduces a subtle bias that RLOO avoids entirely.

### The GRPO Baseline Is Biased

When you include a response's own reward in the baseline,
the advantage estimate is no longer statistically clean.

- If a response received an unusually **high** reward: it pulls the group
  average up, making the advantage **smaller** than it should be.
- If a response received an unusually **low** reward: it pulls the group
  average down, making the advantage **larger** (less negative) than it
  should be.

Both directions of contamination make the advantage estimates systematically
inaccurate. The training signal is weaker and noisier than it needs to be.

### Quantifying the Bias

In a group of k responses, GRPO's baseline for response i includes
(1/k) of that response's own reward.

The advantage computes as:

```
advantage_GRPO = r_i - baseline
               = r_i - (r_i/k + average_of_others)
               = ((k-1)/k) * (r_i - average_of_others)
```

The factor (k-1)/k is less than one —
GRPO **systematically underestimates** the true advantage.

| Group size k | Shrinkage factor (k-1)/k | Signal captured |
|---|---|---|
| k = 2 | 0.50 | 50% |
| k = 4 | 0.75 | 75% |
| k = 8 | 0.875 | 87.5% |
| k = 16 | 0.9375 | 93.75% |

RLOO does not have this shrinkage.
It captures the full advantage signal regardless of group size.

### Why RLOO's Baseline Is Unbiased

RLOO excludes the response's own reward from its baseline,
so there is no self-contamination.

```
advantage_RLOO = r_i - average_of_all_others
```

The expected value of this estimate equals the true advantage of response i —
no shrinkage, no distortion.
This is a strict improvement over GRPO's baseline in terms of statistical
accuracy.

### The Practical Implication

With an unbiased advantage estimate, the model gets a cleaner training
signal. It more accurately learns which responses are genuinely better
than expected and which are genuinely worse.

This translates to more efficient training — fewer steps needed
to achieve the same quality improvement.

The Ahmadian et al. paper (2024) showed that RLOO with this unbiased
baseline matches PPO performance on instruction-following tasks
while using significantly fewer computational resources.

### The Cost Is Zero

Switching from GRPO's baseline to RLOO's leave-one-out baseline costs
nothing. You have already generated all the responses.
You are just computing the average differently — excluding one reward
from each baseline computation rather than including it.
The computational difference is negligible.
The statistical improvement is real.

---

## No Clipping Needed — Deep Dive

One of RLOO's most practically significant advantages is that it does not
need importance sampling ratio clipping.

To understand why this matters, we need to understand what importance
sampling is and why PPO introduced clipping in the first place.

### What Importance Sampling Is

In PPO, the policy that collects data (generates responses) and the policy
that is being updated by gradients are **not the same** at every step.

PPO runs multiple gradient update steps on the same batch of data —
the first step uses the current policy, but subsequent steps use an
increasingly different policy. To correct for this discrepancy, PPO uses an
importance sampling ratio:

```
ratio = probability under new policy / probability under old policy
```

This ratio adjusts the gradient to account for the fact that the data
was generated under different conditions.

### Why Clipping Was Introduced

The importance sampling ratio can become very large if the current policy
has drifted far from the data-collecting policy.
A large ratio means a large gradient update — which can destabilise training
and produce erratic behaviour.

PPO clips the ratio to a range around 1.0, typically [0.8, 1.2],
to prevent these large updates.

The clipping makes training stable but introduces its own complications:

- It is a non-smooth operation, creating discontinuities in the gradient
- The choice of clipping range is a hyperparameter that requires tuning
- The clipping prevents beneficial updates that would otherwise be appropriate
- It adds conceptual complexity that makes PPO harder to reason about

### Why RLOO Does Not Need This

RLOO is an **on-policy algorithm**.

At each training step:
1. Generate a fresh batch of responses using the **current** policy
2. Compute the rewards for all responses
3. Compute leave-one-out advantages
4. Update the policy **once** on this data
5. Discard the batch — never reuse it

Because the data is always generated by the current policy
and the policy is updated exactly once on each batch,
the importance sampling ratio is always exactly 1.0.

When the ratio is 1.0, clipping does nothing —
it clips values that are already in the valid range.
So you can simply **drop the clipping mechanism entirely**.

### Benefits of Dropping Clipping

- **Simpler update rule** — advantage times log probability, nothing more
- **One fewer hyperparameter** — no epsilon to tune
- **Smooth gradient** — no discontinuities from the clipping operation
- **Cleaner theory** — easier to reason about what the algorithm is doing
- **No tension** between "take a big enough step" and "don't destabilise"

### The On-Policy Constraint

Running on-policy means you cannot reuse data across multiple gradient steps,
making each step somewhat less data-efficient than PPO's multi-step updates.

However, this is offset by:

- The simplicity of the algorithm
- The elimination of the value model
- The cleaner gradient from the unbiased baseline

In practice, the Ahmadian et al. paper found that RLOO reaches competitive
performance with PPO even with this constraint.

---

## The Math in Plain English — Deep Dive

The mathematics of RLOO is significantly simpler than PPO,
because there is no value model, no importance sampling ratio, and no clipping.
The core computation can be described in plain language
without any loss of precision.

### The Seven Steps of RLOO

**Step 1 — Generate responses:**
For each prompt in the batch, generate k responses using the current policy.
Each response is a sequence of tokens.
Typical values of k are 4, 6, or 8.

**Step 2 — Score responses:**
Send each response through the reward model to get a scalar reward score.
Now you have k reward scores for each prompt.

**Step 3 — Compute leave-one-out baselines:**
For each response, compute the baseline as the average reward of all other
responses for the same prompt.

```
baseline(r1) = (r2 + r3 + r4) / 3
baseline(r2) = (r1 + r3 + r4) / 3
baseline(r3) = (r1 + r2 + r4) / 3
baseline(r4) = (r1 + r2 + r3) / 3
```

**Step 4 — Compute advantages:**
The advantage for each response is its reward minus its leave-one-out baseline.

- Positive advantage → this response was better than its siblings
- Negative advantage → this response was worse
- Zero advantage → exactly average

**Step 5 — Compute the policy gradient:**
For each response, multiply its advantage by the log probability of that
response under the current policy.
Sum over all tokens in the response.
Average over all responses in the batch.

**Step 6 — Update the policy:**
Apply this gradient to the policy parameters using Adam or similar.

- Responses that outperformed their siblings → increase probability
- Responses that underperformed → decrease probability

**Step 7 — Add the KL penalty (optional but recommended):**
Compute the per-token KL divergence between the current policy
and a reference policy.
Subtract a scaled version of this KL from the reward before computing
advantages.
The scaling coefficient beta controls how far the model is allowed to drift.

### What You Do Not Need

The following things are required by PPO but entirely absent from RLOO:

- A value model
- An importance sampling ratio
- A clipping mechanism
- Off-policy correction
- A separate critic network
- Multiple gradient steps per batch
- GAE (generalised advantage estimation) parameters

The algorithm is genuinely simpler — not just in theory but in implementation.

---

## RLOO vs GRPO vs PPO Comparison — Deep Dive

Understanding where RLOO sits relative to PPO and GRPO requires looking
at several dimensions: the baseline, the presence or absence of a value model,
the importance sampling mechanism, memory requirements, and practical
performance.

### Comparing the Baselines

**PPO:**
Uses a learned value function to estimate the expected reward from any state.
This is a separate neural network trained alongside the policy,
often the same size as the policy, doubling the memory footprint.
The advantage: can generalise across prompts in theory.
The disadvantage: difficult to train well for language model fine-tuning —
tends to underfit or overfit, leading to noisy advantage estimates.

**GRPO:**
Uses the full-group average reward as the baseline.
Simple, requires no extra model.
Flaw: biased — the response's own reward contaminates its own baseline.
Shrinks the effective advantage by the factor (k-1)/k.

**RLOO:**
Uses the leave-one-out average as the baseline.
Equally simple — no extra model, negligible compute cost.
Unlike GRPO's baseline, it is mathematically unbiased.
Gives the full advantage signal without shrinkage.

### Comparing Importance Sampling

**PPO:**
Uses importance sampling ratios to allow multiple gradient steps per batch.
Requires computing new/old policy probability ratios for every token.
Ratios must be clipped to [1-eps, 1+eps] to prevent instability.

**GRPO:**
Also uses clipped importance sampling — inherits this from PPO despite
simplifying other aspects.

**RLOO:**
Does not use importance sampling at all.
On-policy by design: one gradient step per fresh batch.
No ratio computation. No clipping. No associated hyperparameters.

### Comparing Memory Requirements

**PPO:** Requires four models simultaneously:
1. The RL policy being trained (full gradients)
2. The frozen reference policy (no gradients)
3. The reward model (forward pass only)
4. The value / critic model (full gradients)

**GRPO:** Requires three models (same as RLOO — no value model).

**RLOO:** Requires three models (same as GRPO — no value model).

### Full Comparison Table

| Dimension              | PPO                      | GRPO                   | RLOO                  |
|------------------------|--------------------------|------------------------|-----------------------|
| Baseline type          | Learned value model      | Full-group mean        | Leave-one-out mean    |
| Baseline bias          | Low (if well-trained)    | Yes — (k-1)/k factor   | Mathematically zero   |
| Value model required   | Yes                      | No                     | No                    |
| Importance sampling    | Yes (clipped)            | Yes (clipped)          | No                    |
| Clipping required      | Yes                      | Yes                    | No                    |
| On-policy or off       | Off-policy               | Off-policy             | On-policy             |
| Models in memory       | 4                        | 3                      | 3                     |
| Hyperparameter count   | High                     | Medium                 | Low                   |
| Implementation effort  | High                     | Medium                 | Low                   |
| Empirical performance  | Strong                   | Strong                 | Matches PPO           |

### Which to Choose

**Choose PPO** for tasks with very long horizons where step-level value
estimation provides meaningful signal beyond what group comparison can offer.

**Choose GRPO** when you want an off-policy algorithm without a value model
and are willing to accept the (k-1)/k baseline bias.

**Choose RLOO** when you want a simple, principled, memory-efficient algorithm
with no bias in the baseline, no clipping hyperparameter, and no value model
to train or tune. Particularly well-suited for researchers who want to
understand exactly what the algorithm is doing without layers of
approximations stacked on top.

---

## Memory and Speed Advantages — Deep Dive

The practical case for RLOO over PPO comes down significantly to resource
efficiency. Fine-tuning large language models is expensive — in GPU memory,
compute time, and engineering complexity.

### Memory: Eliminating the Value Model

PPO requires four models in memory simultaneously during training.
For a 7B parameter policy:

```
4 models × 7B parameters × 4 bytes (bf16) ≈ 112 GB
```

...before accounting for gradients, activations, and optimiser states.

RLOO eliminates the value model, reducing the simultaneous model count
from four to three.

The savings are not just 25% of model weight memory —
eliminating the value model also reduces:

- Gradient memory (no backward pass through the critic)
- Activation memory (no critic forward pass per batch)
- Compute time (no critic training objective per step)

For very large models (70B+), the difference between needing three vs four
large models can determine whether fine-tuning fits on a given cluster at all.
RLOO makes certain experiments feasible that PPO would require
twice as many GPUs to run.

### Compute: No Importance Sampling Ratio

PPO computes, for every token in every response,
the ratio of the current policy's probability to the old policy's probability.
This effectively requires **two forward passes** through the policy per batch:

1. One to generate responses (standard)
2. One to compute updated probabilities for the importance sampling ratio

RLOO's on-policy design means only **one forward pass** per batch is needed.
The saving is approximately one full policy forward pass per training step.

### Speed: No Clipping Overhead

The clipping operation itself is cheap, but it is embedded in a more complex
objective requiring:

- Storage of old policy probabilities across gradient steps
- Ratio computation for every token
- Minimum of clipped and unclipped objective

RLOO's objective is simply: advantage times log probability, summed and
averaged. This means fewer tensors to track, simpler automatic differentiation
graphs, and faster per-step wall-clock time.

### Engineering Complexity

PPO implementations require careful handling of:

- Value model initialisation, training, and learning rate scheduling
- Importance sampling ratio computation and storage
- Clipping boundary choices and epsilon tuning
- Generalised advantage estimation (GAE) parameters
- Separate data pipelines for on-policy generation
- Handling of variable-length sequences in off-policy settings

The HuggingFace RLHF implementation details post lists 37 distinct
implementation choices for PPO, many of which are subtle and consequential.

RLOO requires none of this complexity.
The algorithm is short enough to explain in a few paragraphs and implement
in a few dozen lines of well-commented code.
Fewer implementation decisions means fewer bugs, faster iteration,
and easier reproduction across teams and codebases.

---

## How it Works in Practice

### Initialisation

- Start with a supervised fine-tuned (SFT) model as the policy
- Keep a frozen copy as the reference policy for KL computation
- Load a reward model trained on human preference data
- No value model is initialised — it is not needed

### The Training Loop

For each iteration:

1. Sample a batch of training prompts (typically 64–256 prompts)
2. Generate k responses per prompt (typically k=4 or k=8) at temperature 1.0
3. Send all responses to the reward model — obtain scalar reward scores
4. Compute leave-one-out baseline for each response
5. Subtract baseline from reward to get the raw advantage
6. Compute per-token KL from the reference policy — aggregate to scalar
7. Subtract beta × KL from the advantage to apply the KL penalty
8. Optionally normalise advantages: subtract batch mean, divide by batch std
9. Compute policy gradient: advantage × sum of log token probabilities
10. Take one gradient step using Adam
11. Discard the batch entirely — never reuse it

### Key Hyperparameters

| Parameter | Typical range | Notes |
|---|---|---|
| k (samples per prompt) | 4 to 8 | More = better baseline, more generation cost |
| beta (KL weight) | 0.01 to 0.1 | Higher = closer to reference policy |
| Learning rate | 1e-6 to 1e-5 | Similar to SFT fine-tuning |
| Batch size | 64 to 256 | Larger = more stable gradient estimates |
| Advantage normalisation | Batch-level z-score | Usually recommended for stability |

### Monitoring During Training

Track these metrics to verify the training is working correctly:

- **Mean reward per batch** — should increase over time
- **Mean KL divergence** — should stay bounded, not explode
- **Fraction of positive advantages** — should be around 50% if working well
- **Reward standard deviation within a group** — measures batch diversity

### Convergence Characteristics

RLOO typically requires fewer training steps than PPO to reach the same
quality level, due to the cleaner gradient signal from the unbiased baseline.

However, because each step generates more data (k responses per prompt),
the wall-clock time per step is longer than a single-sample REINFORCE step.

The overall training time is comparable to PPO or slightly better,
depending on whether generation or gradient computation is the bottleneck.

---

## Common Misconceptions

### Misconception 1: "RLOO is just GRPO with a small fix"

RLOO and GRPO both generate multiple responses per prompt and use group-based
baselines. But RLOO is not a patch on GRPO — they have different theoretical
foundations.

GRPO inherits from PPO's off-policy framework and retains importance sampling
with clipping. RLOO is a pure on-policy REINFORCE algorithm with a specific
leave-one-out baseline that comes with a clean unbiasedness proof.

Calling RLOO a variant of GRPO understates the differences in their
underlying assumptions and mechanisms.

### Misconception 2: "No clipping means RLOO is less stable than PPO"

Stability and clipping are not the same thing.
Clipping was introduced in PPO to handle instabilities caused by off-policy
updates — the risk that the policy drifts far from the data-generating policy
between gradient steps.

RLOO avoids this risk entirely by staying on-policy.
It never needs to correct for a discrepancy between data-generating
and current policies because they are always the same.
The source of instability that clipping addresses simply does not exist
in RLOO.

### Misconception 3: "RLOO wastes compute by generating multiple samples"

Multiple samples per prompt are required to compute the leave-one-out baseline.
But GRPO also requires multiple samples per prompt, and PPO effectively
requires multiple forward passes through the policy for importance sampling.

The total compute cost is comparable.
What RLOO gains is that each sample is used more efficiently —
with an unbiased baseline — whereas GRPO uses them with a biased baseline.

### Misconception 4: "RLOO cannot handle length differences between responses"

RLOO computes rewards at the response level, not the token level.
If responses vary significantly in length and the reward model does not
account for length, longer responses might receive systematically higher
or lower rewards.

This is a real limitation, but it is shared by GRPO and is not unique to RLOO.
Dr. GRPO (the next episode) addresses this specifically.
The leave-one-out mechanism itself is orthogonal to length normalisation.

### Misconception 5: "All responses for a prompt must be on the same GPU"

This is true in the naive implementation — all k responses for a prompt
must be together to compute the leave-one-out baseline.
But in practice, most training frameworks batch responses by prompt anyway,
so this is not a practical constraint.
Responses for a given prompt are generated and processed together naturally.

### Misconception 6: "RLOO is only appropriate for simple tasks"

RLOO was originally applied to combinatorial optimisation problems,
which can be quite complex.
Its adaptation to LLM fine-tuning has shown strong results on
instruction following, mathematical reasoning, and code generation.

There is nothing in the algorithm that limits it to simple tasks.
If anything, the cleaner gradient signal from the unbiased baseline
may help more on complex tasks where the advantage landscape is subtle.

---

## Connections to Other Topics

### REINFORCE

RLOO is a direct descendant of the classic REINFORCE algorithm
from Williams (1992).

REINFORCE says: for each response, multiply its reward by the log probability
of the response and update the policy in that direction.
The problem with vanilla REINFORCE is high variance —
reward scores fluctuate a lot and the gradient estimates are noisy.

RLOO adds a baseline to REINFORCE to reduce variance,
specifically the leave-one-out baseline.
The baseline does not change the expected gradient (it preserves unbiasedness)
but substantially reduces its variance.

Understanding RLOO is understanding REINFORCE plus one well-chosen
design decision.

### PPO

PPO also builds on policy gradient ideas and also uses a baseline
(its learned value function).
But PPO took the design in a different direction: off-policy updates
with importance sampling ratios and clipping.

RLOO is the simpler path: stay on-policy, use a free baseline,
avoid all the complexity that off-policy requires.

### GRPO

GRPO shares the group-sampling structure with RLOO but uses a biased
full-group baseline and retains importance sampling clipping.

Many practitioners encounter GRPO before RLOO.
If you already understand GRPO, RLOO is what you get when you
correct the baseline bias and drop the off-policy mechanism.

### KL Divergence

Like PPO and GRPO, RLOO uses a KL penalty to prevent the trained model
from drifting too far from the reference policy.

Without it, the model can find reward model exploits by generating responses
that the reward model scores highly but that are actually low quality.
RLOO's KL penalty works exactly as in the other algorithms:
subtract a scaled KL divergence from the reward before computing advantages.

### Reward Models

RLOO's advantage computation depends entirely on the reward model's scores
being meaningful. If the reward model assigns similar scores to good and bad
responses, the leave-one-out baseline will not distinguish them
and the advantages will all be near zero.

The quality of the reward model is a prerequisite for RLOO to work.

### Value Models (Absence Thereof)

RLOO's most important connection to the value model topic is negative —
it shows that a value model is not always necessary.

Value models exist to provide a low-variance baseline for advantage
estimation. RLOO demonstrates that you can get an unbiased,
reasonably low-variance baseline without any learned model at all,
simply by generating multiple responses per prompt and using the
leave-one-out average.

### Verifiable Rewards

RLOO is particularly well-suited for verifiable reward settings —
tasks like mathematical reasoning or code generation where correctness
can be checked automatically and the reward is binary
(correct or incorrect).

In such settings, the reward signal is already clean and unambiguous.
The leave-one-out baseline cleanly reflects whether this specific response
was more or less likely to be correct than the model's other attempts.
Several recent mathematical reasoning systems use RLOO or close variants
for exactly this reason.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| RLOO | REINFORCE Leave-One-Out: generate k responses per prompt, use others' rewards as your baseline |
| Leave-one-out baseline | Exclude your own reward from the group average when computing your baseline |
| Unbiased estimator | The advantage estimate has zero systematic error — expected value equals true advantage |
| GRPO bias | GRPO includes the response's own reward in its baseline, shrinking advantages by (k-1)/k |
| No importance sampling | RLOO stays on-policy (one gradient step per fresh batch), so no ratio correction needed |
| No clipping | With no importance sampling ratio, there is nothing to clip — simpler code, fewer hyperparameters |
| No value model | The leave-one-out average replaces the learned value function — saves roughly 25% model memory |
| KL penalty still used | RLOO still applies KL divergence penalty to prevent drifting too far from reference policy |
| Empirical performance | Matches PPO on instruction-following benchmarks with significantly lower resource requirements |
| Best use case | Tasks with clear outcome-level rewards: instruction following, maths reasoning, code generation |
| Key limitation | No built-in length normalisation — longer responses may have inflated or deflated rewards |
| Next: Dr. GRPO | Addresses length normalisation by dividing advantage by response length before comparison |

---

## Up Next

The next topic in this series is **Dr. GRPO** —
a refined version of GRPO that addresses the length normalisation problem
that RLOO and standard GRPO both leave unresolved.

When a model generates responses of different lengths, the cumulative reward
can be influenced by length in ways unrelated to actual quality.
A longer response that is merely verbose may score higher than a concise,
correct response simply because it has more tokens for the reward model
to react to.

Dr. GRPO introduces a normalisation step that divides the advantage by the
length of the response, correcting for this effect and encouraging the model
to be appropriately concise rather than verbose.

Understanding Dr. GRPO requires everything covered in this chapter —
the group-sampling structure, the baseline computation, the relationship
between response length and reward — and adds one targeted fix on top.

By this point in the series, you have the foundation to understand why
length normalisation matters and how a simple division operation can
substantially improve the quality of a fine-tuned model's outputs.

**Up next:** Dr. GRPO — Length-Normalised Group Relative Policy Optimisation
