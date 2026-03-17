# Dr. GRPO — GRPO Done Right

> **Sources used:**
> - Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
>   Models*, DeepSeek AI 2024 — [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
> - Liu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, ByteDance
>   Seed 2025 — [arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)
> - Zeng et al., *GRPO Done Right*, Hugging Face Blog 2025 —
>   [huggingface.co/blog/grpo-done-right](https://huggingface.co/blog/grpo-done-right)
> - Schulman et al., *Proximal Policy Optimization Algorithms*, OpenAI 2017 —
>   [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
> - DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement
>   Learning*, 2025 — [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)

---

## The Big Idea

Group Relative Policy Optimisation (GRPO) is the reinforcement learning algorithm that powers
models like DeepSeek-R1. It works by sampling multiple responses to the same prompt, comparing
their rewards, and nudging the model to produce more of what earned higher scores and less of
what earned lower scores. On paper this sounds clean and sensible. In practice, however, the
original GRPO formulation contains a subtle but serious flaw in the way it computes and
aggregates the training loss.

The flaw is not obvious — it hides inside a denominator — but its consequences compound over
thousands of training steps and ultimately corrupt the quality of the learned model. The model
ends up learning to generate longer and longer responses not because longer responses are more
helpful or more accurate, but because the loss computation assigns them more gradient signal
simply by virtue of having more tokens.

Dr. GRPO, whose name is a play on "GRPO Done Right", is a corrected version of the algorithm
that addresses this flaw directly. The fix boils down to a single conceptual change: normalising
the loss correctly so that every token contributes equally to the gradient signal, regardless of
how long or short the response it belongs to happens to be. This one change — often implemented
in just a handful of lines of code — makes training more stable, prevents a pathological
behaviour called length hacking, and produces models whose reasoning quality is genuinely better
rather than merely longer.

Understanding Dr. GRPO requires understanding the bug it fixes, which in turn requires
understanding how GRPO computes its loss. This guide walks through each layer in plain language,
without assuming any prior knowledge of the loss computation details. The goal is to build up an
intuition for why the bug exists, why it matters, what the fix does, and what happens in
practice when you apply it.

GRPO itself emerged as an alternative to Proximal Policy Optimisation (PPO) for training
language models. PPO is effective but computationally expensive because it requires maintaining
a separate critic (value function) network alongside the policy network. GRPO eliminates the
critic by using the group of sampled responses as a self-contained baseline. Instead of asking
"how good is this response compared to what the value network predicts?", GRPO asks "how good
is this response compared to the other responses we just sampled for the same prompt?" This is
more efficient and often easier to implement, which is why it became attractive for large-scale
training. But cutting out the critic introduced new failure modes that were not fully anticipated
at the time, and the length bias problem is one of them.

---

## Real-Life Analogy

Imagine a school where teachers grade essays on a scale from zero to ten. The goal is to
encourage students to write better essays, so the grading system is supposed to reward quality
of thinking. However, someone has accidentally programmed the grade spreadsheet to add up each
sentence's individual contribution rather than averaging them across all the sentences in the
essay. An essay with fifty sentences therefore naturally accumulates more raw points in the
spreadsheet than an essay with five sentences — even if each sentence in the short essay is
brilliantly written and each sentence in the long essay is mediocre.

Under this system, a teacher marks the content of each sentence fairly: a sharp, precise
sentence might get 0.8 out of 1.0, while a vague, repetitive sentence might get 0.3 out of
1.0. But the final grade reported to the student is the sum of all sentence scores, not the
average. This means a student who writes fifty vague sentences scores 50 times 0.3, which
equals 15, while a student who writes five sharp sentences scores 5 times 0.8, which equals 4.
The first student wins despite writing worse sentences in every individual case.

A diligent student who catches on to this bug will quickly learn that the way to get a high
final grade is not to write better sentences but to write more of them. Even if many sentences
are redundant, rambling, or only loosely connected to the question, the raw totals keep
climbing. The student who writes a crisp, precise five-sentence essay that nails the question
gets a low total score simply because there were fewer sentences to accumulate.

This is exactly the bug in the original GRPO loss. Longer responses accumulate more gradient
signal simply because they have more tokens. Shorter responses — even correct, precise ones —
get proportionally less gradient pushed back into the model. Over training, the model learns the
wrong lesson: longer is better. The quality of each token matters, but the quantity of tokens
matters even more in terms of how much each response shapes the model's future behaviour.

Dr. GRPO fixes the spreadsheet. It divides the total score by the number of sentences before
reporting the final grade. Now each sentence contributes equally, short essays are judged fairly
on their content quality, and students have no incentive to pad their work with filler. A
student who writes five brilliant sentences and a student who writes fifty average sentences can
now be directly compared on the quality of their thinking, not on who wrote more.

The analogy is deliberately simple, but the consequences it captures are entirely real. In
language model training, the "student" is the model, the "sentences" are tokens, the "grade"
is the gradient update, and the "filler sentences" are the padding and repetitive content that
length-hacking models learn to generate.

---

## The Length Bias Problem in GRPO — Deep Dive

To understand the length bias problem precisely, it helps to walk through what GRPO actually
does during a single training step.

For a given prompt, GRPO samples a group of responses from the current model — typically between
four and sixteen completions. Each completion gets a reward signal from some external reward
function, which might be a trained reward model that scores helpfulness, or a rule-based
verifier that checks whether the answer to a math problem is numerically correct. GRPO then
computes a group-relative baseline by averaging the rewards across the group, and for each
response it calculates an advantage: how much better or worse this particular response was
compared to the group average. A response with a positive advantage should be reinforced; a
response with a negative advantage should be suppressed.

The loss for each response is computed token by token. For every token in the response, GRPO
considers the log-probability the model assigned to that token and multiplies it by the
advantage for that response. These per-token contributions are then summed across all the tokens
in the response to get the total loss contribution for that response.

Here is where the length bias enters with full force. If one response is five hundred tokens
long and another is five tokens long, and both receive the exact same advantage — say, both were
equally correct answers to the prompt — the five-hundred-token response contributes roughly one
hundred times more to the overall gradient. The model's weights get nudged very heavily based on
the long response, and barely at all based on the short response. The reward signal was
identical, but the gradient signal was not.

This asymmetry has a compounding effect over training. The model is implicitly being trained
with this message: responses like the long one deserve one hundred times more weight in shaping
my future behaviour than responses like the short one. Since the long response gets so much more
gradient, the model gradually shifts its probability distribution toward generating outputs that
look like the long response, regardless of whether that length was actually helpful.

The problem is not just an abstract fairness concern — it actively corrupts the quality signal.
Suppose you are training a reasoning model and you have two responses to a math problem.
Response A is five tokens: "Yes, since 6 plus 4 equals 10." Response B is five hundred tokens
of elaborate working-out that arrives at the wrong answer. Even if the reward function correctly
gives Response A a higher reward than Response B, the sheer volume of gradient signal from
Response B may dominate the weight updates in ways that the reward difference cannot fully
compensate for. The model sees five hundred times as much gradient from the wrong response as
from the right one, and over many steps, this drags it toward verbose-but-wrong behaviour.

There is also a within-group effect. Even when all responses in a group are of comparable
quality, if the positive-reward responses happen to be longer than the negative-reward responses
in a particular batch, the positive-reward gradient updates will be larger simply because of
length. The reward signal is contaminated by a length-correlated noise source, making it much
harder for the model to learn reliably.

The length bias problem is especially insidious because it is not obviously wrong on the
surface. Summing over tokens is a natural thing to do — it is how cross-entropy loss works in
standard language model training. The mistake is applying that same summing logic in the
reinforcement learning context without accounting for the fact that responses are now being
compared against each other, and comparisons should be made on a per-token basis, not a
per-response-sum basis.

### Why Short Correct Responses Lose

Consider a group with four responses to a simple question. Three of them are correct and terse:
around ten tokens each. One is incorrect and verbose: around three hundred tokens. The reward
function correctly assigns positive advantage to the three short correct responses and negative
advantage to the long incorrect one.

In terms of raw gradient contributions:
- Each short correct response contributes 10 tokens of positive gradient.
- The long incorrect response contributes 300 tokens of negative gradient.

The total positive gradient from three correct responses is 30 token-units. The total negative
gradient from one incorrect response is 300 token-units. The negative direction dominates by a
factor of ten, even though the majority vote of the group is correct. The model gets pushed
more in the direction of suppressing the style of the correct responses than in the direction
of reinforcing them, which is the opposite of what we want.

---

## Per-Token Loss Normalization — Deep Dive

The conceptually cleanest fix for length bias is per-token loss normalisation. The idea is
straightforward: instead of summing the per-token loss contributions across a response and
using that raw sum as the response's loss, divide by the number of tokens in the response to
get a per-token average.

When you sum across all tokens in a response, the magnitude of the loss contribution scales
linearly with length. A fifty-token response with high per-token loss can contribute less total
gradient than a five-hundred-token response with low per-token loss, even if both responses
deserve the same weight in training. This is the bug.

When you instead average across all tokens in a response, every response has the same budget
of gradient signal to contribute to a training step. A five-token response and a five-hundred-
token response each contribute exactly one unit of normalised loss, and the question of which
one influences the model's weights more is decided entirely by the magnitude of the advantage —
which is determined by the reward, which is what we actually care about.

Per-token loss normalisation makes the gradient signal reward-driven rather than length-driven.
If two responses have the same advantage, they push the model's weights by the same amount. If
one response has a higher advantage (earned a better reward), it influences the model more.
Length is no longer a factor.

### What Counts as an Effective Token

There is an important subtlety about what counts as the number of tokens for normalisation.
The denominator should be the number of effective tokens: tokens that the model genuinely
generated as part of its response and that carry semantic content.

Specifically, these token types should be excluded from the effective count:

- **Prompt tokens** — the input given to the model, not part of what it generated
- **Padding tokens** — appended to make batches uniform in length, no semantic content
- **Special separator tokens** — depending on the template format, these may not represent
  meaningful decisions by the model

Only tokens in the generated output, before any padding, should count toward the effective
token total. This requires maintaining a boolean mask per sequence that marks which positions
are effective. The denominator for normalisation is the number of True values in this mask for
the given response.

### Zero Effective Tokens

A robust implementation must handle the edge case where a response produces zero effective
tokens — for instance, if the model immediately generates an end-of-sequence token and
produces no content. Such responses should be excluded from the loss computation entirely,
rather than contributing a zero or triggering a division-by-zero error.

### Effect on Learning Rate

Per-token normalisation systematically reduces the magnitude of gradient contributions
compared to unnormalised summing. If you switch from unnormalised to normalised GRPO, the
effective gradient magnitude per step decreases. Practitioners sometimes compensate by
adjusting the learning rate upward to maintain the same effective step size. This is a minor
engineering consideration, but worth noting when comparing experiments across the two setups.

---

## Denominator Normalization — Deep Dive

Per-token normalisation handles the within-response length bias, but there is a second
normalisation concern at the level of the group itself. In GRPO, multiple responses are sampled
per prompt, and the losses from all of them are combined into a single training signal.

In the original GRPO formulation, after computing the per-response loss (as a sum over tokens),
the group-level aggregation divides by the number of responses sampled. If you sample eight
responses per prompt, the group-level loss is the average of the eight response losses. This
sounds natural, but it interacts badly with within-response length variation.

Consider a group of eight responses where seven are brief, correct answers using around ten
tokens each, and one is a long, wrong answer using five hundred tokens. When you sum the eight
per-response losses before dividing by eight, the single long wrong answer can dominate the
gradient update for that prompt. Its raw loss magnitude, driven by its five-hundred-token
length, can easily exceed the combined magnitude of the seven correct short answers. Dividing
by eight normalises across response count but does nothing to normalise across the massive
variation in raw loss scale.

### The Group-Level Token Count

Dr. GRPO's approach is to normalise at the token level across the entire group. Rather than
computing a per-response average loss and then averaging those across responses, the total loss
across all responses is divided by the total number of effective tokens across the entire group.

This gives every individual token — from any response in the group — equal weight in the final
gradient. A response that generated five hundred tokens contributes proportionally more gradient
than one that generated ten tokens, but only in proportion to its actual content, not due to
any bonus from being longer.

When you combine per-response token normalisation and group-level token normalisation, the
combined effect is that the gradient signal for each training step is proportional to the
advantage, and only to the advantage. Length, at both the individual response level and the
group composition level, is no longer a confounding variable. The model receives clean,
reward-proportional feedback.

### Implementation of Group Normalisation

The implementation requires computing the total effective token count across all responses in
the group before any per-response losses are aggregated. This is a modest additional
bookkeeping requirement. Most training frameworks already have per-token masks available for
padding removal, so extending this to group-level counting is a small additional step.

---

## Why This Matters for Reasoning Models — Deep Dive

The length bias problem is particularly severe for reasoning models, and this is not a
coincidence. It is a direct consequence of what reasoning models are trained to do and how they
do it.

Reasoning models are trained specifically to produce chain-of-thought outputs: extended
step-by-step working-out that makes the model's thought process explicit before it commits to
a final answer. These chains can be hundreds or even thousands of tokens long, especially for
complex mathematical or logical problems.

When you apply GRPO without proper normalisation to a model that is simultaneously learning to
reason and to generate text, you create a feedback loop that pulls the training in the wrong
direction. The model is supposed to be rewarded for correct final answers, but the gradient
signal it actually receives is heavily amplified by the length of the response.

### The Feedback Loop

Any response that happened to be correct and long gets disproportionately emphasised. Any
response that happened to be correct and short gets de-emphasised. Over many training steps,
this creates pressure toward longer reasoning chains, regardless of whether the extra length
improves the quality of the reasoning.

The problem compounds because the model's output distribution determines what gets sampled
during training. If the model is being pushed toward longer outputs, the responses sampled in
subsequent training steps will be longer on average, which will in turn generate even larger
raw loss values, which will push the model even further toward length. This is a positive
feedback loop — a runaway process that does not stabilise until it hits some external ceiling,
such as a context length limit or a hard maximum response length constraint.

### Verbosity Drift in Practice

This pathology has been directly observed and documented in reasoning model training. It goes
by several names: verbosity drift, length hacking, padding behaviour. Models trained without
normalisation exhibit a systematic increase in average response length over the course of
training that is not matched by a commensurate increase in answer quality.

The model learns to fill space. Each additional token it generates is, in some sense, a bet
that more tokens means more gradient, means more training signal, means the model gets updated
more in the direction of generating responses like this one.

### Inference Cost Implications

The irony is sharp and practically important. Reasoning models are supposed to be precise,
step-following thinkers. A reasoning model that has internalised length hacking is not thinking
more carefully — it is generating more words while thinking at roughly the same depth. The cost
at inference time is real: longer responses require more computation, higher memory bandwidth,
and greater latency. For production systems where inference costs are significant, a model that
has learned to pad its reasoning chains represents a direct economic cost that has nothing to do
with quality improvements.

Dr. GRPO breaks the feedback loop by removing the gradient amplification that length provides.
With proper normalisation, the model receives the same gradient signal for a three-hundred-token
reasoning chain and a thirty-token direct answer, as long as their rewards are the same.

---

## Before vs After Dr. GRPO — Deep Dive

The differences between training with original GRPO and training with Dr. GRPO are measurable,
consistent, and appear both during training and at inference time. They manifest in several
distinct ways that practitioners can monitor and evaluate.

### Response Length Distributions During Training

With original GRPO, the distribution of response lengths tends to shift rightward over training.
Average length increases, and the distribution develops a longer tail of very verbose responses.
This shift often accelerates after the model discovers that certain verbose response patterns
receive consistently positive rewards, even when the verbosity itself is not the reason for
the positive reward.

With Dr. GRPO, average response length is much more stable during training. The length
distribution reflects the task distribution: problems that require long reasoning chains produce
long responses, while problems with simple answers produce short responses.

### Quality Versus Length Correlation

One diagnostic to run after training is to examine whether higher-quality responses tend to be
longer or shorter. With original GRPO, this correlation is often positive but for the wrong
reason: the model has been trained to be verbose, and its rewards tend to be decent because the
task rewards are somewhat correlated with effort. The correlation is partially spurious.

With Dr. GRPO, the correlation between quality and length is much weaker and more task-specific.
Simple tasks produce both short high-quality responses and short low-quality responses. Complex
tasks produce both long high-quality responses and long low-quality responses. Length is
determined by content, not by training pressure.

### Training Stability

With original GRPO, gradient magnitude spikes occur regularly whenever a batch contains several
unusually long responses. These spikes can destabilise training, causing sudden large weight
updates that partially undo the gains from previous steps. Managing these spikes requires either
aggressive gradient clipping (which can also suppress legitimate strong learning signals) or a
very conservative learning rate (which slows convergence).

With Dr. GRPO, gradient magnitudes are naturally bounded by the per-token normalisation.
Individual responses cannot spike the gradient regardless of their length. Training is more
predictable and easier to tune hyperparameters for.

### Reward Score Trajectories

In both setups, the reward score tends to increase over training — that is the point of the
algorithm. But the character of the increase differs. With original GRPO, reward scores can
increase rapidly in ways that outpace genuine quality improvement, partly because the model is
learning to generate long responses that happen to score well with reward models that themselves
have some length bias.

With Dr. GRPO, reward score increases are more tightly coupled to actual quality improvements,
because the training signal has been cleaned of the length amplification effect.

### Final Model Behaviour at Inference

At the end of training, models trained with Dr. GRPO tend to produce responses whose length is
tightly calibrated to task difficulty. Hard problems get long, detailed reasoning chains; easy
problems get short, direct answers. Models trained with unnormalised GRPO tend to produce
somewhat verbose responses across the board, because the implicit training pressure toward
verbosity applied to all prompts uniformly.

---

## Implementation Details — Deep Dive

The changes that Dr. GRPO makes to the loss computation are conceptually simple, even though
they have significant practical consequences. This section describes each change in plain
language, tracing through the full computation step by step.

### Step 1: Per-Token Loss Contributions (Unchanged)

For each response in the group, and for each generated token in that response, compute a loss
contribution. This contribution is proportional to the log-probability of the token under the
current policy, multiplied by the advantage of the response. The advantage is a scalar for the
whole response — it does not vary by token. The log-probability does vary by token. This step
is identical in original GRPO and Dr. GRPO.

### Step 2: Within-Response Aggregation (The Key Change)

In original GRPO: sum all per-token contributions for a response to get the total response loss.

In Dr. GRPO: compute the mean instead of the sum — divide the total by the number of effective
tokens in the response. Effective tokens are generated tokens that are not padding and not
prompt tokens. Only the tokens that the model actually decided to produce as part of its output
should count.

This change requires maintaining a boolean mask for each sequence identifying which positions
are effective tokens. The denominator is the count of True values in this mask. If this count
is zero, the response should be skipped entirely rather than contributing a zero or causing a
division-by-zero error.

### Step 3: Group-Level Aggregation (Second Change)

In original GRPO: average the per-response losses across the N responses in the group.

In Dr. GRPO: divide the sum of all per-token contributions across all responses by the total
number of effective tokens across all responses. This means every token in the group — from
any response — gets equal weight in the final gradient, regardless of which response it came
from or how long that response was.

### Step 4: Clipping and Ratio Computation (Unchanged)

Dr. GRPO does not change how the probability ratio between the current policy and the reference
policy is computed, nor does it change how clipping is applied. The standard GRPO clipping
mechanism (keeping the ratio within a range around 1.0 to prevent overly large updates) remains
in place. Only the normalisation of the loss changes.

### Step 5: Handling Edge Cases

Robust implementations need to handle several edge cases:

- Responses with zero effective tokens should be excluded from the loss computation.
- The group-level token count should be computed before any response exclusions to ensure
  the denominator is consistent.
- If all responses in a group are excluded, the entire group's gradient should be zeroed out
  rather than skipped, to avoid biasing toward prompts where the model produces output.

### What Does Not Change

The GRPO algorithm's outer structure — sampling a group of responses, computing rewards,
computing advantages, performing a clipped policy gradient update — is entirely unchanged. The
change is only in how the loss for each response is computed and how those losses are combined
across the group. This makes Dr. GRPO backward-compatible with any existing GRPO
infrastructure: the change is surgical and localized.

---

## How it Works in Practice

When you actually run a training loop with Dr. GRPO, the changes manifest in a number of
observable ways that practitioners can monitor directly.

### Loss Curve Interpretability

The per-step training loss is more interpretable with Dr. GRPO because it no longer fluctuates
based on the lengths of responses in the current batch. With unnormalised GRPO, a batch where
the model generates longer responses shows a higher reported loss than a batch where it
generates shorter responses, even if the quality distribution is identical. This makes the loss
curve noisy and hard to interpret as a signal of training progress.

With Dr. GRPO, the per-step loss is normalised and therefore reflects actual changes in
reward-weighted log-probability, making it a more useful diagnostic.

### Gradient Norm Monitoring

Training practitioners routinely monitor gradient norms to detect instability. With unnormalised
GRPO, gradient norms can spike by an order of magnitude when a batch contains several very long
responses, triggering gradient clipping and potentially destabilising the training trajectory.

With Dr. GRPO, gradient norms are bounded by the per-token normalisation and are much more
consistent across batches. The clipping threshold can be set more conservatively and reliably.

### Response Length Tracking

A useful metric to track during training is the average response length per batch, plotted
alongside the average reward per batch. With unnormalised GRPO, these two quantities tend to
become positively correlated over training, even on tasks where length should not matter.

With Dr. GRPO, the correlation is much weaker. The average reward should increase because the
model is genuinely learning to be more correct. The average response length should remain
relatively stable, varying based on task difficulty rather than training step count.

### Interaction With Context Length Limits

Many training setups impose a maximum response length, after which generation is truncated.
With unnormalised GRPO, models are implicitly rewarded for hitting this length limit, because
longer responses get more gradient. They therefore learn to generate right up to the edge of
the context window on many prompts. This is wasteful and creates pathological training dynamics
because the truncated responses have a distorted loss landscape.

With Dr. GRPO, models have no incentive to pad up to the context limit, and truncation becomes
less frequent, simplifying the training dynamics considerably.

### Checkpoint Evaluation

When evaluating checkpoints on held-out prompts, models trained with Dr. GRPO tend to show a
cleaner relationship between problem difficulty and response length. Hard benchmark problems get
long responses; easy ones get short responses. This is the desired behaviour.

Models trained with unnormalised GRPO tend to produce uniformly long responses across difficulty
levels, because the length pressure was applied uniformly during training regardless of actual
task requirements.

---

## Common Misconceptions

Several misconceptions arise frequently when practitioners first encounter Dr. GRPO, and
addressing them directly helps clarify what the fix actually does and does not accomplish.

### Misconception 1: Per-Token Normalisation Penalises Longer Responses

This is not correct. Normalisation does not penalise length — it removes the implicit reward
for length. A long response and a short response with the same advantage will receive the same
gradient push. The model will still generate long responses when the task content requires them.
A hard proof that takes three hundred tokens to complete will still take three hundred tokens;
the model is not being told to be shorter, only to be length-neutral in how its learning signal
is allocated.

The difference is that the model learns to use the length that serves the answer, rather than
using length as a strategy in itself.

### Misconception 2: Length Bias Only Matters in Extreme Cases

This underestimates the compounding nature of training dynamics. Even modest differences in
average length between positive-reward and negative-reward responses in a group create a
systematic bias that accumulates over thousands of training steps. A bias that seems small in
any single step can shift the model's behaviour significantly over the course of training.

Small consistent biases are often more dangerous than large intermittent ones precisely because
they are harder to notice and easier to dismiss as noise.

### Misconception 3: Gradient Clipping Is a Sufficient Remedy

Gradient clipping prevents training instability due to large gradient magnitudes. It does not
address the directional bias that length introduces. Even with very aggressive clipping, if long
responses systematically contribute more gradient signal than short responses in the correct
direction, the model will still drift toward verbosity over time. Clipping reduces the magnitude
of individual problematic steps; it does not change the direction in which the model is being
systematically pushed. You need to fix the normalisation, not just clip the consequences.

### Misconception 4: Dr. GRPO Prevents Models From Learning Long Reasoning Chains

Some researchers worry that normalising away length effects will prevent the model from learning
to generate extended chain-of-thought reasoning. This concern is unfounded. Chain-of-thought
reasoning is valuable because it actually helps get correct answers to hard problems — the
reasoning process earns a better reward than guessing without reasoning. With Dr. GRPO, the
model can still learn that long reasoning chains are useful; it just learns this because the
reasoning genuinely improves the reward. The incentive to reason carefully is preserved; only
the spurious incentive to be verbose for its own sake is removed.

### Misconception 5: Dr. GRPO Is a Completely Different Algorithm From GRPO

Dr. GRPO is better understood as a corrected implementation of GRPO. The outer algorithm
structure — sample a group, compute advantages, update the policy with a clipped objective — is
entirely unchanged. What changes is the mechanics of how the loss is computed and aggregated.
Dr. GRPO is GRPO with the loss normalisation fixed. Existing GRPO code can be updated to
Dr. GRPO with a small, targeted change.

### Misconception 6: All Implementations of GRPO Have This Bug

Some implementations of GRPO, including some that appeared before the Dr. GRPO paper, already
apply correct per-token normalisation without labelling it as such. The naming of Dr. GRPO
drew attention to the bug and the fix, but the fix itself was discovered and applied
independently by several groups. If you are using an existing GRPO implementation, it is worth
checking whether the loss is summed or averaged over tokens — this is the critical question.

---

## Connections to Other Topics

Dr. GRPO connects to several broader themes in language model training and reinforcement
learning that are worth understanding in context.

### Reward Scale Sensitivity in Reinforcement Learning

The length bias problem is a specific instance of a general challenge in RL called reward scale
sensitivity. Whenever the signal used to update a policy is not properly normalised, the policy
can learn to optimise for the scale or magnitude of the signal rather than its direction. In
GRPO, the scale is set by response length, and the policy learns to optimise for length. The
fix is to normalise the signal so that its scale is invariant to response length. This is a
general principle: training signals should be invariant to irrelevant confounders.

### Connection to PPO and the KL Constraint

PPO, the other main RL algorithm used for language model training, also sums log-probabilities
over tokens when computing its policy gradient. However, PPO includes a KL penalty term that
keeps the model close to its reference distribution, and this penalty provides some indirect
control on verbosity by preventing the model from drifting too far from the original SFT model's
length distribution. GRPO operates without this reference model constraint, which is part of
what makes it efficient. But removing the constraint also removes the indirect length control
that the KL penalty provides. This makes proper normalisation even more important in GRPO-family
algorithms than in PPO-family algorithms.

### Connection to Reward Hacking

In RLHF, reward hacking refers to the model learning to exploit quirks in the reward model to
score highly without actually being more helpful or accurate. Length hacking is a specific form
of reward hacking where the model exploits the implicit gradient-scale reward for verbosity.
Dr. GRPO prevents this specific form of hacking, but it does not address other forms of reward
hacking (such as finding outputs that fool the reward model for semantic rather than length
reasons). Preventing reward hacking more broadly requires careful reward model design, output
diversity measures, and sometimes additional constraints on the policy update.

### Connection to DAPO

DAPO — Decoupled Clip and Dynamic Sampling Policy Optimisation — is the next major development
in the GRPO family. DAPO incorporates the normalisation fix from Dr. GRPO and extends it with
several additional innovations. The decoupled clipping mechanism applies different clip ratios
for responses with positive advantages versus negative advantages, giving the model more freedom
to increase the probability of good responses while being more conservative about suppressing
bad ones. The dynamic sampling mechanism skips prompts for which all responses in the group
received the same reward.

Understanding Dr. GRPO is a natural and necessary prerequisite for understanding DAPO.

### Connection to Token-Level Policy Gradient Methods

Dr. GRPO is closely related to formulations of the policy gradient applied at the token level
rather than the sequence level. When you normalise by token count, you are effectively treating
each token as a separate decision, each receiving a gradient signal proportional to the
response-level advantage. This framing connects GRPO to other token-level RL methods and makes
clear that the algorithm is fundamentally about training each token-generation decision, not
each sequence-generation decision.

### Connection to Training Efficiency

By producing cleaner, length-unbiased gradient signals, Dr. GRPO allows training to converge
faster to genuinely good reasoning behaviour. The model spends fewer training steps exploring
the spurious strategy of verbosity and more training steps learning what actually earns rewards.
This translates to lower compute costs for achieving a given level of reasoning quality — a
significant practical benefit given the enormous compute budgets involved in training state-of-
the-art reasoning models.

---

## Key Takeaways

| Property | Original GRPO | Dr. GRPO |
|---|---|---|
| Loss per response | Summed over all tokens | Averaged over tokens (per-token mean) |
| Group aggregation | Average across N responses | Normalised by total effective tokens |
| Gradient scale | Proportional to response length | Proportional to advantage (reward) only |
| Length bias | Present — longer responses dominate | Removed — all tokens weighted equally |
| Padding token handling | May contaminate denominator | Explicitly masked out |
| Prompt token handling | May contaminate denominator | Excluded from effective count |
| Training stability | Gradient spikes on long batches | Bounded gradient norms throughout |
| Verbosity drift | Common and often severe | Prevented by normalisation |
| Quality signal | Quality and length conflated | Quality signal is clean, reward-driven |
| Reasoning chain learning | Length-hacking risk is high | Chains grow only when task requires it |
| Response length at inference | Biased upward by training pressure | Calibrated to task difficulty |
| Implementation complexity | Simple but subtly incorrect | Slightly more careful masking required |
| Gradient clipping sensitivity | Requires aggressive tuning | Less sensitive, more predictable |
| Relationship to PPO | No KL constraint, length bias present | No KL constraint, bias removed |
| Downstream algorithms | Basis for DAPO | DAPO builds on Dr. GRPO foundations |

The single most important takeaway is this: the original GRPO sums the token-level loss across
all tokens in a response, then averages across responses. Dr. GRPO normalises by the number of
effective tokens, making the loss a per-token mean rather than a per-response sum. This one
change — dividing by length instead of ignoring length — is the entire fix. Everything else
(the group sampling, the advantage computation, the clipping, the reward function) remains the
same. The fix is small, surgical, and correctness-driven, and its consequences for training
quality are large.

---

## Up Next

The next topic is **DAPO** — Decoupled Clip and Dynamic Sampling Policy Optimisation. DAPO
takes the corrected loss normalisation from Dr. GRPO as its starting point and introduces three
additional innovations on top of it.

First, DAPO decouples the clipping ratios for responses with positive advantages and responses
with negative advantages. The intuition is asymmetric: the model should be relatively free to
increase the probability of responses it got right (exploration is good), but should be more
conservative about suppressing responses it got wrong (excessive suppression can destabilise
training). Using separate clip parameters for the two cases gives finer control over this
tradeoff.

Second, DAPO introduces dynamic sampling, which identifies and skips prompts for which all
responses in the group received the same reward. When all group members have the same reward,
the advantages are all zero, and the policy gradient signal is zero — the group provides no
comparative learning signal whatsoever. Skipping these groups is more efficient than processing
them, and doing so concentrates training compute on prompts where the model actually has
something to learn.

Third, DAPO frames the entire algorithm explicitly as a token-level policy gradient method,
making the connection between sequence-level reward signals and token-level training decisions
more precise and theoretically grounded.

Understanding Dr. GRPO — and specifically the insight that token-level normalisation is the
correct way to aggregate the policy gradient signal — is essential preparation for DAPO. DAPO
assumes the normalisation fix is in place and builds its additional contributions on top of a
correctly-normalised foundation.
