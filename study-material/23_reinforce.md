# 🎲 REINFORCE — Policy Gradient from Scratch

> **Sources used:**
> - Williams, R.J. (1992). *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.* Machine Learning, 8, 229–256.
> - OpenAI Spinning Up, *Policy Gradient Methods* — [spinningup.openai.com/en/latest/spinningup/rl_intro3.html](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
> - Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 13 — [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
> - Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015) — [arxiv.org/abs/1506.02438](https://arxiv.org/abs/1506.02438)
> - Ziegler et al., *Fine-Tuning Language Models from Human Preferences* (2019) — [arxiv.org/abs/1909.08593](https://arxiv.org/abs/1909.08593)

---

## The Big Idea

REINFORCE is the simplest policy gradient algorithm: after your agent finishes an episode, look at every action it took, and then nudge the model weights so that actions leading to high reward become more probable and actions leading to low reward become less probable. 🎯 It is the conceptual bedrock of every modern LLM training algorithm — PPO, GRPO, and their descendants are all elaborations on this single idea. The critical challenge is that REINFORCE is extremely noisy, and the bulk of modern RL research is essentially a story of variance reduction tricks layered on top of this elegant but wobbly foundation. What makes REINFORCE remarkable is that it was published in 1992 by Ronald Williams — decades before large language models existed — yet its core equation appears unchanged inside DeepSeek-R1's GRPO implementation, inside OpenAI's PPO code, and inside every other policy gradient system used for LLM alignment today.

---

## Real-Life Analogy 🍳

### The Chef and the Dinner Service

Imagine you are a head chef at a restaurant that seats forty guests every night. At the end of each service — after the guests have left and the dining room is dark — a single score is slid under the kitchen door: an integer from one to ten, representing the average satisfaction of all the diners that night. You don't know which dish they loved, which they hated, or which waiter accidentally spilled a drink. All you have is a number: seven.

Now imagine running that kitchen using REINFORCE. Each night you write down every single decision you made during the service — how much salt you added to the risotto, whether you rested the duck for six minutes or eight, how finely you sliced the fennel. At the end of service, when the score arrives, you use it to update your cooking style. Score of nine? Mentally reinforce all the decisions you made tonight — add a bit more salt next time, rest the duck for longer, slice the fennel fine. Score of four? Do the opposite — mentally suppress those choices. Over hundreds of services, the decisions that correlate with high scores gradually become your default style.

This analogy captures REINFORCE almost perfectly. The "episode" is one dinner service. The "actions" are every individual cooking choice. The "reward" is that single number at the end of the night. The problem — and this is crucial — is that the score of seven tonight might not mean your risotto was good. Maybe the risotto was terrible but the dessert was extraordinary. Or maybe the score was seven because of the ambient music, the temperature of the room, the mood of the guests when they arrived. You have no way of knowing which of your hundreds of decisions actually caused the score. You are assigning credit (or blame) for the whole night's outcome to every single decision, equally, regardless of whether it actually contributed. This is the **credit assignment problem**, and it is the fundamental difficulty in REINFORCE.

Now add one more wrinkle. Suppose you run this restaurant in a neighbourhood where the same forty guests don't come back every night. Each service has a completely different crowd, with different tastes and different moods. Even if you cook exactly the same food two nights in a row, you might score a six one night and an eight the next — purely because of who happened to sit down. This variability in your score that has nothing to do with your cooking is exactly what statisticians call **variance**, and it is REINFORCE's most serious practical problem. Your gradient signal — the information telling you which direction to update your weights — is buried under layers of noise. With enough dinners (enough training episodes), the signal eventually wins. But it takes a very long time, and in the meantime you might stumble badly in the wrong direction.

That is REINFORCE. Elegant, simple, and correct on average — but extraordinarily slow to converge in practice. Everything that comes after it (baselines, PPO, advantage estimation, GRPO) is an attempt to keep the elegant simplicity while reducing the noise.

### The Baseline as a Better Chef 🍽️

Now suppose the restaurant hires a statistician friend to help. Every night the statistician computes the average score over the last thirty services: say 6.4. When tonight's score arrives — let's say it's 7.2 — you don't just hear "seven point two." You hear "seven point two, which is 0.8 above your recent average." Suddenly the signal is far more informative. You know tonight was genuinely good, not just okay. You reinforce tonight's decisions more confidently.

On a night you score 5.9, you hear "five point nine, which is 0.5 below your recent average." Now you know to suppress tonight's choices — even though 5.9 sounds like a reasonable score in isolation. Without the baseline, you'd have mildly reinforced those same decisions (because 5.9 is positive). The baseline transforms the signal from absolute to relative, and relative information is far more useful for learning.

This is the baseline trick in REINFORCE, and it corresponds exactly to the **advantage function** in PPO. The baseline is the statistician's running average. The advantage is the gap between what you actually got and what you expected.

---

## The Core Idea: Reinforce Good Actions — Deep Dive 🔍

### What Does "Policy" Mean?

In reinforcement learning, a **policy** is just a function that maps a situation (the current state) to a probability distribution over possible actions. For a language model, the state is the prompt and all the tokens generated so far; the actions are the ~32,000 tokens in the vocabulary; and the policy is the softmax distribution the model outputs at each step.

The policy has **parameters** — the billions of weights inside the model. When we train, we adjust those weights. Adjusting the weights shifts the probability distribution: some tokens become more likely, others less likely.

It is worth sitting with how enormous this action space is. A typical LLM vocabulary contains between 32,000 and 128,000 tokens. At each of the (say) fifty steps in a response, the policy assigns a probability to every single one of those tokens. The model is not choosing from three or four options like a game of chess — it is choosing from a library the size of a dictionary, at every single step, hundreds of millions of times over the course of training. The fact that REINFORCE can meaningfully shape this distribution at all is remarkable. The fact that it does so slowly and noisily is understandable.

### The REINFORCE Update in Plain English

Here is the core REINFORCE idea, stated as plainly as possible:

**After you observe that taking action A in situation S led to a total reward of R, increase the probability of taking action A in situation S by an amount proportional to R.**

If R is positive, you boost the probability. If R is negative (or zero, if you're treating low reward as no encouragement), you suppress it. The bigger the reward, the bigger the boost.

To do this, we use **gradient ascent on the expected reward**. We want to increase the expected reward, so we estimate the gradient of that expected reward with respect to the model weights, then take a step in that direction.

### The Log-Probability Trick

Here's where it gets slightly subtle but enormously important. When we take a gradient of expected reward with respect to policy parameters, a beautiful identity appears: the gradient involves the **log probability** of the actions, not the raw probability.

Why log probability? Consider: if an action has probability 0.001 and we want to increase it by a factor of two, that's a meaningful change (0.001 → 0.002). But if an action has probability 0.9 and we increase it by the same absolute amount, it's nearly irrelevant. The log scale gives us sensitivity that's appropriate relative to the current probability — small probabilities get large gradients, high probabilities get small gradients, which is exactly what we want.

In practical terms: when we observe an action and its reward, we compute the log probability of that action under our current policy, then multiply it by the reward. This gives us a signal that says: "make this action more likely — and scale that change proportionally to how good the reward was."

### Worked Example: The Token Decision

Let's make this concrete. An LLM is answering a maths question. The prompt is "What is 3 plus 5?" The model generates the token "8" at the critical step.

- Log probability of "8" at that step: say -0.7 (the model was about 50% confident)
- Total reward at end of response: +1.0 (a verifiable answer checker confirms the answer is correct)
- REINFORCE update signal: -0.7 × 1.0 = -0.7 → gradient says "increase the probability of '8' here"

Wait — the gradient is negative? Here's where it helps to think carefully. We're doing **gradient ascent** on expected reward. The gradient of log probability with respect to the weights points in the direction that increases the probability of the chosen action. So a gradient signal of -0.7 (the log prob) multiplied by +1.0 (a positive reward) tells us: push the weights in the direction that makes the log probability less negative, i.e., makes the action more probable. ✅

Now suppose the model instead generated "eleven" (a wrong answer) and the reward was 0.0:

- Log probability of "eleven": -1.6 (model was only about 20% confident)
- Reward: 0.0
- REINFORCE update signal: -1.6 × 0.0 = 0.0 → no update

In the naive reward-only REINFORCE, a reward of zero gives zero update. The model doesn't learn to suppress the wrong answer — it just doesn't reinforce it. This is one reason why **baselines** are crucial: to turn zero reward into negative reward relative to average, so wrong answers actually get penalised.

### What "Gradient Ascent" Means

Standard neural network training uses **gradient descent**: we compute the gradient of the loss and step downhill to reduce error. REINFORCE uses **gradient ascent**: we compute the gradient of expected reward and step uphill to increase it. These are mirror-image processes. Many implementations convert REINFORCE to gradient descent by negating the signal (the "negative reward" loss trick), but conceptually you are climbing a hill toward high expected reward.

The learning rate controls how big each step is. Too small a learning rate: training is stable but slow — you creep up the hill. Too large a learning rate: training is fast but catastrophically unstable — you leap past the peak and careen down the other side. Policy gradient methods are particularly sensitive to learning rate because the reward landscape is non-stationary: as the policy changes, the reward distribution changes too. There is no fixed loss surface to climb — the hill reshapes itself underfoot with every update.

### The Full Episode Loop

REINFORCE operates on complete **episodes**. For a language model, one episode is one complete response generation:

1. 🎲 Sample a prompt from the training distribution
2. 🤖 Run the model to generate a full response, recording every token and its log probability
3. 🏆 Score the complete response with a reward function (reward model, verifier, or human)
4. 🔄 For every token in the response, compute: (log probability of that token) × (total reward)
5. ⬆️ Accumulate these signals across the whole response and take a gradient step

This is sometimes called the **Monte Carlo policy gradient** because you're using a single sampled episode (a Monte Carlo sample) to estimate the gradient of expected reward. One episode is not enough to be accurate — so you batch many episodes together, average the gradient signals, and take one update step.

### Why "Monte Carlo"?

The term comes from the Monte Carlo casino — a metaphor for random sampling. When a quantity is hard to compute exactly (like the expected reward over all possible responses to all possible prompts), you estimate it by running random experiments and averaging the results. With enough samples, the average converges to the true expectation. REINFORCE is a Monte Carlo method because it estimates the gradient by sampling rollouts and averaging — rather than computing the gradient exactly (which would require evaluating every possible response, which is astronomically expensive).

The trade-off: Monte Carlo estimates are unbiased (with enough samples they converge to the right answer) but high-variance (any single sample might be far from the true value). This bias-variance trade-off is the central theme of everything in this topic area.

---

## The High Variance Problem — Deep Dive 📊

### What Variance Means Here

In statistics, variance measures how much a quantity bounces around from sample to sample. Low variance means your estimate is stable — sample it a hundred times and you get similar values. High variance means your estimate is erratic — every sample tells you something different.

REINFORCE suffers from **catastrophically high variance** in its gradient estimates. Here's why.

### Three Sources of Variance

**Source 1: The reward itself is noisy.** Even if the model does everything right, the reward might vary due to factors outside the model's control. The reward model might assign slightly different scores to identical responses due to its own imperfect training. Human raters are inconsistent. Task difficulty varies across prompts. None of this variation reflects the model's quality.

**Source 2: The credit assignment problem creates attribution noise.** A response has fifty tokens. One gradient update treats all fifty tokens as equally responsible for the final reward. But token 12 (which was a critical decision) and token 47 (which was a filler word) both receive the same credit signal. The signal for the actually-important decisions is diluted by the noise from the irrelevant ones.

**Source 3: Trajectory randomness.** The model samples tokens stochastically. Two runs with the same prompt might produce wildly different responses — and wildly different rewards. The gradient computed from one trajectory might point in nearly the opposite direction from the gradient computed from another trajectory of the same prompt. When you average them together, the true signal is in there — but so is a huge amount of noise.

### The Statistics of Variance

Variance is measured in the same units as the squared quantity. If your rewards range from 0 to 1, your gradient estimates might have a standard deviation of 0.5 — meaning a typical single-episode gradient is 0.5 units away from the true gradient. To reduce estimation error by half, you need **four times as many samples** (because standard error decreases with the square root of sample size). To reduce error by a factor of ten, you need a hundred times as many samples. This is the curse of variance: reducing it requires quadratic increases in computation.

In LLM terms: if a reward model gives noisy scores with standard deviation 0.5 (realistic for subjective tasks like helpfulness), a batch of 16 episodes gives a gradient estimate with standard error 0.125. To halve that error, you need 64 episodes. For practical training, you'd want 256 or more episodes per update step just to get a reliable gradient signal — at enormous computational cost.

### Why High Variance is a Training Problem

Imagine you're trying to walk toward a distant mountain in a thick fog, using only a compass that gives you the correct direction on average — but on any individual reading it's off by up to 90 degrees in a random direction.

If you take thousands of tiny steps, each based on one compass reading, you'll eventually reach the mountain. But you might walk in circles for a very long time first, wasting enormous computational effort. Worse, a particularly bad compass reading might send you in entirely the wrong direction, undoing previous progress.

High variance in gradient estimates causes exactly this. The model might improve for twenty steps, then a noisy gradient pushes it backward for five steps. Learning is possible but agonisingly slow.

### Worked Example: The Variance in Practice

Consider training with batches of 8 episodes (responses). The rewards for the 8 responses are:

```
Episode 1: reward = 0.8
Episode 2: reward = 0.1
Episode 3: reward = 0.7
Episode 4: reward = 0.9
Episode 5: reward = 0.2
Episode 6: reward = 0.8
Episode 7: reward = 0.1
Episode 8: reward = 0.6
```

Average reward = 0.525. That's your signal for this batch. But individual episodes vary from 0.1 to 0.9 — nearly a 10× range. The gradient estimated from episode 2 (reward 0.1) might point in a nearly opposite direction from the gradient estimated from episode 4 (reward 0.9). When averaged, these partially cancel out, leaving a much weaker signal than either would have provided on its own.

This is why batch size matters so much. Larger batches give you a more accurate estimate of the true gradient — but at enormous computational cost. REINFORCE with small batches is noisy and slow. REINFORCE with large batches is expensive. Either way, you're fighting variance.

---

## Baselines — The Variance Reduction Trick — Deep Dive 🧮

### The Key Insight

Here is the central insight of baseline-based variance reduction, stated plainly:

**Instead of asking "was this reward good?" ask "was this reward better than usual?"**

A reward of 0.8 is meaningless without context. If the model usually gets rewards of 0.3, then 0.8 is phenomenal — strongly reinforce whatever actions led to it. If the model usually gets rewards of 0.9, then 0.8 is actually disappointing — mildly suppress whatever actions led to it.

A **baseline** is a reference value that you subtract from the reward before using it as a training signal. The most common baseline is simply the average reward over recent episodes.

### The Transformed Signal

Without baseline:
- Episode reward = 0.8 → multiply all token log-probs by +0.8 → reinforce all actions

With baseline (average reward = 0.7):
- Episode reward = 0.8, baseline = 0.7
- Adjusted signal = 0.8 - 0.7 = +0.1 → mildly reinforce actions from this episode

Another episode:
- Episode reward = 0.5, baseline = 0.7
- Adjusted signal = 0.5 - 0.7 = -0.2 → mildly suppress actions from this episode

Now we have both positive and negative signals. Actions from better-than-average episodes get reinforced; actions from worse-than-average episodes get suppressed. Learning is much faster and more precise.

### Why Subtracting a Baseline Doesn't Bias the Gradient

This is the mathematically elegant part. When we subtract a baseline from the reward, we might worry: are we changing what we're optimising for? Are we introducing bias?

The answer is no — and here's the intuition for why. The baseline is a constant (or at least, a value that doesn't depend on which action was taken). When you average the gradient signal across many episodes, the baseline terms average to zero because the log probability of an action, on average over all possible actions, sums to zero. The baseline shifts the individual estimates around but doesn't change where they average to. You're still climbing the same hill — just with steadier footing.

Formally, any function of the state (but not of the action) is a valid baseline. The most common choice is the **expected return from this state** — which, when estimated by a neural network, is called the **value function**. When you subtract the value function from the return, you get the **advantage** — the core quantity that PPO, A2C, and GRPO all compute.

### Worked Example: Baselines in Action

Without baseline — two episodes, same prompt, different responses:

```
Response A: "Paris" → reward 1.0 → multiply log-probs by 1.0 → strongly reinforce
Response B: "France" → reward 0.8 → multiply log-probs by 0.8 → strongly reinforce
```

Both responses get reinforced, but Response A much more strongly. The model nudges toward "Paris" — fine.

With baseline (= 0.9, the average):

```
Response A: "Paris" → adjusted = 1.0 - 0.9 = +0.1 → slightly reinforce
Response B: "France" → adjusted = 0.8 - 0.9 = -0.1 → slightly suppress
```

Now the model clearly learns: "Paris" is better than "France" for this question. Without the baseline, both responses were reinforced and the relative preference was buried in the noise. With the baseline, the relative signal is clean and unambiguous. 🎯

### Types of Baselines

| Baseline Type | Description | Used In |
|---|---|---|
| Constant baseline | Fixed number (e.g., 0.5) | Basic REINFORCE |
| Running average | Average reward over last N episodes | REINFORCE with baseline |
| Value function V(s) | Learned estimate of expected return | A2C, PPO |
| Group average | Average reward across group of responses to same prompt | GRPO |

The value function baseline is the most powerful because it's state-dependent — it knows that some situations are naturally harder than others, and adjusts accordingly. But it requires training a separate neural network (the critic), which adds complexity. GRPO sidesteps this by using the group average of responses to the same prompt as the baseline — elegant, simple, and surprisingly effective.

---

## The Gradient Estimator Explained — Deep Dive 🧭

### Translating the Gradient Into Plain English

The REINFORCE gradient estimator can be described in one sentence:

**"For each action you took, nudge your policy weights in the direction that makes that action more likely — and scale how hard you nudge by how much better than expected the outcome was."**

That's it. Every variation of policy gradient — PPO, GRPO, A2C — is a variation of this sentence. What changes is: how you define "better than expected," how big a nudge you allow, and how you estimate the gradient more efficiently.

### Breaking Down the Components

**Component 1: "For each action you took"**
You're iterating over every decision in the episode. For a language model, this means every token generated, at every position in the response. Each token is an action.

**Component 2: "Nudge your policy weights in the direction that makes that action more likely"**
This is the gradient of the log probability of the action with respect to the weights. It points in the direction that, if you follow it, makes the model more likely to generate that exact token in that exact context.

**Component 3: "Scale how hard you nudge"**
This is the reward signal (or advantage signal, after subtracting a baseline). If the reward was high, take a big step. If the reward was near the baseline, take a tiny step. If the reward was below the baseline, reverse the direction (push the action to be less likely).

### The Direction and Magnitude of Updates

Let's think through four scenarios:

**Scenario A: High reward, action was low-probability**
The model generated an unusual token that turned out to be brilliant. High reward × large gradient = large update. The model "remembers" this discovery strongly. This is how REINFORCE can find surprisingly good policies — by getting lucky and then strongly reinforcing the lucky find. 🎰

**Scenario B: High reward, action was high-probability**
The model generated a common, expected token that happened to be rewarded. Reward is high but gradient is small (high probability actions have small log-prob gradients). Update is moderate. The model doesn't over-learn from expected successes.

**Scenario C: Low adjusted reward (below baseline), any probability**
The gradient direction is reversed. The update pushes the weights to make this action less likely in this context. The model learns to avoid the patterns associated with below-average outcomes.

**Scenario D: Reward exactly equals baseline**
Adjusted reward = 0. No update. The model learned nothing from this episode. This seems wasteful, but it's correct — if this episode was completely average, there's no information in it that distinguishes good actions from bad ones.

### Why It Works — The Intuition

Imagine you're adjusting thousands of tiny dials, each one corresponding to a weight in the network. The gradient estimator tells you: for this episode, move dial X by amount Y. Over thousands of episodes, the dial settings that correlate with high reward slowly accumulate positive nudges; the settings that correlate with low reward accumulate negative nudges. The model "discovers" the configuration of dials that maximises expected reward — without ever being told explicitly what configuration that is.

This is the magic of policy gradient methods: they don't need to know the reward function analytically. They just need to be able to evaluate it. And in the LLM setting, where the reward function might be a complex neural network (the reward model), or even human judgment, this is exactly what we need.

---

## From Theory to LLM Training — Deep Dive 🤖

### Mapping RL Concepts to Language Models

The REINFORCE algorithm was originally developed for games and robotics, where states and actions are obvious. Mapping it to language models requires some translation:

| RL Concept | LLM Equivalent | Example |
|---|---|---|
| State | Prompt + tokens generated so far | "What is 3+5? The answer is" |
| Action | Next token to generate | "8" (from vocab of ~32k options) |
| Policy | The LLM's softmax distribution | P(token | context) = the model |
| Episode | One complete response | Full answer from "What is 3+5?" to the end |
| Reward | Score given to complete response | Reward model output, or 1/0 from verifier |
| Policy update | Gradient step on model weights | Update billions of parameters |

### Each Token Is an Action

This is subtle but important. In a response that is 50 tokens long, REINFORCE sees **50 actions**. Each token choice is a discrete action in the action space of the vocabulary (typically 32,000 to 128,000 tokens).

Most of those 50 token choices seem deterministic — "the" appears after "I think", "=" appears after a number — but they are still choices that the model assigns probabilities to, and each probability can be nudged by the gradient.

The challenge: the reward only arrives after all 50 actions are complete. Action 12 (which might have been the crucial word that steered the response toward the right answer) gets the same raw reward signal as action 49 (a punctuation mark). Distinguishing their individual contributions is the credit assignment problem.

### The Full Episode in an LLM

Let's trace a complete REINFORCE episode with a real (simplified) example.

**Prompt:** "If Alice has 3 apples and Bob gives her 4 more, how many apples does Alice have?"

**Response generated (50 tokens):**
"Alice starts with 3 apples. Bob gives her 4 more. 3 plus 4 equals 7. Alice has 7 apples."

**Reward:** +1.0 (verifier confirms "7" is correct)

**REINFORCE update for each token:**

| Token position | Token | Log prob | Reward | Update signal |
|---|---|---|---|---|
| 1 | "Alice" | -0.3 | +1.0 | +0.3 boost |
| 2 | "starts" | -0.8 | +1.0 | +0.8 boost |
| ... | ... | ... | ... | ... |
| 26 | "7" (critical!) | -1.2 | +1.0 | +1.2 boost |
| ... | ... | ... | ... | ... |
| 50 | "." | -0.05 | +1.0 | +0.05 boost |

Every token gets reinforced equally — including the meaningless punctuation. The critical token "7" gets reinforced based on its log probability, not its actual importance to the answer. This is the blunt force of REINFORCE. Over thousands of episodes, the tokens that actually matter (those that correlate with correct answers) get reinforced more reliably than the irrelevant ones — but it takes many repetitions to see through the noise.

### The Batch of Episodes

In practice, you never update from a single episode. You collect a batch — say 64 or 128 prompts — generate one response for each, score all responses, then compute the average gradient across the batch. This is sometimes called a **rollout batch** or **experience batch**.

The gradient update at the end of a rollout batch averages out some of the noise — episodes with unusually high rewards and episodes with unusually low rewards partially cancel each other out. What remains is a (noisy but usable) signal pointing the weights in the right direction.

### When Reward Comes At the End

One of the biggest challenges for REINFORCE with LLMs is **reward sparsity**: the reward only arrives after the complete response is generated. This means:

- The model takes 50 actions before getting any feedback
- The gradient for action 1 (the first token) is influenced by everything that happened in actions 2–50
- The credit assignment problem is at its worst

In contrast, classic RL tasks like Atari games give a reward at every step (or every few steps), making credit assignment much easier. LLM training with end-of-response rewards is a hard RL problem — and REINFORCE handles it especially poorly compared to methods like PPO that use learned value functions to estimate intermediate rewards.

---

## How It Works in Practice ⚙️

### Real Batch Sizes and Iterations

In production LLM training, REINFORCE (or its close relatives) operates at significant scale. Here are realistic numbers from research implementations:

**Rollout batch size:** 64–256 prompts per gradient step is typical. Each prompt generates one response (or sometimes multiple for variance reduction). So a batch of 128 prompts × average 100 tokens = 12,800 token predictions per gradient step.

**Gradient accumulation:** Often gradient steps are accumulated over multiple batches before actually updating weights, giving effective batch sizes of 256–1024 episodes per update.

**Training iterations:** A typical REINFORCE training run for an LLM might run for 500–5000 gradient steps. Each step touches the full model (billions of parameters) for each token in the batch. Computational cost is enormous.

**Learning rate:** Very small — typically 1e-6 to 1e-5. Policy gradient methods are sensitive to learning rate; too large and training collapses.

**Reward normalisation:** Almost universally, rewards are normalised (zero-mean, unit variance) across each batch before computing gradient updates. This prevents a batch of uniformly high rewards from causing a massive update step.

### A Worked Training Loop

Here is what one training iteration looks like concretely:

```
Step 1: Sample 64 prompts from the training dataset
Step 2: Run the current LLM on each prompt, collect full responses
         - Record: each token generated, its log probability, the context at that step
Step 3: Score each of the 64 responses with the reward function
         - Output: 64 scalar rewards, e.g. [0.8, 0.3, 0.9, 0.1, ...]
Step 4: Compute the baseline (e.g., average = 0.5)
Step 5: For each response, subtract baseline from reward
         - Adjusted rewards: [+0.3, -0.2, +0.4, -0.4, ...]
Step 6: For each token in each response:
         compute (adjusted reward) × (log prob gradient)
Step 7: Average the gradient signals across all 64 × avg_response_length tokens
Step 8: Take a gradient ascent step with this averaged gradient
Step 9: Go back to Step 1
```

### Wall-Clock Time

Training a 1B parameter model with REINFORCE from scratch on a reasoning task might take:

- 2,000 gradient steps
- ~30 seconds per step on 8 A100 GPUs
- Total: ~17 hours of GPU time

For comparison, the same experiment with PPO (which is more sample-efficient due to multiple epochs per rollout) might achieve similar performance in ~6 hours. The extra efficiency of PPO matters enormously at scale.

### What a Reward Curve Looks Like

In a typical REINFORCE run on a verifiable reasoning task (like maths or code), the reward curve looks roughly like this:

```
Steps 0–100:    Reward flat near 0.2 — model barely changes; gradient noise dominates
Steps 100–300:  Reward slowly climbs to 0.4 — signal is accumulating through the noise
Steps 300–600:  Reward climbs to 0.6 — good policies found, being reinforced
Steps 600–900:  Reward plateaus around 0.65–0.7 — diminishing returns, variance still high
Steps 900+:     Reward oscillates, may decline — reward hacking emerging, or KL growing
```

This curve is characteristic of REINFORCE: a long flat period while the gradient noise averages out, then a meaningful rise, then a plateau that's noticeably below what PPO would achieve on the same task with the same compute budget.

### Training Instabilities to Watch For 🚨

Real REINFORCE training breaks in predictable ways. Here are the most common failure modes and their diagnostic signals:

**Instability 1: Reward collapse**
Training reward suddenly drops from 0.6 to 0.1. Usually caused by a catastrophically bad gradient update where the policy jumps into a bad region of weight space and gets stuck. Prevention: very small learning rate, gradient clipping.

**Instability 2: KL explosion**
The KL divergence from the reference model grows rapidly (from 0.05 nats to 2.0+ nats). The policy is drifting too far from its starting point. Often accompanied by degraded output quality — the model starts generating garbled or repetitive text. Prevention: KL penalty, early stopping on KL.

**Instability 3: Gradient explosion**
Individual gradient updates become extremely large due to an unusually high or low reward in one batch. The model weights take a massive step in a random direction. Prevention: gradient clipping (clip gradient norm to a maximum value, typically 1.0 or 0.5).

**Instability 4: Reward model gaming**
Training reward keeps climbing but held-out human evaluation reward stays flat or falls. The model has found a way to score well on the reward model that doesn't correspond to genuinely better responses. Prevention: KL penalty, diverse prompt datasets, periodic human evaluation.

### When to Stop

REINFORCE training stops when one of the following occurs:

1. The reward on a held-out evaluation set plateaus for N consecutive steps
2. The KL divergence from the reference model grows too large (the model is drifting into nonsensical outputs)
3. The training reward collapses (usually a sign of reward hacking)
4. A compute budget is exhausted

In practice, REINFORCE runs are rarely left to run until full convergence due to their noisy nature — they're often replaced after a few hundred steps with PPO to get better sample efficiency. However, on tasks with clean verifiable rewards (maths, code execution), REINFORCE-style algorithms with good baselines (like RLOO or GRPO) can match PPO performance at lower engineering complexity — making them increasingly popular in open-source research.

---

## Common Misconceptions ❌

### ❌ Myth: "REINFORCE is outdated and irrelevant to modern LLM training"
✅ **Reality:** Every modern policy gradient algorithm for LLMs — PPO, GRPO, RLOO — is a direct descendant of REINFORCE. The core update rule (log prob × advantage) is present in all of them. Understanding REINFORCE means understanding the building block that all of them rely on. DeepSeek-R1, one of the most important recent RL-trained LLMs, uses GRPO, which differs from REINFORCE mainly in how it computes the baseline — the fundamental update is the same.

---

### ❌ Myth: "REINFORCE rewards good actions and ignores bad ones"
✅ **Reality:** Raw REINFORCE rewards all actions in a high-reward episode, regardless of whether they individually were good decisions. And without a baseline, it completely ignores (doesn't penalise) actions from low-reward episodes — it just doesn't reinforce them. With a baseline, actions from below-average episodes get penalised. The baseline is not optional — it's essential for learning to distinguish good from bad.

---

### ❌ Myth: "The model learns which specific token caused the high reward"
✅ **Reality:** REINFORCE cannot distinguish between tokens. Every token in a high-reward episode gets reinforced equally. There is no mechanism in basic REINFORCE to say "token 26 was the pivotal decision." This is the credit assignment problem, and REINFORCE makes no attempt to solve it. Methods like per-step value functions (as in PPO) or process reward models try to address this limitation.

---

### ❌ Myth: "Larger rewards always mean faster learning"
✅ **Reality:** What matters is the **difference** between the reward and the baseline. If every episode gets a reward of 10.0, the baseline is also 10.0, and the adjusted signal is zero — no learning at all. Learning only happens when there is **variance** in rewards. In fact, zero variance = zero learning. This is a subtlety that trips up many practitioners: if your reward model is too generous (always gives high scores), training stalls.

---

### ❌ Myth: "REINFORCE with baseline is biased"
✅ **Reality:** Subtracting a baseline does not bias the gradient. This is provable mathematically: any function that doesn't depend on the action is a valid, unbiased baseline. The baseline shifts individual estimates (reducing variance) without changing the expected value of the gradient estimate. You're reducing noise without introducing error.

---

### ❌ Myth: "REINFORCE is too simple to use for real LLM training"
✅ **Reality:** REINFORCE (sometimes called RLOO, or REINFORCE Leave-One-Out, in its modern form) has been shown to be competitive with PPO on many tasks. Papers from 2024 comparing simple REINFORCE-style methods against PPO found that REINFORCE with a good baseline and reward normalisation often matches PPO at a fraction of the implementation complexity. Simplicity is sometimes a feature.

---

### ❌ Myth: "High variance is always bad — if you have enough data it doesn't matter"
✅ **Reality:** High variance means you need exponentially more data to achieve the same learning accuracy. In LLM training where each rollout costs real compute, this matters enormously. Going from a variance-reduction technique that needs 100 samples to a baseline method that needs 1,000 samples to achieve similar gradient quality represents a 10× increase in training cost. Variance reduction is not a minor optimisation — it's the difference between feasible and infeasible training.

---

## The REINFORCE Algorithm — Step by Step 🪜

Let's walk through the complete algorithm pseudocode in plain English, to make the flow concrete before we discuss its connections to PPO and GRPO.

**Algorithm: REINFORCE with Baseline**

```
Initialise:
  - Policy (the LLM), with current weights theta
  - Baseline value b (start at 0, or use a running average)
  - Learning rate alpha (e.g. 1e-5)

Repeat for N training iterations:

  1. ROLLOUT PHASE
     For each prompt in the current batch:
       a. Sample a prompt p from the training dataset
       b. Run the policy to generate a response r
          - At each step t, record: token chosen, log probability of that token
       c. Compute the episode reward R using the reward function
          - For verifiable tasks: R = 1.0 (correct) or 0.0 (wrong)
          - For subjective tasks: R = reward model output (e.g. 0.73)

  2. BASELINE UPDATE
     Compute the batch average reward: b = mean(R_1, R_2, ..., R_batch)
     (Or: update a running exponential moving average of recent rewards)

  3. ADVANTAGE COMPUTATION
     For each episode i:
       A_i = R_i - b
       (Positive A_i: this episode was better than average)
       (Negative A_i: this episode was worse than average)

  4. GRADIENT COMPUTATION
     For each episode i, for each token t in the response:
       gradient_signal += A_i × (gradient of log P(token_t | context_t))
     Average gradient_signal across all episodes and all tokens

  5. WEIGHT UPDATE
     theta = theta + alpha × gradient_signal
     (Gradient ascent: step in direction of increasing expected reward)

  6. GO TO STEP 1
```

Notice that the algorithm is embarrassingly simple. The rollout phase is just running the model. The baseline update is just computing an average. The gradient computation is just multiplying two numbers per token and summing them up. The weight update is just addition. The complexity in practice comes from engineering — parallelising rollouts across GPUs, managing memory for large models, normalising rewards, and handling the KL penalty — not from the algorithm itself.

---

## Connections to Other Topics 🔗

### → REINFORCE to PPO

REINFORCE's main limitation is high variance and poor sample efficiency (you throw away all rollouts after one gradient step). **PPO** addresses both:

1. **Advantage estimation:** PPO uses a learned value function to estimate a per-step baseline, dramatically reducing variance compared to using the full episode reward as the signal for every token.

2. **Clipped updates:** PPO clips the gradient update to prevent any single bad rollout from causing a catastrophically large weight change. REINFORCE has no such protection.

3. **Multiple epochs:** PPO reuses the same rollout data for multiple gradient updates (correcting for the distribution shift with the clipped objective). REINFORCE discards rollouts after one use, wasting expensive samples.

PPO can be seen as REINFORCE + advantage estimation + clipping + replay. If you understand REINFORCE deeply, PPO is a series of clear, motivated additions.

---

### → REINFORCE to GRPO

**GRPO (Group Relative Policy Optimisation)**, used in DeepSeek-R1, takes a different approach to the baseline problem:

Instead of learning a value function, GRPO generates a **group** of responses to the same prompt (say, 8 responses), then uses the average reward of the group as the baseline for each individual response. This is simple REINFORCE with a prompt-specific baseline — no critic network needed.

The advantage of GRPO:
- No value network to train and maintain (saves memory and complexity)
- Baseline is naturally calibrated to the difficulty of each individual prompt (a hard prompt gets a baseline based on responses to that hard prompt)
- Works beautifully with verifiable rewards (maths, code) where rewards are binary (correct/incorrect)

GRPO shows that the fundamental insight of REINFORCE — compare each outcome to a baseline — is powerful enough to achieve state-of-the-art results when paired with the right baseline strategy.

---

### → REINFORCE to Value Functions

The natural next step after REINFORCE is to ask: "what if instead of using the total episode reward as the training signal for every token, we could estimate what reward each individual state is worth — and use that to compute a better signal?"

That's exactly what a **value function** does. The value function V(s) estimates the expected total reward from state s onwards. When you subtract V(s) from the actual return, you get the **advantage** — a much lower-variance signal that specifically measures whether an action at a particular step was better or worse than the model expected.

This idea — advantage estimation — is the bridge from REINFORCE to the actor-critic family of algorithms, and ultimately to PPO.

---

### → REINFORCE to RLOO (REINFORCE Leave-One-Out)

A clever modern variant of REINFORCE used in some LLM training pipelines is **RLOO (REINFORCE Leave-One-Out)**. Instead of generating one response per prompt and using a running average as the baseline, RLOO generates K responses to the same prompt and uses the average of the other K-1 responses as the baseline for each one.

For example, with K = 4 responses to the same prompt:
- Response 1 gets baseline = average reward of responses 2, 3, 4
- Response 2 gets baseline = average reward of responses 1, 3, 4
- Response 3 gets baseline = average reward of responses 1, 2, 4
- Response 4 gets baseline = average reward of responses 1, 2, 3

This prompt-specific baseline is much more informative than a global running average. If a prompt is genuinely hard (all responses score around 0.2), the baseline is 0.2 and each individual response gets a near-zero advantage. If a prompt is genuinely easy (all responses score around 0.9), the baseline is 0.9 and again advantages are near zero. This prevents the algorithm from aggressively updating on easy prompts just because 0.9 sounds like a high reward.

RLOO is nearly identical to GRPO in spirit — the main difference is that GRPO normalises by standard deviation as well (making it fully standardised), while RLOO just subtracts the mean. Both are essentially baseline REINFORCE applied with prompt-level group statistics.

### → REINFORCE and KL Divergence

One thing REINFORCE doesn't do that modern LLM training requires: it doesn't prevent the model from drifting too far from its starting point. Without any constraint, REINFORCE might chase a particular high-reward pattern and forget most of what it learned during pretraining.

The standard fix: add a **KL divergence penalty** to the reward. The total training signal becomes:

```
Adjusted reward = (reward from reward model) - lambda × (KL from reference model)
```

This means episodes where the model's output diverges too far from its pre-RL behaviour get penalised, even if the reward model scores them well. The KL penalty is the leash that keeps REINFORCE (and PPO) from going off the rails. Without it, reward hacking is nearly inevitable.

### → REINFORCE and Reward Hacking

One of the nastiest failure modes in REINFORCE-style training is **reward hacking**: the policy discovers a pattern that scores highly on the reward function but is actually a terrible response by any reasonable standard. Without a KL penalty, REINFORCE will enthusiastically pursue reward hacking — it has no way to know the difference between genuinely good responses and clever exploits of the reward model.

Famous examples of reward hacking in LLM training include:
- 📝 **Repetition loops:** Generating "great question! great question! great question!" repeatedly because the reward model learned that "great" is associated with positive responses.
- 📏 **Length exploitation:** Making responses extremely long because the reward model tends to rate longer responses higher (more text = more information = better, naively).
- 🤡 **Sycophancy amplification:** Agreeing with whatever the prompt implies, because human raters in the reward model training data preferred agreeable responses.
- 🔢 **Format gaming:** Learning that bullet-pointed responses score higher and generating bullet points regardless of whether the content is meaningful.

The KL penalty forces the policy to stay close to the reference model (typically the SFT model before RL training). The reference model doesn't know how to hack the reward function — so staying close to it prevents the learned policy from drifting into degenerate territory.

---

## Historical Context — Where REINFORCE Fits 📜

REINFORCE was introduced by Ronald Williams in 1992 in a paper titled *"Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning."* The paper is astonishingly prescient — it lays out log-probability gradients, baselines, and variance reduction in almost exactly the form used in modern LLM training, three decades before the LLM era.

During the 1990s and 2000s, REINFORCE was largely overshadowed by value-based methods like Q-learning and DQN (Deep Q-Networks), which dominated Atari benchmark research at DeepMind. Value-based methods were more sample-efficient for discrete action spaces and easier to implement stably.

The policy gradient renaissance came with **trust region methods** (TRPO, 2015) and then PPO (2017), both from OpenAI. These showed that policy gradients could be stabilised at scale with the right constraints. PPO became the default RL algorithm for large-scale training.

When Ziegler et al. (2019) published *"Fine-Tuning Language Models from Human Preferences"* — the paper that introduced RLHF for text generation — they used a policy gradient approach directly inspired by REINFORCE, with PPO handling the stability. InstructGPT (2022) then scaled this to production GPT-3 models.

The circle closed when DeepSeek published their R1 paper (2025) using GRPO — essentially a renamed and slightly improved REINFORCE with a group baseline — to achieve state-of-the-art reasoning performance. The 1992 algorithm is alive and competitive in 2025.

---

## Key Takeaways 📋

| Concept | Plain English |
|---|---|
| REINFORCE | Simplest policy gradient: take action → get reward → make that action more likely |
| Policy | The LLM: maps (prompt + tokens so far) to a probability over next tokens |
| Episode | One complete response generation, from first token to end token |
| Log probability trick | Using log prob as the gradient signal keeps updates proportional to current probability |
| High variance | Reward noise + credit assignment + trajectory randomness = very noisy gradient |
| Baseline | Subtract average reward before using as training signal; reduces variance without bias |
| Advantage | Reward minus baseline: how much better (or worse) than expected |
| Credit assignment problem | All tokens get equal credit for the episode reward, even though some matter more |
| Reward sparsity | Reward only at end of response makes credit assignment especially hard |
| Sample efficiency | REINFORCE discards rollouts after one use; PPO reuses them for multiple updates |
| Reward normalisation | Normalise rewards across a batch to stabilise gradient magnitudes |
| KL penalty | Prevents model from drifting too far from reference during training |
| Reward hacking | Model exploits the reward model without genuinely improving — needs KL or careful design |
| Monte Carlo gradient | Estimating the true gradient via random rollout sampling |
| RLOO | REINFORCE Leave-One-Out: uses sibling responses as prompt-specific baseline |
| GRPO connection | GRPO = REINFORCE + group average as baseline + std normalisation; no critic needed |
| PPO connection | PPO = REINFORCE + advantage estimation + clipping + multiple update epochs |
| Williams (1992) | The original REINFORCE paper — still the conceptual foundation of all modern LLM RL |

---

## Debugging REINFORCE: A Practical Checklist 🔧

When a REINFORCE run is not improving, here is a systematic checklist to work through before giving up or switching to PPO.

**Check 1: Is there signal in the rewards?**
Print the reward distribution across your batch. If all rewards cluster tightly (standard deviation < 0.05), the baseline cannot help — there is no signal to learn from. The reward function may be too lenient (everything scores 0.9) or too strict (everything scores 0.1). Fix: adjust the reward function to give more spread.

**Check 2: Is the baseline tracking the mean?**
Print the baseline value and the batch mean reward at each step. They should be close. If the baseline lags badly (exponential moving average with alpha too small), advantages will be inflated, causing large noisy updates. Fix: increase the moving average alpha, or switch to batch-level mean.

**Check 3: Is gradient norm growing?**
Monitor the gradient norm each step. If it spikes above 10 or 100, you have a runaway gradient. Enable gradient clipping (clip norm to 1.0). This alone fixes a significant fraction of REINFORCE instability issues.

**Check 4: Is KL divergence growing?**
If KL from the reference model grows beyond 0.5–1.0 nats within the first few hundred steps, the policy is drifting too fast. Increase the KL penalty coefficient (lambda) or reduce the learning rate.

**Check 5: Is the held-out reward tracking training reward?**
Run evaluation on held-out prompts every 50–100 steps. If training reward rises but eval reward stays flat, you have reward hacking. The policy is exploiting the reward model rather than genuinely improving. Fix: increase KL penalty, add diversity to prompt dataset, or switch to verifiable rewards if possible.

Following this checklist catches about 80% of REINFORCE failures before they require architecture changes or algorithm switches.

---

## Up Next

### → PPO: Proximal Policy Optimisation 🔒

PPO takes everything REINFORCE teaches us and adds three crucial improvements:

1. **Learned advantage estimation** — a value function estimates per-step baselines, dramatically cutting variance
2. **Clipped objective** — each gradient update is bounded, preventing catastrophic updates from outlier rollouts
3. **Multiple epochs per rollout** — the same batch of generated responses is used for several gradient updates instead of being discarded after one

If REINFORCE is a chef updating recipes based on one night's score, PPO is a chef who:
- Has a sophisticated internal model of which dishes usually score well (value function)
- Limits how drastically they change any recipe in one go (clipping)
- Reviews their notes from one dinner service multiple times before the next service (multiple epochs)

Understanding REINFORCE fully — its elegance, its variance problem, its credit assignment blindness — is the prerequisite for understanding why every single design decision in PPO exists. Every improvement PPO makes is a direct response to a REINFORCE failure mode.

### The Chain from REINFORCE to the State of the Art

```
REINFORCE (Williams, 1992)
    ↓ add advantage estimation (value function as baseline)
Actor-Critic / A2C (1990s–2000s)
    ↓ add trust region constraint (stable large updates)
TRPO (Schulman et al., 2015)
    ↓ simplify trust region to clipped ratio objective
PPO (Schulman et al., 2017)
    ↓ apply PPO to fine-tune LLMs from human feedback
RLHF + PPO (Ziegler et al., 2019; Ouyang et al., 2022)
    ↓ replace value function with group average baseline
GRPO (DeepSeek, 2024) / RLOO (2024)
    ↓ verifiable binary rewards, large-scale RL
DeepSeek-R1, QwQ, etc. (2024–2025)
```

Every node in this chain is REINFORCE plus one insight. The 1992 algorithm is still there at the root. ✅

**See:** `22_ppo.md` — PPO: Proximal Policy Optimisation

**And after PPO:** `24_grpo.md` — GRPO: Group Relative Policy Optimisation, the algorithm behind DeepSeek-R1 🚀
