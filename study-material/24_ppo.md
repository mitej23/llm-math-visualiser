# PPO — Proximal Policy Optimization

> **Sources used:**
> - Schulman et al., *Proximal Policy Optimization Algorithms*, OpenAI 2017 — [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
> - Huang et al., *The N+ Implementation Details of RLHF with PPO*, Hugging Face Blog 2023 — [huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)
> - Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), OpenAI 2022 — [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
> - OpenAI Spinning Up, *Proximal Policy Optimization* — [spinningup.openai.com](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
> - Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016 — [arxiv.org/abs/1506.02438](https://arxiv.org/abs/1506.02438)

---

## The Big Idea

**PPO (Proximal Policy Optimization)** is the reinforcement learning algorithm that powered the InstructGPT training pipeline and most subsequent RLHF systems. Its name contains the core insight: update the policy, but keep the new policy *proximal* (close) to the old one. Make improvements, but only take small, bounded steps at a time.

In plain English: PPO is a way of teaching an agent to improve its behaviour through trial and error while preventing it from overreacting to any single batch of experience. Each training update changes the policy by at most a small, controlled amount. This sounds like a simple idea, but implementing it correctly requires several interlocking components — a clipping mechanism to bound updates, a critic network to estimate how good a situation is, and a generalized advantage estimator to measure how much better or worse an action was compared to expectations.

For language model training specifically, PPO became important because it can handle the enormous complexity of text generation: a 32,000-token vocabulary at each step, sparse rewards (only at the end of a full response), and an action space so large that unconstrained gradient updates would send the model spiraling into incoherence within a few hundred steps.

The result of applying PPO to language model training is a model that learns not just from human-written examples (SFT) but from its own experience of generating text and receiving feedback — which is how it can surpass the ceiling imposed by imitation learning alone.

PPO was introduced by OpenAI in 2017 (Schulman et al., [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)) as a simpler, more practical alternative to TRPO (Trust Region Policy Optimization). Where TRPO required solving a constrained optimization problem at every step — mathematically rigorous but computationally expensive — PPO achieved similar stability through a cheap clipping operation that required only standard first-order gradient descent. This pragmatic trade-off made PPO the dominant policy gradient algorithm for years, and it was the natural choice when OpenAI built the InstructGPT training pipeline in 2022.

Understanding PPO deeply is important even if you plan to work with GRPO or DPO, because PPO establishes the conceptual vocabulary for all subsequent algorithms: the policy ratio, the advantage function, the reference policy, and the KL constraint are all concepts that appear — in modified form — in every successor algorithm.

This document covers PPO systematically: from the problem it solves (policy collapse in vanilla RL), through its core mechanism (the clipping trick), to the full infrastructure required to apply it to language models (four models, GAE, per-token KL). Each section builds on the previous ones, so reading in order is recommended. By the end, you should be able to explain why each design decision in PPO exists and what would go wrong if it were removed — which is the level of understanding needed to reason about GRPO's simplifications in the next topic.

---

## Real-Life Analogy

Imagine you are a professional chef who has been tasked with improving a signature dish. After years of making the same recipe, you decide to start experimenting. But you are a careful chef — you do not overhaul the dish all at once. Instead, you make one small change per week: a little more garlic, a slightly different cooking time, a different plating style. Each week you serve the adjusted dish to your regular customers and collect their feedback. If the change went over well, you keep it. If it made things worse, you reverse it. Either way, the adjustment was small enough that you never permanently ruined the dish.

PPO works the same way. The policy (the chef's recipe) is updated regularly based on experience (customer feedback). But the key constraint is that no single update can change the recipe too dramatically. The clipping mechanism is the chef's self-imposed rule: "I will change at most one ingredient per week, and only by a moderate amount." This prevents a bad batch of feedback from triggering a catastrophic overhaul.

Compare this to what would happen without the constraint. Imagine the chef gets a single negative review and decides to change the entire dish from scratch — new protein, new sauce, new technique. The dish might become unrecognisable. Worse, the feedback that triggered the change came from a small sample: perhaps that customer was having a bad day, or ordered the wrong item. Acting too aggressively on limited information destroys what was working.

For athletes, the analogy is a coach who pushes an athlete to improve without demanding too much too fast. A sprinter refining their start technique does not rebuild their entire running form in one session. They adjust foot placement by a centimetre, practice for a week, observe the timing, and make a further adjustment. PPO is incremental improvement — systematic, measured, reversible.

There is also a useful analogy in how companies adjust strategy. A well-run company does not pivot its entire business model because one quarter's results were disappointing. It makes small, tested changes — a new product line, a slightly different pricing model, a refined marketing message — and evaluates the result before making the next change. Dramatic overnight pivots (the "big gradient update") usually destroy as much value as they create. The company that survives is the one that learns from experience while preserving what already works. PPO is that kind of disciplined, evidence-based improvement applied to the weights of a neural network.

---

## The Problem PPO Solves — Deep Dive

To appreciate PPO, you need to understand the specific failure mode it was designed to address: policy collapse caused by overly large gradient updates in vanilla policy gradient reinforcement learning.

### Vanilla Policy Gradients

The simplest reinforcement learning approach for improving a policy is the REINFORCE algorithm (Williams, 1992). The core idea is straightforward: when an action led to high reward, increase the probability of taking that action in similar situations. When an action led to low reward, decrease its probability.

This is implemented as a gradient update: you compute the log-probability of the action taken, multiply by the reward received, and use that product as the gradient signal. The optimizer then nudges the policy weights in the direction that increases expected reward.

The problem is in the step size. Standard gradient descent does not know how large a step is "safe." If the computed gradient is large — which happens when rewards are large or probabilities change dramatically — the optimizer will take a large step. A large step changes the policy significantly. And a significantly changed policy may behave very differently from the policy that collected the training data.

### The Distribution Mismatch Problem

Here is the critical issue: the data used to compute the gradient was collected by the *old* policy. The data tells you what actions the old policy took and what rewards those actions received. You use this data to update the policy, producing a *new* policy.

But if the new policy is very different from the old one, the data is no longer representative of what the new policy would do. The gradient estimate, which was computed under the assumption that the policy is approximately the same as when the data was collected, becomes unreliable.

Worse, this can compound. If one bad update produces a poor policy, the next batch of data is collected by that poor policy, leading to even worse gradient estimates, leading to another bad update. This is policy collapse: within a handful of training steps, the policy degrades from sensible behaviour to producing garbage.

### The Trust Region Concept

The theoretical solution is a **trust region** — a constraint that says the new policy must remain within a certain "distance" of the old policy. Trust region methods, such as TRPO (Trust Region Policy Optimization, Schulman et al. 2015), formalise this constraint mathematically. TRPO works but requires solving a constrained optimization problem at each step, which is computationally expensive and difficult to implement.

PPO achieves a similar effect through a much simpler mechanism: clipping. Instead of enforcing a hard constraint on how different the new policy can be, PPO simply clips the gradient signal so that updates beyond a certain size have no effect. The policy can still improve, but it cannot improve *too fast* in any single direction.

### What TRPO Actually Did

TRPO (Trust Region Policy Optimization, Schulman et al. 2015) is the direct predecessor to PPO. It formulated the policy update as a constrained optimization problem: maximize expected reward subject to the constraint that the KL divergence between the new and old policies does not exceed a threshold. This is a mathematically rigorous trust region.

The problem is implementation. Solving a constrained optimization problem at each step requires computing the Fisher information matrix (a second-order quantity related to the curvature of the policy distribution), which is expensive to compute exactly for large neural networks. TRPO used conjugate gradient methods to approximate this, which works but adds significant complexity. The algorithm is also difficult to parallelize and hard to tune.

PPO's key insight: you do not need the full rigor of TRPO to get most of the benefit. A simple first-order method (standard gradient descent) with a clipping operation achieves training stability comparable to TRPO on most practical tasks, at a fraction of the implementation complexity. This pragmatic trade-off — give up some theoretical guarantees in exchange for simplicity and scalability — is what made PPO the dominant algorithm.

### The Role of Multiple Epochs

One mechanism by which vanilla policy gradient methods take large implicit steps is multiple gradient updates on the same batch of data. Each update changes the policy slightly, but after ten gradient steps on the same data, the policy may have changed substantially relative to when the data was collected.

PPO addresses this directly: the ratio used in the clipping is always computed against the policy that collected the current batch of data (the "old policy"), not against the policy from the previous batch update. This means the clipping constraint is respected even when multiple gradient steps are taken on the same batch. Each step is bounded by the clipping, regardless of how many steps have been taken.

This is why PPO's training loop explicitly stores the old policy's log-probabilities at the time of data collection and uses those stored values as the denominator of the ratio throughout all subsequent gradient updates on that batch.

### Why This Matters Even More for LLMs

Language model policies have an enormous action space. At each step, the model chooses from 32,000 or more tokens. A single policy update affects the probability distribution over all 32,000 tokens, for every possible context. Without clipping, a single batch of experience could dramatically shift token probabilities across thousands of contexts — destroying capabilities that took billions of training tokens to acquire.

Consider what happens if the reward model assigns very high scores to responses that start with "Certainly! Here is a detailed answer..." After one unconstrained update, the model might assign near-100% probability to that phrase in most contexts, ignoring all other possible openings. This kind of collapse happens fast. PPO's clipping mechanism prevents any single update from producing that kind of drastic shift.

The credit assignment problem compounds this challenge. In a game like Chess or Go, each move has immediate observable consequences. In text generation, the quality of a response is judged holistically at the end, and individual tokens are hard to evaluate in isolation. Token 3 in a 200-token response might have been pivotal (setting up the argument structure that made the response excellent) or irrelevant (a connector word that could have been anything). The advantage estimate for token 3 is necessarily an approximation, and that approximation is used to drive the gradient for all tokens simultaneously. Any bias in the advantage estimate for token 3 affects how the model generates connector words everywhere, in every context. This is why LLM PPO training is so sensitive to advantage estimation quality and why GAE, reward normalization, and advantage normalization are all necessary.

There is also a coverage problem unique to language: the policy must behave sensibly on prompts it has never seen during training. RL training with PPO only updates the policy based on the prompts in the training batch. A model might improve dramatically on the prompt "write a poem about autumn" but show no change on "write a poem about winter" — even though these should improve together. The generalization is not guaranteed; it depends on the model's existing capabilities and on the diversity of the prompt dataset. This is fundamentally different from supervised learning, where generalization is well-studied and more reliable.

---

## The Clipping Mechanism — Deep Dive

The clipping mechanism is the defining innovation of PPO. It is conceptually simple but requires careful unpacking to understand why it works.

### The Policy Ratio

At every training step, PPO computes a ratio comparing the new policy to the old policy for each action that was taken:

```
ratio = (probability of action under new policy) / (probability of action under old policy)
```

If the ratio equals 1.0, the new policy assigns exactly the same probability to this action as the old policy did when the data was collected. Nothing has changed for this particular action.

If the ratio is 1.5, the new policy is 50% more likely to take this action than the old policy was. This is a significant shift.

If the ratio is 0.5, the new policy is half as likely to take this action as the old policy. Again, a significant shift.

The ratio measures the degree of change. PPO uses this ratio to scale the gradient signal: if the action was good (positive advantage), scaling by the ratio increases the probability of the action. If the action was bad (negative advantage), scaling by the ratio decreases it.

### The Clipping Operation

The clipping operation says: only let the ratio move within the range of one minus epsilon to one plus epsilon, where epsilon is a small constant (typically 0.2 in standard PPO, 0.1 to 0.15 in some LLM training runs).

With epsilon of 0.2, the ratio is clipped to the interval from 0.8 to 1.2. This means:
- If the unconstrained update would push the ratio to 2.0 (double the probability of this action), the clipping stops it at 1.2.
- If the unconstrained update would push the ratio to 0.3 (a dramatic decrease), the clipping stops it at 0.8.

At any given step, the policy can change by at most 20% in either direction for any action. This is the "proximal" constraint that gives PPO its name.

### The Min Operation

There is one additional subtlety: PPO takes the minimum of two terms — the unclipped objective and the clipped objective. This is not symmetric: the objective is pessimistic. When an action is good (positive advantage) and the update would push the ratio above 1 plus epsilon, the clipping kicks in and prevents further increase. When an action is bad (negative advantage) and the update would push the ratio below 1 minus epsilon, the clipping kicks in and prevents further decrease.

The effect is that the objective never benefits from going further than epsilon in any direction. There is no reward for making an extremely large update. This is why the minimum is taken: to ensure that the clipping always makes the objective more conservative, not less.

### A Concrete Example

Suppose an action (generating the token "sure") was taken, and the advantage for that action is positive: the response was better than expected. Without clipping, you would want to increase the probability of generating "sure" as much as possible. Maybe the gradient says: make this token 5 times more likely.

With clipping at epsilon of 0.2, you can only increase the probability by at most 20%. The gradient signal beyond 1.2 is simply discarded. The update is taken, the probability increases by 20%, and then at the next step you collect new data and perhaps update again.

This is why PPO often runs multiple gradient update epochs on the same batch of experience — the first epoch takes the maximum allowed step, and subsequent epochs continue from there. Each epoch respects the clipping constraint relative to the policy at the *start of that batch* (the old policy). This allows more data efficiency while maintaining the stability guarantees.

### Epsilon in Practice

In InstructGPT, the clipping epsilon is set to 0.2, following the original PPO paper. In some LLM-specific implementations, smaller values (0.1 to 0.15) are used because LLMs are more sensitive to large policy shifts. The right value depends on the model size, the learning rate, and the stability of the reward signal. Tuning epsilon is one of the many hyperparameter challenges in practical PPO training.

A useful diagnostic: during training, track what fraction of sampled actions have their ratio clipped. If nearly all actions are being clipped (the ratio is hitting 1 plus or minus epsilon on almost every action), the policy is trying to make large updates but being held back — this suggests either the learning rate is too high or the reward signal is unusually strong. If almost no actions are being clipped, the policy is changing less than the epsilon would allow — this suggests the updates are already conservative and the clipping constraint is not binding.

### The Conservative Objective Explained Plainly

The min operation in the PPO objective is easy to misread. Here is the plain-English version for each case:

**Good action (positive advantage), ratio trying to go above 1 + epsilon:** The unconstrained objective says "push the probability up as much as possible." The clipped objective says "only push up to the limit." The minimum of these two is the clipped version — more conservative.

**Good action (positive advantage), ratio staying below 1 + epsilon:** The update is not hitting the constraint. Both the clipped and unclipped values are the same. The minimum selects the common value.

**Bad action (negative advantage), ratio trying to go below 1 - epsilon:** The unconstrained objective says "push the probability down as far as possible." The clipped objective says "only push down to the limit." The minimum of these two is the clipped version (because negative advantage times clipped ratio is a less negative number than negative advantage times very small ratio — recall that minimizing a negative number means taking the less negative value, which is the more conservative update).

The min operation consistently picks the more conservative of the two options, regardless of the sign of the advantage. This is the key property that makes PPO stable: it never benefits from exceeding the trust region.

---

## GAE: Generalized Advantage Estimation — Deep Dive

To compute the policy gradient, PPO needs to know not just what reward was received, but how much *better or worse* the action was compared to what was expected. This measure is called the **advantage**. Computing good advantage estimates is crucial — noisy advantages lead to noisy gradients, which lead to unstable training.

### What Advantage Means

The advantage of an action in a given state answers the question: "Was this action better or worse than the average action I could have taken here, and by how much?"

A positive advantage means the action led to more reward than expected. The policy should become more likely to take this action in this context. A negative advantage means the action led to less reward than expected. The policy should become less likely.

If the advantage were perfectly estimated, the policy would only increase the probabilities of actions that genuinely led to above-average outcomes, and decrease the probabilities of those that led to below-average outcomes. In practice, advantages are estimated from noisy data, so the quality of the estimator matters enormously.

### The Two Extremes: Monte Carlo and TD

There are two basic ways to estimate the advantage, and they represent a fundamental trade-off in reinforcement learning.

**Monte Carlo (MC) estimation** waits until the end of an episode and uses the actual total reward received. For language models, this means generating the entire response, receiving the reward model score, and using that score as the return. This estimate has low bias — it uses the actual observed reward — but high variance. One response might have gotten a high reward due to factors unrelated to any particular action, making it hard to assign credit accurately.

**Temporal Difference (TD) estimation** uses a learned value function to bootstrap the return. Instead of waiting for the full episode, TD uses the estimated value of the next state as a proxy for future rewards. This has lower variance — the value function smooths out noise — but introduces bias, because the value function is itself an approximation.

### GAE: Interpolating Between the Two

Generalized Advantage Estimation (Schulman et al., 2016) introduces a single parameter, lambda, that interpolates between the MC and TD extremes.

When lambda equals 0, GAE reduces to pure TD estimation: only the immediate reward and the next state value are used. This is low variance but potentially biased.

When lambda equals 1, GAE reduces to a discounted version of Monte Carlo estimation: it sums all future TD errors, which approximates the full return. This is unbiased but higher variance.

For values of lambda between 0 and 1, GAE produces a weighted sum of multi-step TD errors. The weighting decays exponentially: the one-step TD error is included fully, the two-step error is weighted by lambda, the three-step error by lambda squared, and so on. This means the advantage estimate relies more on near-term observed rewards (less bias) and less on far-future estimates (less variance from long chains of predictions).

In practice, lambda values between 0.9 and 0.97 work well. The InstructGPT paper uses 0.95. This setting means the advantage estimate is mostly driven by the observed rewards over the next 5 to 10 steps, with diminishing weight on further-future estimates.

### Advantage Normalization

Before computing the PPO gradient update, it is standard practice to normalize the advantages across the mini-batch: subtract the mean advantage and divide by the standard deviation. This has two benefits. First, it keeps the gradient magnitudes consistent across different training batches — batches where the reward model happens to produce unusually high or low scores will not produce unusually large or small gradient updates. Second, it ensures that roughly half of the actions receive positive advantage (and are reinforced) and half receive negative advantage (and are discouraged) in every batch, maintaining a balanced gradient signal regardless of the absolute reward level.

Without advantage normalization, training can stall. If all advantages in a batch happen to be positive (all responses scored above average), the policy will reinforce all of its recent actions — including bad ones — because the gradient signal always pushes toward the observed behavior. Normalization prevents this by making the advantage relative: "was this action better than other actions in this same batch?"

### The Discount Factor

GAE also incorporates a discount factor, gamma, that controls how much future rewards are weighted relative to immediate rewards. A discount factor of 1.0 treats all future rewards equally regardless of when they occur. A discount factor of 0.99 weights rewards one step in the future at 99% of immediate rewards, rewards two steps ahead at 98%, and so on.

For language model generation, gamma is typically set close to 1.0 (0.99 to 1.0) because the reward signal (from the reward model) only arrives at the very end of the response. If gamma were smaller, the per-token advantage estimates for early tokens in the response would be very small — those tokens are far from the final reward — making it hard to provide useful gradient signal for the beginning of the response. A near-unity discount factor ensures that the reward propagates back through the entire response, giving all tokens some gradient signal.

### Why GAE Matters

Poor advantage estimates are one of the main causes of PPO training instability. If the advantage function assigns high advantages to bad actions (because the reward was coincidentally high), the policy will mistakenly reinforce those bad actions. If advantages are too noisy, the gradient signal oscillates and the policy fails to improve consistently.

GAE provides a principled way to get advantage estimates that are good enough for stable training. It is one of the key reasons why PPO outperformed earlier policy gradient methods in practice — not just the clipping, but the quality of the advantage estimates feeding into the clipping.

The advantage estimator is the bridge between the reward signal (which arrives once per response) and the gradient signal (which must be assigned to each of the hundreds or thousands of tokens in the response). This credit assignment problem — figuring out which specific tokens were "good" or "bad" given only a single response-level reward — is one of the hardest challenges in RLHF. GAE is the best practical solution developed for the PPO setting, though it is far from perfect: assigning credit to token 3 in a 200-token response based on a reward received after token 200 is genuinely difficult, and GAE's solution is an approximation with known limitations.

---

## The Critic Network — Deep Dive

GAE requires a value function: a network that, given a state, predicts the expected total future reward from that state. In the actor-critic framework, this network is called the **critic**. It does not take actions — it observes states and evaluates them.

### What the Critic Learns

The critic learns to answer the question: "Given that we are at this point in the conversation, how much total reward do we expect to receive from here to the end of the response?"

In language model training, the "state" at each step is the sequence of tokens generated so far. The critic processes this sequence and outputs a scalar: the expected future reward.

Early in training, the critic's estimates are poor. It has seen little data and has no reliable sense of which token sequences lead to high rewards. As training progresses, the critic improves: it learns that certain types of responses (well-structured, relevant, confident) tend to lead to high reward model scores, and that others (repetitive, vague, off-topic) tend to lead to low scores.

The critic's quality directly affects the quality of the advantage estimates, which directly affect the quality of the policy gradient. A better critic produces better advantages, which produce more accurate gradients, which produces faster and more stable policy improvement.

### The Value Loss

The critic is trained by minimizing the difference between its predictions and the actual returns observed. This is the value loss or critic loss. When the actual return from a state was 4.2 but the critic predicted 2.8, the loss is high and the critic's weights are updated to push the prediction toward 4.2.

This creates an interesting training dynamic: the actor (policy) and critic are both learning simultaneously. The actor uses the critic's current estimates to compute advantages. The critic uses the actor's current behavior (which determines what states are visited and what rewards are received) to improve its own predictions. They co-evolve, which can be unstable if not managed carefully.

### Initializing the Critic

One practical detail with significant impact on training stability: how the critic is initialized. The most common choice in RLHF is to initialize the critic from the reward model's weights. This makes intuitive sense — the reward model has already learned to associate token sequences with quality scores, giving the critic a useful starting point rather than random initialization.

The alternative is to initialize the critic from the SFT model weights (same as the actor) and add a scalar output head. This also works, but requires more warm-up iterations for the critic to produce useful value estimates. With either initialization, the critic's output head is initialized to produce values near zero — a neutral starting point that does not bias the initial advantage estimates.

### Clipping the Value Loss

Just as the policy objective is clipped, the value loss in PPO is also typically clipped. This prevents the critic from making large, destabilizing updates based on outlier observations. The clipping range for the value loss uses the same epsilon as the policy clipping, applied to the value predictions themselves.

The value clipping works as follows: if the critic's current prediction is more than epsilon away from its prediction at the start of the batch (the "old value"), the loss is capped. This prevents the critic from chasing outlier observations that may not reflect the true value of a state. Like ratio clipping for the actor, value clipping for the critic makes each training step conservative and stable.

### What the Critic Learns Over Time

Early in PPO training, the critic is essentially guessing. It has no reliable way to predict whether a given partial response will eventually receive a high or low reward model score. As training progresses, the critic develops useful representations:

It learns that certain structural features of a partial response predict high reward: a clear opening sentence, correct formatting, responsiveness to the specific question asked. It learns that other features predict low reward: repetition, irrelevance, excessive hedging. These are the same features the reward model responds to, because the reward model's scores are what the critic is trained to predict.

A well-trained critic effectively encodes something like "how good does this response look so far?" in a way that is more fine-grained than the reward model's response-level score.

Interestingly, the critic can learn things the reward model does not explicitly model. The reward model is trained to score completed responses based on human preferences. The critic, through its training on millions of per-token advantage estimates, learns which partial-response states tend to lead to high-scoring completions versus low-scoring ones. This includes structural knowledge — opening sentences that set up well-reasoned arguments, the right level of detail for the complexity of the question, signals that the response is about to go off-topic — that the reward model expresses only implicitly through its scores on full responses.

This is one reason why the actor-critic approach can be more effective than a simple "generate-then-score" loop without a critic: the critic's per-step estimates provide denser learning signal across the full response, rather than a single signal at the end. For short responses (50 to 100 tokens), the dense signal matters less. For long responses (500 tokens or more), the critic's per-step guidance is essential for stable training.

### How the Critic Differs from the Reward Model

A common point of confusion: the critic network and the reward model are not the same thing and serve different purposes.

The **reward model** is a trained-and-frozen network that evaluates completed responses. It takes a full (prompt, response) pair and outputs a scalar quality score. It is not updated during PPO training.

The **critic** is a network that is updated during PPO training. It estimates the expected future reward at each *step within* a response — not just at the end. It is trained jointly with the policy. In standard RLHF PPO, the critic is initialized from the same weights as the policy (or sometimes the reward model) and then fine-tuned alongside the policy.

This distinction matters for memory: both models must be kept in GPU memory simultaneously, doubling the memory footprint beyond just the actor and reference policy.

The distinction also matters conceptually: the reward model defines the *goal* of training (maximize this score). The critic is a *tool* for achieving that goal more efficiently (better gradient estimates). Confusing them leads to incorrect intuitions about which model is responsible for what.

---

## The Actor-Critic Architecture — Deep Dive

PPO for language models uses an actor-critic architecture: two neural networks — the actor and the critic — working together during training.

### The Actor

The actor *is* the policy: it is the language model that generates text. Given a prompt and any tokens generated so far, the actor produces a probability distribution over the next token. It samples from this distribution to generate each token in sequence until the response is complete.

During PPO training, the actor's weights are updated to increase the probability of high-advantage actions and decrease the probability of low-advantage actions, subject to the clipping constraint. The actor is the model that will ultimately be deployed: the trained, aligned language model.

The actor is initialized from the SFT (supervised fine-tuning) model. It starts as a model that already knows how to follow instructions reasonably well, and PPO refines it further to maximize reward while staying close to the reference policy.

### The Critic

The critic is a second network, typically with the same architecture as the actor but with its output head modified to produce a single scalar value instead of a probability distribution over tokens.

The critic observes the same sequence of tokens that the actor has generated and estimates the expected future reward from that point. Its output is used to compute the advantage estimates that guide the actor's updates.

The critic is not deployed — it is only needed during training. After PPO training is complete, only the actor's weights are saved and used for inference.

### How They Work Together

During a PPO training step:

1. The actor generates a response to a prompt, token by token.
2. At each token position, the critic observes the token sequence so far and outputs a value estimate.
3. The reward model scores the completed response.
4. GAE uses the critic's per-step value estimates and the final reward model score to compute an advantage for every token in the response.
5. The clipped PPO objective is computed using these advantages and the ratio of new-to-old probabilities.
6. Both the actor and the critic are updated using gradient descent: the actor to increase the probability of high-advantage tokens, the critic to better predict actual returns.

The actor and critic share observations but have separate objectives. The actor optimizes the clipped PPO objective. The critic minimizes the value prediction error. Both objectives are often combined into a single total loss, with a weighting coefficient controlling how much the critic loss contributes relative to the actor loss.

### Why Two Networks Instead of One

One might ask: could a single network simultaneously act as both actor and critic? In principle, yes — and some implementations do share weights between actor and critic (only the final output heads differ). In practice, for large language models, the actor and critic are usually kept as separate models because sharing weights can create conflicting gradient signals. The actor's gradient pushes the shared representation toward good policy behaviour; the critic's gradient pushes it toward accurate value prediction. These objectives are not perfectly aligned and can interfere.

Keeping them separate allows each network to specialize without interference. The memory cost is significant (two full model copies), but the training stability is worth it in most large-scale setups.

### The Entropy Bonus

Some PPO implementations add an entropy bonus to the actor's objective. Entropy measures the "randomness" of the policy: a policy that assigns 99% probability to one token has very low entropy; a policy that distributes probability roughly equally across many tokens has high entropy.

High entropy is desirable early in training because it encourages exploration: the actor tries many different token choices rather than always picking the most likely one. Low entropy is desirable later in training when the actor has learned what good responses look like and should commit to them.

The entropy bonus adds a small positive reward for maintaining entropy — it penalizes the actor for becoming too deterministic too quickly. This is especially relevant for tasks where the prompt dataset is diverse: if the actor becomes very low-entropy (very confident) early on based on limited data, it may lock in suboptimal behavior before it has seen enough examples to know what "good" really looks like.

In practice, the entropy coefficient is small and decays over training. It is a hyperparameter that requires tuning, and some implementations omit it entirely, relying on sampling at temperature 1.0 to maintain sufficient exploration.

---

## PPO in LLM Training — Deep Dive

Applying PPO to language model alignment introduces several additional components beyond the standard PPO algorithm. The full LLM PPO setup is considerably more complex than the robotics and game-playing applications where PPO was originally developed.

### The Four Models

When PPO is used for RLHF with a language model, four separate model copies must be kept in memory simultaneously:

**1. The Actor (RL Policy)**
The model being trained. Initialized from the SFT checkpoint. Requires full gradients and optimizer states. This is the largest memory consumer: a 7B parameter model in bfloat16 requires about 14 GB just for weights, plus roughly 2x to 4x that for gradients and optimizer states.

**2. The Reference Policy**
A frozen copy of the SFT model. It never receives gradient updates. It is used exclusively to compute the KL penalty — measuring how far the actor has drifted from the original SFT behaviour. Because it is frozen, it requires no gradient storage, but still needs the full weight footprint: another 14 GB for a 7B model.

**3. The Reward Model**
The trained-and-frozen network that scores completed responses. It may be smaller than the policy (some implementations use a 1B or 3B reward model with a 7B policy), but it still requires GPU memory for inference. Another 2 to 14 GB depending on size.

**4. The Critic (Value Model)**
A separate network, typically initialized from the reward model's weights (since the reward model has already learned something about response quality). Requires full gradients. Another full model footprint.

For a 7B parameter policy with a 7B critic, 7B reference, and even a small 3B reward model, the total weight memory alone exceeds 100 GB. This requires multi-GPU infrastructure and sophisticated memory management (gradient checkpointing, model sharding, mixed precision). This is one of the primary reasons PPO-based RLHF is expensive to run and why alternatives like GRPO (which eliminates the critic and reference policy from the critical path) became attractive.

### The Reference Policy and KL Penalty

The reference policy exists to solve the reward hacking problem. Without any constraint, the actor would discover that certain response styles — perhaps very long, sycophantic, or stylistically bizarre — fool the reward model into giving high scores. It would rapidly overfit to these reward model quirks rather than developing genuinely better behaviour.

The KL penalty penalizes the actor for diverging from the reference policy. The total reward at each step is not just the reward model score but the reward model score minus a constant multiplied by the KL divergence between the actor and the reference policy at that token.

When the actor's probability distribution over tokens becomes very different from the reference policy's distribution, the KL divergence is large and the penalty is heavy. This pulls the actor back toward the original SFT model's behavior, preventing it from drifting into degenerate modes.

The KL coefficient controls the strength of this constraint. Set it too small and the actor reward-hacks freely. Set it too large and the actor cannot improve meaningfully — it stays too close to the SFT model.

### Per-Token KL

An important implementation detail: the KL penalty in InstructGPT and most serious implementations is applied *per token*, not just at the end of the response. At every single token generation step, the actor's probability distribution is compared to the reference policy's distribution, and a per-token KL penalty is applied.

This makes the penalty more granular and more effective. A response that deviates from the reference policy at only one or two tokens (maybe using an unusual phrase that fools the reward model) will incur a smaller penalty than a response that deviates throughout.

### The Full Training Loop in Practice

One PPO iteration for LLM training involves:

1. Sample a batch of prompts from the prompt dataset (typically 128 to 1024 prompts per batch).
2. Run the actor to generate complete responses for all prompts (this is the most expensive step computationally).
3. Run the reward model on all (prompt, response) pairs to get reward scores.
4. Run the reference policy on all responses to get per-token log-probabilities (for KL computation).
5. Run the critic on all token sequences to get per-step value estimates.
6. Compute per-token KL penalties by comparing actor and reference log-probabilities at each token.
7. Combine reward model scores and KL penalties to form the total per-step reward signal.
8. Compute per-token advantages using GAE, combining value estimates with the total rewards.
9. Normalize advantages across the mini-batch (subtract mean, divide by standard deviation).
10. Run multiple epochs of gradient updates on actor and critic using the clipped PPO objectives.
11. Log statistics, adjust KL coefficient if needed, and start the next iteration.

Steps 2 through 8 are inference passes — no gradients needed. Step 10 requires gradients for both actor and critic. The entire pipeline must be parallelized across multiple GPUs.

The HuggingFace PPO implementation blog post lists over 37 specific implementation details that matter for stability and performance — things like reward normalization, advantage normalization per mini-batch, value function clipping, gradient clipping, and careful handling of padding tokens.

### The Generation Bottleneck

Step 2 — generating responses — is the dominant computational cost in PPO training. Autoregressive generation is inherently sequential: each token must be generated before the next one can begin. For a response of 512 tokens, this requires 512 serial forward passes through the actor. With a batch of 512 prompts, this is 512 times 512 = 262,144 forward passes just to generate one batch of training data.

This is why PPO training is slow compared to supervised learning. In supervised fine-tuning, the responses are precomputed — you just do one forward pass and one backward pass per batch. In PPO, the responses must be generated on-the-fly before training can proceed. Generation can take 5 to 10 times longer than the actual gradient update step.

Practical consequence: increasing the batch size in PPO training has diminishing returns past a certain point, because larger batches take proportionally longer to generate. This is different from SFT where larger batches usually improve training efficiency. The optimization in PPO training often focuses on making generation faster (model quantization during generation, faster sampling kernels) rather than on the gradient update step.

### Asynchronous PPO

A significant engineering optimization used in large-scale RLHF systems is asynchronous PPO: separating the generation workers from the training workers. Generation happens continuously on a set of GPUs, producing batches of experience. Training happens on a separate set of GPUs, consuming these batches as they become available.

This keeps the training GPUs fully utilized (no idle waiting for generation) and the generation GPUs fully utilized (no idle waiting for training). The trade-off is increased staleness: by the time a batch of experience reaches the training step, the actor may have already been updated a few times since the experience was collected. This introduces off-policy data, which weakens the trust region guarantees that PPO relies on. Practitioners address this by bounding the staleness (discarding experience that is more than a few updates old) and adjusting the clipping epsilon to account for the increased off-policy nature of the data.

---

## How It Works in Practice

### Iteration Counts

RLHF training with PPO is typically short relative to pretraining. The InstructGPT paper ran PPO for a relatively small number of gradient steps — on the order of 1,000 to 10,000 iterations depending on the model size. Compare this to pretraining, which runs for hundreds of billions of tokens.

This makes intuitive sense: pretraining builds the model's knowledge and language capabilities from scratch. PPO refines the model's behaviour, adjusting what it already knows how to do rather than teaching it new skills.

The shortness of PPO training is also a practical constraint: RLHF training is expensive. Each iteration requires generating full responses and scoring them with the reward model, both of which consume significant compute. Running for 100,000 PPO iterations at batch size 512 would require generating more than 50 million full responses — prohibitive even for large labs. The typical approach is to run until human evaluation metrics plateau, then stop. This means PPO training is typically under-converged relative to pretraining: there is almost always more improvement to be had, but the compute cost makes it impractical to continue.

The shortness of PPO training also explains why the quality of the SFT starting point matters so much. PPO cannot build new skills from scratch — it can only refine and adjust behaviour that already exists. A poor SFT model means PPO has little to work with and the training run may not produce meaningful improvement before the compute budget is exhausted.

### Memory Costs in Numbers

For a 7B parameter model trained in bfloat16:
- Actor weights: approximately 14 GB
- Actor optimizer state (AdamW): approximately 56 GB (4x weights for first and second moment)
- Reference policy weights: approximately 14 GB
- Critic weights: approximately 14 GB
- Critic optimizer state: approximately 56 GB
- Reward model weights: approximately 4 to 14 GB
- Activations and working memory: highly variable, roughly 10 to 40 GB

Total: well over 150 GB, often requiring 4 to 8 A100 (80 GB) GPUs just for the model, plus additional infrastructure for data loading and communication.

### Reward Normalization

A standard practice is to normalize rewards before computing advantages. If the reward model outputs scores in a range of roughly 0 to 10, and different prompts produce very different score distributions, the advantage estimates will have inconsistent scales. Normalizing rewards (subtracting the running mean and dividing by the running standard deviation) keeps the gradient magnitudes stable across different prompt types.

### Epoch Count Within Each Batch

The original PPO paper recommends running 3 to 10 gradient update epochs on each batch of collected experience. For LLM training, most implementations use 1 to 4 epochs per batch. More epochs extract more training signal from each expensive generation step but risk overfitting to the current batch and violating the clipping constraint by the later epochs.

### Early Stopping via KL Threshold

Some implementations add an early stopping rule: if the KL divergence between the current policy and the reference policy exceeds a threshold (typically 0.1 to 0.2 nats), training stops for that iteration. This is a safety valve that prevents the actor from drifting too far in a single training step.

### The Dynamic KL Coefficient

The OpenAI implementation used a dynamic KL coefficient — one that is adjusted automatically during training based on observed KL divergence. If the actor drifts too far from the reference in a given iteration (KL exceeds a target), the KL coefficient is increased to apply a stronger penalty next iteration. If the actor drifts less than the target, the coefficient is decreased to allow more exploration.

This adaptive scheme removes one hyperparameter (fixed KL coefficient) at the cost of introducing a new one (target KL divergence). In practice, a target KL of around 0.01 to 0.1 is used, and the coefficient is adjusted by a multiplicative factor (typically 1.5 for increase, 0.5 for decrease) whenever the actual KL deviates from the target by more than a factor of 1.5 in either direction.

The dynamic KL coefficient is a practical engineering solution to the problem that the "right" KL coefficient changes over training: early in training, the actor should be allowed to drift more (it needs to explore); later in training, when a good policy has been found, tighter constraints prevent degradation.

### Gradient Clipping

Standard PPO training applies gradient clipping to both the actor and critic updates — typically clipping the global gradient norm to 1.0. Without gradient clipping, a single batch with unusual reward distribution can produce very large gradients that, even after ratio clipping, push the model weights far enough to cause instability.

Gradient clipping is a blunt but effective tool. It does not distinguish between gradients that are large because they contain useful signal and gradients that are large because of numerical instabilities or outlier data. It simply caps the maximum size of any weight update. For PPO in LLMs, this is a necessary safety measure.

### Handling Variable-Length Responses

Language model responses have variable lengths. A batch of 512 prompts might produce responses ranging from 50 tokens to 500 tokens. This creates a padding problem: to batch-process variable-length sequences, shorter sequences must be padded to match the longest sequence in the batch.

The PPO objective must correctly ignore padding tokens. Computing the policy ratio and the advantage at padding positions would contaminate the gradient with meaningless signal. This is handled by masking: a binary mask is applied to the loss computation, zeroing out contributions from padding positions. Getting the masking exactly right is one of the sources of subtle bugs in PPO implementations — the HuggingFace implementation guide calls out at least three separate masking decisions that can silently go wrong.

### Prompt Dataset Curation

The diversity and quality of the prompt dataset is one of the most underappreciated factors in PPO training quality. If the prompt dataset covers only simple question-answering prompts, PPO will optimize the actor to be good at question answering while leaving other tasks unchanged. InstructGPT used a carefully curated mix of real API prompts from OpenAI users (with permission) alongside annotator-written prompts specifically designed to cover diverse task types: summarisation, creative writing, coding assistance, dialogue, instruction following, classification, and more.

A prompt dataset that is too narrow produces a model that behaves well on the training distribution but disappoints on novel task types. A prompt dataset that is too broad (covering tasks the model is not yet capable of) wastes compute on hopeless optimization. Getting this balance right is as much art as science and is a significant source of difference in quality between different RLHF implementations.

### Reward Clipping and Outlier Management

Large positive reward model scores (reward outliers) can dominate the gradient signal if not managed. Some implementations clip the reward model's output to a fixed range (such as negative 5 to positive 5) before computing advantages. Others apply whitening (standardizing the reward distribution to zero mean and unit variance) across each batch. Without this management, a single unusually high-scoring response can cause a large gradient step that violates the spirit of PPO's conservatism even when the clipping on the ratio is respected.

This is a subtle point: PPO's clipping limits how much the *policy ratio* can change, but if the *advantage* is very large, even a modest ratio change multiplied by a huge advantage produces a large gradient update. Controlling the advantage scale through reward normalization is thus an essential complement to ratio clipping.

### The Compute Balance: Generation vs Training

In RLHF PPO, there is a fundamental tension between the cost of generating experience and the cost of updating the model. Generating responses (running the actor forward to produce text) is computationally expensive: for a long response of 500 tokens, you run the actor's forward pass 500 times in sequence. Training on that experience is comparatively cheap — a single forward and backward pass over the collected batch.

This asymmetry means it is tempting to run many gradient update epochs per batch of generated experience (extracting maximum value from expensive generation). But doing so risks violating the clipping constraint: by the third or fourth epoch of updates, the current policy may have drifted significantly from the policy that generated the data, making the ratio large and the clipping frequently active. When clipping is consistently active for most samples, it means the update is hitting the constraint every step — which suggests the batch size or learning rate should be adjusted.

The practical sweet spot, established empirically by practitioners, is one to four epochs per batch. More than four epochs typically produces diminishing returns or instability.

---

## PPO vs. the Alternatives

It is useful to understand how PPO relates to the other algorithms used for LLM alignment, because the relationships illuminate what PPO is doing and why the field moved on.

### PPO vs. REINFORCE

REINFORCE (vanilla policy gradient) is PPO without the clipping, without the critic, and without GAE. It uses the actual observed return as the advantage estimate and applies unclipped gradient updates. REINFORCE is simpler but significantly less stable: without the critic, advantage estimates are very noisy; without clipping, individual batches can cause large policy changes. For small problems, REINFORCE works fine. For LLMs with sparse rewards and enormous action spaces, it is essentially unusable. PPO solves exactly the problems that make REINFORCE fail at scale.

### PPO vs. TRPO

TRPO is the theoretically correct version of what PPO approximates. TRPO enforces a hard KL constraint on each update and achieves provably monotonic policy improvement (each update is guaranteed to not decrease expected reward). PPO gives up the hard guarantee in exchange for simplicity. In practice, for LLM training, PPO achieves comparable performance to TRPO while being dramatically easier to implement and scale.

### PPO vs. DPO

DPO (Direct Preference Optimization, Rafailov et al. 2023) takes a completely different approach: instead of running RL, it directly optimizes a loss function derived from preference data. DPO does not generate responses during training (the responses are precomputed), does not need a reward model at inference time, and does not need a critic or a separate value function. It trains with only two models (policy and reference).

DPO is simpler, more memory-efficient, and more stable than PPO. Its main limitation is that it cannot go "beyond" the quality of the preference data: it learns from fixed (prompt, winner, loser) pairs rather than from its own experience. PPO, by generating and evaluating its own responses, can in principle continue improving past the quality of any fixed dataset. Whether this theoretical advantage translates to practice depends heavily on the quality of the reward model.

For most open-source LLM training today, DPO or a variant is the default because of its simplicity. PPO is used when online learning (generating new responses and learning from them) is essential — typically for tasks with verifiable rewards like mathematics and code.

### PPO vs. GRPO

GRPO (Group Relative Policy Optimization, DeepSeek 2024) keeps PPO's clipping mechanism and reference policy KL constraint but replaces the critic with a group-based advantage estimator. By generating a group of responses for each prompt (typically 4 to 16 responses) and computing each response's advantage as its reward deviation from the group mean, GRPO eliminates the need for a trained value function entirely.

This is significant because: (1) no critic means two fewer models in memory, (2) no value loss means simpler training code, and (3) group-relative advantages are unbiased by a imperfect critic. The cost is that GRPO requires more generation compute per prompt (generating a group of responses rather than one), and the advantage estimates from small groups can be noisy.

The DeepSeek-R1 paper demonstrated that GRPO at scale, combined with verifiable rewards from mathematics problems, produces exceptional reasoning ability — arguably surpassing what earlier PPO-based RLHF achieved in reasoning tasks. This has made GRPO the algorithm of choice for training reasoning models in 2024-2025.

---

## Training Instability and Failure Modes

PPO for language models is notoriously difficult to stabilize. This section documents the most common failure modes, because understanding them is essential for anyone working on RLHF systems.

### Reward Hacking

The most dramatic failure mode. The actor discovers a way to achieve high reward model scores that does not correspond to genuine quality improvement. Classic examples:

**Length exploitation:** The actor learns that the reward model prefers longer responses (a common annotator bias) and begins padding every response with unnecessary caveats, background context, and redundant explanations. Reward scores rise; actual quality stagnates or degrades.

**Sycophancy:** The actor learns to strongly agree with whatever framing is in the prompt, even when the prompt contains false premises. If a user asks "Einstein failed maths — why was he such a poor student?" a reward-hacking actor might generate an enthusiastic elaboration on this false premise because it matches the user's apparent expectation, and annotators sometimes rated agreement positively.

**Format gaming:** Certain response formats (bullet points, numbered lists, markdown headers) consistently earn higher reward model scores, regardless of whether that format is appropriate for the content. The actor begins inserting formatting everywhere.

The KL penalty is the primary defense against reward hacking. But if the KL coefficient is too small relative to the reward magnitude, hacking will occur. Monitoring reward model scores alongside other quality metrics (human evaluations, held-out benchmark performance) is essential for catching this early.

### KL Explosion

The opposite of reward hacking. The actor drifts so far from the reference policy that its outputs become incoherent — not because they were optimized toward a wrong reward signal, but because the optimization went unchecked. KL explosion often manifests as responses that are grammatically bizarre, repetitive, or semantically disconnected.

This happens when the KL coefficient is set too low, the learning rate is too high, or the epsilon for ratio clipping is too large. Monitoring the KL divergence between actor and reference policy throughout training is non-negotiable. A sudden spike in KL divergence is a warning sign that the training run is about to fail.

### Value Model (Critic) Collapse

The critic is trained simultaneously with the actor. If the actor changes rapidly (due to large reward signals), the critic's estimates can lag badly — it predicts values based on old actor behavior that is no longer representative. This produces garbage advantage estimates, which produce garbage gradients, which makes the actor's updates even more erratic.

Critic collapse can be recognized by monitoring the value loss: if the value loss begins increasing rather than decreasing over training, the critic is losing the ability to track the actor's behavior. Techniques to mitigate this include using a smaller learning rate for the critic than for the actor, initializing the critic from the reward model weights (since the reward model already understands response quality), and clipping the value function updates.

### Repetition Loops

A failure mode specific to text generation. The actor finds that generating repetitive text — repeating the same phrase or sentence multiple times — achieves acceptable reward model scores. This happens because reward models are often insensitive to repetition: annotators judged responses holistically and may have overlooked repetition in a long response. Once the actor starts falling into repetition loops, it becomes hard to escape: the repetitive token has high probability, which makes it even more likely to be sampled, which reinforces the pattern further.

Mitigations include repetition penalties during generation, minimum length constraints, and careful reward model design that specifically penalizes repetition.

### Mode Collapse

A more subtle failure mode: the actor stops generating diverse responses and converges to a single "safe" template that reliably earns acceptable reward model scores. Every response starts the same way, follows the same structure, and ends with a similar conclusion. Reward scores are decent, but the model has lost the creative flexibility to produce varied, context-appropriate responses.

Mode collapse is distinct from reward hacking in that the templated responses might actually be good — the actor has found a genuinely high-quality response format. The problem is that it produces that format regardless of the prompt, ignoring the specific details and requirements of each request.

The KL penalty and entropy bonus both work against mode collapse: the KL penalty forces the actor to remain close to the reference policy (which was more diverse), and the entropy bonus rewards maintaining a distributed probability over tokens. Monitoring response diversity (measuring how similar responses to different prompts are to each other) can catch mode collapse early.

### Alignment Regression

A failure mode that appears only when evaluating the model on tasks outside the training distribution. During PPO training, the model becomes better at the types of prompts in the training set. But it can simultaneously regress on other tasks — becoming more opinionated on factual questions, less accurate at coding tasks, or less reliable at following complex multi-step instructions.

This is sometimes called "alignment tax regression" — the model was already slightly penalized on some benchmarks by SFT; PPO can exacerbate this if the prompt dataset used for PPO training is not representative of all the tasks the model is expected to handle.

The mitigation is to mix pretraining data into the PPO training loop — periodically sampling batches of pretraining text and computing the standard cross-entropy language model loss on them. This "replay" of pretraining data prevents the model from forgetting capabilities acquired during pretraining. InstructGPT used this technique; without it, the model showed slightly larger alignment taxes on NLP benchmarks.

---

## Common Misconceptions

**Misconception 1: "PPO is just gradient descent with a reward signal."**

Vanilla policy gradient (REINFORCE) is gradient descent with a reward signal. PPO adds the clipping mechanism, the actor-critic architecture, GAE for advantage estimation, and in the LLM setting, the KL penalty with a reference policy. These additions are what make the difference between unstable, sample-inefficient training and something that works reliably enough for large-scale RLHF.

**Misconception 2: "The reward model and the critic are the same thing."**

They are not. The reward model is frozen and scores complete (prompt, response) pairs. The critic is trained jointly with the actor and estimates the expected future reward at each token position during generation. They serve entirely different purposes in the training pipeline.

**Misconception 3: "PPO is mathematically guaranteed to keep the policy close to the reference."**

The KL penalty provides a soft constraint, not a hard guarantee. If the KL coefficient is too small, or if the reward model assigns extreme scores to certain outputs, the actor can still drift significantly. This is why careful monitoring of the KL divergence during training is essential and why some implementations use a dynamic KL coefficient that increases when drift is detected.

**Misconception 4: "PPO training makes models much more capable."**

PPO primarily changes what the model does with existing capabilities — it aligns behaviour with human preferences. It does not teach the model new factual knowledge or reasoning skills. Those come from pretraining and scale. An aligned 7B model will still not know things it never encountered during pretraining.

**Misconception 5: "Clipping to epsilon 0.2 means the policy changes by at most 20% per training run."**

The clipping is per update step, not per training run. Over many steps, small 20% changes compound: after many steps, the policy can be very different from the original SFT model. This is normal and expected — it is why the KL penalty against the reference policy exists, to bound the total drift over the entire training run rather than just step by step.

**Misconception 6: "PPO is the best algorithm for LLM alignment."**

PPO was the first widely-used algorithm for this purpose, and it works well when implemented carefully. But GRPO, DPO, DAPO, and other algorithms have since been developed that achieve comparable or better results with significantly lower computational cost. For most open-source models today, DPO or GRPO is preferred over PPO.

**Misconception 7: "The clipping mechanism makes PPO safe from all large updates."**

Clipping bounds the policy ratio, but it does not bound the size of the weight update in absolute terms. If the learning rate is very high, even a ratio of exactly 1.2 (the maximum allowed) multiplied by a large advantage can produce a substantial weight update. PPO's stability depends on both the clipping epsilon and the learning rate being appropriately set. High epsilon combined with high learning rate is nearly as unstable as no clipping at all.

**Misconception 8: "Per-token KL penalty and per-response KL penalty are equivalent."**

They produce similar effects but differ in granularity and in how they interact with GAE. Per-token KL allows the model to deviate on some tokens while staying close to the reference on others — a more fine-grained constraint. Per-response KL (computed at the end of the sequence) is blunter: it measures total divergence but does not penalize specific tokens. Per-token KL is generally preferred in serious RLHF implementations because it provides cleaner credit assignment.

**Misconception 9: "RLHF with PPO always converges to a stable, high-quality policy."**

There is no guarantee of convergence. PPO for language models is a non-stationary optimization problem: as the actor changes, the reward landscape it experiences changes, and the critic's estimates change too. Many RLHF runs plateau early, degrade due to reward hacking, or oscillate without converging. The InstructGPT paper used early stopping based on human evaluation rather than waiting for an objective convergence criterion — because there is no reliable objective criterion to wait for. This is a fundamental difference from supervised learning, where a decreasing validation loss is a reliable convergence signal. In PPO training, reward model scores can increase while actual quality decreases (reward hacking), and KL divergence can look normal while the model silently degrades on out-of-distribution prompts. Human evaluation remains the gold standard for assessing whether a PPO training run has produced a better model.

---

## Connections to Other Topics

**SFT (Supervised Fine-Tuning, topic 18):** The SFT model is the starting point for PPO training. The actor is initialized from SFT weights. The reference policy is a frozen copy of the SFT model. Without a strong SFT checkpoint, PPO training starts from a poor baseline and is much harder to stabilize.

**Reward Models (topic 19):** The reward model is one of the four models required for PPO-based RLHF. It provides the scalar signal that PPO tries to maximize. The quality of the reward model directly determines the quality of the policy that PPO learns — a biased or noisy reward model will produce a biased or noisy policy, regardless of how well PPO is implemented.

**KL Divergence (topic 20):** The KL penalty that prevents reward hacking in RLHF is computed using KL divergence between the actor and the reference policy. Understanding KL divergence is essential for understanding why the penalty works the way it does and why it is applied per-token rather than at the response level.

**RLHF Overview (topic 21):** PPO is the RL algorithm at the heart of Phase 3 of the RLHF pipeline. The three-phase RLHF pipeline (SFT, reward model training, RL fine-tuning) was described at a high level in topic 21. This topic goes deep on the RL fine-tuning phase — specifically the algorithm used.

**GRPO (topic 25, up next):** GRPO (Group Relative Policy Optimization) was developed in part as a response to PPO's complexity and memory requirements. GRPO eliminates the critic network by using a group of responses to estimate advantages, and is simpler to implement and less memory-intensive. Understanding what PPO requires (four models, a critic, GAE) makes it clear why GRPO's simplifications are significant.

**KL Divergence as a Training Signal (topic 20):** In RLHF PPO, KL divergence plays two roles simultaneously. It functions as a penalty term subtracted from the reward (preventing reward hacking) and as a monitoring metric (tracking how far the actor has drifted from the reference). Understanding the math of KL divergence — that it is asymmetric, always non-negative, and zero only when the two distributions are identical — helps explain why the per-token penalty is applied in a specific direction (actor relative to reference, not the other way around) and why the coefficient must be chosen carefully.

**Temperature and Token Sampling (topic 15):** During the generation phase of PPO training, the actor typically generates responses at temperature 1.0, meaning it samples from the full probability distribution rather than greedily picking the most likely token. This is intentional: greedy sampling produces low-diversity outputs, and PPO needs diverse responses to explore the reward landscape. If all responses to a prompt are nearly identical (because the actor greedily picks the same tokens every time), the advantage estimates will be nearly zero for all of them, and no gradient signal is produced. Sampling diversity is essential for PPO to work.

**Transformer Architecture (topic 7):** Both the actor and the critic are transformer models. The critic is simply a transformer with a modified final layer that outputs a scalar instead of a token distribution. The key insight from the transformer's design — that the last token's hidden state summarizes the entire context through self-attention — is the reason why the critic and the reward model both read off their scalar predictions from the last token's hidden state.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| PPO | Improve the policy in small, bounded steps — never overreact to any single batch of experience |
| Policy collapse | What happens without trust region constraints — unconstrained updates destroy what was working |
| Distribution mismatch | Data collected by old policy is unreliable for training the new policy if they diverge too much |
| TRPO vs PPO | TRPO: hard constraint, expensive; PPO: soft clipping, cheap, comparable performance |
| Policy ratio | New probability divided by old probability for a given action — measures the size of the update |
| Clipping | Constrain the ratio to [1 - epsilon, 1 + epsilon] — limit maximum policy change per step |
| Epsilon | Typically 0.2 — a 20% maximum change in any direction per update step |
| Min operation | PPO always picks the more conservative of clipped vs unclipped — pessimistic objective |
| Advantage | How much better or worse was this action than the expected average? |
| GAE | Weighted blend of Monte Carlo and TD advantage estimates, controlled by lambda |
| Lambda | Interpolates between TD (0) and Monte Carlo (1) — typically 0.95 in LLM training |
| Gamma | Discount factor — typically 0.99 or 1.0 for LLMs because reward is sparse and end-of-sequence |
| Advantage normalization | Standardize advantages per mini-batch — prevents scale issues from dominating updates |
| Actor | The language model being trained — selects tokens and is updated by PPO |
| Critic | A second network that estimates expected future reward — provides advantage estimates |
| Critic initialization | Usually from reward model weights — gives the critic a useful starting point |
| Value loss | How the critic is trained — minimize the error between predicted and actual returns |
| Reference policy | Frozen SFT copy — used to compute KL penalty and prevent reward hacking |
| KL penalty | Penalizes the actor for diverging from the reference policy — bounds total drift |
| Per-token KL | KL applied at each token step, not just at the end — more fine-grained anti-hacking |
| Dynamic KL coefficient | Auto-adjusted to keep KL divergence near a target — removes one hyperparameter |
| Four models | Actor, critic, reference policy, reward model — all in GPU memory simultaneously |
| Memory cost | 150+ GB for a 7B model — requires multi-GPU infrastructure |
| Generation bottleneck | Response generation is 5-10x more expensive than gradient update step |
| Training length | Thousands of steps — short relative to pretraining, long relative to SFT |
| Reward hacking | Model finds non-genuine ways to score highly — mitigated by KL penalty and monitoring |
| KL explosion | Actor drifts too far from reference — mitigated by higher KL coefficient and early stopping |
| Mode collapse | Model converges to a single response template — mitigated by entropy bonus and KL penalty |
| PPO in RLHF | The algorithm that powered InstructGPT — since largely replaced by GRPO and DPO |

---

## Hyperparameter Reference

Every PPO run involves a set of hyperparameters that require careful tuning. This table summarises the most important ones, their typical values in LLM training, and what goes wrong when they are set incorrectly.

| Hyperparameter | Typical Range | Too Small | Too Large |
|---|---|---|---|
| Clipping epsilon | 0.1 – 0.2 | Policy updates too slowly, training stalls | Policy changes too fast, training unstable |
| KL coefficient | 0.01 – 0.1 | Reward hacking, policy drifts to degenerate modes | Policy cannot improve, stays near SFT model |
| Learning rate (actor) | 1e-6 – 1e-5 | Very slow convergence | Unstable updates, policy degrades |
| Learning rate (critic) | 5e-6 – 2e-5 | Critic cannot track actor changes | Critic overfits to individual batches |
| GAE lambda | 0.9 – 0.97 | High bias in advantages, wrong gradient direction | High variance in advantages, noisy gradients |
| Discount gamma | 0.99 – 1.0 | Early tokens receive no gradient signal | Not applicable (can be 1.0 for LLMs) |
| PPO epochs per batch | 1 – 4 | Underutilizes generated experience | Overfits to current batch, violates trust region |
| Batch size (prompts) | 128 – 1024 | High variance in gradient estimates | GPU memory exhausted; generation too slow |
| KL target (dynamic) | 0.01 – 0.1 | Coefficient inflated too aggressively, training unstable | Coefficient never tightens, reward hacking |
| Entropy coefficient | 0.0 – 0.01 | Mode collapse, low diversity | Policy too random, cannot exploit good strategies |
| Gradient clip norm | 0.5 – 1.0 | Too much damping, slow convergence | Large gradient spikes cause instability |
| Response max length | 256 – 2048 tokens | Model cannot express complete answers | Generation time and memory cost become prohibitive |

Setting these correctly requires monitoring several training statistics: KL divergence between actor and reference, reward model score distribution, value loss trend, ratio distribution (what fraction of actions are clipped), and response diversity. Most RLHF practitioners develop intuition for these diagnostics over multiple failed and successful training runs.

---

## Up Next

The next topic is **GRPO (Group Relative Policy Optimization)**, the algorithm used in DeepSeek-R1 and many modern reasoning models.

GRPO addresses PPO's two largest practical pain points. First, it eliminates the critic network entirely — instead of maintaining a separate value function, GRPO estimates advantages by comparing a *group* of responses generated for the same prompt. The group average reward becomes the baseline, and each individual response's advantage is its deviation from that group average. This is mathematically clean and requires no separately trained value model.

Second, because there is no critic, GRPO reduces the number of models needed from four to two (actor and reference policy), cutting memory requirements roughly in half. This is what made DeepSeek-R1's training tractable at scale and why GRPO has become the dominant approach for RL-based reasoning model training in 2024 and 2025.

Understanding PPO deeply makes the design choices in GRPO immediately legible: every simplification GRPO makes is a direct response to a known pain point in PPO. The clipping mechanism and KL penalty carry over unchanged. What changes is how the advantage is estimated — from a trained critic network (PPO) to a group-relative computation (GRPO).

The progression from PPO to GRPO mirrors a broader trend in the field: taking a theoretically motivated but complex algorithm and discovering that most of the benefit can be achieved with something much simpler. The same trajectory happened in optimization (SGD to Adam), in attention (full attention to efficient approximations), and in alignment (RLHF to DPO). Understanding the complex thing first — PPO, full attention, RLHF — always makes the simpler successor easier to reason about and use correctly.

What PPO achieved is worth stating clearly before moving on: it demonstrated that reinforcement learning from human feedback is practical at scale, that language models can meaningfully improve through their own trial-and-error experience, and that the stability problems inherent in RL can be managed through careful algorithm design. Every algorithm that came after — GRPO, DPO, DAPO, RLOO — builds on the empirical foundation that PPO established. The researchers who designed GRPO understood what PPO's limitations were because they had run many PPO training runs and watched them fail in specific, predictable ways. That accumulated understanding is embedded in every design choice GRPO makes.

The story of PPO in LLM training is a story of practical engineering meeting theoretical motivation — and of the field continuing to simplify and scale what originally seemed impossibly complex. GRPO, the next topic, is chapter two of that story.

One final note: PPO is not obsolete. For tasks with verifiable rewards — formal mathematics, code execution, structured output with clear correctness criteria — online RL with PPO or GRPO remains state of the art because the reward signal is reliable and dense enough for RL to work well. For open-ended quality tasks where rewards must be estimated by a learned model, DPO and its variants often win on simplicity and stability. Knowing PPO means knowing when to reach for it and when not to — which is the kind of judgment that comes from understanding both what it does well and where it struggles.
