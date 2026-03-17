# 🎮 RL Foundations for LLMs

> **Sources used:**
> - Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), MIT Press 2018 — [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
> - OpenAI Spinning Up, *Key Concepts in RL* — [spinningup.openai.com](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
> - Williams, R.J. (1992), *Simple statistical gradient-following algorithms for connectionist reinforcement learning*, Machine Learning 8, 229–256
> - Sheng et al., *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, 2025

---

## The Big Idea

Reinforcement learning (RL) is the science of learning through trial and error: an agent takes actions in an environment, receives rewards or penalties, and gradually learns a policy that maximises its long-run returns. 🎯 When this framework is applied to language models, the "agent" is the LLM, the "actions" are token choices, the "environment" is the reward model judging the completed response, and the "reward" is a scalar signal telling the model whether that response was good or not. This foundational vocabulary — agent, environment, state, action, policy, reward, trajectory, return — is the shared language of every modern LLM training paper, from InstructGPT to DeepSeek-R1, and understanding it precisely is the prerequisite for everything in the series ahead. Unlike supervised learning, RL never tells the model what the correct answer is — it only signals whether the model's own freely chosen answer was good or bad, forcing the model to discover improvement on its own. That single shift — from "here is the right answer" to "here is how well you did" — unlocks the ability to train models that surpass the quality of any human demonstration in the training set.

---

## Real-Life Analogy 🎮

Imagine you are learning to play a completely new board game — let's call it **Gridfall**. Nobody gives you the rulebook. Nobody tells you which moves are legal or which strategies are best. You sit down at the board, you make a move, and after the game ends, someone tells you whether you won or lost. That single piece of information — win or lose — is your entire feedback signal. Over thousands of games, you gradually figure out which opening moves tend to lead to wins, which mid-game patterns are dangerous, and which endgame sequences are decisive. You are doing reinforcement learning. The board is your **environment**, your current board position is your **state**, your move selection is your **action**, the game's outcome is your **reward**, and the strategy you build up over time — the mental model of "what move to make in this situation" — is your **policy**.

Now extend the analogy. Suppose you are not just learning one board game but you are a writer learning to write the perfect response to any question someone might ask you. You write a response, and after you finish, a panel of readers rates it from 0 to 10. You can't see their reasoning — you just see the number. Over thousands of attempts across thousands of different prompts, you start to learn: long responses with specific examples tend to score higher, vague generalities score lower, a confident tone beats a hedging tone for certain question types, and cheerful curiosity in the opener reliably bumps the score up. You are learning a **policy** — a mapping from "prompt situation" to "writing strategy" — purely from reward signals.

This is exactly what happens when we train an LLM with reinforcement learning. 🧠 The model starts from a base policy (its behaviour after supervised fine-tuning), explores different ways of completing each prompt by sampling from its probability distribution, receives a score for each completed response from a reward model, and gradually shifts its weights toward producing responses that score higher. The brilliance of this approach — and its main difficulty — is that you never need to tell the model *how* to improve. You just need to tell it *whether* it improved. The gradient does the rest, nudging the probability of high-reward token sequences up and the probability of low-reward sequences down.

---

## The Three Branches of Machine Learning — Where RL Fits 🗺️

Before diving into the mechanics, it helps to understand where reinforcement learning sits relative to the two paradigms most ML practitioners know well.

**Supervised learning** is learning from labelled examples. You have a dataset where every input comes paired with the correct output. The algorithm's job is to find a mapping from inputs to outputs that generalises to new inputs. The entire feedback mechanism is: "here is the right answer — get closer to it." Language model pretraining is supervised learning on a massive scale: the input is a sequence of tokens, the correct output is the next token, and the model trains on trillions of such pairs.

**Unsupervised learning** is learning structure from unlabelled data — finding clusters, patterns, latent representations, or generative models without any human-provided labels. Autoencoders, VAEs, and certain embedding methods fall here.

**Reinforcement learning** is fundamentally different from both. There are no pre-provided correct answers. There is no fixed dataset. Instead, the agent interacts with an environment, makes decisions, and receives scalar reward signals that indicate — roughly and often noisily — whether those decisions were good. The agent must figure out the right strategy entirely from this indirect feedback signal. 🧭

The critical implication for LLM training: with supervised fine-tuning, the maximum quality of the model is bounded by the quality of the human demonstrations it imitates. But with reinforcement learning, the model can explore the space of possible responses freely and discover strategies that no human demonstrator thought to write down. This is precisely how DeepSeek-R1 developed its extended chain-of-thought reasoning patterns — not by imitating human reasoning traces, but by discovering through RL that step-by-step reasoning led to higher reward (correct answers) than jumping straight to conclusions.

---

## Agent, State, Action, Environment — Deep Dive 🔬

### The Agent

The **agent** is the learner and decision-maker. It exists inside an environment, perceives what's happening, takes actions, and receives rewards. In classical RL, the agent might be a robot navigating a warehouse, a chess engine choosing moves, or a video-game bot learning to run a level. In LLM training, the agent is the language model itself — specifically, the version of the model with its current set of weights. The agent's goal is always the same: learn a way of acting that accumulates as much reward as possible over time.

An important subtlety: the agent is not a static object. Every time a gradient update runs, the agent's weights change slightly, which means its behaviour changes slightly. The training process is the story of the agent evolving — starting from a random or pre-trained set of weights and slowly converging on a policy that maximises expected reward.

### The Environment

The **environment** is everything the agent is not. It is the world the agent acts in. The environment receives the agent's actions and responds with new states and reward signals. In classical RL, the environment is a physics simulator, a game engine, or the real physical world. In LLM training, the environment is a combination of two things: (1) the **text context** that grows with each token the model generates, and (2) the **reward model** that evaluates the completed response at the end. The environment is what the agent cannot directly control — it can only influence it through actions.

### The State

The **state** is the environment's current configuration — a snapshot of all information relevant to the agent's decision. In a chess game, the state is the current board position (which pieces are where). In a video game, the state might be the pixels on the screen. In LLM token generation, the state is the **entire token sequence so far** — the original prompt plus every token the model has already generated. Each token generated changes the state, because the new state includes the previous token.

This makes LLM token generation a **sequential decision problem**: each action (token choice) changes the state, which then influences every future decision. The state grows by one token with each step, until the model generates an end-of-sequence token or hits a length limit. The complete trajectory of states and actions is called an **episode**.

### The Action

The **action** is what the agent does. In each state, the agent chooses one action from the available action space. In LLM generation, the action at each step is **choosing the next token** — a discrete choice from the model's vocabulary, typically 32,000 to 100,000+ tokens. The model does not output a single deterministic token; instead it outputs a probability distribution over all tokens, and then **samples** from that distribution. This sampling is what introduces randomness into generation — two runs with the same prompt produce different completions because different tokens are sampled.

### Worked Example: One Generation Episode 📝

Let's trace through a concrete example to make this concrete.

- **Prompt (initial state):** "Explain why the sky is blue in one sentence."
- **State 0:** [prompt tokens]
- **Action 0:** Model samples token "The" (probability: 0.35, other options: "Light" 0.18, "Sunlight" 0.12, ...)
- **State 1:** [prompt tokens] + "The"
- **Action 1:** Model samples token "sky" (probability: 0.41)
- **State 2:** [prompt tokens] + "The sky"
- **Action 2:** Model samples token "appears" (probability: 0.28)
- ... (this continues for ~25 tokens) ...
- **Final token:** Model generates `<eos>` (end of sequence)
- **Reward:** Reward model evaluates the full response and returns r = 7.4 out of 10

The episode is complete. The model now has a trajectory: a sequence of (state, action) pairs plus a final reward. The policy gradient algorithm uses this trajectory to decide which token choices to make more or less probable in the future.

---

## The Credit Assignment Problem 🕵️

One of the deepest challenges in RL — and particularly in RL for LLMs — is the **credit assignment problem**. When a response earns a high reward, which of the hundreds of individual token choices deserves the credit? When a response earns a low reward, which token was the culprit?

Consider a response that is 80 tokens long and scores 8.5 out of 10. Tokens 1–30 set up a strong introduction. Tokens 31–55 provide an excellent explanation. Tokens 56–75 include a minor factual error. Tokens 76–80 are a solid conclusion. The reward model gives 8.5 — good but not perfect, probably because of the error in tokens 56–75. But the policy gradient algorithm doesn't know any of this. It sees the full 80 tokens, the score of 8.5, and has to decide how much of that 8.5 to attribute to each individual token choice.

Simple approaches — like assigning the same reward signal to every token in the episode — are blunt instruments. More sophisticated approaches exist:

- **Monte Carlo returns:** Compute the discounted cumulative reward from each timestep onward. Tokens later in the episode see the full reward; earlier tokens see slightly discounted versions. This helps but doesn't fully solve the problem.

- **Advantage functions:** Instead of using the raw reward, compute how much better this trajectory was than the expected value from this state. This baseline subtraction reduces variance and sharpens the signal, but the problem of assigning credit to individual tokens remains.

- **Process reward models (PRMs):** Rather than giving a single reward at the end of the episode, train a separate model to evaluate the quality of each step or reasoning trace intermediate. This gives dense, step-level reward signals and dramatically improves credit assignment — but requires training a separate step-level reward model, which is expensive.

- **GRPO's group baseline:** The DeepSeek approach. For each prompt, generate a group of 8 responses and use their average reward as the baseline. Every token in a response with above-average reward gets a positive signal; every token in a below-average response gets a negative signal. Simple, effective, and computationally efficient.

The credit assignment problem is one of the core reasons why RL for LLMs is harder than RL for games. In Atari, each game step is a single action from an 18-button controller, and rewards arrive frequently. In LLM generation, each episode involves hundreds of actions from a 32,000-button controller, and reward only arrives at the very end. The signal-to-noise ratio is much lower, and the assignment problem is much harder. 🔎

---

## Policy and the Reward Signal — Deep Dive 📐

### What Is a Policy?

A **policy** is the agent's decision-making strategy — a mapping from states to actions (or, in the probabilistic case, a mapping from states to probability distributions over actions). The policy answers the question: "Given that I am in state s, what action should I take?"

Policies come in two flavours:

- **Deterministic policy:** For each state, the policy specifies exactly one action. (Always take action a when in state s.) Useful in chess or game playing where you want the single best move.
- **Stochastic policy:** For each state, the policy specifies a probability distribution over actions. (In state s, take action a with probability 0.7, action b with probability 0.2, action c with probability 0.1.) This is what LLMs use. The distribution is the softmax output over the vocabulary.

For LLMs, the stochastic policy is essential. We want the model to explore different completions — to sometimes choose the second-best word rather than always picking the top word. Exploration is what allows the model to discover that certain token sequences it initially thought were unlikely are actually very high-reward. A purely deterministic policy would always generate the same response and never improve.

### The Parameters of the Policy

We write the policy as **π_θ**, where θ represents the neural network weights. The subscript θ is crucial: it tells us that the policy depends on these weights, and changing the weights changes the policy. When we say "update the policy," we mean "update θ."

The policy outputs the probability of taking action a in state s. For an LLM, this means: given all the tokens so far (state s), what is the probability of generating token a next? The policy outputs one probability for each of the ~32,000 tokens in the vocabulary. The sum of all these probabilities is exactly 1. Sampling from this distribution gives us the next token.

### The Reward Signal

The **reward signal** is the scalar feedback the environment provides after each action (or, in the delayed-reward case, after the episode ends). The reward tells the agent whether what it just did was good or bad — but crucially, *it does not tell the agent what to do differently*. That's the agent's job to figure out.

In LLM training with a reward model:
- Each response gets a single scalar reward at the end (sparse reward)
- Typical reward scales: 0 to 10, or −2 to +2, depending on the reward model calibration
- A score of +8/10 means this was a high-quality response
- A score of +2/10 means this response was poor
- The model does not receive token-level rewards during generation — it only knows if the complete response was good or bad

### The Policy Gradient Theorem — In Plain English

The **policy gradient theorem** answers the question: "How should we update the policy parameters θ to increase expected reward?"

The answer is elegant. For each (state, action, reward) tuple in a trajectory:

**Increase the probability of action a in state s if the reward from that trajectory was higher than expected. Decrease the probability of action a in state s if the reward was lower than expected.**

More precisely: compute the **log probability** of each action the agent took (the log of "how likely was this token choice?"). Multiply that log probability by the **advantage** — how much better was this trajectory than the average trajectory? Then add this to the policy's gradient update.

In words: if the response ended up scoring +8 and the average is +5, the trajectory had a positive advantage (+3). Every token choice in that response gets a positive nudge — we increase the probability of those token choices slightly. If the response scored +2 (below average), every token in that response gets a negative nudge — we decrease their probabilities.

### J(θ) — The Objective Function in Plain English

J(θ) is **the expected total reward the policy earns, averaged over all possible trajectories**.

Think of it this way: if you ran the policy for a million different prompts and averaged all the rewards, you would get J(θ). A higher J(θ) means the policy is, on average, producing better responses. The entire goal of RL training is to **increase J(θ)** — to push the expected reward up.

When we compute the gradient of J with respect to θ and take a step in that direction, we are making a change to the weights that is predicted to increase J(θ). This is gradient ascent on the expected reward. Over thousands of gradient steps, J(θ) should increase, meaning the policy produces progressively better responses on average.

The magic is that we never directly compute J(θ) — we estimate it from sampled trajectories. Each batch of generated responses gives us a noisy estimate of J(θ) and its gradient, and we use that estimate to take a step in the right direction.

---

## Value Functions — The Model's Expectations 🔭

Closely related to the policy is the **value function** — the agent's estimate of how much total reward it can expect to receive from a given state onward, following its current policy.

The **state value function V(s)** answers: "If I am in state s right now and follow my current policy, what is my expected total discounted return?" For an LLM, V(s) answers: "Given the prompt and everything generated so far, how good is my current position — how likely am I to produce a high-reward response from here?"

The **action value function Q(s, a)** answers: "If I am in state s and take action a (a specific token), and then follow my policy, what is my expected return?" Q(s, a) tells you the expected reward from choosing a specific next token, given the context so far.

The **advantage function A(s, a)** is the difference: A(s, a) = Q(s, a) − V(s). It answers: "How much better is it to take this specific action a in state s, compared to whatever action my policy would typically take?" A positive advantage means "this token choice is better than average from this state." A negative advantage means "this token choice is worse than average."

Value functions are central to many RL algorithms:

- **PPO** maintains a separate **value model** (also called a critic) — a neural network trained to estimate V(s) for every state during generation. The value model is used to compute advantage estimates, which drive the policy gradient. This is why PPO requires four models in memory simultaneously during RLHF training: the RL policy, the reference policy, the reward model, and the value model.

- **GRPO** skips the value model entirely. Instead, it estimates the baseline from the average reward within a group of responses to the same prompt. No separate critic network needed — much lower memory footprint and simpler implementation.

- **REINFORCE** uses a simple Monte Carlo estimate: the return G_t (actual cumulative discounted reward from timestep t onward) acts as a noisy estimate of Q(s_t, a_t). The average return across episodes acts as a noisy estimate of V(s_t). This is conceptually simple but high-variance.

Understanding value functions is essential for reading any RL paper about LLMs — the choice of how to estimate advantages is one of the most consequential algorithmic decisions in the whole training pipeline. 🔑

---

## The Trajectory and Discounted Returns — Deep Dive 🛤️

### What Is an Episode?

An **episode** is one complete run from start to finish. For LLMs, an episode is:
- **Start:** A prompt is presented to the model
- **Middle:** The model generates tokens one by one (state → action → new state → action → ...)
- **End:** The model generates the end-of-sequence token, and the reward model evaluates the full response

Each episode produces a **trajectory**: the complete sequence of (state, action) pairs that occurred during generation, plus the final reward received at the end.

### What Is a Trajectory?

Formally, a trajectory τ (tau) is the complete sequence:

```
τ = (s₀, a₀, s₁, a₁, s₂, a₂, ..., s_T, a_T, r_T)
```

Where:
- s₀ is the initial state (the prompt)
- a₀ is the first token chosen
- s₁ is the state after that first token (prompt + first token)
- ... and so on until ...
- a_T is the final token (end-of-sequence)
- r_T is the reward received at the end

In LLM training, a trajectory is essentially one complete (prompt, response, reward) tuple. The trajectory is the fundamental unit of data that the policy gradient algorithm works with.

### Discounted Returns — Why γ Matters 💡

The **return** from a trajectory is the total reward collected over the episode. But in RL, we don't just sum up rewards naively — we **discount** future rewards by a factor γ (gamma, pronounced "gamma"), where γ is between 0 and 1 (typically 0.95 to 0.99).

The discounted return G at timestep t is:

```
G_t = reward_at_t + γ × reward_at_(t+1) + γ² × reward_at_(t+2) + ...
```

In words: the return at time t is the reward you get right now, plus 0.99 times the reward you'll get next step, plus 0.99² times the reward the step after that, and so on. Rewards further in the future count for less.

### Why γ < 1? Three Reasons 🔢

**Reason 1: Mathematical convergence.** If γ = 1 and episodes can be long, the sum of rewards can grow without bound. With γ < 1, the geometric series converges to a finite number. This keeps the return values numerically manageable for gradient computations.

**Reason 2: Uncertainty about the future.** In real environments, the further ahead you look, the less certain you are about what will happen. Discounting reflects this uncertainty: near-term rewards are more reliable than distant ones. A reward you'll definitely get in the next step is worth more than a reward you might get 50 steps from now.

**Reason 3: Preference for sooner rewards.** In many practical settings, a good outcome sooner is genuinely more valuable than the same outcome later. For LLMs, this is less critical — the only reward is at the end of the episode — but for multi-step reasoning tasks where intermediate rewards exist, discounting helps the model prioritise efficient paths to good outcomes.

**Worked example with γ = 0.99:**
- A response has 10 tokens before the end-of-sequence, then receives reward +8
- Token 1 was generated 10 steps before the end: its discounted contribution is 0.99^10 × 8 ≈ 7.23
- Token 5 was generated 5 steps before the end: 0.99^5 × 8 ≈ 7.61
- Token 9 was generated 1 step before the end: 0.99^1 × 8 ≈ 7.92

With a high γ like 0.99, the discounting effect is minimal for short responses — all tokens effectively carry almost the full reward signal. This is appropriate for LLM responses, where responses are usually short enough that discounting doesn't matter much in practice.

---

## On-Policy vs Off-Policy Learning 🔁

RL algorithms divide into two broad families based on whether the data used for learning was generated by the same policy that is being trained.

**On-policy learning** means the algorithm learns from data generated by the current policy. PPO and GRPO are on-policy algorithms. After each batch of gradient updates, the old data is discarded because it was generated by a slightly different version of the policy. On-policy learning is more stable — the data always reflects the current policy's behaviour — but less sample-efficient, since data cannot be reused across updates.

**Off-policy learning** means the algorithm can learn from data generated by a different (typically older) policy. Q-learning, DQN, and SAC are off-policy algorithms. Data can be stored in a **replay buffer** and reused many times, making off-policy methods more sample-efficient. However, they require careful correction to handle the mismatch between the data-generating policy and the current policy being updated.

For LLM training, on-policy methods dominate:

- **PPO** is technically slightly off-policy (it reuses data for multiple gradient steps per batch), but uses the clipping mechanism to ensure the policy doesn't drift too far from the data-generating version within each batch.
- **GRPO** is strictly on-policy — a new group of responses is sampled for every gradient update.
- **DPO** is technically off-policy — it trains on a fixed dataset of (prompt, winner, loser) triplets — but it avoids the RL formulation entirely, so the policy mismatch issue is handled differently.

The practical implication of on-policy training for LLMs: **you must generate fresh responses at training time**. This means running LLM inference (which is computationally expensive) as part of the training loop. A single training step involves: sample prompts → run the LLM to generate responses → score with reward model → compute gradients → update weights. The inference cost of generating those responses is a significant fraction of total training compute. This is why GRPO's elimination of the value model (and its associated inference cost) was such a meaningful engineering improvement.

---

## From Supervised Learning to RL — Deep Dive 🔄

### The Fundamental Difference

In **supervised learning** (which includes standard language model pretraining and supervised fine-tuning):
- You have a dataset of (input, correct output) pairs
- The loss function measures how different the model's output is from the correct output
- The gradient pushes the model's output toward the correct output
- You always know exactly what the right answer is

In **reinforcement learning**:
- You do not have correct outputs — you only have rewards
- The loss function measures expected reward (not distance from a correct answer)
- The gradient pushes the model toward outputs that historically earned higher rewards
- You never know what the "right" answer is — only whether your answer was good or bad

This distinction is profound. Supervised learning is like a teacher marking your exam with red pen and writing the correct answer next to each mistake. RL is like getting your exam back with only a grade at the top — 67% — and no other feedback. You have to figure out yourself which answers were wrong and how to do better next time.

### Why Cross-Entropy Loss Can't Capture "Good Response" ✍️

Cross-entropy loss, the workhorse of language model training, measures: **how surprised was the model by the actual next token?** It pushes the model to assign high probability to the tokens that actually appeared in the training data.

This works brilliantly for pretraining because you have billions of tokens of human-written text, and predicting those tokens well means learning grammar, facts, reasoning patterns, and stylistic conventions. But it breaks down for alignment for a critical reason:

**There is no unique "correct" response to a prompt.**

If someone asks "What are three benefits of exercise?", there are hundreds of valid answers — different orders, different examples, different phrasings, different levels of detail. Cross-entropy loss would only push the model toward one particular response (whichever was in the training data), while heavily penalising all the other valid responses.

Worse, for creative tasks, the "correct" response is completely undefined. There is no ground truth for "write me a poem about autumn." Cross-entropy loss would require a specific poem to imitate, but any of millions of poems could be wonderful answers. RL solves this elegantly: you don't need a specific target — you just need a reward function that tells you whether any given poem was good.

### The Key Differences Summarised 📊

| Dimension | Supervised Learning | Reinforcement Learning |
|---|---|---|
| Feedback | Token-level: "correct answer was X" | Episode-level: "your response scored 7/10" |
| Data | Fixed dataset of labeled examples | Generated on-the-fly from the policy |
| Loss | Cross-entropy (distance from target) | Expected reward (harder to compute) |
| Exploration | None — model sees correct answers | Essential — must try different outputs |
| Credit assignment | Exact — every token gets a direct signal | Hard — which of 50 tokens caused the good score? |
| Unique answer | Yes (one correct completion per input) | No (many valid responses exist) |
| Improvement ceiling | Limited by quality of demos | Can exceed human-level demonstrations |

### Why RL Can Exceed the Quality Ceiling of Supervised Learning 🚀

This is one of the most important ideas in modern LLM training. With supervised learning (SFT), the model learns to imitate human demonstrations. It can only be as good as those demonstrations — if the training data contains mediocre responses, the model learns to produce mediocre responses. The quality ceiling is the quality of the training data.

With RL, the model can **discover response strategies that humans never demonstrated**. By exploring different completions and learning which ones score highest, the model can find novel response patterns that score better than any example in the supervised dataset. This is how models like DeepSeek-R1 developed extended chain-of-thought reasoning without explicitly being shown that reasoning format — the RL reward signal for correct answers caused it to emerge organically.

---

## Exploration and Exploitation — The Core Tension ⚖️

Every RL agent faces the **exploration-exploitation trade-off**: should I use what I already know works well (exploitation), or should I try new things in case something better exists (exploration)?

**Pure exploitation** means always choosing the action with the highest estimated reward. For an LLM, this means always choosing the highest-probability token — greedy decoding. If the model starts believing that a certain response style earns high rewards, it will always generate that style and never discover whether other styles might be even better. Training stagnates. The model gets stuck in a local optimum.

**Pure exploration** means choosing actions randomly, ignoring what has been learned. For an LLM, this means sampling from a flat or high-temperature distribution, generating incoherent responses. The training signal becomes noise. No learning happens.

**Balanced exploration** means maintaining enough randomness to discover new high-reward strategies while still using what has been learned to produce generally good responses. For LLMs, temperature-1.0 sampling strikes this balance: the model samples from its learned distribution (which already concentrates probability on reasonable tokens) while still occasionally choosing less-likely tokens that might lead to better responses.

There are more sophisticated exploration strategies beyond temperature control:

- **Entropy regularisation:** Add a term to the loss function that rewards the policy for maintaining a diverse probability distribution. If the policy becomes too "peaked" (almost deterministic), the entropy term penalises it and forces more exploration.

- **Group sampling:** The GRPO approach generates multiple responses per prompt (typically 8). This guarantees that even if the top-1 response is the same every time, other members of the group explore different completion paths. The diversity within the group is itself a form of structured exploration.

- **Nucleus sampling (top-p):** Instead of sampling from the full vocabulary, sample only from the smallest set of tokens whose cumulative probability exceeds p (e.g., 0.9). This removes the long tail of low-probability, often-incoherent tokens from exploration, focusing the search on plausible continuations.

The exploration-exploitation trade-off never fully disappears — it is managed and tuned throughout training. Getting it right is part engineering, part art, and is one of the key hyperparameter decisions that separates stable from unstable RL training runs. 🎯

---

## How RL Applies to LLMs Specifically 🤖

### The Mapping

| RL Concept | LLM Equivalent |
|---|---|
| Agent | The language model (with parameters θ) |
| Environment | Reward model + growing text context |
| State | Current prompt + all tokens generated so far |
| Action | Choosing the next token (~32k options) |
| Policy | LLM's softmax distribution over the vocabulary |
| Episode | One complete prompt → response generation |
| Reward | Reward model scalar at end of response |
| Trajectory | The complete (prompt, response, reward) tuple |

### The Prompt as State

When the model begins generating a response, the initial state is the **prompt** — all the tokens in the user's question or instruction. Each token the model generates becomes part of the new state. By the time the model is generating token 30, the state includes the prompt plus the first 29 generated tokens. The state grows with every action.

This is fundamentally different from classical RL environments like Atari games or robotics, where the state is provided by an external environment and doesn't depend on the agent's own actions (or depends on them only indirectly through physics). In LLM generation, **every action directly and permanently modifies the state**. You can't undo a token choice — once "However" is in the context, all subsequent token choices are conditioned on that word being there.

### Token Sampling as Action 🎲

Rather than choosing a single deterministic next token, the LLM **samples** from its output distribution. This sampling is the action-selection mechanism. The temperature parameter controls how "peaked" or "flat" this distribution is:

- **Temperature 0 (greedy):** Always pick the highest-probability token. Deterministic. No exploration. The same prompt always gives the same response.
- **Temperature 1.0 (standard sampling):** Sample from the raw softmax distribution. Good balance of following the policy and exploring alternatives.
- **Temperature 2.0 (high temperature):** Distribution is flattened, making unusual tokens more likely. More exploration, more randomness, less coherent.

During RL training, temperature is typically kept at 1.0 to ensure meaningful exploration. Too low and the model stops exploring alternatives; too high and the responses become incoherent noise.

### The Reward Model as Environment 🏆

The reward model is a pre-trained neural network — typically the same architecture as the LLM but with the language modelling head replaced by a scalar output head — that takes a (prompt, response) pair and outputs a scalar score.

This score plays the role of the environment's reward signal. Once the LLM has finished generating a complete response (reached the end-of-sequence token), the reward model evaluates it and returns a number. In DeepSeek-R1, verifiable tasks use binary rewards (+1 for correct, 0 for wrong). In RLHF-style training, rewards might range from −2 to +4 based on learned human preferences.

**The reward model is not perfect** — it is itself a learned approximation of human preferences, trained on a finite dataset of human comparisons. This imperfection is a major source of problems (reward hacking, discussed in RLHF overview) and motivates the KL penalty that prevents the LLM from straying too far from its reference policy.

### Why This Is Hard 🔥

RL for LLMs is one of the most challenging applied RL problems, for several reasons:

**1. Enormous action space.** At each step, the model chooses from 32,000–100,000 possible tokens. Standard RL algorithms were designed for much smaller action spaces (Atari has ~18 actions; board games have hundreds; robotics joints have single-digit dimensions). With 100k options per step, even small vocabulary distributions are complex objects.

**2. Long episodes.** A response might be 200–1000 tokens long. Each token choice is an action in the episode. The credit assignment problem — "which of the 500 token choices caused the response to score 8 instead of 6?" — is extremely hard to solve.

**3. Sparse reward.** The reward only arrives at the end of the episode. During generation, the model receives zero feedback. It must wait until the entire response is complete before learning anything. This contrasts with games where you get reward at every step.

**4. Non-stationarity.** As the model's weights change through training, the distribution of generated responses changes — which means the reward model evaluations change, the baseline expectations change, and the advantage estimates shift. Everything is moving simultaneously, making convergence difficult.

**5. Reward model imperfection.** The reward model is itself a noisy, fallible estimator. Optimising against it too aggressively causes reward hacking — the policy finds high-reward responses that the reward model rates highly but that humans would not.

---

## Reward Hacking — When the Signal Lies 🚨

One of the most important failure modes in RL — and particularly RL for LLMs — is **reward hacking** (also called reward gaming or specification gaming). This occurs when the agent finds a way to earn high reward that does not correspond to the intended behaviour.

The classic toy example: a boat-racing agent is trained to maximise the score in a racing game. It discovers it can spin in circles collecting score bonuses without ever completing a lap — this earns higher reward than racing normally. The agent "hacked" the reward function by finding a strategy the designers hadn't anticipated.

For LLMs, reward hacking takes more subtle forms:

**Verbosity hacking:** The agent discovers that reward models tend to rate longer, more detailed responses higher. So it learns to pad responses with additional caveats, examples, and restatements — producing responses that are technically longer but not actually better. The reward goes up; quality plateaus or degrades.

**Style mimicry:** The agent learns that certain superficial stylistic patterns (bullet points, headers, specific phrases like "Certainly!" or "Great question!") are associated with higher reward model scores. It begins including these patterns regardless of whether they improve the actual content.

**Hedging exploitation:** If the reward model rates "appropriately uncertain" responses highly, the agent may over-hedge — adding "I'm not sure but..." and "You should verify this" to responses even when it is confident and correct, because hedging reliably bumps the score.

**Sycophancy amplification:** If the reward model values responses that align with perceived user preferences, the agent may learn to agree with everything the user says, even factual errors, because disagreement is penalised.

**The KL penalty as a guard:** Every modern LLM RL system includes a KL divergence penalty between the current policy and the frozen reference policy. This penalty limits how far the policy can drift. Reward hacking typically requires dramatic changes to response style or content — changes that push the KL divergence high. By penalising large KL values, the training system naturally limits the severity of reward hacking. But it doesn't prevent it entirely; it just bounds how extreme the hacking can become before the KL cost outweighs the reward gain.

Reward hacking is the reason evaluation on held-out human judges is essential throughout RL training — the reward signal alone cannot be trusted as a measure of true quality improvement. 🔍

---

## How It Works in Practice — Real Numbers 🔢

### A Concrete GRPO Training Step (DeepSeek Style)

Modern LLM RL training often uses **GRPO (Group Relative Policy Optimization)**, which works like this:

**Step 1: Sample a batch of prompts.**
Take 128 prompts from the training dataset. Each prompt is a question or task.

**Step 2: Generate a group of responses per prompt.**
For each prompt, generate G = 8 different completions by sampling from the current policy at temperature 1.0. This gives 128 × 8 = 1,024 total responses.

**Step 3: Score all responses.**
Run each of the 1,024 responses through the reward model. Each gets a scalar score. For verifiable tasks (maths, code), this might just be 1 (correct) or 0 (wrong).

**Step 4: Compute group-relative advantages.**
For each group of 8 responses to the same prompt:
- Compute the mean reward across the 8 responses (e.g., mean = 0.5)
- Subtract the mean from each response's reward to get the advantage: if one response scored 1.0, its advantage is +0.5; if another scored 0.0, its advantage is −0.5
- Normalise by the standard deviation of rewards in the group

This group-relative baseline replaces the value model needed in PPO — no separate critic network needed.

**Step 5: Update the policy weights.**
For each token in each response: multiply the log probability of that token by the advantage of the response it belongs to. Sum these across the batch. Add a KL penalty term. Take a gradient step.

Responses with positive advantages (better than the group average) get their token probabilities nudged up. Responses with negative advantages get their token probabilities nudged down.

**Repeat** with the next batch of prompts. Over thousands of steps, the policy improves.

### Reward Scales and What They Mean 📏

| Training setup | Reward range | What 0 means | What max means |
|---|---|---|---|
| RLHF (InstructGPT style) | −2 to +4 | Neutral / average response | Excellent response |
| GRPO on math (DeepSeek) | 0 or 1 | Wrong answer | Correct answer |
| RLHF (general) | 0 to 10 | Completely unhelpful | Outstanding response |
| Constitutional AI | −1 to +1 | Preferred response (loser) | Preferred response (winner) |

The absolute values of rewards don't matter much — what matters is the **relative difference** between rewards within a batch. If all responses score between 6.8 and 7.2, the advantage estimates will be tiny and the policy will barely change. If responses range from 2 to 9, the advantages will be large and the policy update will be substantial.

### Learning Rate and Stability 🎛️

The learning rate during RL fine-tuning is typically much smaller than during SFT pretraining. Common values:

- **SFT fine-tuning:** learning rate ~ 1e-5 to 5e-5
- **RL fine-tuning (PPO/GRPO):** learning rate ~ 1e-6 to 5e-6 (often 5x–10x smaller)

Why smaller? Because the RL loss landscape is noisier (gradient estimates come from sampled trajectories, not labelled ground truth), and because the model has already converged to a good SFT baseline — small nudges are enough to shift the policy in the desired direction. Large steps risk destabilising the policy, causing reward hacking or KL explosion.

### Training Duration 🕰️

RL fine-tuning is typically much shorter than SFT training:

- **Pretraining:** trillions of tokens, weeks to months of compute
- **SFT fine-tuning:** tens of millions of tokens, hours to days
- **RL fine-tuning:** hundreds of millions of policy-generated tokens, hours to days

The brevity of RL training is by design. The goal is not to reshape the model's entire knowledge base — that was done in pretraining — but to adjust the policy's behavioural tendencies. Small, targeted adjustments are sufficient, and the KL penalty ensures the model doesn't drift far enough to forget what it learned in SFT.

### What Happens to Loss During RL Training 📉

Unlike SFT training, where the cross-entropy loss steadily decreases, RL training loss curves look different:

- The policy gradient loss is not monotonically decreasing — it fluctuates as the policy explores different regions of the response space
- The KL penalty term tends to increase gradually as the policy drifts from the reference
- The reward (on held-out evaluation prompts) should increase — this is the true signal of progress
- A flat or decreasing reward on held-out prompts, even if training reward increases, signals reward hacking

Monitoring held-out evaluation reward — not training reward — is the correct diagnostic during RL training.

### A Note on Verifiable vs Learned Rewards 🧮

One of the most exciting developments in LLM RL is the shift toward **verifiable rewards** for tasks where objective correctness can be determined automatically.

For a maths problem: the answer is either numerically correct or not. No reward model needed — just check the answer. For a coding problem: the code either passes all test cases or it doesn't. For a formal logic puzzle: the proof is either valid or invalid. These binary signals (1 for correct, 0 for wrong) are clean, unambiguous, and impossible to game through superficial stylistic hacking.

The implication is dramatic: the reward hacking problem largely disappears for verifiable tasks, because you cannot game a correct/incorrect signal the way you can game a learned reward model's aesthetic preferences. This is why GRPO applied to verifiable maths and code problems — as in DeepSeek-R1 — produces such reliable reasoning improvements. The signal is clean, and the algorithm simply follows it. 🎯

### Group Size Matters 👥

The number of responses generated per prompt (the group size G) is a key hyperparameter:

- **G = 1 (single response):** No comparison baseline. High variance in gradient estimates. Policy gradient is noisy and slow.
- **G = 4:** Reasonable baseline. Lower variance. Widely used in early RLHF setups.
- **G = 8:** The DeepSeek default. Good balance of compute cost and gradient quality.
- **G = 64:** Used in some research setups. Very low variance but very high compute cost.
- **Larger G** means more diverse coverage of the response space per prompt, better advantage estimation, and more stable training — but proportionally more inference compute.

### KL Penalty in Practice 🔒

The KL penalty is a regularisation term added to the reward that measures how far the current policy has drifted from the reference policy (the frozen SFT model). Typical coefficient: λ = 0.01 to 0.1.

- **λ too small:** Policy drifts freely, reward hacks, catastrophic forgetting of alignment
- **λ too large:** Policy barely moves from the reference, reward barely improves, training is stuck
- **λ = 0.04 (DeepSeek default):** Policy improves significantly while staying close enough to the reference that quality remains high

---

## The Markov Property — Why Token-by-Token Works 🔗

A deep theoretical assumption underlying all of RL as normally formulated is the **Markov property**: the future depends only on the current state, not on the history of how we got there. Formally: knowing the full history of states and actions before this moment gives you no additional information about the future beyond what the current state already contains.

For LLMs, the "current state" is the entire token sequence so far — the prompt plus all generated tokens. Is this Markovian? Yes, by construction. The LLM's probability of generating any next token depends only on the current context window (all tokens so far). Given the current sequence, the model's future outputs are fully determined (in distribution) — there is no additional history that would change the prediction.

This is why the RL framework applies cleanly to LLM generation. Each step is Markovian: the state (current token sequence) fully determines the distribution of future states and rewards, without needing to know which specific path was taken to reach this sequence. The attention mechanism, which looks back at all previous tokens, is exactly the machinery that makes the current state fully sufficient — nothing is "forgotten" from the history because it all lives in the context window. 🔑

The practical implication: we don't need to track the full history of states visited during training — just the current batch of (state, action, reward) tuples. This makes storage and computation tractable, even for long episodes.

---

## Tabular RL vs Deep RL — Why Neural Networks Are Necessary 🧮

In introductory RL courses, algorithms are often presented in the **tabular** setting: a lookup table maps every possible (state, action) pair to a value or Q-value. This works beautifully for small problems — GridWorld, simple mazes, Tic-Tac-Toe — where the state space is small enough to enumerate.

For LLMs, the state space is so astronomically large that tabular methods are impossible even in principle. The state is a sequence of tokens. With a vocabulary of 32,000 tokens and sequences up to 2,000 tokens long, the number of distinct possible states is 32,000 raised to the power of 2,000 — a number larger than the number of atoms in the observable universe. 🌌

**Deep RL** solves this by using a neural network as a function approximator. Instead of looking up Q(s, a) in a table, we run a neural network with weights θ that approximates Q(s, a) for any state-action pair it has never seen before. The network generalises from the states it has seen during training to new states.

For LLMs, the policy network is the LLM itself — the same transformer that generates text is also the function approximator for the policy. The policy network maps from the current state (token sequence) to a probability distribution over actions (next token probabilities). This is exactly what the LLM does during inference. RL training is the process of adjusting the LLM's weights so that this probability distribution is shaped toward actions that earn high reward.

The value model (used in PPO but not GRPO) is a separate neural network — typically initialised from the same weights as the LLM, with a different output head — that approximates V(s) for every state. It learns to predict: "given this context, how much total reward can I expect the current policy to earn from here?"

The use of neural networks as function approximators introduces the instabilities that make deep RL challenging (gradient issues, non-stationary targets, bootstrapping errors) — but it is the only approach that scales to the astronomical state spaces of language generation. 🏗️

---

## Common Misconceptions 🧹

**❌ Myth: The model learns the right answer directly from RL.**
✅ Reality: RL never tells the model what the right answer is — it only tells the model whether its answer was good or bad. The model must figure out what "better" looks like by comparing its own outputs. This is fundamentally different from supervised learning, where the target answer is always known.

---

**❌ Myth: A higher reward always means learning is happening.**
✅ Reality: Reward can go up for the wrong reasons — reward hacking. The model might learn to produce responses that fool the reward model (long, confident-sounding, using words the reward model associates with quality) without actually improving response quality. This is why the KL penalty, held-out evaluation sets, and human spot-checks are essential.

---

**❌ Myth: RL for LLMs is just fine-tuning with different data.**
✅ Reality: The key structural difference is that the training data is **generated by the model itself, on-the-fly, during training**. The model produces responses, those responses get scored, and those scored responses become the training signal. The data distribution is constantly changing as the model improves — this is fundamentally different from fine-tuning on a fixed dataset.

---

**❌ Myth: More training steps always improve the policy.**
✅ Reality: RL training can overfit, collapse, or diverge. The policy can start performing worse after a certain number of steps if the KL penalty is too weak, the reward model is imperfect, or the learning rate is too high. Early stopping, monitoring on held-out prompts, and careful KL tuning are essential.

---

**❌ Myth: The "reward" in RL for LLMs is just human ratings.**
✅ Reality: Modern LLM RL uses diverse reward sources: reward models trained on human preferences, verifiable outcomes (code execution, maths answer checking), constitutional AI principles applied by another model, or hybrid combinations. The reward signal is an engineering choice, not a fixed thing.

---

**❌ Myth: Token-level log probabilities are the "actions" the policy gradient cares about.**
✅ Reality: The policy gradient cares about the **advantage** — how much better was this episode than average? — multiplied by the log probability. If all responses score similarly, the advantage is near zero and no learning happens, regardless of the log probabilities. The advantage signal is what actually drives learning.

---

**❌ Myth: RL for LLMs always needs a separate reward model.**
✅ Reality: For verifiable tasks (maths problems, coding challenges, logical puzzles), you can use **executable correctness** as the reward: run the code, check if it compiles and passes tests; verify if the maths answer matches the ground truth. No reward model needed. DeepSeek-R1 used this for its core reasoning capability development.

---

**❌ Myth: The policy gradient always points in the right direction.**
✅ Reality: The policy gradient is a **noisy estimate** of the true gradient of J(θ). It is computed from a small batch of sampled trajectories, and the true expected gradient would require averaging over all possible trajectories — infinitely many. With small batches, the estimate is noisy, and gradient steps sometimes move in the wrong direction. This is why multiple gradient steps, momentum, and careful learning rate scheduling are essential. High variance in gradient estimates is one of the core challenges of RL algorithms like REINFORCE and is one of the main reasons more sophisticated variance-reduction techniques (baselines, advantage normalisation, clipping) were developed.

---

**❌ Myth: RL for LLMs is a completely new technique from 2022 onward.**
✅ Reality: The theoretical foundations of RL — Markov decision processes, policy gradients, the REINFORCE algorithm — were established in the 1950s–1990s. The Bellman equations date to 1957. REINFORCE was published by Williams in 1992. Q-learning and temporal difference methods were developed through the 1980s and 1990s. What changed in 2022 with InstructGPT was the engineering: combining these decades-old algorithms with modern large-scale neural networks, reward models trained from human feedback, and the compute necessary to make the whole system work. The ideas are old; the scale and application are new.

---

## The RL Vocabulary Map — Quick Reference 🗂️

When you read an RL paper about LLMs, you will encounter dense vocabulary that can feel overwhelming at first. Here is a plain-English mapping of every term you are likely to encounter:

| Term | Symbol | Plain English |
|---|---|---|
| Markov Decision Process | MDP | The formal framework: states, actions, transitions, rewards |
| State | s | The full context: prompt + all generated tokens |
| Action | a | Next token chosen from the vocabulary |
| Action space | A | All possible tokens (~32k to 100k) |
| State space | S | All possible token sequences (astronomically large) |
| Transition | T(s, a, s') | How the state changes after an action (deterministic for LLMs) |
| Policy | π_θ | The LLM's probability distribution over next tokens |
| Deterministic policy | π(s) | Always outputs one specific action |
| Stochastic policy | π(a|s) | Outputs a probability for each action |
| Reward | r | Scalar signal from the reward model |
| Return | G_t | Total discounted reward from step t onward |
| Discount factor | γ | How much future rewards are discounted (0.95–0.99) |
| Episode | — | One prompt → response → reward cycle |
| Trajectory | τ | The full (state, action, ..., reward) sequence of an episode |
| Value function | V(s) | Expected return from state s under the current policy |
| Q-function | Q(s,a) | Expected return from state s after taking action a |
| Advantage | A(s,a) | Q(s,a) − V(s): how much better is this action vs average? |
| Policy gradient | ∇J(θ) | The direction to update θ to increase J(θ) |
| Objective | J(θ) | Expected total reward across all trajectories |
| Baseline | b(s) | A state-dependent value subtracted from returns to reduce variance |
| KL divergence | KL(π‖π_ref) | How different the current policy is from the reference policy |
| Entropy | H(π) | How spread out the policy's distribution is (high = more exploration) |
| On-policy | — | Training data generated by the current policy |
| Off-policy | — | Training data generated by a different (older) policy |
| Replay buffer | — | Storage of past experiences for off-policy reuse |
| Value model / critic | V_φ | Separate neural network that estimates V(s) for PPO |
| Reference policy | π_ref | Frozen SFT model used for KL penalty computation |

---

## Connections to Other Topics 🔗

**→ Reward Models (Topic 19):** The reward model is the "environment" in the RL framework for LLMs. Understanding what reward models learn, how they score responses, and why they can be gamed is essential for understanding why RL training is difficult. The reward model's imperfections directly limit how much RL training can help. A reward model that is biased toward verbose responses will cause the policy to become verbose; a reward model that confuses confidence for accuracy will cause the policy to become overconfident. The reward model is the single most consequential design choice in any LLM RL system.

**→ KL Divergence (Topic 20):** The KL divergence penalty is the safety constraint in every RL training loop. It prevents the policy from drifting so far from the SFT reference that it loses coherence or alignment. Without it, RL training degenerates into reward hacking within a few hundred steps. The KL penalty coefficient λ controls the fundamental trade-off between reward maximisation and policy stability — too small and the model reward-hacks, too large and the model barely improves.

**→ RLHF Overview (Topic 21):** RLHF is the most famous application of RL to LLMs. The RL foundations in this chapter provide the theoretical scaffolding for understanding why RLHF works, what PPO is doing, and why DPO was developed as a simpler alternative. Every element of the RLHF pipeline maps cleanly onto the RL vocabulary: SFT initialises the policy, the reward model defines the reward function, and PPO implements the policy gradient update with stability constraints.

**→ Temperature and Sampling (Topic 15):** The temperature setting during RL training controls the exploration-exploitation balance. RL training typically uses temperature 1.0 to ensure diverse exploration. The connection between sampling temperature and RL exploration is a key engineering choice. Nucleus sampling (top-p) and other sampling strategies similarly affect which parts of the token distribution the policy explores during training.

**→ Logits and Token Selection (Topic 14):** The policy's action at each step is determined by the logits — the raw scores the model assigns to each token before softmax. Understanding how logits translate to probabilities, and how those probabilities are sampled, is essential for understanding how RL gradient updates actually modify the model's behaviour. A gradient update that "increases the probability of token X in state S" is, mechanically, a small adjustment to the weights of the final linear layer that increases the logit for token X relative to all other tokens.

**→ Training Loop and Loss Functions (Topic 17):** RL training uses a completely different loss function from standard language model training. Instead of cross-entropy loss (distance from a target token), RL uses the policy gradient objective (expected reward scaled by log probabilities and advantages). Understanding how the standard training loop works — forward pass, loss computation, backward pass, weight update — is the prerequisite for understanding how the RL training loop modifies each of these steps.

**→ REINFORCE Algorithm (Topic 23 — next):** REINFORCE is the simplest policy gradient algorithm and the mathematical foundation for everything else. It implements the policy gradient theorem directly: multiply the log probability of each action by the return, sum across the trajectory, and take a gradient step. Understanding RL foundations here makes REINFORCE straightforward. Every subsequent algorithm — PPO, GRPO, DAPO, RLOO — is recognisable as a refinement of REINFORCE that addresses its high variance, sample inefficiency, or instability.

---

## Key Takeaways 📋

| Concept | Plain English |
|---|---|
| Agent | The LLM — the thing that makes decisions (token choices) |
| Environment | The reward model + growing text context that evaluates those decisions |
| State | The full token sequence: prompt + all tokens generated so far |
| Action | Choosing the next token from the vocabulary (~32k options) |
| Policy (π_θ) | The LLM's softmax distribution over all tokens — "probability of each next word" |
| Reward | Scalar score from the reward model after the complete response is generated |
| Episode | One complete prompt → response generation → reward cycle |
| Trajectory | The full (prompt, response, reward) record of one episode |
| Return (G_t) | Total discounted reward from this point onward in the episode |
| Discount factor (γ) | How much future rewards are worth relative to current rewards (typically 0.99) |
| J(θ) | Expected total reward — the thing we want to maximise through RL training |
| Policy gradient | Nudge token probabilities up for high-reward responses, down for low-reward ones |
| Advantage | How much better than average this trajectory's reward was |
| Sparse reward | Reward only at episode end — LLMs must handle this by default |
| Exploration | Sampling from the policy distribution rather than always taking the top token |
| Credit assignment | Figuring out which of the 500 token choices caused the score to be 8 not 6 |
| Cross-entropy vs RL | SFT needs a "correct" answer target; RL only needs a "good or bad" signal |
| Why RL > SFT ceiling | RL can discover better responses than any human demonstration |
| Group size G | Number of responses sampled per prompt (typically 4-64) for advantage estimation |
| KL penalty | Regularisation that prevents the policy from drifting too far from the SFT reference |

---

## The Evolution of RL for LLMs — A Timeline 📅

Understanding where we are now requires knowing how we got here.

**1992 — REINFORCE (Williams):** The policy gradient algorithm is published. Simple, high-variance, but theoretically sound. The mathematical foundation that everything else builds on.

**2015 — PPO precursors (TRPO, Schulman et al.):** Trust region policy optimisation introduces the idea of constraining how far each update can move the policy. Stable but computationally expensive.

**2017 — PPO (Schulman et al.):** Proximal policy optimisation replaces the TRPO constraint with a simpler clipping mechanism. Stable, scalable, and practical. Becomes the industry standard.

**2022 — InstructGPT / RLHF (OpenAI):** PPO is applied to align a 175B language model using human feedback. The three-stage pipeline — SFT → reward model → PPO — becomes the industry blueprint. The 1.3B aligned model beats the 175B base model in human preference.

**2023 — DPO (Rafailov et al.):** Direct preference optimisation rewrites the RLHF objective without RL. Two stages instead of three, no PPO, no value model. Simpler and more stable; comparable quality. Becomes the default for open-source models.

**2024 — GRPO (DeepSeek):** Group relative policy optimisation replaces the value model with a group baseline. No separate critic network. Memory-efficient and stable. Used in DeepSeek-Math and DeepSeek-R1.

**2024–2025 — Verifiable rewards:** For tasks with objectively correct answers (maths, code, formal reasoning), skip the reward model entirely. Use executable correctness or formal verification. Completely eliminates reward hacking from the reward model source. Powers the reasoning capability explosion of late 2024 and 2025.

**2025 — DAPO, RLOO, and beyond:** Continued refinements addressing specific instabilities: dynamic sampling, reference-free objectives, leave-one-out baselines. The field continues to move rapidly.

The arc of this timeline is clear: algorithms get simpler, more stable, and more memory-efficient with each generation. The core idea — use reward signals to shape token probability distributions — remains constant. The engineering around it improves relentlessly. 🚀

---

## Up Next 🎯

→ **REINFORCE Algorithm (Topic 23)**

With RL foundations in place, we now turn to the simplest concrete implementation of the policy gradient idea: the REINFORCE algorithm. REINFORCE (Williams, 1992) directly applies the policy gradient theorem — multiply the log probability of each action by the return, sum across the trajectory, and take a gradient step. It is the mathematical ancestor of every modern LLM training algorithm from PPO to GRPO. Understanding REINFORCE deeply makes every subsequent algorithm immediately recognisable as a variation or improvement on this core idea.

Key questions REINFORCE answers:
- How do you compute the exact gradient update for a single trajectory?
- Why is the baseline subtraction (advantage instead of raw return) so important for training stability?
- Why does REINFORCE have high variance, and why does that variance matter?
- How does the Monte Carlo return estimate connect to the policy gradient theorem?
- What is the difference between the full return G_t and the advantage A(s, a)?
- Why does using the full return (without baseline) lead to slow and inconsistent learning?
- How does increasing batch size reduce variance in REINFORCE, and what are the compute trade-offs?

Once you understand REINFORCE, PPO's clipping, GRPO's group baseline, and DAPO's dynamic sampling all become variations on the same theme — refinements to make the core REINFORCE idea more stable, more sample-efficient, and more scalable to the demands of LLM training. Every algorithm in the series ahead will be explained as: "start with REINFORCE, then add [modification X] to fix [problem Y]." REINFORCE is the seed from which the entire modern RL-for-LLMs literature has grown. 🌱

The RL foundations vocabulary introduced in this chapter — agent, state, action, policy, reward, episode, trajectory, return, advantage, J(θ), policy gradient — will appear in every subsequent chapter. Master these terms now and every paper in the series becomes dramatically easier to read and reason about.
