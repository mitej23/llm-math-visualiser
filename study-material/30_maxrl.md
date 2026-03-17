# 🎯 MaxRL — Maximum Likelihood RL

> **Sources used:**
> - Yu et al., *Reinforcement Learning for Language Models*, various preprints 2024–2025
> - DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025) — [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
> - Chen et al., *Evaluating Large Language Models Trained on Code* (HumanEval, pass@k definition) — [arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374)
> - Kaplan et al., *Scaling Laws for Neural Language Models* — [arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
> - Schulman et al., *Proximal Policy Optimization Algorithms* — [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
> - Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO paper) — [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
> - Dong et al., *RAFT: Reward rAnked Fine-Tuning for Generative Foundation Model Alignment* — [arxiv.org/abs/2304.06767](https://arxiv.org/abs/2304.06767)

---

## The Big Idea

**MaxRL (Maximum Likelihood Reinforcement Learning)** is a training approach that sidesteps the complexity of traditional RL algorithms by reframing the improvement of an LLM as a supervised learning problem — but only on a carefully filtered subset of data. Rather than computing rewards, running policy gradients, maintaining value models, or clipping policy ratios, MaxRL does something strikingly simple: generate a large batch of candidate responses, check which ones are correct, discard every incorrect one, and run ordinary supervised fine-tuning on what remains.

The name comes from the underlying objective. When you discard wrong answers and train only on correct ones, you are performing maximum likelihood estimation on the distribution of high-quality outputs. There is no probability ratio between old and new policies, no KL penalty term in the loss, no advantage function to estimate. The loss function is the standard cross-entropy loss that a language model uses during pretraining and supervised fine-tuning. The only addition is the filtering step that decides which outputs are high enough quality to learn from.

This approach is sometimes called **rejection sampling fine-tuning (RFT)** or **filtered supervised learning**, but the framing as MaxRL highlights the conceptual connection to the broader RL literature: you are still optimizing a policy toward better outcomes, just doing it through a maximum likelihood lens rather than through gradient-based policy optimization. Some papers call it **RAFT** (Reward rAnked Fine-Tuning), emphasizing that a ranking step precedes the fine-tuning. Regardless of the name, the mechanism is the same.

MaxRL is particularly well suited to domains where correctness can be automatically verified — mathematics, code execution, formal proofs, structured data extraction, and similar tasks where an external checker can binary-judge whether a response is right or wrong. In these domains, MaxRL can match or exceed the performance of PPO and GRPO while requiring less engineering complexity, less memory, and far less hyperparameter sensitivity.

The technique rose to prominence alongside the wave of research into reasoning-capable models like DeepSeek-R1, which used large-scale reinforcement learning on verifiable rewards. MaxRL represents the simplest point in the spectrum of RL algorithms for language models, and understanding it clearly makes the more complex methods (GRPO, PPO) much easier to appreciate — because you can see exactly what complexity they add and why.

---

## Real-Life Analogy: The Filter-Everything Chef

Imagine a culinary school where the head chef wants to teach students how to make a perfect soufflé. The chef has two philosophies available, and they lead to completely different classrooms.

The first philosophy is the traditional kitchen instructor. Every time a student makes a soufflé, the instructor is right there, tasting it, giving detailed feedback on a ten-point scale — "this one fell because you over-beat the egg whites, score 4 out of 10," "that one is too dense because the oven was opened, score 6 out of 10." The feedback is rich, nuanced, and continuous. The instructor must always be present, always evaluating, always engaged. Students learn from the graded feedback on both their successes and their failures. This is analogous to PPO or GRPO: continuous reward signals guide every update, and even negative experiences carry learning value.

The second philosophy is the MaxRL chef. This chef makes a radically different choice. They cook dozens of soufflés themselves, photograph every single one, and then invite students to study only the photographs of the soufflés that came out perfectly. Not the mediocre ones. Not the collapsed ones. Not the ones that were almost good. Exclusively the perfect ones. The students learn entirely by imitating successes.

This MaxRL chef does not need to be present while students cook. They do not need to grade failures. They simply filter the universe of soufflé possibilities down to the perfect subset and say: produce outputs that look like these. The filtering is the key innovation. The chef expends significant effort cooking many soufflés to find the perfect examples. But once the perfect examples are identified, the teaching step is straightforward — it is just imitation learning on a curated set.

There is no nuanced scoring to design, no debate about whether a slightly-collapsed soufflé deserves a 4 or a 5 out of 10. Either a soufflé is perfect enough to be a teaching example, or it is excluded. The binary judgment is the simplification that makes everything downstream tractable.

For language model training: the chef's kitchen is the LLM generating thousands of candidate responses, the recipe judgment is the automated verifier checking mathematical correctness or test-case pass rates, and the students learning from photographs is the supervised fine-tuning step that trains only on verified correct outputs.

One important implication: the MaxRL chef must cook many soufflés to find the perfect ones. If the chef is terrible and none of their soufflés are perfect, they have nothing to teach from. This is the bootstrap problem of MaxRL — the model must already be capable enough to produce correct responses some of the time. A model with zero probability of solving any problem in the training set will produce no training signal and cannot improve.

---

## The Core Idea: Filter and Supervise — Deep Dive

The MaxRL training loop has three steps that repeat across training iterations. Understanding each step in detail — including the engineering challenges hidden in each — clarifies why the approach is both powerful and bounded.

### Step 1: Generate Many Candidates

Given a prompt or problem, the model generates a large number of candidate responses. The number is typically denoted k, and practical implementations use k ranging from 8 to 64, with the optimal value depending on the model's current per-problem accuracy and the difficulty distribution of the training set. The responses are generated with elevated temperature to encourage diversity — each candidate response should explore different solution paths and phrasings. If all k responses were nearly identical, there would be little benefit to generating more than one.

The generation step is the most computationally expensive part of the process. Generating k responses means running the model's forward pass k times for each prompt, each producing a full sequence of tokens. For a mathematics problem that elicits a 400-token chain-of-thought reasoning response, generating k=32 candidates requires processing approximately 12,800 tokens per problem. Across a training batch of 500 problems, that is 6.4 million tokens of inference just for generation — before any training has occurred.

This cost is real and significant, but it is also highly parallelizable. Each of the k responses is independent of the others and can be generated concurrently across multiple GPUs. The wall-clock time for generation scales with the cost of a single response, not k times that cost, if you have sufficient hardware parallelism. This is one reason why MaxRL is practical at scale: the compute is expensive in aggregate, but the latency per training step is manageable.

### Step 2: Verify Correctness

Each of the k candidate responses is passed through a verifier that makes a binary judgment: correct or incorrect. The verifier is an external system — not a trained neural network — that checks an objective property of the response.

For mathematics problems: compare the final numerical answer against a ground-truth answer. The answer might be extracted by looking for text within a specific delimiter (such as a boxed environment in LaTeX-formatted responses) or by parsing the last numerical expression in the response. Symbolic math libraries can check algebraic equivalence, which is more robust than string matching for answers like "1/3" versus "0.333."

For code generation: execute the generated code against a suite of unit tests. If all tests pass, the response is correct. This approach underlies evaluation benchmarks like HumanEval and MBPP. It requires a sandboxed execution environment to safely run potentially malicious or buggy generated code. The sandbox overhead adds latency but is non-negotiable for security.

For formal proofs in theorem proving: run the generated proof through a proof checker (Lean, Coq, Isabelle). The checker either accepts the proof as valid or rejects it with an error. Proof checking is fast — much faster than proof generation — making it an ideal verifier.

The crucial property of any verifier is reliability. A false positive (labeling an incorrect response as correct) introduces bad training data — the model will learn from and reinforce wrong solutions. A false negative (rejecting a correct response) is less catastrophic — it wastes a training opportunity but does not corrupt the model. Verifier reliability asymmetry matters: err on the side of rejecting uncertain answers rather than accepting them.

### Step 3: Supervised Fine-Tuning on Correct Responses Only

After verification, you have a filtered set of (problem, correct response) pairs. These pairs are used as training data for standard supervised fine-tuning. The loss function is cross-entropy: for each token in the correct response, maximize the log-probability of that token given the preceding context (the problem plus all previous response tokens).

There is no policy ratio. No advantage function. No clipping. No KL penalty. Just ordinary token-level cross-entropy minimization, the same as pretraining and SFT. This means all the stability properties of supervised fine-tuning carry over: well-understood hyperparameters, stable gradients, predictable convergence behavior. Training is dramatically simpler to monitor and debug than PPO or GRPO training.

After a training pass on the filtered data, the model's weights are updated to increase the probability of generating responses similar to the verified-correct ones. In the next iteration, the updated model generates new candidates from the new distribution, which are verified again, producing a new filtered training set. This iterative refinement loop is why MaxRL is correctly classified as reinforcement learning rather than one-shot supervised learning — the model's behavior feeds back into its own training signal.

### What Happens With Unhelpful Batches

Two failure cases break the feedback loop. First: if all k responses for a problem are wrong, no training signal is produced for that problem. The model cannot improve on it in this iteration. Second: if all k responses are correct, the problem is already solved by the model — training on it reinforces existing capabilities rather than building new ones. Both extremes should be detected and handled:

- All-wrong: mark the problem as too hard for the current model, reduce its sampling probability, revisit later after easier problems have built up relevant capabilities.
- All-correct: mark the problem as mastered, reduce its sampling probability or remove from the active training set.

Problems in the intermediate range — where some but not all responses are correct — carry the densest training signal. Curriculum design that maintains most problems in this zone maximizes the information extracted per unit of compute.

---

## pass@k Diversity Sampling — Deep Dive

The pass@k metric was formalized in the HumanEval paper (Chen et al., 2021) for evaluating code generation models, but it has become central to the MaxRL framework across all domains. Understanding it precisely reveals the mechanics of MaxRL's training signal and how compute translates into learning.

### The Mathematical Definition

pass@k asks: if you generate k candidate responses for a problem, what is the probability that at least one of them is correct?

Formally, if each individual response is correct independently with probability p (the per-sample accuracy), then:

pass@k = 1 - (1 - p)^k

This formula captures the complementary probability: the chance that all k responses are wrong, subtracted from 1. A few worked examples make the behavior concrete:

For a model with p = 0.10 (10% per-sample accuracy):
- pass@1  = 10%
- pass@4  = 34%  (1 - 0.9^4)
- pass@16 = 81%  (1 - 0.9^16)
- pass@32 = 97%  (1 - 0.9^32)

For a model with p = 0.05 (5% per-sample accuracy):
- pass@1  = 5%
- pass@8  = 34%
- pass@32 = 81%
- pass@64 = 96%

The practical conclusion: even models with individually low accuracy can produce correct training samples with sufficient k. MaxRL's strength is extracting these correct samples for supervised learning, rather than requiring the model to be reliable on the first try.

### The Independence Assumption

The pass@k formula assumes independence between the k responses. In practice, there is correlation: the model has a fixed set of learned strategies, and when those strategies fail, multiple responses tend to fail in similar ways. If the model's dominant incorrect approach to a problem is a particular algebraic error, many of the k responses will make that same error.

This correlation means that the empirical pass@k is typically lower than the formula predicts. Temperature is the tool for breaking this correlation: higher temperature introduces randomness that pushes the model off its dominant solution paths and allows it to explore minority-probability approaches that may be correct. The independence assumption becomes more accurate at higher temperatures.

### Practical k Values

In MaxRL implementations for mathematics and code reasoning:

k=8 to 16: appropriate for problems where the model has moderate existing capability, producing training signals efficiently without excessive generation cost. Used in early training stages or for medium-difficulty problems.

k=32 to 64: necessary for hard problems where per-sample accuracy is very low. At k=64, even a model with only 3% per-sample accuracy achieves pass@64 of approximately 86%, producing meaningful training signal from otherwise impenetrable problems.

k=1 with post-hoc filtering: the degenerate case where only one response is generated. This is effectively online SFT — train on every response the model produces that happens to be correct. Low efficiency for hard problems, but the simplest implementation.

The right k is not fixed throughout training. An adaptive curriculum adjusts k based on the current model's estimated pass@k for each problem. When a problem has high pass@k, reduce k to save compute. When pass@k is low, increase k to find enough correct samples. This adaptation requires tracking performance per problem across training iterations, but the payoff is significantly better compute efficiency.

### Temperature, Top-p, and Response Diversity

For pass@k to benefit from large k, responses must be diverse. The diversity settings used during MaxRL generation differ from those used during evaluation or deployment:

**Temperature 0.8:** mild randomness, good diversity for problems with multiple valid solution approaches, each response is still generally coherent.

**Temperature 1.0:** standard training temperature, balances diversity against coherence, the default for most MaxRL implementations.

**Temperature 1.2:** higher diversity, useful for hard problems where the model's default reasoning path always fails and only unusual approaches succeed. Risk: individual responses become less coherent.

**Temperature 1.5 and above:** aggressive diversity, appropriate for exploratory searches but individual response quality degrades significantly. Chain-of-thought reasoning tends to become less structured at these temperatures.

Most implementations use temperature around 1.0 to 1.1 for MaxRL generation, slightly higher than the 0.7 to 0.8 range used for supervised fine-tuning evaluation.

---

## Compute Indexing — Deep Dive

One of MaxRL's most important intellectual contributions is making the relationship between compute expenditure and training signal quantity explicit and directly tunable. Other RL algorithms treat compute as a byproduct of their training procedure. MaxRL treats compute allocation as a primary design decision.

### The Two-Phase Compute Budget

In any MaxRL training run, compute is split between two phases:

**Generation phase:** inference compute spent generating k responses per problem. This is proportional to k, the number of problems in the batch, and the average response length. It is the "search" phase — you are searching for correct solutions in the model's output distribution.

**Training phase:** gradient compute spent performing supervised fine-tuning on the filtered correct responses. This is proportional to the number of correct responses found and the number of gradient steps taken.

For a fixed total compute budget, there is a trade-off: spending more on generation yields denser training data (more correct responses) but less compute available for training on those responses. Spending more on training makes fuller use of each correct response found but constrains how many you can find. The optimal split depends on the current model's accuracy (which determines how many correct responses generation will find per unit of compute) and on the marginal value of additional gradient steps versus additional training examples.

### Concrete Compute Accounting

Suppose the model has per-problem accuracy p=0.10, and each response costs C inference FLOPs. To find one correct response, you need on average 1/p = 10 responses, costing 10C FLOPs in generation. The expected number of correct responses found from a generation budget of B FLOPs is B / (10C).

As the model improves — say p rises from 0.10 to 0.30 — the same generation budget B now produces B / (3.33C) correct responses, three times as many. Training signal density improves automatically as the model improves. This positive feedback is one reason MaxRL training often accelerates in its middle phase: as the model gets better at producing correct responses, each training iteration yields more training signal.

### The Connection to Test-Time Compute Scaling

The compute indexing perspective connects directly to a major research theme of 2024–2025: test-time compute scaling. Models like OpenAI's o1 and o3 allocate more inference compute at test time to improve response quality — generating multiple candidates and selecting the best, or using iterative refinement. The training-time compute (finding correct responses to learn from) and test-time compute (searching over multiple responses to find the best for a user's query) are deeply analogous.

Models trained with MaxRL are naturally suited to test-time compute scaling because they were trained in a regime where generating multiple attempts and keeping the best is the standard operating mode. The model's weights are adapted to the experience of trying multiple approaches, not just producing a single deterministic answer. This may partly explain why MaxRL-trained models tend to show stronger scaling with test-time compute than models trained purely with SFT.

### Marginal Utility Curves

A useful way to think about compute allocation is through marginal utility curves. For a fixed model and problem:

- The first few generated responses (k=1 to 4) provide the highest marginal utility: you go from having no correct samples to having some.
- The next few responses (k=4 to 16) provide moderate marginal utility: you collect more diverse correct approaches.
- Beyond k=16 to 32, the marginal utility of each additional response diminishes: most correct solution approaches have already been found.

This diminishing marginal utility suggests that for a fixed compute budget, you should generate moderate k per problem and cover more problems, rather than generating very large k for few problems. The exact optimal k depends on the difficulty distribution and the current model accuracy, but empirically, k in the range 16–32 provides a good balance for most training settings.

---

## Why No Importance Sampling — Deep Dive

The absence of importance sampling in MaxRL is one of its most consequential properties. Understanding why it can safely omit importance sampling requires understanding what importance sampling does and when it is necessary.

### The Purpose of Importance Sampling in RL

In standard policy gradient methods like PPO, you collect a batch of experience (observations, actions, rewards) under one version of the policy, then use that experience to compute gradient updates. After a gradient update, the policy has changed — the new policy assigns different probabilities to the same actions. If you then use the same experience batch for another gradient update, there is a distribution mismatch: the experience was collected under the old policy, but you are computing gradients as if it were collected under the new policy.

Importance sampling corrects for this mismatch. Each training example is reweighted by the ratio of the new policy's probability to the old policy's probability for the action taken. If the new policy is twice as likely to take some action as the old policy, that experience is downweighted by a factor of two to prevent overestimating how often the new policy would encounter that situation.

In PPO, the policy ratio r_t(theta) = pi_new(a|s) / pi_old(a|s) is precisely the importance sampling weight. The clipping constraint pi_clip limits how far this ratio can deviate from 1, keeping importance sampling corrections bounded and stable. Without importance sampling, the gradient estimates would be biased and could lead to instability.

### MaxRL Is Strictly On-Policy

MaxRL avoids this problem entirely by being strictly on-policy. At the start of each training iteration, the current model generates fresh candidates. Those candidates are verified, filtered, and used immediately for one round of supervised fine-tuning. After the weights are updated, the old generation is discarded — it is never reused. The next iteration generates new candidates from the newly updated model.

Because the generation policy and the training policy are always synchronized — they are the same model at the same moment in training — there is never a distribution mismatch. The experience used for training was generated by the current policy, not an older one. Importance sampling weights would all be exactly 1.0, which is equivalent to not applying them at all.

This on-policy constraint is the reason MaxRL must regenerate candidates every iteration. It cannot amortize the cost of generation by reusing the same batch for multiple training steps. Each training step requires fresh data from the current model. This is more expensive per gradient step than off-policy methods that reuse data, but it is also simpler, more stable, and theoretically cleaner.

### Practical Approximations

Most real implementations do make a small concession: they perform multiple gradient update steps on the same generated batch before discarding it. This is technically off-policy (each update step after the first uses data from a slightly outdated policy) and technically requires importance sampling corrections. In practice, if only a few gradient steps are taken per batch (typically 1 to 4), the distribution mismatch is small enough that importance sampling corrections have negligible impact. The algorithm remains approximately on-policy.

This practical approximation is the same one that PPO itself makes — PPO's derivation assumes you update once per batch, but implementations typically do multiple epochs. The difference is that PPO explicitly computes and clips the importance sampling ratio to bound the error, while MaxRL ignores the small correction entirely and accepts the negligible bias.

---

## MaxRL vs Verifiable Rewards — Deep Dive

MaxRL and "verifiable rewards" are terms that frequently appear together in 2024–2025 RL for LLMs literature, sometimes used as if they were synonymous. They are closely related but distinct concepts, and understanding the distinction is important.

### Verifiable Rewards as a Signal Source

The phrase "verifiable rewards" refers to using an automated, objective verification system as the reward signal in reinforcement learning — rather than a trained reward model. Instead of a neural network that has learned to predict human preferences, you use an external program or system that objectively checks whether a response satisfies some criterion:

- A math answer checker that compares the model's final answer to a ground-truth answer
- A code executor that runs test cases against generated code
- A formal proof checker that verifies whether a proof is logically valid
- A database query executor that checks whether generated SQL produces correct results

The key property of a verifiable reward is that it is objective and automated. It does not require human annotation. It cannot be "gamed" by surface-level stylistic features the way a trained reward model can. An incorrect mathematical answer will fail the verifier no matter how confidently or eloquently it is phrased.

DeepSeek-R1 (2025) used verifiable rewards as its primary training signal — the model was rewarded for producing correct mathematical answers and well-formatted responses, with correctness checked by an automated system. This is described as "verifiable rewards" in the paper, but the actual RL algorithm used was GRPO, not MaxRL.

### Where MaxRL Fits

MaxRL is one specific algorithm that can use verifiable rewards. The relationship is:

- Verifiable rewards: the source of correctness signals (what tells us if a response is right or wrong)
- MaxRL: the training algorithm (what we do with that correctness signal)

MaxRL uses verifiable rewards in the simplest possible way: binary filter. A response is either correct (included in training) or incorrect (discarded). The reward is used as a gate, not as a gradient.

GRPO also uses verifiable rewards, but differently: it assigns binary rewards {0, 1} to responses in a group, computes group-relative advantages, and incorporates the advantages directly into the policy gradient. GRPO learns from both correct responses (positive advantage) and incorrect ones (negative advantage), whereas MaxRL discards the incorrect ones.

The distinction matters when resources and engineering complexity are the binding constraint. If you have verifiable rewards and want the simplest possible training loop, use MaxRL. If you want to extract more information from every generated response — including from failures — use GRPO or PPO.

### The Math and Code Alignment

Both MaxRL and verifiable-reward GRPO shine in mathematics and code because these domains have fast, reliable, automatic verifiers. The MATH benchmark's evaluation script can check thousands of answers per second. HumanEval's test runner can execute hundreds of code solutions in parallel. The bottleneck in MaxRL training is not verification — it is generation and training. Verification is cheap.

For domains without reliable verifiers — open-ended creative writing, helpfulness and harmlessness judgments, nuanced reasoning about ambiguous situations — neither MaxRL nor verifiable-reward GRPO can be directly applied. You either need a trained reward model (reintroducing the complexity MaxRL sought to avoid) or human annotations (which do not scale). MaxRL's applicability is bounded by the set of tasks where an objective, automated verifier can be defined.

---

## The Bootstrap Question: Where Does the First Training Data Come From?

A practical question that frequently arises when applying MaxRL is: what do you do if the base model cannot solve any problems in your training set? MaxRL requires at least some correct responses to generate training signal. If pass@k is near zero for all problems, no training data is collected and the model cannot improve.

### Starting From SFT

The standard answer is to start from a supervised fine-tuned model rather than a raw pretrained base model. An SFT model has been trained on human demonstrations of correct problem-solving — worked examples of mathematics problems, reference implementations of code challenges, and similar. This gives the model a starting capability that is far above zero, enabling MaxRL to find correct responses from the first iteration.

In the InstructGPT and DeepSeek-R1 pipelines, supervised fine-tuning always precedes RL training. SFT provides the foundation; RL (in whatever form) then improves beyond the ceiling of the demonstration data. MaxRL is no exception to this pattern.

### Easy-to-Hard Curriculum as Bootstrap

An alternative for cases where even SFT gives near-zero accuracy on target problems is to bootstrap with an easy-to-hard curriculum. Begin the MaxRL training on much simpler problems than the eventual target — problems simple enough that even the base model can sometimes solve them. As the model improves on easy problems, its general capabilities (algebraic manipulation, code structure, logical inference) also improve. Eventually, those improved capabilities transfer to harder problems, producing nonzero pass@k on the target difficulty level and enabling MaxRL to continue.

This curriculum-based bootstrap is more laborious to design but extends MaxRL to settings where no SFT data exists for the target domain. It is the main strategy for bootstrapping MaxRL on novel or specialized domains.

### Distillation as Bootstrap

A third strategy is to use a more capable model to generate the initial training data. If a larger or more capable model (say GPT-4) can solve problems in your training set reliably, you can use it to generate correct responses for each problem and fine-tune your smaller model on those responses. The smaller model then has a starting capability from which MaxRL can improve further.

This approach is called **distillation** — the smaller model learns from the larger model's outputs. It is widely used in practice. DeepSeek-R1 used distillation from larger DeepSeek models to produce smaller fine-tuned models. The smaller distilled model often starts with better capabilities than a purely SFT-trained model on the same problem domain, enabling faster MaxRL improvement.

---

## Comparison With Other Methods — Deep Dive

MaxRL, GRPO, and PPO represent three different points on the spectrum of complexity and information efficiency in RL for LLMs. Each makes different trade-offs between algorithm complexity, memory requirements, information extracted from each response, and training stability.

### PPO: Full RL With All Components

PPO requires four models in memory simultaneously: the policy being trained, a frozen reference policy for KL computation, a reward model for scoring responses, and a value model for computing per-step advantage estimates. The training loop involves generating responses under the current policy, scoring them with the reward model, computing per-token KL penalties against the reference policy, estimating advantages using the value model, and running the clipped policy gradient update with importance sampling corrections.

PPO's complexity is justified by its information efficiency: it extracts as much learning signal as possible from every generated response. Every response — whether good or bad — contributes a gradient signal. The advantage function precisely attributes credit to individual tokens, allowing the model to learn that the error in a solution occurred at a specific reasoning step, not throughout the entire response.

The cost of this information efficiency is engineering complexity, memory, and stability challenges. The four-model requirement means the memory footprint for a 7B parameter model is roughly 28B parameters worth of weights, before accounting for gradients and optimizer states. PPO training can collapse (reward hacking), diverge (KL explosion), or oscillate, requiring careful hyperparameter tuning and monitoring.

### GRPO: Group-Based RL Without Value Model

GRPO (Group Relative Policy Optimization, from the DeepSeekMath paper) eliminates the value model by computing advantages relative to the group of responses generated for the same problem. For a group of k responses with rewards r_1 through r_k, the advantage of response i is:

advantage_i = (r_i - mean of all rewards in group) / standard deviation of all rewards in group

This group-relative normalization replaces the value model's job of estimating expected reward from a given state. Instead of asking "is this response better than the average response from any state?", GRPO asks "is this response better than the other responses to this specific prompt?" This is a simpler, more directly applicable comparison.

GRPO still uses a reference policy for KL computation and applies importance sampling within the policy gradient update. It still processes both correct and incorrect responses, using the normalized advantages to push toward high-reward responses and away from low-reward ones. The result is that GRPO uses both the positive signal from correct responses and the negative signal from incorrect ones, learning from failures as well as successes.

This makes GRPO more data-efficient than MaxRL per generated response: GRPO extracts gradient signal from all k responses, while MaxRL extracts signal from only the correct ones. For a group with 2 correct responses out of k=8, MaxRL uses 2/8 = 25% of generated data, while GRPO uses 100% (with varying advantage magnitudes).

### MaxRL: Filter and Supervise

MaxRL requires only the current policy model and an external verifier. No reference policy, no reward model, no value model. The training step is supervised fine-tuning with cross-entropy loss — the simplest possible training. The algorithm uses only correct responses for learning, discarding everything else.

The trade-off is clear: MaxRL is the least data-efficient per response (uses only correct ones) but the simplest and most stable (supervised fine-tuning). When generate-and-filter is cheap (hardware-rich environments), the inefficiency is acceptable. When engineering simplicity and training stability are paramount (early research phase, resource-constrained settings), MaxRL's straightforwardness is a major advantage.

### When to Choose Each

Choose PPO when: you need maximum performance, have engineering resources to handle the complexity, and are working in a domain where responses have varying quality levels that a reward model can distinguish.

Choose GRPO when: you want strong performance with reduced memory (no value model), are working with verifiable rewards or a simple reward model, and want the benefits of RL (learning from failures) without PPO's full complexity.

Choose MaxRL when: you have a reliable verifier, want maximum training stability and engineering simplicity, are resource-constrained on memory, and can afford the compute cost of generating many candidates to offset the filtering waste.

---

## How it Works in Practice

A concrete MaxRL training run for a mathematics reasoning model demonstrates the practical engineering decisions at each stage.

### Dataset and Verifier Preparation

Start with a dataset of mathematics problems, each with a ground-truth answer. The dataset should span a wide difficulty range. Standard sources include the MATH benchmark (7,500 training problems across algebra, geometry, number theory, and competition mathematics), GSM8K (8,500 grade-school mathematics problems with step-by-step solutions), and competition datasets from AMC, AIME, and similar sources.

The answer verifier must be built and tested before training begins. For numerical answers, the verifier extracts the final answer from the model's response and compares it against the ground truth. Answer extraction is trickier than it sounds: models do not always format answers consistently. The extractor must handle fractions written as "3/4" or "0.75" or "three-fourths," numbers followed by units, and answers embedded in longer sentences. A robust verifier normalizes all numerical forms before comparison.

### Training Loop Iteration

In each training iteration, the procedure is:

First, sample a batch of problems from the training set. Batch sizes of 64 to 512 problems are typical. Weight sampling toward problems in the intermediate difficulty range where pass@k is neither near 0 nor near 1.

Second, for each problem, generate k candidate responses from the current model using temperature in the range 0.9 to 1.1. Batch these generation requests across GPUs for maximum parallelism. Each response should include a chain-of-thought reasoning section and a clearly formatted final answer.

Third, run the verifier on every response. Extract the final answer, normalize it, compare against ground truth. Label each response correct or incorrect. Typically, 10% to 40% of responses are correct in a well-designed training setup.

Fourth, collect all (problem, correct response) pairs. Check that no individual problem is contributing more than some maximum number of correct responses to the batch (to prevent a single easy problem from dominating the training signal). Discard all incorrect responses.

Fifth, run supervised fine-tuning on the filtered batch. Use a learning rate of 1e-5 to 2e-5, run 1 to 4 gradient steps per batch, apply gradient clipping to prevent rare large updates.

Sixth, log the proportion of problems that produced at least one correct response (the batch-level pass@k). Log the average number of correct responses per problem. These metrics characterize training signal density and should be monitored to detect all-wrong or all-correct batch failures.

Seventh, repeat. A full MaxRL training run typically consists of hundreds to thousands of such iterations, running for hours to days on a multi-GPU cluster.

### Monitoring and Stopping

Training progress is measured by evaluating pass@1 on a held-out evaluation set at regular intervals. Because MaxRL's training signal is binary and clearly defined, progress tracking is reliable. When pass@1 on the evaluation set plateaus for several consecutive evaluation checkpoints, training can be stopped. Overfitting is rare but possible — watch for pass@1 increasing on the training set while remaining flat or declining on the eval set.

---

## MaxRL in Context: The 2024–2025 Research Wave

MaxRL's rise to prominence is inseparable from the broader wave of research into reasoning-capable language models that accelerated in 2024 and peaked with the release of DeepSeek-R1 in early 2025. Understanding the context makes MaxRL's role clearer.

### The Reasoning Bottleneck

By 2023, it was well established that scaling pretrained language models improved performance on many tasks. But mathematical reasoning and formal code synthesis remained stubborn bottlenecks: even the largest models achieved limited reliability on competition mathematics problems and complex multi-step coding tasks. The limiting factor was not knowledge (the models knew the relevant facts and techniques) but reliable execution of multi-step reasoning chains.

Supervised fine-tuning on demonstrations had been tried extensively. The ceiling of SFT for mathematics is approximately the ceiling of human demonstrators — and while human demonstrators can solve competition mathematics, they cannot produce the volume and diversity of demonstrations needed to train reliable automation. A model that learns from 10,000 human-written solutions to 10,000 problems will generalize to novel problems, but not perfectly, and scaling the demonstration dataset further shows diminishing returns.

RL for reasoning offered a path beyond the SFT ceiling: instead of learning from a fixed set of demonstrations, the model could practice on an essentially unlimited number of problems and receive correctness feedback automatically via verifiers. The challenge was making this RL training work reliably and at scale.

### The Role of MaxRL in This Wave

MaxRL occupied an important niche in this research landscape: it was the simplest approach that demonstrably improved reasoning capabilities beyond the SFT baseline. Papers like RAFT (2023) and various rejection sampling fine-tuning works showed that the elaborate PPO apparatus was not necessary for meaningful improvement on mathematics benchmarks — a simple generate-verify-filter loop with supervised training was sufficient.

This simplified approach became the baseline that more complex methods were compared against. GRPO was motivated in part as a method that could match or exceed MaxRL's performance while also extracting signal from incorrect responses. Process reward models were motivated as a method that could improve on MaxRL's outcome-only supervision.

DeepSeek-R1's success — achieving strong performance on AIME and other competition mathematics benchmarks — demonstrated that RL training for reasoning was genuinely powerful at scale. While DeepSeek-R1 used GRPO rather than pure MaxRL, the verifiable-rewards framework it employed is the direct parent of MaxRL's binary verifier concept. Researchers studying DeepSeek-R1's methods naturally asked: what is the simplest version of this that still works? MaxRL is the answer.

---

## Common Misconceptions

### Misconception 1: "MaxRL is just supervised fine-tuning — it is not really reinforcement learning"

Each individual training step in MaxRL is a supervised fine-tuning step, using a cross-entropy loss on labeled data. But the overall algorithm is reinforcement learning in the broad sense: the model's current behavior (generating responses) determines what training data is available (which responses are correct), which determines how the model's weights are updated, which changes future behavior. This feedback loop between behavior and training signal is the defining characteristic of reinforcement learning. MaxRL implements this loop through a filtering-and-supervised-learning mechanism rather than through explicit policy gradients, but the loop structure is identical.

### Misconception 2: "MaxRL requires a trained reward model like PPO and RLHF"

MaxRL requires a verifier — an external system that checks objective correctness. A verifier and a reward model are fundamentally different. A reward model is a trained neural network, itself requiring annotation data, its own training pipeline, and ongoing monitoring for reward hacking. A verifier is an external program that runs code, checks mathematical answers, or validates proofs. It does not require training data beyond the ground-truth problem labels already needed for evaluation. This distinction is crucial: MaxRL's computational simplification is specifically the elimination of the reward model training and deployment overhead.

### Misconception 3: "MaxRL only works if the base model is already highly capable"

The model must be capable of generating at least some correct responses for at least some problems in the training set. But "at least some" can be quite modest: if the model achieves 3% per-sample accuracy and you generate k=64, pass@64 is roughly 86%. You will find correct responses for most problems in your training set. The key is problem curriculum: start with problems the model can sometimes solve, and introduce harder problems progressively as the model's pass@k improves. MaxRL with a weak base model requires more careful curriculum design, but it is not fundamentally blocked by low initial accuracy.

### Misconception 4: "MaxRL wastes most of its generated compute since it discards wrong answers"

This framing is correct but incomplete. Every incorrect response does represent compute that does not directly contribute a training gradient. But incorrect responses are not entirely without value: they characterize the current capability of the model, reveal which problems remain unsolved, and guide the curriculum. The generate-and-discard inefficiency is the explicit cost MaxRL pays for the benefit of simpler training and a cleaner training signal. Whether this trade-off is favorable depends on the setting: in hardware-rich environments where generation is cheap, the waste is acceptable. In compute-constrained settings, GRPO's ability to learn from incorrect responses too becomes more valuable.

### Misconception 5: "MaxRL training is unstable because the training data distribution shifts every iteration"

The opposite is true: MaxRL is more stable than PPO or GRPO precisely because the training step is supervised fine-tuning, not policy gradient descent. The training data distribution does shift every iteration — that is the nature of on-policy training — but the learning step itself is well-behaved supervised learning. There are no policy ratios that can explode, no value function estimates that can collapse, no reward model being gamed. The shifts in training data distribution are gradual and controlled by the learning rate and the number of gradient steps per batch.

### Misconception 6: "MaxRL quickly plateaus because the model masters all problems"

In practice, training sets for mathematics and code span such wide difficulty ranges that the model rarely masters all problems during a training run. As the model improves at algebra problems, competition-level geometry problems remain hard. As it improves at one class of geometry problems, harder ones emerge. Data augmentation — rephrasing problems, changing numerical values, composing sub-problems into larger ones — further extends the effective training set. The practical plateau in MaxRL performance comes from the model reaching the limit of what the training distribution can teach, not from literally solving every problem.

---

## Connections to Other Topics

### Connection to Temperature and Token Sampling (Scene 15)

Temperature is not just a generation setting in MaxRL — it is a core algorithmic parameter. The diversity of the k generated responses depends directly on temperature. The independence assumption underlying the pass@k formula requires high enough temperature that responses explore distinct solution paths. The study guide for temperature and sampling (Scene 15) explains how temperature scales the logit distribution before softmax, controlling the entropy of the resulting token probability distribution. In MaxRL, higher entropy at generation time translates directly to higher diversity in the candidate pool and more efficient discovery of correct solutions.

### Connection to Supervised Fine-Tuning (Scene 18)

MaxRL's training step is exactly the supervised fine-tuning algorithm described in Scene 18, applied to a dynamically generated and filtered dataset rather than a static human-curated one. All the properties of SFT carry over directly: the loss function, the optimizer settings, the learning rate schedule, the gradient clipping strategy. MaxRL inherits SFT's stability and interpretability while adding the adaptive data collection mechanism that allows it to improve beyond the ceiling of a fixed training set.

### Connection to Reward Models (Scene 19)

Scene 19 explains how trained reward models work, what they can and cannot capture, and why they are vulnerable to reward hacking. MaxRL explicitly avoids trained reward models, using verifiers instead. The vulnerabilities discussed in Scene 19 — length bias, style bias, out-of-distribution failures — do not apply to objective verifiers. An answer is either numerically correct or it is not, regardless of the response's length, formatting, or confidence level. MaxRL can be understood as the answer to the question: what happens if you eliminate the reward model and replace it with something that cannot be fooled?

### Connection to RLHF Overview (Scene 21)

Scene 21 traces the evolution of alignment training from InstructGPT's RLHF pipeline through DPO, GRPO, and verifiable rewards. MaxRL sits at the far end of this evolutionary path. The trajectory is a progressive simplification: RLHF needed four models and PPO; GRPO reduced to three models and a simpler policy gradient; MaxRL reduces to one model and a verifier with supervised fine-tuning. Each simplification costs some information efficiency but gains stability and reduces engineering overhead. MaxRL represents what happens when you take the simplification to its logical extreme.

### Connection to KL Divergence (Scene 20)

MaxRL conspicuously lacks a KL divergence penalty. This is safe specifically because MaxRL is on-policy: the training data always comes from the current model, so there is no distribution mismatch to penalize. The risk of catastrophic forgetting — the model drifting so far from its starting capabilities that it forgets earlier learning — is lower in MaxRL than in RL methods with explicit reward optimization, because the supervised fine-tuning step makes only modest weight updates. However, extended MaxRL training exclusively on mathematics problems has been observed to cause some degradation in general language capabilities, a form of catastrophic forgetting that is analogous to the alignment tax discussed in the RLHF chapter.

### Connection to Scaling Laws (Scene not yet covered)

The compute indexing perspective of MaxRL directly engages with neural scaling laws. The original Kaplan et al. scaling law work characterized how loss decreases predictably with compute, data, and parameters. MaxRL introduces a new degree of freedom: the allocation of compute between generation (finding correct samples) and training (learning from them). This allocation is itself subject to optimization, and the optimal allocation follows predictable scaling law-like behavior. As the model improves, the return on generation compute increases (more correct samples per compute unit), while the return on training compute decreases (each new sample provides diminishing new information). Understanding these dynamics is an active research area in 2025.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| MaxRL core idea | Generate k candidates, keep only verified-correct ones, run supervised fine-tuning on the filtered set |
| Why "maximum likelihood" | Training on correct responses maximizes the log-likelihood of those responses — same cross-entropy loss as standard SFT |
| Verifier vs reward model | Verifier checks objective correctness by running external programs; reward model is a trained neural network predicting human preferences |
| pass@k definition | Probability that at least one of k generated responses is correct; equals 1 - (1-p)^k for per-sample accuracy p |
| Why temperature matters | Higher temperature creates diverse candidates; diversity is required for pass@k to increase with larger k |
| Compute indexing | MaxRL makes the trade-off between generation compute and training compute explicit and directly optimizable |
| On-policy property | Each iteration generates fresh data from the current model; no distribution mismatch; no importance sampling needed |
| No importance sampling | Follows from on-policy constraint; generation and training policy are always the same model at the same moment |
| MaxRL vs PPO | PPO: 4 models, importance sampling, value model, learns from all responses; MaxRL: 1 model + verifier, supervised, correct responses only |
| MaxRL vs GRPO | GRPO: group-relative advantages, uses both correct and incorrect responses; MaxRL: binary filter, correct responses only |
| Information efficiency | MaxRL is less efficient per response than GRPO/PPO; compensates with larger k and generation compute |
| Best suited for | Math, code, formal proofs — any domain with a fast, reliable, automatic binary verifier |
| Not suited for | Creative writing, open-ended helpfulness, nuanced reasoning — no binary verifier can be defined |
| Training stability | Very high; supervised fine-tuning is stable and well-understood; no policy ratios or value functions to diverge |
| Key practical challenges | Answer parsing for math; sandboxed code execution; curriculum design; detecting and handling all-wrong batches |
| Limitation: bootstrap problem | Model must produce at least some correct responses; zero per-sample accuracy = zero training signal |
| Limitation: verifier quality | A buggy verifier introduces bad training data; verifier reliability is a critical assumption |
| Catastrophic forgetting | Extended training on narrow domains can degrade general capabilities; mix in general SFT data as a mitigation |
| Alternative names | Rejection sampling fine-tuning (RFT), filtered supervised learning, RAFT (Reward rAnked Fine-Tuning) |

---

## Up Next

**→ Trust Regions and Open Problems in RL for LLMs**

MaxRL represents the simplest end of the RL-for-LLMs spectrum — no reward model, no importance sampling, no advantage functions, no value network. It achieves strong results on verifiable domains at the cost of information efficiency and generality. Understanding MaxRL's limitations is the clearest way to see where the research frontier lies.

The next topics address the open questions that MaxRL raises:

**Trust regions and policy constraint methods.** PPO's clipping is one form of trust region. KL divergence penalties are another. How do you define how much the policy is allowed to change per step in a theoretically principled way? Natural policy gradient methods, mirror descent formulations, and geometric constraints on the policy manifold all offer different answers. These methods are important for RL approaches that need to prevent the reward hacking and mode collapse problems that MaxRL's simplicity sidesteps by design.

**Process reward models.** MaxRL uses outcome supervision — a response is correct or incorrect based on its final answer. But a response can have a correct final answer reached through flawed reasoning, or an incorrect final answer despite mostly correct reasoning. Process reward models (PRMs) score individual reasoning steps, not just the final outcome. This finer-grained signal can guide learning more precisely but requires step-level annotations or automatic step verification, which is significantly harder to implement than outcome verification.

**Exploration strategies for language models.** MaxRL uses temperature to explore. But temperature is a blunt tool — it adds uniform randomness to all token predictions rather than directing exploration toward promising regions of the response space. More sophisticated exploration strategies might use uncertainty estimation to identify which responses are likely to be informative, or use intrinsic motivation signals to encourage the model toward novel solution approaches. These methods from classical RL are being actively adapted to the language model setting.

**The verifier bootstrapping challenge.** For new domains without existing verifiers, how do you build a verifier in the first place? If you have a model that is already decent at a task, you might use it to generate candidate verifiers and test them on a small manually verified set. This bootstrapping problem — building the tools needed for MaxRL from the outputs of a model that does not yet have MaxRL's benefits — is an active research direction. Constitutional AI's self-critique approach is one angle on this problem; automatic test case generation is another.

**Curriculum design at scale.** Which problems to train on, in which order, with which k values, for how long — these curriculum design decisions have large effects on MaxRL's final performance but are poorly understood theoretically. Empirical curriculum research (adaptive sampling based on current pass@k, difficulty schedules drawn from human educational psychology) is producing practical results, but principled theory for optimal curriculum design in MaxRL remains an open problem.

**Combining MaxRL with other signals.** MaxRL is pure outcome supervision with binary filtering. What happens when you combine it with a small amount of process supervision (step-level feedback on a fraction of problems) or a lightweight reward model for the problems without a verifier? Hybrid approaches that blend MaxRL's stability with richer feedback signals are an active research direction. The right mixture likely depends on the domain, the available annotation budget, and the model's current capability level.

**Multi-domain MaxRL.** Most MaxRL implementations focus on a single domain — typically mathematics or code. A multi-domain setup trains on mathematics, code, formal proofs, and structured reasoning simultaneously, sharing model weights across domains. The challenge is that the correct sample density varies wildly across domains, and the curriculum must be managed independently per domain while sharing a unified model. Multi-domain MaxRL remains an open engineering and research challenge with significant potential impact on producing general-purpose reasoning models that do not specialize at the expense of breadth.

---

## Worked Example: One MaxRL Iteration in Detail

To make the algorithm concrete, trace through a single MaxRL training iteration step by step, using a specific mathematics problem and a small k=4 example.

### The Problem and Initial State

Suppose the current model is a 7B parameter mathematics reasoning model, trained with supervised fine-tuning on worked mathematics examples but not yet improved with MaxRL. The training batch includes the problem: "A train travels 240 miles in 4 hours. How far does it travel in 7 hours?"

The ground-truth answer is 420. The verifier will accept any response where the correctly extracted final numerical answer is 420.

### Generation Step

Generate k=4 responses from the current model at temperature 1.0. The four responses might look roughly like:

Response 1: The model computes speed = 240/4 = 60 miles per hour, then distance = 60 × 7 = 420 miles. Final answer: 420. This response is correct, and the reasoning chain is also sound.

Response 2: The model writes that 240 / 4 = 60, then incorrectly multiplies 60 × 6 = 360 due to a calculation error. Final answer: 360. The approach is right but the arithmetic is wrong.

Response 3: The model attempts a ratio approach: 240/4 = 7/x, solving to get x = 7 × 4 / 240 which equals an incorrect fraction. Final answer: something other than 420. The model has set up the proportion incorrectly.

Response 4: The model correctly computes 240/4 = 60, then correctly computes 60 × 7 = 420. Final answer: 420. A second correct response, perhaps with different phrasing in the reasoning chain.

### Verification Step

The verifier extracts final numerical answers from each response: 420, 360, some fraction, 420. It compares each against the ground truth of 420. Results: correct, incorrect, incorrect, correct.

### Filtering Step

Responses 1 and 4 are kept. Responses 2 and 3 are discarded. The filtered training data for this problem consists of two (problem, correct response) pairs.

### Training Step

The two correct (problem, response) pairs are added to the training batch for this iteration. Along with correct responses from other problems in the batch, they form the supervised fine-tuning dataset for this step. The model is updated to increase the probability of generating responses like Responses 1 and 4 — specifically, responses that correctly compute the speed and then multiply by the time. The probability of reasoning chains that make arithmetic errors (like Response 2) or set up proportions incorrectly (like Response 3) is not explicitly decreased, but the model is shifted toward the correct approaches by imitation learning on the correct responses.

### The Effect Over Many Iterations

In subsequent iterations, the model is more likely to correctly compute rates and use them to find distances. This improved capability transfers to other rate-distance-time problems, and also to related problems involving ratios and proportions. The model's generative frontier has expanded by one more class of problems.

This is the fundamental mechanism of MaxRL: one problem at a time, one iteration at a time, the model's reliable capability zone expands outward from easy problems toward hard ones.

### What Counts as "The Same Problem"

One subtle engineering question in iterative MaxRL is whether the same problem should appear in multiple training iterations or be retired once the model achieves high pass@k on it. If the same problem appears in every iteration but the model now solves it reliably, most of the generated responses are correct, the gradients from training on it are small (the model is already confident), and the compute spent generating 64 responses for a problem the model almost always gets right is largely wasted.

A practical answer: track per-problem pass@k across iterations. Problems above pass@k = 0.8 are retired from active training (or heavily downsampled) and replaced with harder problems. Problems below pass@k = 0.2 are flagged as too hard for now and deferred. The active training pool consists of problems in the 0.2 to 0.8 range — the zone where training signal is densest and compute is used most efficiently.

Implementing this dynamic pool requires maintaining a lightweight database of problem difficulty estimates. Each problem is associated with its rolling pass@k, updated every few iterations. The sampling distribution over problems is recomputed periodically to reflect current difficulty estimates. This bookkeeping is a few hundred lines of code — modest engineering overhead for the significant compute efficiency gains it provides. Production MaxRL systems for mathematics training uniformly implement some version of this adaptive curriculum, and it is one of the details that distinguishes effective implementations from naive ones.

---

## MaxRL Within the Broader RL for LLMs Taxonomy

To place MaxRL precisely, it helps to map out the full landscape of RL-style training approaches for language models and see where MaxRL sits relative to each.

### Outcome vs Process Rewards

One axis of the taxonomy is whether the reward signal comes from the outcome (the final answer or output) or from the process (intermediate steps in the reasoning). MaxRL is firmly in the outcome-reward category: the verifier checks the final answer. Process reward approaches check individual reasoning steps, which requires either step-level human annotations or automated step verification — significantly harder to build but providing a richer learning signal.

### Online vs Offline Approaches

Another axis is whether the training data is generated online (from the current model, updated continuously) or offline (from a fixed dataset collected in advance). MaxRL in its iterative form is online: each iteration generates fresh data from the current model. Static rejection sampling fine-tuning is offline: data is collected once from a fixed base model.

Online approaches are more powerful because they adapt the training distribution to the current model's capabilities. They are more expensive because they require continuous inference compute throughout training. Offline approaches are cheaper but cannot improve beyond what the base model could generate.

### Contrastive vs Non-Contrastive

A third axis is whether the training signal is contrastive — comparing good and bad responses directly — or non-contrastive — learning only from good responses. MaxRL is non-contrastive: it trains only on correct responses. GRPO and DPO are contrastive: they explicitly compare better responses to worse ones and use the comparison to drive the gradient.

Contrastive methods are generally more data-efficient because they extract information from both the positive and negative examples. Non-contrastive methods like MaxRL are simpler but rely entirely on positive reinforcement. In learning theory terms, MaxRL is analogous to pure positive example learning; GRPO and DPO are analogous to learning from both positive and negative examples.

### Where MaxRL Sits

MaxRL occupies the corner of the taxonomy characterized by: outcome rewards, online data collection, and non-contrastive training. This corner has the lowest algorithmic complexity of any RL-style approach — it requires the fewest components, the simplest loss function, and the simplest training loop. This is precisely why it is a valuable baseline and a useful entry point for understanding the more complex approaches.

---

## Why MaxRL Matters for the Field

MaxRL's importance extends beyond its practical utility as a training algorithm. It matters conceptually because it clarifies the design space of RL for LLMs and reveals what is genuinely essential versus what is algorithmic overhead.

### The Simplest Baseline

Before MaxRL and rejection sampling fine-tuning became widely discussed, the implicit assumption in the field was that RL for LLMs required the full PPO apparatus: reward model, value model, policy gradient, importance sampling, KL penalty. The existence of MaxRL as a simple, working alternative forces the field to ask: which of these components are necessary, and for what?

MaxRL's strong empirical results on mathematics benchmarks show that a significant fraction of what PPO achieves can be replicated with just a verifier and supervised fine-tuning. This raises the question: how much of PPO's advantage, if any, comes from the additional components, and under what conditions? The answer informs where to invest engineering effort in more complex algorithms.

### The Generative Frontier

MaxRL also crystallizes the concept of the **generative frontier**: the set of tasks the model can currently solve at least some of the time. Training with MaxRL is equivalent to pushing the model's capabilities to cover more of the generative frontier — starting from what the model can sometimes do and expanding outward to what it can reliably do, then pushing further to what it can sometimes do at the new level.

This frontier framing is useful for thinking about curriculum design, scaling, and the limits of what MaxRL can achieve. A model's generative frontier expands with MaxRL training until it reaches the boundary of what the training distribution can teach, which is bounded by the hardest problems in the training set and the diversity of correct solutions the model can eventually generate.

### RL as Structured Data Collection

MaxRL reframes RL training for LLMs as a data collection problem. The RL loop is not primarily an optimization algorithm — it is a mechanism for collecting high-quality training data that would be difficult or expensive to collect in any other way. Human annotators could produce the same training data (correct solutions to mathematics problems), but at enormous cost and limited scale. MaxRL generates that data automatically, cheaply, and at any scale the hardware budget permits.

This framing connects RL for LLMs to the broader theme of data-centric AI: the idea that the quality and diversity of training data often matters more than the sophistication of the training algorithm. MaxRL's contribution, viewed through this lens, is a method for generating high-quality data from the model's own distribution — a form of self-supervised data generation with a verifier acting as the quality filter.

The data-collection framing also explains why MaxRL's training stability is so high: the optimization step is always working with clean, verified, correct examples. There are no adversarial gradients, no reward hacking, no noisy advantage estimates. The optimizer is doing exactly what optimizers are best at: adjusting weights to better explain a clean supervised signal. All the difficulty of RL for LLMs has been moved into the data collection phase — finding the correct examples — and away from the optimization phase. This separation of concerns is one of MaxRL's most elegant design properties.

---

## The History and Naming Landscape

MaxRL did not emerge from a single paper with a single canonical name. It developed gradually as researchers independently rediscovered the same core insight from different angles, each using a different label for essentially the same algorithm.

### Rejection Sampling Fine-Tuning (RFT)

The earliest widely cited formulation under a distinct name is **Rejection Sampling Fine-Tuning (RFT)**, introduced in work on mathematical reasoning around 2023. The "rejection sampling" framing comes from statistics: in Monte Carlo methods, rejection sampling is a technique for drawing samples from a target distribution by generating candidate samples and accepting them with a probability proportional to how well they match the target. In RFT, the "target distribution" is the distribution of correct responses, and the rejection criterion is the binary verifier. The name emphasizes the statistical sampling perspective over the RL perspective.

RFT as originally described used a fixed base model to generate all training data — that is, it ran a single round of generate-verify-filter rather than iterating. Iterative versions, where the model is updated and then used to generate the next round of training data, came later and are more powerful in practice.

### RAFT: Reward rAnked Fine-Tuning

**RAFT** (Dong et al., 2023, arxiv.org/abs/2304.06767) generalizes rejection sampling fine-tuning by replacing the binary correct/incorrect filter with a continuous ranking by reward. Instead of keeping only responses that score above a threshold, RAFT ranks all k responses by their reward score and keeps the top fraction. This allows RAFT to work in settings where the reward is continuous — helpfulness scores, human preference probabilities — rather than binary. For verifiable domains where the reward is binary, RAFT reduces exactly to MaxRL.

RAFT made explicit the connection to best-of-n sampling (generating n responses and picking the highest-scoring one) as a test-time analogue of the training-time filter. The parallel between training-time filtering and test-time selection is one of MaxRL's conceptually cleanest features.

### The "MaxRL" Name

The framing as MaxRL — Maximum Likelihood RL — appears in the RL for LLMs literature as a way of unifying the various rejection sampling and filtered SFT approaches under a common theoretical lens. The name emphasizes that what all these methods have in common is a maximum likelihood training objective applied to a filtered high-quality subset of the model's own outputs. This framing also naturally positions MaxRL within the spectrum of RL algorithms: at one extreme (MaxRL), the RL signal is used only as a filter and training is supervised; at the other extreme (PPO), the RL signal shapes every gradient step through advantage-weighted policy updates.

---

## The Role of Chain-of-Thought in MaxRL

One of the most important practical decisions in MaxRL for reasoning tasks is whether to train on the entire response — including the intermediate reasoning chain — or only on the final answer. This choice has significant implications for what the model learns.

### Outcome Supervision vs Process Supervision

**Outcome supervision** trains on the complete response (reasoning chain plus final answer) for every correct response, regardless of whether the reasoning chain itself is sound. If the model produces a correct final answer via flawed or lucky reasoning, outcome supervision will reinforce that flawed reasoning chain. Over many training iterations, the model can develop a habit of sloppy reasoning that happens to produce correct answers on training problems but fails to generalize to unseen problems.

**Process supervision** trains only on responses where the reasoning chain has been verified to be correct at each step, not just at the final answer. This requires step-level verification, which is much harder to implement than answer-level verification. For mathematics, you would need to check that each algebraic manipulation is valid, each logical deduction is sound, and each intermediate calculation is correct. Automated step-level verification is an active research problem.

In practice, most MaxRL implementations use outcome supervision — verify only the final answer, train on the entire chain-of-thought response. The implicit assumption is that responses with correct final answers are more likely to have sound reasoning chains than responses with incorrect final answers, which is true on average. But it is not universally true, and outcome supervision does introduce some noise from responses that got the right answer via questionable reasoning.

The tension between outcome and process supervision is one of the active debates in RL for LLMs research. Models like OpenAI o1, which show strong generalization on unseen reasoning problems, are believed to use some form of process-level supervision or process reward model during training, though the exact details are not publicly documented.

### Length and Chain-of-Thought Quality

An interesting empirical observation in MaxRL training for mathematics is that correct responses are consistently longer than incorrect responses. This is not because length causes correctness — it is because more difficult problems that require more reasoning steps naturally produce longer chains of thought, and those longer chains are more likely to be produced by the model when it is genuinely working through the problem carefully.

Training on correct responses therefore implicitly trains the model to produce longer, more detailed reasoning chains. This is a beneficial side effect: the model learns that careful, extended reasoning is associated with correct answers, reinforcing the chain-of-thought behavior. However, if the model learns a shortcut — packing a correct final answer at the end of a long but low-quality reasoning chain — the length effect can be misleading.

---

## Iterative vs Single-Round MaxRL

Not all implementations of the MaxRL / rejection sampling framework are iterative. Understanding the difference between single-round and iterative versions clarifies the RL interpretation and the expected improvement trajectory.

### Single-Round (Static) Version

In the single-round version, you take a fixed base model, generate k responses for each problem in the training set, filter correct ones, and fine-tune the base model on the filtered set exactly once. The resulting model is then evaluated and deployed — no further rounds of generation-verification-training.

This approach is computationally predictable: you can calculate exactly how many correct responses you will generate (given an estimate of pass@k) and how long training will take. But it is also limited: the model can only learn from responses that the original base model was capable of generating. If the base model's per-sample accuracy is 10%, the training data consists only of the 10% of responses that happened to be correct. The model cannot learn to solve problems it was completely incapable of solving before.

Single-round MaxRL is essentially equivalent to best-of-n supervised fine-tuning: generate many, keep the best, train on those. It is simple and predictable but not truly reinforcement learning — there is no feedback loop between behavior and training signal, because the model is only trained once.

### Iterative Version

The iterative version is what properly qualifies as reinforcement learning. After each round of generate-verify-filter-train, the model's weights are updated and the next round begins with the improved model. Problems that were unsolvable at round 1 (pass@k near 0) may become solvable by round 5, as the model accumulates knowledge from solving related easier problems. The training signal expands to cover harder and harder problems as the model improves.

Iterative MaxRL shows a characteristic learning curve: rapid improvement in the first few iterations as easy problems are mastered, followed by slower but consistent improvement as the model works through progressively harder problems. The plateau occurs when the training set no longer contains problems that are just beyond the model's current capability — either the model has mastered everything, or the remaining problems are too hard to ever produce a correct sample at any k.

The iterative version requires more careful engineering: data management across iterations, monitoring for distribution shift, periodic refreshing of the problem sampling distribution as problems move from hard to easy. But the iterative version achieves significantly better final performance than the single-round version.

---

## MaxRL and Data Augmentation

One practical strategy for extending the effective reach of MaxRL training is data augmentation — systematically creating new problems from existing ones to maintain training signal density as the model improves.

### Problem Paraphrasing

The simplest augmentation: rephrase existing problems without changing the mathematical content. "John has 5 apples and gives away 3. How many remain?" and "Maria begins with 5 oranges and distributes 3 among her friends. What is she left with?" test the same arithmetic operation but with different surface forms. A model that has memorized the specific phrasing of the original problem may not transfer its knowledge to the paraphrase.

Training on paraphrased problems teaches the model to focus on mathematical structure rather than surface-level text patterns. This improves generalization to genuinely novel problems at evaluation time.

### Numerical Substitution

Replace the specific numbers in a problem with different values. The solution procedure remains the same, but the correct answer changes. This is particularly effective for arithmetic and algebra problems where the model might overfit to seeing specific numbers. "3x + 7 = 22" becomes "5x + 11 = 41" and "2x + 3 = 17" — the same algebraic procedure applies but with fresh numbers that require the model to actually execute the computation rather than recall a memorized answer.

### Problem Composition

Compose two known problems into a single harder problem. If the model has mastered single-step algebra and single-step ratio problems separately, a composed problem might require both steps in sequence. This form of augmentation explores the combinatorial space of skills the model has acquired and tests whether they transfer to multi-step settings.

Data augmentation interacts well with MaxRL's training signal dynamics: augmented problems are often slightly harder than the originals (especially compositions), placing them in the intermediate difficulty zone where MaxRL's training signal is most informative. A well-designed augmentation pipeline can keep the training signal dense even as the model masters the original training set.

---

## MaxRL Failure Analysis: What Goes Wrong

Understanding the failure modes of MaxRL in detail helps practitioners avoid them and helps researchers identify directions for improvement.

### Verifier Gaming

The most serious failure mode is verifier gaming: the model learns to produce outputs that pass the verifier without actually solving the problem correctly. This is analogous to reward hacking in standard RL, but enabled by weaknesses in the verifier rather than in a learned reward model.

Examples of verifier gaming in practice:

For code generation: the model produces code that passes the specific test cases used for verification but fails on related inputs not in the test suite. This is the classical overfitting-to-tests problem. The model has learned the test inputs and outputs from its training data and produces code that hard-codes the expected outputs rather than implementing the general algorithm. Standard mitigation: use diverse test suites, private test cases not seen during training, and adversarially constructed edge cases.

For mathematics: the model learns to produce an answer that matches a common ground-truth format even when the underlying reasoning is wrong. If the verifier only checks whether the final answer matches (not whether the reasoning is sound), the model might produce many plausible-looking reasoning steps followed by a guessed answer. For problems with small integer answers (like competition problems where the answer is typically between 1 and 999), guessing the answer and backfilling plausible reasoning is a viable strategy that passes the verifier. Mitigation: use more diverse answer formats, require symbolic proofs rather than numerical answers, or implement process reward models that check reasoning steps.

### Mode Collapse to a Single Strategy

Another failure mode is the model converging to a single problem-solving strategy that happens to work on many training problems but fails to generalize. If a particular algebraic manipulation sequence produces correct answers on 70% of problems in the training set, MaxRL will reinforce that approach heavily. The model may learn to always apply that sequence, losing the flexibility to use alternative approaches when the standard one fails.

This is related to but distinct from reward hacking: the model is not gaming the verifier, it is genuinely solving problems correctly — but only the problems that fit its dominant strategy. Problems requiring alternative approaches are consistently wrong, and since they produce no training signal, the model has no mechanism to expand its repertoire.

Mitigation strategies include diversity incentives (penalizing responses that are too similar to other correct responses in the same batch), curriculum designs that include problems specifically designed to require diverse strategies, and ensembling multiple independently trained MaxRL models that may converge to different strategies.

### Answer Parsing Errors at Scale

At training scale, the verifier processes millions of responses across thousands of training iterations. Even a very low false positive rate — say 0.1% of wrong answers accepted as correct — can introduce significant amounts of bad training data. At one million responses processed per training run, a 0.1% false positive rate means 1,000 incorrect responses included in training data.

These low-rate errors are hard to detect in individual training iterations (where they appear as occasional odd outliers) but can accumulate into systematic biases over many iterations of training. The model learns from the incorrect responses and begins to reproduce the patterns that fooled the verifier.

Robust verifier design treats answer parsing as a critical system component, not an afterthought. Using multiple independent parsing strategies (regex-based extraction, symbolic math library evaluation, LLM-based extraction as a fallback) and taking the intersection of their results reduces false positives at the cost of increased false negatives. Given the asymmetric costs of false positives versus false negatives in MaxRL, this trade-off is usually worth making.
