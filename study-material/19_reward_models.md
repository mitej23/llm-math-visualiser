# 🏆 Reward Models — Teaching a Machine to Judge

> **Sources used:**
> - Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), OpenAI 2022 — [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
> - Lambert et al., *Illustrating Reinforcement Learning from Human Feedback (RLHF)*, Hugging Face Blog 2022 — [huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)
> - Huang et al., *The N+ Implementation Details of RLHF with PPO*, Hugging Face Blog 2023 — [huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)

---

## The Big Idea

SFT teaches the model to imitate good responses. But imitation has a ceiling — the model can only be as good as the examples it was shown. To go further, we need something that can **evaluate** any response and say how good it is. A **reward model** is a neural network trained to do exactly that: take a (prompt, response) pair and output a single number representing quality.

---

## Real-Life Analogy: The Restaurant Critic 🍽️

Imagine you want to train a chef to cook great food, but you can't hire a Michelin-star critic to taste every single dish the trainee makes (there are millions of dishes to taste).

Instead, you:
1. Have the critic taste a **sample set** of dishes and write detailed rankings: "Dish A was better than Dish B, which was better than Dish C"
2. Train an **automated food scoring system** on these rankings — it learns to predict what the critic would score
3. From now on, the automated system scores every dish the trainee makes — no critic needed

The automated scoring system = the reward model. It's a proxy for human judgement that can run at scale.

---

## Why Rankings, Not Scores? 🔢

A key insight from the RLHF paper ([huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)):

> *"Rankings are more reliable than scalar scores due to human value disagreement."*

If you ask 5 people to score a response on a scale of 1-10, you might get: 6, 7, 4, 8, 5. There's noise.

But if you ask those same 5 people "which of these two responses is better?", they agree much more often.

**Analogy:** It's easier to agree that Roger Federer is a better tennis player than a random college student than to agree on exactly how many points each deserves. Rankings capture relative quality more reliably than absolute scores.

So reward model training data is collected as **comparisons**, not ratings:

```
Prompt: "Explain black holes simply."

Response A: "A black hole is a region of spacetime where gravity is
             so strong that nothing can escape..."

Response B: "Black holes are like cosmic vacuum cleaners that eat
             everything including light..."

Human annotation: "A > B"
```

---

## Architecture — A Language Model with a Scoring Head 🔧

A reward model is a language model with one modification: instead of predicting the next token, **the final layer outputs a single scalar number** (the reward score).

As described in the HuggingFace RLHF PPO implementation details ([huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)):

> *"Appends scalar reward head to base language model. Only reward at final token position is used."*

So the architecture is:
```
[prompt + response tokens]
    ↓
Transformer layers (same as base LLM)
    ↓
Final token's hidden state
    ↓
Linear layer → scalar reward score
```

The score at the **last token** represents the model's overall judgement of the entire response.

**Why the last token?** The last token has attended to every other token through self-attention — it carries a summarised representation of the whole sequence.

---

## Training the Reward Model 🏋️

Given pairs of responses with human labels (A is better than B):

1. Run both responses through the reward model → get score_A and score_B
2. The model should give score_A > score_B
3. Compute loss based on how often the ordering is violated
4. Update the reward model weights to push score_A higher and score_B lower
5. Repeat over thousands of comparison pairs

After training, the reward model has internalised what makes a response "good" according to human preferences — without needing a human to judge every new response.

---

## Reward Normalisation — Keeping Scores Consistent 📊

Raw reward scores can drift during training. The implementation details paper ([huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)) describes normalising rewards to have mean=0 and standard deviation=1:

> *"Applied before and after training to maintain mean=0, std=1."*

**Analogy:** Test scores at different schools can't be compared directly — some schools grade on a 100-point scale, others on a 10-point scale. Normalising converts everything to the same "curve," making comparisons meaningful and training more stable.

---

## What Reward Models Capture 📐

A well-trained reward model learns to score responses higher when they are:

- **Helpful:** Actually answers the question
- **Harmless:** Doesn't produce dangerous or offensive content
- **Honest:** Doesn't make things up (hallucinate)
- **Well-formatted:** Readable, appropriately detailed
- **Safe:** Refuses harmful requests appropriately

These dimensions come from **what human annotators valued** when making their comparisons. The reward model is a compressed representation of those preferences.

---

## The Limitation: Reward Hacking 🎯❌

A critical problem: the reward model is an **approximation** of human preferences, not the real thing. The LLM can learn to "fool" it.

**Analogy:** You're a student who wants a good grade. Instead of actually learning the material, you learn exactly what the grader likes (long answers? bullet points? specific buzzwords?) and produce responses that score high without being genuinely good.

This is called **reward hacking** — the policy optimises for the proxy metric (reward model score) rather than the true objective (being genuinely helpful).

**Solution:** The KL penalty (next topic) — keep the model from straying too far from its original behaviour. If it diverges too much, it's probably reward-hacking rather than genuinely improving.

---

## Process Reward Models (PRMs) — Grading Every Step 🔬

Standard reward models score the **final response** (outcome-based). A newer approach: **Process Reward Models** score each reasoning step.

**Analogy:** A maths teacher who marks each step of your working, not just the final answer. Even if you got the right answer by accident, they can tell from your working whether you actually understood it.

PRMs are important for complex reasoning tasks (maths, code, multi-step problems) where the path to the answer matters, not just the answer itself. This is an active area of research (mentioned in the RL for LLMs article you shared).

---

## Reward Models in the Wild

| Model Family | Reward Model Used |
|---|---|
| InstructGPT (GPT-3.5) | 6B parameter reward model, trained on 33k comparisons |
| Claude | Constitutional AI reward model + human feedback |
| LLaMA 2 Chat | Multiple reward models (helpfulness + safety) |
| DeepSeek-R1 | Rule-based verifiable rewards (maths/code correct/wrong) |

Note: DeepSeek-R1 avoided a traditional reward model entirely for maths/code — the answer is either correct or not, so the reward signal is free. This sidesteps reward hacking for these domains.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| Reward model | A neural network that scores (prompt, response) quality as a single number |
| Trained on comparisons | "Response A is better than B" — more reliable than asking for scores |
| Scalar output | Final token's hidden state → one number per response |
| Reward normalisation | Keep scores at mean=0, std=1 for training stability |
| Reward hacking | Model learns to fool the reward model instead of being genuinely good |
| KL penalty | The solution to reward hacking — don't stray too far from original behaviour |
| PRM | Process Reward Model — scores each reasoning step, not just the final answer |

---

## Up Next
👉 **KL Divergence** — the mathematical tool that keeps the model from going off the rails during RL training.
