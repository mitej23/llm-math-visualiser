# 🎓 Supervised Fine-Tuning (SFT) — From Base Model to Assistant

> **Sources used:**
> - Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), OpenAI 2022 — [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
> - Lambert et al., *Illustrating Reinforcement Learning from Human Feedback (RLHF)*, Hugging Face Blog 2022 — [huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)

---

## The Big Idea

A pretrained LLM has read most of the internet — it knows an enormous amount. But it was trained to **predict the next token**, not to **answer your questions helpfully**. It might respond to "Write me a poem" with more questions, or just continue the text as if it were a web page.

**Supervised Fine-Tuning (SFT)** is the first alignment step: show the model thousands of examples of ideal conversations, and train it to imitate that behaviour.

---

## Real-Life Analogy: The Brilliant but Unpolished Graduate 🎓

Imagine a PhD graduate who has read every textbook, paper, and article ever written. They are incredibly knowledgeable. But when you ask them a question, they might:
- Launch into an unstructured monologue
- Reply with a follow-up question instead of an answer
- Give you the Wikipedia page for the topic rather than a direct answer

**SFT is like professional training** — teaching this brilliant graduate how to:
- Answer questions directly and concisely
- Follow a conversation format
- Behave helpfully and politely
- Know when to say "I don't know"

The knowledge is already there. SFT teaches them *how to use it* in the right way.

---

## Base Model vs. Instruction-Tuned Model

| Aspect | Base Model | SFT Model |
|---|---|---|
| Training signal | Next-token prediction on raw internet text | Demonstrations of desired Q&A behaviour |
| When given a prompt | Continues the text (might ignore the question) | Responds as a helpful assistant |
| Example | "What is the capital of France? The capital..." (continues like an article) | "The capital of France is Paris." |
| Knowledge | Vast, general | Same as base — just better at surfacing it |

The base model isn't "dumb" — it just hasn't learned the *format* of helpful interaction. SFT installs that format.

---

## The SFT Dataset — Teaching by Example 📚

As described in the InstructGPT paper ([arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)):

> *"We collected a dataset of labeler demonstrations of the desired model behavior... then used supervised learning to fine-tune GPT-3 based on these examples."*

The dataset contains thousands of (prompt, ideal response) pairs, written or curated by human annotators:

```
Prompt:  "Explain quantum entanglement to a 10-year-old."
Response: "Imagine you have two magic coins. Whenever you flip one
           and it lands heads, the other one — no matter where it is
           in the universe — instantly lands tails. That's quantum
           entanglement: two particles linked so that what happens to
           one instantly affects the other."
```

The model is trained to produce *exactly this style of response* for *this type of prompt*.

---

## What Kinds of Prompts Are Used?

The InstructGPT paper identified several categories of user intent:

- **Brainstorming:** "Give me 5 ideas for a startup"
- **Classification:** "Is this email spam or not spam?"
- **Closed Q&A:** "What year did World War II end?"
- **Generation:** "Write a birthday message for my grandmother"
- **Rewriting/editing:** "Make this email more formal"
- **Summarisation:** "Summarise this article in 3 bullet points"
- **Open Q&A:** "Why is the sky blue?"
- **Extraction:** "Pull out all the names mentioned in this text"

Covering this diversity ensures the fine-tuned model handles a wide range of real user needs.

---

## How SFT Training Works ⚙️

SFT uses the same training loop as pretraining (cross-entropy loss, backpropagation, gradient descent) — but on a much smaller, curated dataset:

1. Take the pretrained base model (with all its knowledge)
2. For each (prompt, ideal response) pair in the SFT dataset:
   - Feed the full prompt + response to the model
   - Compute cross-entropy loss **only on the response tokens** (not the prompt)
   - Update the weights
3. The model learns: "when I see this type of prompt, generate this type of response"

**Key detail:** Loss is computed only on the response, not the prompt. The model isn't graded on its ability to repeat the question — only on generating the correct answer.

---

## SFT Is "Just" Imitation Learning

SFT is sometimes called **behavioural cloning** or **imitation learning**. The model is literally imitating the human-written examples.

**Analogy:** Learning to cook by watching a chef. You observe exactly what they do (the demonstration), then you try to replicate it step by step. You don't reason about *why* each step is good — you just learn to copy the pattern.

**The strength:** Simple, stable, fast to train.

**The weakness:** The model can only be as good as the demonstrations. It can't exceed the quality of what it was shown. And if a user asks something not covered well in the dataset, the model might produce plausible-sounding but wrong answers (it imitates the *style* of a good response without necessarily having the *content* right).

This limitation is exactly what motivates the next steps: reward models and RLHF.

---

## SFT vs. Pretraining — Scale Difference

| | Pretraining | SFT |
|---|---|---|
| Dataset size | Trillions of tokens (Common Crawl, books, code...) | Thousands to hundreds of thousands of curated examples |
| Training time | Weeks to months on thousands of GPUs | Hours to days on smaller clusters |
| Goal | Learn world knowledge | Learn conversational behaviour |
| Loss function | Cross-entropy on all tokens | Cross-entropy on response tokens only |

The InstructGPT paper noted: their 1.3B instruction-tuned model outperformed the 175B base model on most tasks. **Alignment matters more than size** for practical usefulness.

---

## Chat Templates — Teaching Conversation Format 💬

Modern SFT datasets use structured **chat templates** so the model learns turn-taking:

```
<|system|>
You are a helpful, harmless, and honest assistant.

<|user|>
What is photosynthesis?

<|assistant|>
Photosynthesis is the process by which plants convert sunlight,
water, and carbon dioxide into glucose and oxygen...
```

The special tokens (`<|user|>`, `<|assistant|>`) teach the model:
- Who is speaking at each turn
- Where its response should begin and end
- How to handle multi-turn conversations

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| SFT | Fine-tune on human-written (prompt, ideal response) pairs |
| Base model | Knows a lot but doesn't know how to behave helpfully |
| SFT dataset | Curated examples of ideal assistant behaviour |
| Imitation learning | Model learns by copying human demonstrations |
| Loss on responses only | Only grade the model on what it generates, not the question |
| Chat template | Structured format teaching the model conversation turns |
| SFT limitation | Can't exceed the quality of its demonstrations |

---

## Up Next
👉 **Reward Models** — since SFT is limited by demo quality, we need a way to score responses automatically so we can go beyond imitation.
