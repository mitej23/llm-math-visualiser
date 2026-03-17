# 🎓 Supervised Fine-Tuning (SFT) — From Base Model to Assistant

> **Sources used:**
> - Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), OpenAI 2022 — [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
> - Lambert et al., *Illustrating Reinforcement Learning from Human Feedback (RLHF)*, Hugging Face Blog 2022 — [huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)
> - Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021 — [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
> - Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023 — [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)

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

## 📝 The Instruction Format — Templates and ChatML

Every modern instruction-tuned model uses a specific **chat template** — a structured format that wraps every conversation. This isn't cosmetic: the model is trained on millions of examples in this exact format, so deviating from it at inference time will hurt performance.

**Why templates exist:**
1. The model needs clear markers for "when does the user speak" vs. "when do I speak"
2. In multi-turn conversations, the model must know which history belongs to which speaker
3. System prompts (model persona, instructions) must be separable from user input
4. The end-of-response token tells the model when to stop generating

**ChatML format (used by OpenAI, Mistral, many others):**
```
<|im_start|>system
You are a helpful assistant named Aria. Be concise and friendly.
<|im_end|>
<|im_start|>user
What's the tallest mountain?
<|im_end|>
<|im_start|>assistant
Mount Everest, at 8,849 metres (29,032 feet) above sea level.
<|im_end|>
```

**LLaMA 2 Chat format:**
```
[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What's the tallest mountain? [/INST]
Mount Everest, at 8,849 metres. </s>
```

**Alpaca format (simpler, used in many research SFT datasets):**
```
### Instruction:
What is the tallest mountain?

### Response:
Mount Everest, at 8,849 metres.
```

**Why does the specific format matter?** These tokens (`<|im_start|>`, `[INST]`, etc.) are typically assigned specific token IDs in the tokenizer and become deeply associated with "conversation behavior" during training. Using the wrong format at inference is like trying to talk to someone in a dialect they don't understand — they might still respond, but quality drops.

**Multi-turn example (ChatML):**
```
<|im_start|>user
What is DNA?
<|im_end|>
<|im_start|>assistant
DNA is a molecule that carries genetic information...
<|im_end|>
<|im_start|>user
How does it replicate?
<|im_end|>
<|im_start|>assistant
DNA replication uses an enzyme called DNA polymerase...
<|im_end|>
```
The model sees the full history and generates the next assistant turn.

---

## 🔬 What Actually Changes During SFT?

This is a subtle but important question. The base model is already enormously capable — it has read essentially all of human knowledge. So what does SFT actually modify?

**The surprising answer: not much.** Studies show that SFT changes weights by a very small percentage of their magnitude. The knowledge hasn't moved — only the *activation patterns* for "assistant-style" generation have been reinforced.

**What SFT actually teaches:**
1. **Format:** "Respond as a helpful assistant, not as a document continuation"
2. **Tone:** "Be direct, polite, appropriately detailed"
3. **Turn-taking:** "Know when to stop generating"
4. **Intent recognition:** "A question asking 'what is X' wants an explanation, not more questions"
5. **Safety behaviors:** "Don't explain how to make weapons" (if trained on safe demonstrations)

**What SFT does NOT teach:**
- New factual knowledge (that came from pretraining)
- Genuinely new reasoning capabilities
- Perfect factual accuracy (hallucinations remain)

**The elegant phrasing:** The pretraining learned a *distribution over all possible text*. SFT samples from the narrow slice of that distribution that corresponds to "helpful assistant responses."

**Analogy:** 🎭 Imagine a brilliant actor who knows thousands of characters. Pretraining taught them all those characters. SFT said: "For the next show, you're playing a helpful librarian. Let me show you 50,000 examples of how a librarian would respond." The actor hasn't gained any new knowledge — they've learned which character to inhabit.

**Weight change magnitude:** Research using techniques like LoRA (which measures the effective rank of weight changes) shows that SFT changes are surprisingly low-rank — meaning the conceptual change is actually quite simple, even though it touches all the weights. This is exactly why LoRA works so well for fine-tuning.

---

## ⚖️ Data Quality vs Data Quantity

This is one of the most practically important questions in applied LLM development: *should I collect 10,000 high-quality examples or 100,000 mediocre ones?*

**The research answer: quality usually wins, but only above a threshold.**

**Evidence 1: LIMA (Less Is More for Alignment)**
A 2023 paper by Zhou et al. showed that fine-tuning LLaMA on just **1,000 carefully curated examples** produced an assistant that matched or beat models fine-tuned on much larger datasets. They called this "surface alignment" — because aligning behavior requires surprisingly little data once the base model is strong.

**Evidence 2: InstructGPT**
OpenAI's InstructGPT used ~13,000 high-quality demonstrations. The base model being fine-tuned was 175B — enormous pretraining base. Result: outperformed raw GPT-3 on nearly all tasks.

**Evidence 3: Noisy vs curated datasets**
A widely replicated finding: 100k examples with 20% errors/noise often perform worse than 20k carefully curated examples. The model learns the *wrong patterns* from noisy data.

**What makes data "high quality"?**
1. **Diversity:** Covers many different task types, domains, styles
2. **Correctness:** Factually accurate, well-reasoned responses
3. **Consistency:** Consistent tone and format across examples
4. **Appropriate length:** Responses are the right length — not padded, not truncated
5. **No leakage:** Training examples don't contain answers to your test/eval set

**The long tail problem:** Even with a large dataset, rare task types will be underrepresented. A model trained on mostly Q&A data will be worse at structured output tasks (JSON formatting, tables) because those have fewer examples. Explicitly balancing your dataset across task types matters more than total size.

**Practical rule of thumb:**
- Starting from a very strong base model (>7B): 1k-10k high-quality examples often sufficient
- Starting from a weaker base model: need more data to "teach" more
- Highly specialized tasks (medical, legal, code): 10k-50k domain-specific examples
- General assistant behavior: 50k-500k diverse examples for best results

---

## 💡 LoRA — Parameter-Efficient Fine-Tuning

**The problem with full fine-tuning:** A 70B parameter model has 70 billion weights. Fine-tuning all of them requires storing gradients and optimizer states for all 70B parameters — roughly 4-8× the model size in memory. That's 200-400GB+ just for fine-tuning. Prohibitively expensive.

**LoRA's insight** (Hu et al., 2021): The weight changes during fine-tuning are **low-rank**. Instead of changing the full weight matrix W (size 4096×4096 = 16M parameters), approximate the change as a product of two tiny matrices:

```
W_new = W_original + A × B
```
Where A is [4096 × r] and B is [r × 4096], and r ("rank") is typically 4-64.

With r=16: instead of 16M parameters, you're training only 2 × (4096 × 16) = 131,072 parameters. That's **120× fewer parameters** to update!

**How it works in practice:**
1. Freeze the original model weights (W_original stays fixed, no gradients)
2. Add small trainable matrices A and B at each targeted layer
3. During forward pass: output = W_original × input + (A × B) × input
4. Train only A and B — the frozen weights need no gradient storage
5. At the end, you can merge: W_final = W_original + A×B (zero inference overhead)

**Which layers to apply LoRA to?** Typically the attention layers (Q, K, V projections) and sometimes the FFN layers. The attention layers seem to hold most of the "behavioral" information that SFT changes.

**LoRA hyperparameters:**
- **r (rank):** 4-64 typically. Higher = more expressive but more parameters to train. r=16 is a common default.
- **alpha (α):** Scaling factor. Often set to r (so α/r = 1 scaling) or 2r for slightly stronger updates.
- **Target modules:** Which weight matrices to apply LoRA to. Common: "q_proj, v_proj" or "all linear layers."

**Real efficiency numbers (LLaMA-7B fine-tuning):**
| Method | Trainable params | GPU memory | Time per epoch |
|---|---|---|---|
| Full fine-tuning | 7B (100%) | ~120GB | Baseline |
| LoRA (r=16) | 4.2M (0.06%) | ~20GB | ~4× faster |
| LoRA (r=64) | 16.8M (0.24%) | ~22GB | ~3.5× faster |

**Merging LoRA weights:** After training, A×B can be added directly into W_original. The merged model is identical in size and has zero inference overhead — you get a regular dense model that behaves like the fine-tuned version.

**Analogy:** 🎸 Instead of replacing the entire guitar's internals (full fine-tuning), LoRA adds a small effects pedal (the A×B adapter) that modifies the signal in a targeted, low-dimensional way. The guitar stays the same; only the pedal changes.

---

## ⚡ QLoRA — Quantized LoRA

**The problem LoRA still has:** Even with LoRA, you need to load the full model weights into GPU memory to do the forward pass. A 70B model in float16 is still ~140GB — most people don't have that.

**QLoRA's solution** (Dettmers et al., 2023): Quantize the base model to 4-bit precision (down from 16-bit), then apply LoRA on top.

**What is quantization?** Instead of storing each weight as a 16-bit float (values like 0.4731928...), store it as a 4-bit integer (one of 16 discrete values). This compresses the model by 4×.

**The challenge:** Naively quantizing to 4-bit destroys too much precision. QLoRA uses "NormalFloat4" — a clever 4-bit quantization scheme designed for normally-distributed weights (which neural network weights are) that minimizes information loss.

**QLoRA memory reduction (LLaMA-65B):**
| Method | GPU Memory Required |
|---|---|
| Full 16-bit fine-tuning | ~780GB |
| Full fine-tuning with 8-bit | ~400GB |
| LoRA (16-bit) | ~200GB |
| QLoRA (4-bit base + LoRA) | ~48GB |

That last number is achievable on a single consumer GPU (an RTX 3090 or 4090). This democratized LLM fine-tuning enormously.

**Quality tradeoff:** QLoRA typically degrades quality by <1% compared to full 16-bit LoRA fine-tuning. For most applications, this is negligible. The 4-bit base weights introduce small rounding errors, but the LoRA adapters compensate.

**Analogy:** 📦 QLoRA is like compressing a map from high-resolution to low-resolution (quantization), then adding sticky notes with corrections (LoRA). The map is much smaller, the sticky notes add back precision where needed, and the whole thing fits in your pocket.

---

## 🎯 Multi-Task SFT

Fine-tuning on a diverse mixture of tasks — rather than a single task type — produces a more robust and generalizable assistant.

**Why multi-task?**
1. **Transfer learning between tasks:** Code understanding helps math reasoning; following formatting instructions in one domain transfers to others.
2. **Avoiding forgetting:** Training on a single task causes "catastrophic forgetting" of other capabilities. Multi-task training maintains breadth.
3. **User needs are diverse:** Real users ask about code, science, creative writing, factual questions — all in the same session.

**Typical task mix for general assistant SFT:**
- Instruction following (35%): "Summarize this", "Translate this", "Rewrite more formally"
- Open Q&A (25%): Factual questions, explanations
- Coding (15%): Code generation, debugging, explanation
- Reasoning (10%): Math problems, logic puzzles
- Creative (10%): Stories, poems, brainstorming
- Conversation (5%): Small talk, multi-turn chat

**Mixing strategies:**
- **Fixed ratio:** Always sample in exact proportions above
- **Temperature sampling:** Up-sample rarer task types to prevent majority tasks dominating
- **Curriculum:** Start with easier tasks, gradually mix in harder ones

**Real example — Flan (Google):**
Google's FLAN models are trained on thousands of diverse NLP tasks (translation, classification, summarization, QA...) simultaneously. The result is a model that generalizes to new tasks it hasn't seen before — because it has learned the *meta-skill* of following instructions.

**Data mixing pitfall:** If you include too much of one task type, the model overfits to that format. A model trained on 80% coding will be great at code but will answer general questions in a code-like, terse style even when inappropriate.

---

## ✅ When SFT Is Enough (And When It Isn't)

SFT is a powerful first step, but it has known limitations. Understanding when to go further is practically important.

**When SFT alone works well:**
- **Translation:** Given input text, produce output in another language. Well-defined, verifiable.
- **Summarization:** Given a document, produce a shorter version. Reasonably demonstrable.
- **Classification and extraction:** Given text, categorize it or extract specific information.
- **Question answering:** Given a passage, answer questions about it.
- **Code generation (basic):** Generate code that solves a clearly specified problem.
- **Format transformation:** Convert data formats, restructure text.

**When SFT is insufficient:**
- **Preference alignment:** Users have preferences beyond correctness — tone, style, helpfulness. SFT can teach a format but can't learn subjective preferences from demonstrations alone.
- **Safety:** Getting a model to reliably refuse harmful requests requires more than demonstrations — the model needs to understand *why* something is harmful, not just see examples of refusing.
- **Complex reasoning:** For multi-step math or coding, SFT on demonstrations captures surface patterns but doesn't reliably develop genuine step-by-step reasoning. This is why reinforcement learning (especially RLHF and variants like GRPO) is so valuable for reasoning tasks.
- **Novel situations:** If a user asks something outside the distribution of the SFT dataset, the model can fail silently — producing a plausible-sounding wrong answer because it's imitating the format of a correct answer.

**The ceiling of imitation learning:** SFT can never produce responses better than its training demonstrations. If your annotators are good (but not expert-level), your model will be good (but not expert-level). To go beyond human-level demonstrations, you need reward-based training.

**The typical pipeline:**
```
Base model → SFT → Reward Model → RLHF (PPO or DPO) → Aligned assistant
```

SFT creates a well-behaved starting point. The subsequent RLHF stages push it toward preferences and safety behaviors that are hard to demonstrate directly.

---

## 📂 Open-Source SFT Datasets

Many high-quality SFT datasets are publicly available. Here are the most widely used:

**Alpaca (Stanford, 2023)**
- 52k instruction-response pairs
- Generated by GPT-3.5 using a self-instruct approach
- Covers general instructions
- Limitation: some noise and hallucinations from GPT-3.5

**Dolly (Databricks, 2023)**
- 15k examples
- Written by Databricks employees — human-generated, not model-generated
- High quality, diverse tasks
- License: CC BY-SA 3.0 — commercially usable

**OpenAssistant (LAION, 2023)**
- 160k messages in multi-turn conversations
- Crowdsourced human conversations and rankings
- Available in multiple languages

**FLAN (Google)**
- Thousands of NLP tasks reformatted as instruction-following
- Very diverse, strong for generalizable instruction following
- Research-focused

**ShareGPT**
- Real user-ChatGPT conversations scraped from ShareGPT.com
- Very large (100k+ conversations)
- Contains complex multi-turn dialogue
- Quality varies: includes both excellent and poor conversations

**Orca / Open-Orca**
- Responses generated by GPT-4 (much higher quality than Alpaca)
- 1M+ examples with GPT-4 explanations of reasoning
- Significantly better than Alpaca for training capable models

**WizardLM**
- Alpaca instructions rewritten to be more complex using GPT-4
- "Evol-Instruct" approach: take simple instructions, make them harder step by step
- Better coverage of complex, multi-step instructions

**Practical note:** For production models, companies typically mix multiple datasets, add proprietary annotated data, and filter heavily for quality. The open-source datasets above are excellent starting points but often need cleanup.

---

## 🚫 Common Misconceptions

**"SFT makes the model know more things"**
Wrong. SFT does not add new factual knowledge. The model's knowledge was fixed at the end of pretraining. SFT only changes *how the model expresses and applies* that knowledge.

**"More SFT data is always better"**
Not necessarily. As shown by the LIMA paper, 1k high-quality examples can match 100k noisy ones. Beyond a certain point, adding more low-quality data actively hurts performance by introducing inconsistent patterns.

**"SFT removes hallucinations"**
SFT reduces hallucinations by teaching the model to say "I don't know" in some contexts — because some demonstrations show that behavior. But it doesn't eliminate hallucinations, which are a fundamental property of the pretraining objective (predict next token confidently, even when uncertain).

**"LoRA always matches full fine-tuning quality"**
Mostly, but not always. For very difficult tasks or large distribution shifts, full fine-tuning can outperform LoRA. The gap is usually small, but it exists. For most practical purposes (assistants, chatbots, domain adaptation), LoRA is sufficient.

**"Fine-tuned models can learn from their own outputs"**
Not during supervised fine-tuning — the model receives ground-truth responses from the dataset, not its own outputs. A model can't improve itself through SFT. Self-improvement requires reinforcement learning approaches where the model generates responses and receives feedback.

**"SFT teaches the model to 'think' like an assistant"**
SFT teaches the model to *output text that looks like* an assistant's output. There's no evidence that SFT instills genuine understanding or reasoning — it's pattern matching at a very high level. The "thinking" capability came from pretraining. SFT just selects the right behavior mode.

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
| LoRA | Train tiny low-rank adapters instead of all 7B+ weights — 100× fewer parameters |
| QLoRA | Quantize base model to 4-bit, add LoRA on top — fits on a single consumer GPU |
| Data quality > quantity | 1k excellent examples often beats 100k noisy ones |
| Multi-task SFT | Mix diverse task types to build a robust, general assistant |
| When SFT fails | Preference alignment, safety, novel situations → need RLHF |

---

## Up Next
👉 **Reward Models** — since SFT is limited by demo quality, we need a way to score responses automatically so we can go beyond imitation.
