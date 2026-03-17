# 🔤 Tokenization — How Text Becomes Numbers

## The Big Idea

Computers can't read words. They only understand numbers. **Tokenization** is the process of converting text into a sequence of numbers that a neural network can process.

---

## Real-Life Analogy: The Library Card Catalog 📚

Imagine a vast library where every book is identified by a unique number:
- "The Great Gatsby" → Book #4521
- "To Kill a Mockingbird" → Book #8832
- "Pride and Prejudice" → Book #1173

Tokenization is like the library's card catalog — it assigns a unique ID to every piece of language. When the LLM sees ID #4521, it knows that means something about "The Great Gatsby."

But instead of whole books, we assign IDs to **pieces of words (tokens)** rather than whole words.

---

## What Is a Token?

A **token** is a chunk of text. It's not always a full word — it can be:
- A whole word: `hello` → token #15339
- Part of a word: `un` + `believable` → two tokens
- A word + space: ` the` → one token (note the space!)
- Punctuation: `,` → its own token
- A number: `42` → one token

**Analogy:** Think of tokens like Lego bricks. Words are built from common bricks. Common words get their own single brick (whole word = one token). Rare words get broken into smaller standard bricks.

---

## Why Not Just Use Letters?

You could tokenize letter by letter: `h-e-l-l-o` = 5 tokens. But then:
- "Hello" becomes 5 numbers to process instead of 1
- Context windows fill up fast with more tokens
- The model has to learn word meaning from scratch each time

**Analogy:** Imagine learning to read by sounding out every individual letter every time instead of recognizing common words at a glance. You can read "the" faster when you recognize it as a whole unit.

---

## Why Not Just Use Whole Words?

The problem: there are too many words! English has over 170,000 words, plus names, technical terms, new words, code, emojis...

A vocabulary of 170,000 is too large to be efficient. And what about the word "unhappiness"? It's made of parts (`un` + `happiness`) that the model should understand separately.

---

## The Goldilocks Solution: Subword Tokenization 🎯

Modern LLMs use **subword tokenization** — a middle ground:
- Common words get one token: `the`, `is`, `and`
- Rare words get split into common pieces: `tokenization` → `token` + `ization`
- Very rare words get split further: `Schwarzenegger` → `Sch` + `war` + `zen` + `egg` + `er`

**The sweet spot:** A vocabulary of ~32,000–128,000 tokens that covers almost all text efficiently.

---

## BPE — Byte Pair Encoding 🔀

The most common tokenization algorithm is **BPE (Byte Pair Encoding)**:

1. Start with every individual character as its own token
2. Count which pairs of tokens appear most often together
3. Merge the most frequent pair into a new single token
4. Repeat until you have the desired vocabulary size

**Analogy:** It's like a postal code system. Initially each street has its own code. Then you notice that `Main Street + Springfield` always appears together, so you create one code for the whole thing. Keep merging common combinations until you have an efficient system.

---

## The Vocabulary — The "Dictionary" 📖

After training the tokenizer, you end up with a **vocabulary** — a fixed list mapping tokens to IDs:

```
Token #0:    <pad>       (padding)
Token #1:    <bos>       (beginning of sequence)
Token #2:    <eos>       (end of sequence)
...
Token #278:  " the"
Token #279:  " is"
Token #280:  " and"
...
Token #15339: "hello"
...
```

**Every piece of text can be represented as a sequence of these IDs.**

---

## Special Tokens 🎪

Most tokenizers have special tokens for structural purposes:

| Token | Meaning |
|---|---|
| `<bos>` | Beginning of sequence — "start here" |
| `<eos>` | End of sequence — "I'm done" |
| `<pad>` | Padding — "nothing here, ignore this" |
| `<unk>` | Unknown — "I've never seen this" |
| `<|user|>` | Marks user's turn in a conversation |
| `<|assistant|>` | Marks model's turn |

**Analogy:** Special tokens are like stage directions in a screenplay — they tell the actors (the model) what's happening structurally, not just what words to say.

---

## A Real Example

Let's tokenize: `"The cat sat on the mat."`

Using GPT-2's tokenizer:
```
"The"   → 464
" cat"  → 3797
" sat"  → 3332
" on"   → 319
" the"  → 262
" mat"  → 2603
"."     → 13
```

Seven words become seven tokens (lucky coincidence — common words often get their own token).

Now tokenize: `"Supercalifragilistic"`
```
"Super"  → 12403
"cal"    → 9948
"if"     → 361
"rag"    → 22562
"il"     → 346
"istic"  → 6017
```

One rare word → 6 tokens.

---

## Context Window and Tokens 🪟

When people say an LLM has a "128k context window," they mean it can process 128,000 tokens at once.

That's roughly:
- ~100,000 words of text
- ~200–300 pages of a book
- Several hours of conversation

**Tokens ≠ words.** English text typically averages ~1.3 tokens per word. Code and other languages vary widely.

---

## Byte Pair Encoding — Step-by-Step 🔬

Let's walk through BPE concretely. Imagine training a tiny tokenizer on the word "lower" appearing many times in text.

**Start:** Every character is its own token: `l`, `o`, `w`, `e`, `r`

**Step 1 — Count pairs.** The pair `(l, o)` appears most often. Merge it:
- Before: `l o w e r`
- After: `lo w e r`
- New token added to vocab: `lo`

**Step 2 — Count again.** Now `(lo, w)` is the most common. Merge:
- Before: `lo w e r`
- After: `low e r`
- New token: `low`

**Step 3 — Count again.** `(low, er)` merges:
- Before: `low e r`
- After: `lower`
- New token: `lower`

After enough merges, the full word "lower" becomes a single token — because it was common enough. But a word like "Schwarzenegger" never gets merged into one token because it's rare.

**The key insight:** BPE is frequency-driven. It naturally gives efficient representations to common patterns and falls back to character-level representations for rare ones.

---

## The Vocabulary — What's In It 🗂️

The vocabulary of a modern LLM contains a surprisingly varied mix:

- **Single characters:** `a`, `b`, `z`, `1`, `!` — always present as fallback
- **Common short words:** ` the`, ` is`, ` and`, ` to` — note the leading space!
- **Word fragments:** `tion`, `ing`, `ness`, `un`, `re` — very common suffixes and prefixes
- **Full common words:** `hello`, `world`, `Python`, `model`
- **Code tokens:** `def`, `import`, `{`, `//`, `<br>` — if trained on code
- **Special control tokens:** `<|endoftext|>`, `<|user|>`, `[INST]`
- **Digits and numbers:** `0`–`9`, sometimes multi-digit numbers like `100`, `2024`

**Why does the leading space matter?** In BPE, ` the` (with a space) and `the` (without) are different tokens. ` the` appears after another word, while `the` might start a sentence. This distinction is surprisingly important for model quality.

**Vocabulary sizes in real models:**
| Model | Vocab Size |
|---|---|
| GPT-2 | 50,257 |
| GPT-4 / GPT-3.5 | ~100,000 |
| LLaMA 3 | 128,256 |
| Gemma | 256,128 |
| Claude (est.) | ~100,000+ |

Larger vocabularies mean fewer tokens per sentence (more efficient), but require more memory for the embedding table.

---

## Special Tokens — [BOS], [EOS], [PAD], and More 🎭

Special tokens are the "punctuation marks" of the tokenization world — they're not real language, but they give the model crucial structural information.

### [BOS] — Beginning of Sequence
Every input typically starts with a BOS token. It signals: "a new sequence is starting here." Without it, the model wouldn't know if it was in the middle of a conversation or at the start.

### [EOS] — End of Sequence
When the model generates an EOS token, it means "I'm done." Training the model to produce EOS at the right time is how you get it to stop talking. If you've ever seen a model that rambles on forever, it may have been poorly trained on EOS.

### [PAD] — Padding
When processing a batch of sequences (multiple prompts at once for efficiency), they need to be the same length. Shorter sequences get padded with [PAD] tokens. The model learns to ignore these.

### [SEP] — Separator
Used in models like BERT to separate two pieces of text (e.g., a question and a passage). It's the model's way of knowing "these are two different chunks."

### [MASK] — Mask
Used during BERT-style training. A random word is replaced with [MASK] and the model has to predict what it was — this is how BERT learns language understanding.

### Chat-specific tokens
Modern instruction-tuned models have tokens like:
- `<|user|>` and `<|assistant|>` in LLaMA
- `[INST]` and `[/INST]` in earlier Mistral
- `<|im_start|>` and `<|im_end|>` in Qwen / ChatML format

These let the model know who said what in a multi-turn conversation.

---

## Why Tokenization Matters for Model Behavior 🧠

Tokenization isn't just an implementation detail — it has real consequences for how the model reasons:

**1. Arithmetic struggles.** Numbers get tokenized inconsistently. "1000" might be one token, "1001" might be two. The model doesn't "see" digits — it sees arbitrary chunk IDs. This is a major reason LLMs struggle with precise arithmetic.

**2. Spelling and anagrams.** The model never sees individual letters unless they're separate tokens. Asking "spell this word" or "reverse these letters" is surprisingly hard — the model has to reason about the internal structure of a token it treats as an atomic unit.

**3. Language efficiency.** English is well-represented in most vocabularies. Other languages (especially non-Latin script languages like Chinese, Arabic, Thai) often need more tokens per word, making them less efficient and sometimes lower quality in model outputs.

**4. Code tokenization.** Whitespace in code matters. Python's indentation, for example, creates many space tokens. Models trained heavily on code develop specialised tokenizers that handle this better.

**5. Token boundary effects.** If "New York" tokenizes as `New` + ` York` but "NewYork" tokenizes as `New` + `York` (without space), the model sees them differently even though they mean the same thing.

---

## Tokenization Gotchas and Surprises 😱

Some real-world surprises that trip people up:

**"ChatGPT" splits unexpectedly:**
```
"ChatGPT" → ["Chat", "G", "PT"]
```
The model doesn't see "ChatGPT" as one unit — it sees three separate tokens. This can cause subtle issues when the model refers to itself.

**"tokenization" splits too:**
```
"tokenization" → ["token", "ization"]
```
This is actually good! The model can reuse its knowledge of "token" and the suffix "ization."

**Numbers are chaotic:**
```
"100"  → ["100"]          (one token — common)
"1001" → ["100", "1"]     (two tokens — because 1001 is rare)
"9999" → ["999", "9"]     (two tokens)
```

**Whitespace matters enormously:**
```
"hello"  → [15339]    (no leading space)
" hello" → [23748]    (with leading space — different token!)
```

**Emojis are expensive:**
```
"😀" → ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"]   (4 tokens for one emoji!)
```
Because emojis are encoded as multi-byte UTF-8 sequences and each byte becomes its own token in byte-level BPE.

**Non-English languages cost more:**
A simple English sentence: "The dog runs" → 3 tokens
Same meaning in Thai: ม้าวิ่ง → 6 tokens

---

## Different Tokenizers for Different Models 🔧

Not all models use the same tokenizer — and you can't mix them!

| Model Family | Tokenizer | Vocab Size | Notable Feature |
|---|---|---|---|
| GPT-2 / GPT-3 | BPE (tiktoken) | 50,257 | Byte-level fallback |
| GPT-4 / o1 | cl100k_base | 100,277 | Handles code much better |
| LLaMA 1/2 | SentencePiece BPE | 32,000 | Based on Google's library |
| LLaMA 3 | tiktoken-style | 128,256 | Much larger vocab for multilingual |
| Gemma | SentencePiece | 256,128 | Very large vocab for coverage |
| Claude | Proprietary | ~100k+ | Unknown specifics |
| BERT | WordPiece | 30,522 | Different algorithm, ## prefix |
| T5 | SentencePiece | 32,100 | Used sentencepiece with unigram |

**WordPiece vs BPE:** BERT uses WordPiece, which is similar to BPE but picks merges based on a different likelihood criterion. You'll see `##` prefix for continuation tokens in BERT: `tokenization` → `token` + `##ization`.

**SentencePiece:** A library (not an algorithm) that implements both BPE and unigram models. Used by Google's models. It treats the input as raw bytes, handling any language without preprocessing.

---

## How Many Tokens is One Word? 📏

The ratio of tokens to words depends heavily on the content:

**English prose:** ~1.3 tokens per word
- "The quick brown fox" → 4 words, ~5 tokens

**Technical/scientific text:** ~1.5–2 tokens per word
- More rare words = more splitting

**Code (Python):** ~2–4 tokens per logical unit
- Indentation spaces, keywords, brackets all count separately

**Non-Latin languages:** ~2–5 tokens per word
- Chinese, Japanese, Arabic, etc. are less well-covered in most vocabularies

**Emojis:** ~3–4 tokens per emoji
- Each is a multi-byte character that often splits

**Rule of thumb for budgeting:**
- 1 page of English ≈ 500 words ≈ 650–700 tokens
- 1 minute of speech (transcribed) ≈ 150 words ≈ 200 tokens
- GPT-4's 128k context ≈ 96,000 words ≈ 300 pages

This matters when you're trying to fit long documents into a model's context window, or when you're paying per token for API calls.

---

## Key Takeaways

| Concept | Plain English |
|---|---|
| Token | A chunk of text assigned a unique number |
| Tokenization | Converting text → sequence of numbers |
| Vocabulary | The complete list of all tokens and their IDs |
| BPE | The algorithm that creates a vocabulary by merging common character pairs |
| Subword | Breaking rare words into known smaller pieces |
| Context Window | Maximum number of tokens the model can "see" at once |
| Special Tokens | Control tokens like [BOS], [EOS], [PAD] that give the model structural info |
| Token efficiency | How many tokens it takes to represent a given amount of text |

---

## Up Next
👉 **Embeddings** — once we have token IDs, how do we give them meaning?
