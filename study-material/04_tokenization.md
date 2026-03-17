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

## Key Takeaways

| Concept | Plain English |
|---|---|
| Token | A chunk of text assigned a unique number |
| Tokenization | Converting text → sequence of numbers |
| Vocabulary | The complete list of all tokens and their IDs |
| BPE | The algorithm that creates a vocabulary by merging common character pairs |
| Subword | Breaking rare words into known smaller pieces |
| Context Window | Maximum number of tokens the model can "see" at once |

---

## Up Next
👉 **Embeddings** — once we have token IDs, how do we give them meaning?
