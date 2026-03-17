# LLM Math Visualiser

Plain-English study material + Manim animations covering all core LLM/Transformer concepts.

---

## Structure

```
llm-math-visualiser/
├── study-material/          # Markdown study guides (16 topics)
│   ├── 01_neural_networks.md
│   ├── 02_linear_layers.md
│   ├── 03_activation_functions.md
│   ├── 04_tokenization.md
│   ├── 05_embeddings.md
│   ├── 06_positional_encoding.md
│   ├── 07_transformer_architecture.md
│   ├── 08_attention_mechanism.md
│   ├── 09_multi_head_attention.md
│   ├── 10_feed_forward_networks.md
│   ├── 11_layer_normalization.md
│   ├── 12_kv_cache.md
│   ├── 13_prefill_and_decode.md
│   ├── 14_logits_and_token_selection.md
│   ├── 15_temperature_top_k_top_p.md
│   └── 16_mixture_of_experts.md
│
└── scenes/                  # Manim animation scripts (16 topics)
    ├── utils.py             # Shared helpers, colour palette, LLMScene base class
    ├── 01_neural_networks.py
    ├── 02_linear_layers.py
    ├── ...
    └── 16_mixture_of_experts.py
```

---

## Prerequisites

```bash
pip install manim
# Also needs LaTeX for math rendering — see https://docs.manim.community/en/stable/installation/
```

---

## Rendering Animations

```bash
cd scenes

# Low-quality preview (fast)
manim -pql 01_neural_networks.py NeuralNetworksScene

# High-quality render
manim -pqh 01_neural_networks.py NeuralNetworksScene

# All scenes at once (low quality)
for f in *.py; do
  scene=$(python -c "import ast, sys; t=ast.parse(open('$f').read()); print([n.name for n in ast.walk(t) if isinstance(n, ast.ClassDef)][-1])" 2>/dev/null)
  [ -n "$scene" ] && manim -pql "$f" "$scene"
done
```

---

## Scene Name Reference

| File | Scene Class |
|---|---|
| `01_neural_networks.py` | `NeuralNetworksScene` |
| `02_linear_layers.py` | `LinearLayersScene` |
| `03_activation_functions.py` | `ActivationFunctionsScene` |
| `04_tokenization.py` | `TokenizationScene` |
| `05_embeddings.py` | `EmbeddingsScene` |
| `06_positional_encoding.py` | `PositionalEncodingScene` |
| `07_transformer_architecture.py` | `TransformerArchitectureScene` |
| `08_attention_mechanism.py` | `AttentionMechanismScene` |
| `09_multi_head_attention.py` | `MultiHeadAttentionScene` |
| `10_feed_forward_networks.py` | `FeedForwardScene` |
| `11_layer_normalization.py` | `LayerNormScene` |
| `12_kv_cache.py` | `KVCacheScene` |
| `13_prefill_and_decode.py` | `PrefillDecodeScene` |
| `14_logits_and_token_selection.py` | `LogitsTokenSelectionScene` |
| `15_temperature_sampling.py` | `TemperatureSamplingScene` |
| `16_mixture_of_experts.py` | `MixtureOfExpertsScene` |

---

## Curriculum

1. Neural Networks — nodes, layers, hidden states
2. Linear Layers & Matrix Multiplication
3. Activation Functions — ReLU, SiLU, GELU
4. Tokenization — text → numbers
5. Embeddings — numbers → meaning
6. Positional Encoding — where in the sentence?
7. Transformer Architecture — the big picture
8. Attention Mechanism — relating words
9. Multi-Head Attention — multiple perspectives
10. Feed-Forward Networks — the thinking layers
11. Layer Normalization — keeping values stable
12. KV Cache — memory for efficiency
13. Prefill & Decode — two phases of inference
14. Logits & Token Selection — picking the next word
15. Temperature, Top-k, Top-p — controlling creativity
16. Mixture of Experts — specialization at scale
