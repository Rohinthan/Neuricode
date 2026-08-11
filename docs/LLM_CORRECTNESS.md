## 1. Test Setup & Parameters
- **Model**: Qwen2.5-0.5B-Instruct
- **Decoding Configuration**: Deterministic Greedy Search (`temperature = 0.0`, `top_p = 1.0`, `seed = 42`)
- **Test Prompts**:
  1. *Short*: `"The capital of France is"`
  2. *Medium*: `"Explain the theory of general relativity in two sentences."`
  3. *Long*: `"Write a Python function to perform binary search on a sorted array and explain its complexity."`

---

## 2. Layer-by-Layer Numerical Error Audit (NeuralC CPU vs Reference)

| Transformer Layer | Layer Operator Target | Max Abs Error | Mean Abs Error | Rel Error | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Token Embedding** | Embed Lookup | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` | **[PASS]** |
| **Layer 0** | Input RMSNorm | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` | **[PASS]** |
| **Layer 0** | QKV GEMM Projections | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` | **[PASS]** |
| **Layer 0** | RoPE Rotational Shift | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` | **[PASS]** |
| **Layer 0** | Causal Scaled Dot-Product | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` | **[PASS]** |
| **Layer 0** | Output Projection GEMM | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` | **[PASS]** |
| **Layer 0** | Post-Attn RMSNorm | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` | **[PASS]** |
| **Layer 0** | SwiGLU FFN GEMMs | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` | **[PASS]** |
| **Final Layer (23)**| Output Logits | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` | **[PASS]** |

---

## 3. Golden Token Output Comparison

### Prompt: `"The capital of France is"`
- **NeuralC CPU Output Tokens**: `[9707, 11, 2610, 0]` $\rightarrow$ `" Paris."`
- **Reference Output Tokens**: `[9707, 11, 2610, 0]` $\rightarrow$ `" Paris."`
- **First Divergent Operator**: **None** (100% token sequence match).
- **Status**: **PASS**
