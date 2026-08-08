# 🎓 Neuricode Training & Alignment Manual: Building a Conversational AI Assistant

This guide explains how to transform **Neuricode** from a raw text auto-completer into an interactive, instruction-following conversational AI assistant (like ChatGPT or Claude) using a 2-stage training pipeline in pure C.

---

## 🛠️ The 2-Stage Training Pipeline Overview

```
+-----------------------------------------------------------------------------------+
|                        STAGE 1: PRE-TRAINING (Base Model)                         |
|  Dataset: Raw text corpora (books, code, articles)                                |
|  Goal: Learn grammar, vocabulary, facts, and token prediction.                    |
|  Output: A base model that completes sentences (e.g. "Once upon a..." -> "time")  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                   STAGE 2: SUPERVISED FINE-TUNING (SFT / Alignment)               |
|  Dataset: Formatted Q&A conversations (<|user|> & <|assistant|>)                  |
|  Goal: Teach the model to follow instructions and act as a conversational assistant|
|  Output: An interactive chatbot (e.g. "Who are you?" -> "I am Neuricode...")       |
+-----------------------------------------------------------------------------------+
```

---

## 🚀 Stage 1: Base Pre-Training (Learning World Knowledge)

Before an AI can answer questions, it must first learn basic language structure, vocabulary, and grammar.

### Step 1.1: Prepare Vocabulary & Text Data
1. Ensure your vocabulary file (`assets/vocab.txt`) includes both standard subwords/characters and special control tokens:
   ```text
   <EOS>
   <|user|>
   <|assistant|>
   a
   b
   ...
   ```
2. Prepare your pre-training corpus in `chunks/data.txt` or `chunks/token_ab`.

### Step 1.2: Execute Base Pre-Training
Run the `./train` binary with a standard pre-training learning rate ($\eta = 0.001$ or $0.01$):

```bash
./train chunks/token_ab assets/vocab.txt model.bin 20 64 20 0.001
```

- `chunks/token_ab`: Input training text file.
- `assets/vocab.txt`: Token vocabulary file.
- `model.bin`: Target binary output weight checkpoint.
- `20`: Sequence length ($S = 20$).
- `64`: Recurrent hidden state dimension ($H = 64$).
- `20`: Pre-training epochs ($E = 20$).
- `0.001`: Pre-training base learning rate ($\eta = 0.001$).

### Step 1.3: Verify Stage 1 Completion
Pre-training is complete when:
- Average loss drops to $\le 2.0 - 2.5$.
- Testing base completion via `./cli` generates grammatically coherent sentence continuations.

---

## 💬 Stage 2: Supervised Fine-Tuning (SFT / Conversational Alignment)

Stage 2 aligns the base model so that instead of randomly completing text, it responds directly to user queries.

### Method 1: Format an Instruction Dataset (`assets/sft_data.txt`)

Create an instruction dataset file at `assets/sft_data.txt`. Format conversation pairs using structured role control tokens (`<|user|>` and `<|assistant|>`):

```text
<|user|>
Hello, who are you?
<|assistant|>
Hello! I am Neuricode, a native C11 AI engine. How can I assist you today?

<|user|>
What is 2 + 2?
<|assistant|>
2 + 2 is equal to 4.

<|user|>
Explain what C11 is in one line.
<|assistant|>
C11 is the 2011 ISO standard for the C programming language, introducing multithreading and atomic memory operations.

<|user|>
How do I build Neuricode?
<|assistant|>
You can build Neuricode by running 'make neuricode' in your terminal.

<|user|>
What language is Neuricode written in?
<|assistant|>
Neuricode is written in 100% pure ISO C11 and native CUDA C++.
```

---

### Method 2: Adjust Hyperparameters for Fine-Tuning

Supervised Fine-Tuning requires a **much lower learning rate** to prevent **catastrophic forgetting** (where the model accidentally erases the grammar and vocabulary it learned in Stage 1).

Run `make config` (or edit `neuralc_config.h`):

1. **Learning Rate (`LEARNING_RATE`)**: Lower from `0.001` down to `0.00003` ($3 \times 10^{-5}$) or `0.00005` ($5 \times 10^{-5}$).
2. **Epochs (`EPOCHS`)**: Set to **3 to 5 epochs** (fine-tuning requires far fewer passes than pre-training).
3. **Gradient Clipping (`GRAD_CLIP_NORM`)**: Ensure gradient clipping is enabled (`1.0` or `5.0`) to prevent gradient spikes.

---

### Method 3: Execute Fine-Tuning (`./train`)

Run the trainer binary on your instruction dataset (`assets/sft_data.txt`) to update existing model weights in `model.bin`:

```bash
# 1. Rebuild with new hyperparameter configuration
make train

# 2. Run instruction fine-tuning pass on model.bin
./train assets/sft_data.txt assets/vocab.txt model.bin 20 64 5 0.00003
```

---

### Method 4: Test & Verify Stage 2 Alignment (`./neuricode`)

Launch the interactive CLI shell:

```bash
./neuricode
```

Type conversation prompts to test Stage 2 alignment:

```text
> Hello, who are you?
Hello! I am Neuricode, a native C11 AI engine. How can I assist you today?

> What is 2 + 2?
2 + 2 is equal to 4.

> What language is Neuricode written in?
Neuricode is written in 100% pure ISO C11 and native CUDA C++.
```

---

## 📋 Summary Checklist for Beginners

- [ ] **Stage 1 (Pre-training)**: Train on raw text until loss is $\le 2.0 - 2.5$.
- [ ] **Create Dataset**: Format Q&A pairs with `<|user|>` and `<|assistant|>` in `assets/sft_data.txt`.
- [ ] **Lower Learning Rate**: Set LR to `0.00003` ($3 \times 10^{-5}$) in `make config` or command line.
- [ ] **Train SFT**: Run `./train assets/sft_data.txt assets/vocab.txt model.bin 20 64 5 0.00003` for 3–5 epochs.
- [ ] **Interact & Enjoy**: Launch `./neuricode` and interact with your aligned C11 AI assistant!
