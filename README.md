# Transformer from Scratch
This repository implements the Transformer architecture from scratch, inspired by the seminal paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al. Unlike typical PyTorch workflows, this project **avoids using high-level APIs like `torch.nn.Transformer`** to give you a complete understanding of the inner workings of the model.

![Transformer Architecture](<Transformer Architecture.png>)

---

## 📌 Objectives

* Understand the core components of the Transformer architecture.
* Build each component from scratch:
  * <span style="color:green">[X]</span> Tokenizer & Data Loader
  * <span style="color:green">[X]</span> Token and Positional Embeddings
  * <span style="color:green">[X]</span> Scaled Dot-Product Attention
  * <span style="color:green">[X]</span> Multi-Head Attention
  * <span style="color:red">[X]</span> Feedforward Network
  * <span style="color:red">[X]</span> Encoder and Decoder Layers
  * <span style="color:red">[X]</span> Full Transformer Model
  * <span style="color:red">[X]</span> Training and Evaluation

> ✅ Completed steps are already implemented in the notebook. The rest are in progress.

---

## 📚 Dataset

The notebook uses a plain-text sample from the book *The Verdict*, available at:  
[https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt](https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt)

---

## ⚙️ Components Implemented So Far

### 1. Data Preparation
- Download and clean the raw text
- Custom tokenizer (`SimpleTokenizer`) with special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`)
- Encoding and decoding utility

### 2. Data Loader
- Simple sliding window context-based batching (`SimpleDataLoader`)

### 3. Embeddings
- Random token embeddings
- Positional encoding added manually

### 4. Scaled Dot-Product Attention
- Basic attention mechanism from scratch
- Scaling and softmax applied
- Visualization using `seaborn.heatmap`

### 5. Multi-Head Attention
* Implemented `self_attention` class with query, key, and value matrices
* Implemented `masked_self_attention` to apply causal masking for autoregressive modeling
* Implemented `multihead_attention` class combining multiple attention heads
* Demonstrated multi-head attention output on positionally encoded embeddings

---

## 🚧 Work in Progress

* Feedforward Network
* Encoder and Decoder Layers
* Full Transformer Model
* Training and Evaluation pipelines

---

## 🧠 Goal of This Project

To demystify the internals of Transformer models by building everything from the ground up. This implementation is meant for **educational purposes** and **experimentation**, not production use.

---

## 🔧 Dependencies

- Python 3.8+
- NumPy
- Requests
- Seaborn

Install them using:

```bash
pip install numpy requests seaborn
```
