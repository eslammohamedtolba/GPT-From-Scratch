# Transformer from Scratch using PyTorch

This repository implements the Transformer architecture from scratch, inspired by the seminal paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al. Unlike typical PyTorch workflows, this project **avoids using high-level APIs like `torch.nn.Transformer`** to give you a complete understanding of the inner workings of the model.

---

## 📌 Objectives

- Understand the core components of the Transformer architecture.
- Build each component from scratch:
  - [x] Tokenizer & Data Loader  
  - [x] Token and Positional Embeddings  
  - [x] Scaled Dot-Product Attention  
  - [ ] Multi-Head Attention  
  - [ ] Feedforward Network  
  - [ ] Encoder and Decoder Layers  
  - [ ] Full Transformer Model  
  - [ ] Training and Evaluation  

> ✅ Completed steps are already implemented in the notebook. The rest are in progress.

---

## 📁 Project Structure

```bash
.
├── transformer_from_scratch.ipynb  # Main implementation notebook
├── README.md                       # Project documentation
```

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

---

## 🚧 Work in Progress

- `MultiHeadAttention` class (being implemented next)
- Modular and reusable PyTorch layers
- Support for training and evaluation pipelines

---

## 🧠 Goal of This Project

To demystify the internals of Transformer models by building everything from the ground up. This implementation is meant for **educational purposes** and **experimentation**, not production use.

---

## 🔧 Dependencies

- Python 3.8+
- NumPy
- Requests
- Seaborn
- (Later: PyTorch)

Install them using:

```bash
pip install numpy requests seaborn
```
