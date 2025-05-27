# GPT from Scratch: A Step-by-Step Implementation 

This repository provides a clear and organized implementation of a simplified GPT (Generative Pre-trained Transformer) model from scratch using PyTorch. The goal is to understand the core components and mechanisms behind transformer-based language models by building one in a modular and educational manner.

![GPT Block](<GPT Block.png>)

---

## 🎯 Overview

This implementation walks through the complete process of building a GPT model from the ground up, covering:

### 1. **Data Preparation**
- Load raw text data from an online source
- Implement a simple tokenizer to convert text into tokens with special tokens for unknown words, beginning/end of sequence, and padding
- Create a custom data loader to generate training batches with sliding windows over token sequences

### 2. **Core Components**
- Define token embeddings and positional embeddings to represent input tokens in vector space
- Implement masked multi-head self-attention to allow the model to attend to previous tokens only (causal masking)
- Build essential building blocks such as Layer Normalization, GELU activation, and Feedforward networks

### 3. **Transformer Block**
- Combine multi-head attention and feedforward layers with residual connections and dropout for regularization
- Stack these components to form a single transformer block

### 4. **GPT Model Architecture**
- Assemble embedding layers, transformer block, normalization, and output projection into a complete GPT model
- The model predicts the next token in a sequence, enabling language modeling and text generation

---

## 🔧 Technical Details

### Architecture Components
- **Embedding Layer**: Converts tokens to dense vectors
- **Positional Encoding**: Adds position information to embeddings
- **Multi-Head Attention**: Allows model to focus on different parts of the sequence
- **Feed-Forward Networks**: Applies non-linear transformations
- **Layer Normalization**: Stabilizes training
- **Residual Connections**: Helps with gradient flow

### Training Features
- **Gradient Clipping**: Prevents exploding gradients
- **Dropout Regularization**: Reduces overfitting
- **Validation Monitoring**: Tracks model performance

---

## ⚙️ Configuration

The model is configured with the following hyperparameters:

- **Context window size**: 10 tokens
- **Embedding dimension**: 300
- **Batch size**: 8
- **Stride for sliding window**: 3
- **Number of attention heads**: 10
- **Number of layers**: 12
- **Dropout rate**: 0.1
- **Learning rate**: 5e-5

---

## 📁 Repository Structure

```
├── GPT Block.png              # Architecture diagram
├── GPT_From_Scratch.ipynb     # Main implementation notebook
└── README.md                  # This file
```

---

## 🚀 Getting Started

### Prerequisites

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import re
```

### Running the Code

1. **Clone the repository**
2. **Open the Jupyter notebook**: `GPT_From_Scratch.ipynb`
3. **Run all cells sequentially** to:
   - Load and preprocess the text data
   - Build the model components
   - Train the GPT model
   - Generate text samples

### Key Components Implemented

#### 1. SimpleTokenizer
- Custom tokenizer with special tokens (`<UNK>`, `<BOS>`, `<EOS>`, `<PAD>`)
- Handles text-to-token and token-to-text conversion

#### 2. SimpleDataLoader
- Creates training batches using sliding windows
- Handles input-target pair generation for language modeling

#### 3. MaskedMultiHeadAttention
- Implements causal self-attention mechanism
- Prevents the model from attending to future tokens
- Supports multiple attention heads

#### 4. TransformerBlock
- Combines attention and feedforward layers
- Includes residual connections and layer normalization
- Applies dropout for regularization

#### 5. GPT Model
- Complete language model architecture
- Token and positional embeddings
- Multiple transformer blocks
- Output projection to vocabulary

---

## 📊 Model Performance

The model is trained using:
- **Loss Function**: Cross-entropy loss
- **Optimizer**: Adam optimizer
- **Training/Validation Split**: 90%/10%
- **Model Size**: ~13.5M parameters (approximately 52MB)

---

## 🔍 Text Generation

The repository includes a text generation function that:
- Takes a seed text as input
- Generates continuation using the trained model
- Uses greedy decoding (argmax) for token selection

---

## 🤝 Contributing

This is an educational project. Feel free to:
- Fork the repository for your own learning
- Suggest improvements to the explanations
- Report issues with the implementation
