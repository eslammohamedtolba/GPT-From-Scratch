# GPT from Scratch

A complete implementation of a Generative Pre-trained Transformer (GPT) model built entirely from scratch using PyTorch. This project demonstrates the fundamental concepts behind modern language models through clean, educational code.

![GPT Block](<GPT Block.png>)

---

## 🎯 Project Overview

This repository contains a full implementation of a GPT-style language model, including:
- Custom tokenizer with special token handling
- Multi-head attention mechanism with causal masking
- Complete transformer architecture
- Training pipeline with validation
- Text generation with temperature sampling

The model is trained on Shakespeare's complete works and can generate coherent text in a similar style.

---

## 🏗️ Architecture

### Model Specifications
- **Context Length**: 20 tokens
- **Embedding Dimension**: 300
- **Attention Heads**: 15
- **Transformer Layers**: 15
- **Vocabulary Size**: 39,070 tokens
- **Parameters**: ~39.7M
- **Model Size**: ~151.57 MB

### Key Components
- `SmallTokenizer`: Custom regex-based tokenizer
- `SmallDataLoader`: Efficient data loading with configurable batching
- `MaskedMultiHeadAttention`: Causal self-attention mechanism
- `TransformerBlock`: Complete transformer layer with residual connections
- `GPT`: Main model architecture

---

## 📁 Repository Structure

```
gpt-from-scratch/
├── GPT_From_Scratch.ipynb      # Main notebook with complete implementation
├── README.md                   # This file
└── GPT Block.png               # GPT block image 
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/eslammohamedtolba/GPT-From-Scratch
cd GPT-From-Scratch
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio requests numpy jupyter
```

3. Run the notebook:
```bash
jupyter notebook GPT_From_Scratch.ipynb
```

---

## 🔧 Usage

### Training a New Model

```python
# Configure model parameters
cfg = {
    "context_size": 20,
    "vector_dimension": 300,
    "batch_size": 32,
    "stride": 5,
    "n_heads": 15,
    "n_layer": 15,
    "drop_rate": 0.1,
    "learning_rate": 5e-5
}

# Initialize and train model
gpt_model = GPT(cfg)
# ... training loop (see notebook for details)
```

### Generating Text

```python
# Generate text with temperature sampling
sentence = "To be or not to be"
tokens = tokenizer.encode(sentence)
input_tensor = torch.tensor(tokens).unsqueeze(0)

generated = generate_text(
    model=gpt_model,
    idx=input_tensor,
    max_tokens=50,
    context_size=cfg["context_size"],
    temperature=0.8
)

output_text = tokenizer.decode(generated[0].tolist())
print(output_text)
```

---

## 📊 Training Results

The model was trained for 20 epochs on Shakespeare's complete works:

| Metric | Training | Validation |
|--------|----------|------------|
| Initial Loss | 6.05 | 6.26 |
| Final Loss | 3.11 | 6.83 |
| Training Time | ~1.5 hours (GPU) | - |

### Sample Generated Text
The model generates coherent text in Shakespearean style based on the training corpus.

---

## 🧠 Technical Details

### Tokenization Strategy
- Regex-based splitting on punctuation and whitespace
- Special tokens: `<UNK>`, `<BOS>`, `<EOS>`, `<PAD>`
- Vocabulary built from complete Shakespeare corpus

### Attention Mechanism
- Scaled dot-product attention
- Causal masking for autoregressive generation
- Multi-head attention with configurable head count

### Training Features
- Custom cross-entropy loss implementation
- Gradient clipping for stability
- Learning rate scheduling
- Dropout for regularization

---

## 🛠️ Customization

### Hyperparameter Tuning
Easily modify the `cfg` dictionary to experiment with:
- Model size (layers, dimensions, heads)
- Training parameters (learning rate, batch size)
- Context length and vocabulary size

### Dataset Replacement
The tokenizer and data loader can work with any text corpus:
```python
# Replace with your dataset
url = "your_dataset_url_here"
response = requests.get(url)
text = response.text
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional text generation strategies
- More efficient attention implementations
- Support for larger datasets
- Model compression techniques
