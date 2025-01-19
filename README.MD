# Shakespeare GPT

A GPT-2 model fine-tuned on Shakespeare's works, capable of generating Shakespeare-style text.

## Project Overview

This project implements a GPT-2 architecture trained on Shakespeare's works to generate Shakespeare-style text. The model uses a context window of 1024 tokens and implements various optimizations including gradient accumulation and learning rate scheduling.

## Model Architecture

- Base Architecture: GPT-2 (124M parameters)
- Layers: 12
- Heads: 12
- Embedding Dimension: 768
- Context Length: 1024 tokens
- Total Parameters: ~124M

## Training Details

- Dataset: Shakespeare's complete works
- Training Device: GPU/MPS (Apple Silicon)
- Batch Size: 16 (Effective batch size: 64 with gradient accumulation)
- Learning Rate: 6e-4 with cosine decay
- Weight Decay: 0.1
- Training Steps: 10,000

## Performance

- Best Validation Loss: [Insert your best validation loss]
- Training Time: [Insert your training time]

## Requirements
- bash
- pip install -r requirements.txt

## Project Structure
├── src/
│ ├── train_shakespeare.py # Training script
│ ├── app.py # Gradio interface
│ └── input.txt # Training data
├── requirements.txt
└── README.md

## Usage

### Training

To train the model:

bash
python src/train_shakespeare.py


### Inference

- To run the Gradio interface locally:
- bash
- python src/app.py

bash
python src/app.py

