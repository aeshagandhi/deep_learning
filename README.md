# Vision-Language Pretraining for Image-to-Text Generation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Investigating whether Vision-Language Pretraining models can effectively reverse the text-to-image relationship by generating accurate, semantically meaningful descriptions from images.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Models](#models)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [References](#references)

## Overview

While text-to-image generation has received significant attention, the reverse task of generating rich, context-aware text from images remains less explored and more challenging. This project compares two state-of-the-art architectures for image captioning:

- **ViT-GPT2**: Pure vision encoder combined with autoregressive text generation
- **BLIP**: Multimodal encoder-decoder with joint vision-language pretraining

Our research demonstrates that models grounded in joint vision-language pretraining (BLIP) significantly outperform traditional encoder-decoder approaches, achieving **102% improvement** in combined metrics after fine-tuning.

## Key Features

- **Dual Architecture Comparison**: Comprehensive evaluation of ViT-GPT2 vs. BLIP
- **Custom Evaluation Metrics**: Combined BLEU + BERTScore metric (65% BERTScore, 35% BLEU) for semantic and lexical quality
- **Fine-tuned Models**: Pre-trained and fine-tuned checkpoints available for both architectures
- **Efficient Training Pipeline**: Optimized for NVIDIA A100 GPU with mixed-precision training
- **Reproducible Results**: Fixed random seeds and documented hyperparameters

## Models

### Vision Transformer + GPT-2

**Architecture**:
- **Encoder**: ViT (12 transformer blocks) - Processes 16×16 image patches
- **Decoder**: GPT-2 (12 transformer blocks with cross-attention) - Autoregressive caption generation
- **Inference**: Beam search (4 beams) with early stopping

**Training Configuration**:
- Epochs: 6
- Batch Size: 8 (effective, with gradient accumulation)
- Learning Rate: 5×10⁻⁵
- Optimizer: AdamW
- Mixed Precision: FP16

### Bootstrapping Language-Image Pre-Training (BLIP)

**Architecture**:
- **Multimodal Encoder-Decoder (MED)** with three components:
  1. Unimodal encoder (Image-Text Contrastive loss)
  2. Image-grounded text encoder (Image-Text Matching loss)
  3. Image-grounded text decoder (Language Modeling loss)

**Training Configuration**:
- Base Model: `Salesforce/blip-image-captioning-base`
- Epochs: 10
- Learning Rate: 1×10⁻⁴ with CosineAnnealingWarmRestarts
- Weight Decay: 1×10⁻⁶
- Gradient Accumulation: 4 steps
- Max Sequence Length: 128 tokens

## Dataset

**Source**: [DiffusionDB](https://github.com/poloclub/diffusiondb) - Large-scale text-to-image dataset with 14M+ Stable Diffusion 2.0 generations

**Our Subset**:
- **Total Images**: ~10,000
- **Resolution**: 512×512 pixels
- **Prompts**: Variable length English text with rich descriptions
- **Split**: 60% train / 20% validation / 20% test (seed=42)

**Sampling Strategy**: Random sampling for broad, unbiased coverage while maintaining computational efficiency.

## Results

### Performance Comparison

| Model | BERTScore | BLEU | Combined Score | Improvement |
|-------|-----------|------|----------------|-------------|
| **BLIP (Pretrained)** | 0.44 | 0.27 | 0.38 | - |
| **BLIP (Fine-tuned)** | **0.76** | **0.78** | **0.77** | **+102%** |
| **ViT-GPT2 (Pretrained)** | 0.42 | 0.00 | 0.27 | - |
| **ViT-GPT2 (Fine-tuned)** | 0.43 | 0.00 | 0.28 | +3% |

### Key Findings

- **BLIP achieves 73% improvement in BERTScore** and 189% in BLEU after fine-tuning
- **Strong semantic alignment**: BERTScore effectively captures contextual meaning even with varied wording
- **Clear advantage for multimodal pretraining**: Joint vision-language training significantly outperforms encoder-decoder approaches
- **Limitations**: Both models struggle with abstract concepts, complex relationships, and generating prompts as expressive as ground truth

### Example Predictions

**BLIP - Strong Performance**:
```
Ground Truth: "A portrait of a beautiful female cyborg with a cracked porcelain 
              face by wlop, exposed inner structure, brightly glowing white eyes, 
              art nouveau card, trending on artstation"

Generated:    "a portrait of a beautiful female cyborg wearing an intricate 
              venetian mask by Wlop"

BLEU: 0.79 | BERTScore: 0.69 | Combined: 0.73
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: NVIDIA A100 or similar)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/vision-language-pretraining.git
cd vision-language-pretraining

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

The project relies on PyTorch, HuggingFace Transformers, and standard ML libraries for training, evaluation, and data processing. All required packages are specified in `requirements.txt`.

## Usage

### Training

The repository includes Jupyter notebooks for training both architectures:
- **BLIP Training**: Fine-tuning workflow with CosineAnnealingWarmRestarts scheduler, gradient accumulation, and validation monitoring
- **ViT-GPT2 Training**: Encoder-decoder training pipeline with mixed-precision FP16 optimization

Both notebooks include data loading, preprocessing, training loops, and checkpoint saving functionality.

### Inference

The trained models can generate captions for new images using the HuggingFace Transformers library. The inference pipeline:
1. Loads the fine-tuned model and processor
2. Preprocesses input images (resize, normalize, convert to tensors)
3. Generates captions using beam search decoding
4. Post-processes and returns human-readable text

### Evaluation

Evaluation scripts compute BLEU, BERTScore, and the combined metric across the test set, generating detailed performance reports and prediction samples for qualitative analysis.

## Evaluation Metrics

### Custom Combined Score

We developed a weighted combination of BLEU and BERTScore:

```
Score = 0.65 × BERTScore + 0.35 × BLEU
```

**Rationale**:
- **BERTScore (65%)**: Captures semantic alignment using contextual embeddings, allowing flexibility in wording
- **BLEU (35%)**: Measures lexical precision through n-gram overlap with reference captions

This weighting emphasizes semantic correctness while maintaining lexical accuracy.

## Project Structure

```
vision-language-pretraining/
│
├── model-results/              # Training outputs and model checkpoints
│
├── BLIP_training.ipynb         # BLIP model training notebook
│
├── VIT_Model_Training.ipynb    # ViT-GPT2 model training notebook
│
├── Final_ViT.ipynb             # ViT evaluation and results
│
├── Deep_Learning_FinalReport.pdf   # Comprehensive project report
│
├── requirements.txt            # Python dependencies
│
└── README.md                   # Project documentation
```

**Note**: The repository is organized around Jupyter notebooks for ease of experimentation and visualization. Model checkpoints and results are stored separately for version control efficiency.

## Future Work

- **Larger Datasets**: Scale training to full DiffusionDB for improved generalization
- **Advanced Architectures**: Explore models like CLIP, Flamingo, or GPT-4V
- **Improved Evaluation**: Incorporate human evaluation and more nuanced semantic metrics
- **Refinement Techniques**: Implement iterative refinement and prompt engineering strategies
- **Bidirectional Training**: Develop architectures optimized for both text-to-image and image-to-text generation
- **Real-world Applications**: Deploy for accessibility tools, content moderation, and automated alt-text generation

## Contributors

- **Pranshul Bhatnagar** - [GitHub](https://github.com/pranshul) | [LinkedIn](https://linkedin.com/in/pranshul)
- **Aesha Gandhi** - [GitHub](https://github.com/aesha) | [LinkedIn](https://linkedin.com/in/aesha)
- **Gaurav Law** - [GitHub](https://github.com/gaurav) | [LinkedIn](https://linkedin.com/in/gaurav)

## References

1. Li, J., Li, D., Xiong, C., & Hoi, S. (2022). BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. *ICML 2022*.

2. Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.

3. Radford, A., et al. (2021). Learning Transferable Visual Models from Natural Language Supervision. *ICML 2021*.

4. Wang, Z., et al. (2022). DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models. *arXiv preprint*.

5. Papineni, K., et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. *ACL 2002*.

6. Zhang, T., et al. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR 2020*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Salesforce Research for the BLIP model
- HuggingFace for the transformers library
- Poloclub team for DiffusionDB dataset
- NeurIPS 2022 conference submission guidelines

---

**Note**: This project was developed as part of a Deep Learning course final project. For questions or collaboration opportunities, please open an issue or reach out via email.

If you find this project useful, please consider giving it a star!
