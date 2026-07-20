<div align="center">

# Mini-CLIP

### A lightweight CLIP-style model for fashion image–text retrieval

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-Educational-lightgrey)](#license)

Mini-CLIP is a compact implementation inspired by CLIP, designed for learning and experimenting with multimodal image–text retrieval on fashion data.

[Demo Video](https://www.youtube.com/watch?v=unUhRajTKJU) •
[Model Checkpoint](./best_mini_clip.pth) •
[Source Code](https://github.com/AIVIETNAM-AIO-Nhhoang1207/Mini-CLIP)

</div>

---

## Overview

Mini-CLIP learns a shared embedding space for images and text.

During training, matching image–caption pairs are pulled closer together, while non-matching pairs are pushed farther apart. After training, the model can perform two retrieval tasks:

* **Text-to-Image Retrieval:** given a caption, retrieve the most relevant images.
* **Image-to-Text Retrieval:** given an image, retrieve the most relevant captions.

The project focuses on implementing the essential ideas behind CLIP using smaller and more accessible pretrained encoders.

---

## Demo

A demonstration of the project is available on YouTube:

[![Mini-CLIP Demo](https://img.youtube.com/vi/unUhRajTKJU/maxresdefault.jpg)](https://www.youtube.com/watch?v=unUhRajTKJU)

---

## Model Architecture

Mini-CLIP contains two independent encoders:

| Component          | Architecture               | Output dimension |
| ------------------ | -------------------------- | ---------------: |
| Image Encoder      | Pretrained ResNet-18       |              512 |
| Text Encoder       | Pretrained DistilBERT      |              512 |
| Similarity         | Cosine similarity          |                — |
| Training objective | Symmetric contrastive loss |                — |

### Image Encoder

The image encoder uses a pretrained ResNet-18 backbone.

The original classification layer is removed, and the extracted image representation is projected into a 512-dimensional embedding space.

```text
Input image
    │
    ▼
Pretrained ResNet-18
    │
    ▼
Visual features
    │
    ▼
Linear projection
    │
    ▼
512-dimensional image embedding
```

### Text Encoder

The text encoder uses `distilbert-base-uncased`.

The representation of the first output token is passed through a linear projection layer to obtain a 512-dimensional text embedding.

```text
Input caption
    │
    ▼
DistilBERT tokenizer
    │
    ▼
Pretrained DistilBERT
    │
    ▼
Text representation
    │
    ▼
Linear projection
    │
    ▼
512-dimensional text embedding
```

### Shared Embedding Space

Both image and text embeddings are L2-normalized before their similarity is calculated.

```text
Image ──► Image Encoder ──► Image Embedding ──┐
                                               ├──► Similarity Matrix
Text  ──► Text Encoder  ──► Text Embedding  ──┘
```

---

## Contrastive Learning Objective

For a batch containing matching image–caption pairs:

[
(I_1,T_1), (I_2,T_2), \ldots, (I_N,T_N),
]

the model computes the similarity between every image embedding and every text embedding.

The diagonal entries of the similarity matrix represent matching pairs. The remaining entries represent non-matching pairs.

Mini-CLIP applies cross-entropy loss in both retrieval directions:

[
\mathcal{L}
===========

\frac{
\mathcal{L}*{image\rightarrow text}
+
\mathcal{L}*{text\rightarrow image}
}{2}.
]

A learnable temperature parameter controls the scale of the similarity logits.

---

## Repository Structure

```text
Mini-CLIP/
├── Data.py                 # Dataset for loading images and captions
├── models.py               # ResNet-18 and DistilBERT encoders
├── train.py                # Training, validation and early stopping
├── benchmark.py            # Recall@K evaluation
├── visualize.py            # Similarity visualization
├── visualize_compare.py    # Comparison visualization
├── best_mini_clip.pth      # Trained model checkpoint
├── .gitignore
├── .gitattributes
└── README.md
```

---

## Dataset Format

The project expects CSV files containing at least two columns:

```csv
image_filename,caption
image_0001.jpg,a black short sleeve shirt
image_0002.jpg,a red floral summer dress
image_0003.jpg,white running shoes
```

Required columns:

| Column           | Description                               |
| ---------------- | ----------------------------------------- |
| `image_filename` | Name of the corresponding image file      |
| `caption`        | Natural-language description of the image |

The dataset loader can search for images across one or multiple image directories.

Suggested dataset structure:

```text
Full_Data/
└── dataset/
    ├── images/
    │   ├── image_0001.jpg
    │   ├── image_0002.jpg
    │   └── ...
    └── ML_test/
        ├── train.csv
        ├── val.csv
        └── test.csv
```

> The data paths are currently configured directly inside `train.py`, `benchmark.py`, `visualize.py`, and `visualize_compare.py`. Update them to match your local dataset location before running the scripts.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AIVIETNAM-AIO-Nhhoang1207/Mini-CLIP.git
cd Mini-CLIP
```

### 2. Create a virtual environment

Using `venv`:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision transformers pandas pillow tqdm matplotlib seaborn
```

Main dependencies:

* Python 3.9+
* PyTorch
* Torchvision
* Hugging Face Transformers
* Pandas
* Pillow
* tqdm
* Matplotlib
* Seaborn

---

## Configuration

Before training or evaluation, update the dataset paths inside the Python scripts.

Example configuration:

```python
train_csv_path = "path/to/train.csv"
val_csv_path = "path/to/val.csv"
test_csv_path = "path/to/test.csv"

img_dirs = [
    "path/to/images"
]
```

The project automatically uses CUDA when a compatible GPU is available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## Training

Run:

```bash
python train.py
```

The default training configuration uses:

| Parameter                      |  Value |
| ------------------------------ | -----: |
| Batch size                     |     64 |
| Maximum epochs                 |    100 |
| Embedding dimension            |    512 |
| Backbone learning rate         | `1e-5` |
| Projection learning rate       | `1e-3` |
| Early-stopping patience        |      5 |
| Minimum validation improvement |  0.005 |

The image and text backbones use smaller learning rates than the projection layers and learnable logit scale.

The best model is saved as:

```text
best_mini_clip.pth
```

Training stops early when the validation loss does not improve sufficiently for five consecutive epochs.

---

## Evaluation

Run:

```bash
python benchmark.py
```

The benchmark script:

1. Loads `best_mini_clip.pth`.
2. Encodes all images and captions in the test set.
3. Normalizes the embeddings.
4. Computes the image–text similarity matrix.
5. Evaluates retrieval performance using Recall@K.

---

## Evaluation Metrics

The project evaluates retrieval in both directions.

### Text-to-Image Retrieval

A caption is used as the query, and the model retrieves the most similar images.

* T2I Recall@1
* T2I Recall@5
* T2I Recall@10

### Image-to-Text Retrieval

An image is used as the query, and the model retrieves the most similar captions.

* I2T Recall@1
* I2T Recall@5
* I2T Recall@10

Recall@K is calculated as:

[
Recall@K
========

\frac{
\text{number of queries whose correct result appears in top }K
}{
\text{total number of queries}
}
\times 100%.
]

A higher Recall@K indicates better retrieval performance.

---

## Visualization

### Similarity Visualization

Run:

```bash
python visualize.py
```

This script visualizes image–text similarity scores and helps inspect whether matching image–caption pairs receive higher similarity than non-matching pairs.

### Model Comparison

Run:

```bash
python visualize_compare.py
```

This script provides an additional visual comparison of retrieval or similarity results.

Before running the visualization scripts, verify that:

* the dataset paths are correct;
* `best_mini_clip.pth` exists;
* all required dependencies are installed.

---

## Using the Pretrained Checkpoint

The repository already contains the trained checkpoint:

```text
best_mini_clip.pth
```

It can be loaded as follows:

```python
import torch

from models import ImageEncoder, TextEncoder
from train import MiniCLIP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_encoder = ImageEncoder(embed_dim=512)
text_encoder = TextEncoder(embed_dim=512)

model = MiniCLIP(
    image_encoder=image_encoder,
    text_encoder=text_encoder
).to(device)

state_dict = torch.load(
    "best_mini_clip.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()
```

The architecture used to load the checkpoint must match the architecture used during training.

---

## Project Workflow

```text
Fashion images and captions
            │
            ▼
       Dataset loader
            │
            ▼
   Image and text encoders
            │
            ▼
   Normalized embeddings
            │
            ▼
 Image–text similarity matrix
            │
            ▼
  Symmetric contrastive loss
            │
            ▼
     Trained Mini-CLIP
            │
            ▼
 Text-to-image / Image-to-text retrieval
```

---

## Current Features

* Pretrained ResNet-18 image encoder
* Pretrained DistilBERT text encoder
* Shared 512-dimensional embedding space
* Learnable similarity temperature
* Symmetric image–text contrastive loss
* Validation-based early stopping
* Automatic CPU/CUDA device selection
* Text-to-image retrieval evaluation
* Image-to-text retrieval evaluation
* Recall@1, Recall@5 and Recall@10
* Similarity visualization
* Included pretrained checkpoint

---

## Limitations

The current implementation is intended primarily for learning and experimentation.

Current limitations include:

* dataset paths are hard-coded inside the scripts;
* no command-line configuration is currently provided;
* the model is specialized toward the training dataset;
* retrieval assumes aligned image–caption pairs in the evaluation set;
* no standalone interactive search application is included;
* the repository does not yet provide a pinned `requirements.txt`;
* training can require significant GPU memory because both pretrained encoders are fine-tuned.

---

## Possible Improvements

Future development may include:

* moving configuration into command-line arguments or YAML files;
* adding a `requirements.txt`;
* adding inference scripts for custom images and captions;
* creating a Streamlit or Gradio retrieval demo;
* comparing Mini-CLIP with pretrained CLIP and SigLIP models;
* supporting alternative image and text encoders;
* logging experiments with TensorBoard or Weights & Biases;
* reporting model size, inference time and GPU memory usage;
* adding Mean Reciprocal Rank and median rank;
* supporting multiple captions per image;
* adding unit tests and continuous integration.

---

## Educational Purpose

This project was developed as an educational implementation of multimodal contrastive learning.

It demonstrates:

* how pretrained vision and language models can be combined;
* how images and text can be mapped into a common vector space;
* how contrastive learning aligns matching multimodal samples;
* how cross-modal retrieval systems are evaluated.

The project is inspired by the core idea of CLIP but is not an official OpenAI implementation.

---

## References

* Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021.
* PyTorch documentation.
* Torchvision ResNet documentation.
* Hugging Face Transformers documentation.
* DistilBERT: *DistilBERT, a distilled version of BERT*.

---

**Nguyễn Huy Hoàng**

* GitHub: [AIVIETNAM-AIO-Nhhoang1207](https://github.com/AIVIETNAM-AIO-Nhhoang1207)
* Project demo: [YouTube](https://www.youtube.com/watch?v=unUhRajTKJU)

---

## License

This repository is provided for educational and research purposes.

Before using the project or dataset commercially, verify the licenses of the dataset, pretrained models and external dependencies.

---

<div align="center">

If this project is useful, consider giving the repository a star.

</div>

