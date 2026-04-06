<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dvc&logoColor=white" alt="DVC"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
</p>

# 🔬 Vision Transformer (ViT) — From Scratch

A **complete, from-scratch implementation** of the [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) paper (*"An Image is Worth 16x16 Words"*) using PyTorch, trained and evaluated on the **CIFAR-10** dataset. This project follows a modular, production-grade ML pipeline architecture with configuration management, structured logging, data versioning, and a Streamlit-based inference UI.

---

## 📑 Table of Contents

- [Highlights](#-highlights)
- [Architecture Overview](#-architecture-overview)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Model Configuration](#%EF%B8%8F-model-configuration)
- [Pipeline Stages](#-pipeline-stages)
- [Research Notebooks](#-research-notebooks)
- [Tech Stack](#-tech-stack)
- [License](#-license)
- [Author](#-author)

---

## ✨ Highlights

- **Pure PyTorch implementation** — no pretrained weights, no HuggingFace wrappers. Every layer is hand-built.
- **Modular pipeline architecture** — clean separation of data ingestion, validation, transformation, training, evaluation, and prediction.
- **Configuration-driven** — all hyperparameters and paths managed through YAML files (`config.yaml`, `params.yaml`).
- **Structured logging** — every pipeline step is logged to both console and `logs/running_logs.log`.
- **Streamlit app** ready for interactive image classification inference.
- **DVC integration** for data pipeline reproducibility.
- **Docker-ready** project scaffold.

---

## 🧠 Architecture Overview

The Vision Transformer re-imagines image classification by treating an image as a **sequence of patches**, much like words in a sentence, and processing them with a standard Transformer encoder.

```
Input Image (32×32×3)
        │
        ▼
┌─────────────────────┐
│   Patch Embedding    │  Split image into 4×4 patches → 64 patches
│   (Conv2d projection)│  Project each patch to 256-dim embedding
│   + CLS Token        │  Prepend a learnable [CLS] token → 65 tokens
│   + Position Encoding│  Add learnable positional embeddings
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Transformer Encoder│  ×8 layers (depth)
│ ┌─────────────────┐ │
│ │  Layer Norm     │ │
│ │  Multi-Head Attn│ │  8 attention heads
│ │  + Residual     │ │
│ │  Layer Norm     │ │
│ │  MLP (GELU)     │ │  256 → 768 → 256
│ │  + Residual     │ │
│ └─────────────────┘ │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│     Layer Norm      │
│  Classification Head│  [CLS] token → 10 class logits
└─────────────────────┘
        │
        ▼
   Predictions (10 classes)
   airplane | automobile | bird | cat | deer
   dog | frog | horse | ship | truck
```

### Key Components

| Module | File | Description |
|--------|------|-------------|
| **Patch Embedding** | `Patch_Embedding.py` | Splits image into patches via `Conv2d`, adds CLS token & positional encoding |
| **MLP** | `MLP.py` | Two-layer feed-forward network with GELU activation & dropout |
| **Transformer Encoder** | `Transformer_EncoderLayer.py` | Pre-norm encoder block with Multi-Head Self-Attention, MLP, & residual connections |
| **Vision Transformer** | `Vision_Transformer_Class.py` | Full model: patch embed → N× encoder → layer norm → classification head |

---

## 📁 Project Structure

```
Vision_Transformer_from_Scratch/
│
├── config/
│   └── config.yaml                 # Paths & artifact directories
├── params.yaml                     # Model hyperparameters
│
├── src/vision_Transformer/
│   ├── Components/
│   │   ├── ViT_Component/
│   │   │   ├── Patch_Embedding.py          # Patch + positional embedding
│   │   │   ├── MLP.py                      # Feed-forward network
│   │   │   ├── Transformer_EncoderLayer.py # Encoder block
│   │   │   └── Vision_Transformer_Class.py # Full ViT model
│   │   ├── data_ingestion.py       # CIFAR-10 download
│   │   ├── data_validation.py      # Dataset integrity checks
│   │   ├── data_transformation.py  # Augmentation & normalization
│   │   ├── model_training.py       # Training loop
│   │   ├── model_evaluation.py     # Evaluation metrics
│   │   └── prediction_pipeline.py  # Inference logic
│   ├── Entity/
│   │   └── __init__.py             # Dataclass configs for each stage
│   ├── config/
│   │   └── configuration.py        # ConfigurationManager
│   ├── constants/
│   │   └── __init__.py             # File path constants
│   ├── logging/
│   │   └── __init__.py             # Structured logger setup
│   ├── pipeline/
│   │   ├── Stage_01_data_ingestion.py
│   │   ├── Stage_02_data_validation.py
│   │   ├── Stage_03_data_transformation.py
│   │   ├── Stage_04_model_training.py
│   │   ├── Stage_05_model_evaluation.py
│   │   └── Stage_06_prediction_pipeline.py
│   └── utils/
│       └── common.py               # YAML reader, directory creator, etc.
│
├── research/                       # Jupyter notebooks for experimentation
│   ├── ViT_From_Scratch.ipynb
│   ├── ViT_From_Scratch_final.ipynb
│   ├── Stage_01_data_ingestion.ipynb
│   ├── Stage_02_data_validation.ipynb
│   ├── Stage_03_data_transformation.ipynb
│   ├── Stage_04_Model_Training.ipynb
│   ├── Stage_05_Model_Evaluation.ipynb
│   └── Stage_07_Prediction_Pipeline.ipynb
│
├── model/                          # Saved model weights
│   ├── complete_model.pth
│   └── model_weights.pth
│
├── artifacts/                      # Pipeline stage outputs
├── logs/                           # Runtime logs
├── test/                           # Test dataset (CIFAR-10)
│
├── app.py                          # Streamlit inference app
├── main.py                         # Pipeline orchestrator
├── template.py                     # Project scaffolding script
├── Dockerfile                      # Container definition
├── dvc.yaml                        # DVC pipeline definition
├── pyproject.toml                  # Project metadata & dependencies
├── requirements.txt                # pip dependencies
└── LICENSE                         # MIT License
```

---

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.9
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Priyanshu1303d/Vision_Transformer_from_Scratch.git
   cd Vision_Transformer_from_Scratch
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > This also installs the `vision_Transformer` package in editable mode via `-e .`

### Running the Pipeline

```bash
python main.py
```

### Running the Streamlit App

```bash
streamlit run app.py
```

---

## ⚙️ Model Configuration

All hyperparameters are centralized in [`params.yaml`](params.yaml):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BATCH_SIZE` | 128 | Training batch size |
| `EPOCHS` | 25 | Number of training epochs |
| `LEARNING_RATE` | 3e-4 | AdamW learning rate |
| `PATCH_SIZE` | 4 | Patch dimensions (4×4 pixels) |
| `NUM_CLASSES` | 10 | CIFAR-10 classes |
| `IMAGE_SIZE` | 32 | Input image resolution |
| `CHANNELS` | 3 | RGB channels |
| `EMBED_DIM` | 256 | Transformer embedding dimension |
| `NUM_HEADS` | 8 | Multi-head attention heads |
| `DEPTH` | 8 | Number of Transformer encoder layers |
| `MLP_DIM` | 768 | MLP hidden dimension |
| `DROPOUT_RATE` | 0.1 | Dropout probability |
| `WEIGHT_DECAY` | 0.01 | AdamW weight decay |

> A separate `HyperParameterTuning` profile with 60 epochs is also available in `params.yaml`.

---

## 🔄 Pipeline Stages

The project follows a **staged ML pipeline** pattern, where each stage is independently configurable and executable:

| Stage | Name | Description |
|-------|------|-------------|
| **01** | **Data Ingestion** | Downloads CIFAR-10 train (50,000) and test (10,000) datasets via `torchvision.datasets` |
| **02** | **Data Validation** | Verifies dataset integrity — checks sample counts (50K/10K) and class counts (10) |
| **03** | **Data Transformation** | Applies data augmentation: `RandomCrop`, `RandomHorizontalFlip`, `ColorJitter`, normalization |
| **04** | **Model Training** | Trains the ViT model with the configured hyperparameters |
| **05** | **Model Evaluation** | Evaluates trained model on the test set, computes accuracy and loss metrics |
| **06** | **Prediction Pipeline** | Loads saved model for single-image inference |

### Data Augmentation Pipeline

```python
transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

---

## 📓 Research Notebooks

The `research/` directory contains Jupyter notebooks documenting the full development process:

| Notebook | Purpose |
|----------|---------|
| `ViT_From_Scratch.ipynb` | Initial end-to-end ViT implementation & experimentation |
| `ViT_From_Scratch_final.ipynb` | Refined final version of the model |
| `Stage_01–07` notebooks | Per-stage prototyping before modularization into the `src/` package |

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch, TorchVision |
| **Data** | CIFAR-10, NumPy, Matplotlib |
| **MLOps** | DVC, MLflow |
| **App** | Streamlit |
| **Tooling** | Black, isort, python-box, PyYAML |
| **Packaging** | setuptools, pyproject.toml |
| **Infra** | Docker |

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Priyanshu Kumar Singh**

- GitHub: [@Priyanshu1303d](https://github.com/Priyanshu1303d)

---

<p align="center">
  <i>⭐ If you found this project helpful, consider giving it a star!</i>
</p>