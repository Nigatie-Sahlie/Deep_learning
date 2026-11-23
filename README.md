# Deep Learning with PyTorch

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)

A practical PyTorch portfolio that documents the progression from tensor fundamentals to reusable computer-vision training pipelines, transfer learning, and experiment tracking.

This repository is designed to show not only how to train a model, but also how to organize machine-learning code into understandable, reusable components.

## What this project demonstrates

- PyTorch tensors, datatypes, shapes, indexing, and core operations
- End-to-end classification workflows: data preparation, training, evaluation, and model persistence
- Device-agnostic code that supports CPU and CUDA when available
- Computer-vision models built with convolutional neural networks
- Custom image datasets using `ImageFolder`, transforms, and `DataLoader`
- Modular training code separated into data, model, engine, evaluation, and utility modules
- Transfer learning with a pretrained `EfficientNet_B0`
- Repeatable model experiments with tracked metrics and checkpointed models
- Practical analysis of model performance, class imbalance, and misclassification

## Learning path

| Module | Focus | Evidence in the repository |
| --- | --- | --- |
| **M01 — Introduction** | PyTorch foundations, classification, CNNs, and custom datasets | [`M01_Introduction/notebooks`](M01_Introduction/notebooks) |
| **M02 — Modularity** | A reusable TinyVGG pipeline with separated responsibilities | [`M02_Modularity/going_modular`](M02_Modularity/going_modular) |
| **M03 — Transfer Learning** | Fine-tuning a pretrained EfficientNet model for image classification | [`M03_Transfer/transfer.ipynb`](M03_Transfer/transfer.ipynb) |
| **M04 — Experiment Tracker** | Comparing multiple training runs and selecting a stronger experiment | [`M04_Experiment_Tracker/main.ipynb`](M04_Experiment_Tracker/main.ipynb) |

## Technical highlights

### Reusable training architecture

The modular pipeline in `M02_Modularity` separates concerns into focused files:

- `data_setup.py` — dataset download, transforms, and data loaders
- `model_builder.py` — TinyVGG model definition
- `engine.py` — training and testing loops
- `evaluation.py` — prediction and evaluation helpers
- `utils.py` — model saving and loading utilities
- `main.py` — orchestration entry point

This structure makes the code easier to test, reuse, and extend than a single large notebook.

### Computer vision and transfer learning

The project applies CNN concepts to image classification and uses transfer learning with `EfficientNet_B0`. The transfer-learning workflow demonstrates how to reuse pretrained visual features and adapt a modern architecture to a custom multi-class problem.

### Experiment-driven improvement

The experiment-tracking module compares eight training configurations, records performance over epochs, and identifies a stronger run for further prediction and evaluation. The notebook documents an experiment reaching approximately 90% training accuracy and above 90% test accuracy during its reported run.

## Reported results

Results are documented in the notebooks and may vary with hardware, random seeds, dataset versions, and training settings.

- FashionMNIST CNN experiments report accuracy, precision, recall, and F1 scores of approximately 88% for the stronger model.
- The custom food-image workflow uses three classes: **pizza**, **steak**, and **sushi**.
- The experiment-tracking notebook identifies a best-performing run with test accuracy above 90% during the recorded experiments.
- The notebooks also inspect failure cases, including confusion between visually similar food classes.

## Repository structure

```text
Deep_learning/
├── M01_Introduction/
│   ├── notebooks/           # Guided PyTorch and computer-vision studies
│   └── src/                 # Small reusable helpers for custom data
├── M02_Modularity/
│   ├── going_modular/       # Data, model, engine, evaluation, and utilities
│   ├── index.ipynb          # Walkthrough and visual analysis
│   └── main.py              # Modular pipeline entry point
├── M03_Transfer/
│   ├── transfer.ipynb       # EfficientNet transfer-learning workflow
│   └── script.py            # Supporting script
├── M04_Experiment_Tracker/
│   ├── main.ipynb           # Experiment comparison and tracking
│   └── src/                 # Training and helper functions
├── requirements.txt
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Nigatie-Sahlie/Deep_learning.git
cd Deep_learning
```

### 2. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Install the project's Python dependencies:

```bash
pip install -r requirements.txt
```

For a new machine, install a compatible PyTorch and torchvision build from the [official PyTorch selector](https://pytorch.org/get-started/locally/) if the pinned environment packages are not suitable for your operating system.

### 3. Run the notebooks

```bash
pip install jupyter
jupyter notebook
```

Start with `M01_Introduction`, then follow the modules in order. The notebooks download or prepare their datasets as described in each module.

### 4. Run the modular pipeline

```bash
python M02_Modularity/main.py
```

The pipeline automatically uses CUDA when available and otherwise falls back to CPU.

## Engineering practices shown

- Clear separation between data preparation, model definition, training, and evaluation
- Device-aware PyTorch code
- Reusable functions instead of repeating training logic in every notebook
- Model checkpointing with `state_dict`
- Data augmentation and preprocessing for image inputs
- Evaluation beyond loss alone, including accuracy and classification metrics
- Explicit inspection of model errors rather than reporting only the best score

## Current scope and next improvements

This is an educational and portfolio project rather than a production deployment. Natural next steps would be:

- Add automated tests for data, training, and evaluation helpers
- Replace the environment snapshot in `requirements.txt` with a minimal, platform-neutral dependency file
- Add fixed random seeds and a formal experiment configuration
- Add confusion matrices and a consistent metrics report
- Add a small inference API or command-line prediction interface
- Add continuous integration for linting and notebook validation

## Why it is relevant to a machine-learning role

This project shows a progression from understanding the framework fundamentals to building maintainable training workflows. It demonstrates practical skills in PyTorch, computer vision, transfer learning, experiment comparison, model evaluation, and communicating technical results through reproducible notebooks and modular source code.

## Author

**Nigatie Sahlie**  
GitHub: [@Nigatie-Sahlie](https://github.com/Nigatie-Sahlie)
