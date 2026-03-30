# QSE-Enhanced CNN for Indian Sign Language Recognition

> **"QSE-Enhanced CNN Architecture for Efficient Indian Sign Language Recognition"**  
> Aditya Mohanty · Amrit Kumar Mishra · Nikhil Singh · Sourabh Paul  
> School of Electronics Engineering (SENSE), VIT Chennai  
> *Submitted to IEEE ACCESS*

---

## 📌 Overview

This repository accompanies the research paper introducing **QSE-CNN** — a novel deep learning architecture that fuses a **Quantum Squeeze-and-Excitation (QSE)** attention block with a classical CNN backbone for robust Indian Sign Language (ISL) recognition.

Standard CNNs treat all feature channels equally, limiting their ability to adaptively focus on salient hand gesture features — especially under noisy, real-world conditions. Classical attention mechanisms (SE, CBAM, ECA, CoordAtt, Triple Attention) partially address this, but introduce high parameter overhead and restricted representational capacity.

**QSE-CNN replaces** the first fully connected layer of the classical SE block with a **parameterized quantum circuit** using amplitude embedding and CNOT gates — leveraging quantum superposition and entanglement for efficient, globally correlated channel recalibration.

---

## ✨ Key Contributions

1. **First quantum-enhanced architecture for ISL recognition** — integrating quantum mechanical principles with deep learning for the Indian Sign Language domain.
2. **QSE Block design** — the classical SE excitation MLP is replaced by a quantum circuit that performs amplitude-encoded, entanglement-driven dimensionality reduction, yielding globally correlated channel descriptors with fewer parameters.
3. **Comprehensive benchmarking** across three dataset difficulty levels (Clean, Moderately Complex, Extremely Complex) against four competitive baselines.
4. Demonstrates that QSE-CNN achieves **superior noise resilience** and recognition accuracy versus all classical attention counterparts.

---

## 🏗️ Architecture

The QSE-CNN pipeline consists of four stages:

```
Input Image (224×224 RGB)
        │
        ▼
┌─────────────────────┐
│  Data Preprocessing │  ← Geometric + Photometric Augmentation, Random Erasing
└─────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│       CNN Feature Extraction Backbone    │
│                                          │
│  Block 1: Conv–BN–ReLU  (32 ch)  + QSE  │
│  Block 2: Conv–BN–ReLU  (64 ch)  + QSE  │
│  Block 3: Conv–BN–ReLU  (128 ch) + QSE  │
│  Block 4: Conv–BN–ReLU  (256 ch) + QSE  │
└──────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                  QSE Block (per stage)                        │
│                                                               │
│  Global Avg Pool → Amplitude Embedding → Quantum Circuit      │
│  (CNOT gates / entanglement) → Classical FC → Sigmoid → Scale│
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────┐
│   Classifier     │  ← Fully Connected → Softmax (26 ISL classes)
└──────────────────┘
```

Each CNN block uses **3×3 convolutions** followed by Batch Normalization and ReLU. The QSE block after each stage recalibrates channel-wise feature responses using quantum-inspired operations.

---

## 🔬 The QSE Block

The Quantum Squeeze-and-Excitation block extends the classical SE block (Hu et al., CVPR 2018) by replacing the first FC layer with a quantum circuit:

| Component | Classical SE | QSE (Ours) |
|---|---|---|
| Squeeze | Global Average Pooling | Global Average Pooling |
| Excitation (Step 1) | FC layer (linear) | Quantum circuit with amplitude embedding + CNOT entanglement |
| Excitation (Step 2) | FC layer + Sigmoid | Classical FC + Sigmoid |
| Channel scaling | Element-wise multiplication | Element-wise multiplication |

**Quantum circuit design:**
- **Amplitude Embedding** encodes channel-squeezed feature vectors into quantum state amplitudes
- **CNOT Gates** introduce entanglement across channels, capturing non-linear inter-channel dependencies
- Achieves dimensionality reduction with **fewer parameters** than classical FC alternatives

*Inspired by: Peng et al., "Quantum Squeeze-and-Excitation Networks," IEEE QCE 2024.*

---

## 📊 Experiments

### Dataset

The **Indian Sign Language (ISL)** dataset covers all **26 alphabets (A–Z)** as static hand gesture images.  
Source: [Kaggle — Indian Sign Language ISL](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)

Three evaluation conditions are tested:

| Dataset | Description |
|---|---|
| **Clean** | Uniform lighting, centered gestures, controlled indoor background |
| **Moderately Complex** | Brightness variation, mild camera noise, slight pose/orientation shifts |
| **Extremely Complex** | Motion blur, salt-and-pepper noise (2%), severe lighting variation, background clutter |

### Baselines

| Model | Attention |
|---|---|
| Standard CNN | None |
| CNN + SE | Squeeze-and-Excitation only |
| CNN + SE + CBAM | SE + Convolutional Block Attention Module |
| CNN + ECA + CoordAtt + Triple Attention | Multi-attention ensemble |
| **CNN + QSE (Ours)** | **Quantum Squeeze-and-Excitation** |

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Input resolution | 224 × 224 |
| Loss function | CrossEntropy with label smoothing (ε = 0.1) |
| Optimizer | AdamW (Adam + decoupled weight decay) |
| LR Scheduler | Cosine Annealing Warm Restarts |
| Gradient clipping | Norm limit = 1.0 |
| Early stopping | Patience = 25 epochs |

### Evaluation Metrics

Macro-averaged metrics are used across all 26 classes: **Accuracy (%)**, **Macro Precision**, **Macro Recall**, and **Macro F1 Score**.

---

## 📈 Results

All models achieve near-perfect accuracy (~100%) on the **Clean** dataset, so the meaningful comparison is on the noisy conditions.

### Table 3 — Quantitative Performance Across All Models

| Model | Dataset | Accuracy (%) | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|---|
| Standard CNN | Moderate | 82.7 | 0.86 | 0.83 | 0.83 |
| Standard CNN | Extreme | 31.6 | 0.46 | 0.33 | 0.30 |
| SE + CBAM | Moderate | 85.4 | 0.87 | 0.85 | 0.85 |
| SE + CBAM | Extreme | 34.2 | 0.60 | 0.34 | 0.37 |
| ECA + CoordAtt + Triple Attention | Moderate | 86.5 | 0.87 | 0.86 | 0.85 |
| ECA + CoordAtt + Triple Attention | Extreme | 41.7 | 0.64 | 0.42 | 0.42 |
| SE + CNN | Moderate | 88.5 | 0.91 | 0.88 | 0.88 |
| SE + CNN | Extreme | 43.4 | 0.61 | 0.44 | 0.45 |
| **QSE-CNN (Ours)** | **Moderate** | **90.8** | **0.90** | **0.91** | **0.90** |
| **QSE-CNN (Ours)** | **Extreme** | **44.3** | **0.56** | **0.44** | **0.43** |

### Table 4 — Improvement over Baselines on Extreme Dataset

| Baseline Model | Baseline Accuracy (%) | QSE-CNN Accuracy (%) | Improvement |
|---|---|---|---|
| Standard CNN | 31.6 | 44.3 | **+40.19%** |
| SE + CBAM | 34.2 | 44.3 | **+29.53%** |
| ECA + CoordAtt + TripleAtt | 41.7 | 44.3 | **+6.23%** |
| SE + CNN | 43.4 | 44.3 | **+2.07%** |

### Table 5 — Improvement over Baselines on Moderate Dataset

| Baseline Model | Baseline Accuracy (%) | QSE-CNN Accuracy (%) | Improvement |
|---|---|---|---|
| Standard CNN | 82.7 | 90.8 | **+9.78%** |
| SE + CBAM | 85.4 | 90.8 | **+6.32%** |
| ECA + CoordAtt + TripleAtt | 86.2 | 90.8 | **+5.34%** |
| SE + CNN | 88.5 | 90.8 | **+2.60%** |

---

## 🔬 Ablation Study

### Table 6 — Effect of Dropout Rate (QSE reduction ratio fixed at 16)

| Dropout Rate | QSE-CNN Accuracy (%) |
|---|---|
| 0.0 | 85.19 |
| **0.3** | **90.80** ✅ |
| 0.5 | 85.38 |

### Table 7 — Effect of QSE Reduction Ratio (dropout fixed at 0.3)

| Reduction Ratio (r) | QSE-CNN Accuracy (%) |
|---|---|
| **16** | **90.80** ✅ |
| 8 | 86.92 |
| 4 | 85.00 |

### Table 9 — Effect of QSE Block Placement

| QSE Placement | Description | Accuracy (%) |
|---|---|---|
| Shallow QSE | QSE after Block 1 only | 88.46 |
| Deep QSE | QSE after Block 4 only | 87.31 |
| **Full QSE-CNN** | QSE after all conv blocks | **90.80** ✅ |

> The ablation confirms that applying QSE at **every convolutional stage** (shallow + deep features) is critical — partial placement significantly degrades performance.

---

## 🛠️ Setup & Usage

### Requirements

```bash
pip install torch torchvision pennylane numpy scikit-learn pillow
```

> PennyLane (or equivalent quantum ML library) is required for the quantum circuit components.

### Data Preparation

1. Download the ISL dataset from [Kaggle](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)
2. Organize into `train/` and `val/` folders with one subfolder per class (A–Z)

```
data/
├── train/
│   ├── A/
│   ├── B/
│   └── ...
└── val/
    ├── A/
    ├── B/
    └── ...
```

### Training

```bash
python train.py --data_dir ./data --epochs 100 --batch_size 32 --lr 1e-3
```

### Evaluation

```bash
python evaluate.py --data_dir ./data/val --checkpoint ./checkpoints/best_model.pt
```

---

## 📂 Repository Structure

```
├── models/
│   ├── qse_block.py        # Quantum Squeeze-and-Excitation block
│   ├── cnn_backbone.py     # CNN feature extraction stages
│   └── qse_cnn.py          # Full QSE-CNN architecture
├── data/
│   ├── augmentation.py     # Data augmentation pipeline
│   └── dataset.py          # ISL dataset loader
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── utils/
│   └── metrics.py          # Macro precision, recall, F1
├── checkpoints/            # Saved model weights
└── README.md
```

---

## 👥 Authors

| Name | Affiliation |
|---|---|
| Aditya Mohanty | B.Tech ECE, VIT Chennai (2026) |
| Amrit Kumar Mishra | B.Tech ECE, VIT Chennai |
| Nikhil Singh | B.Tech ECE, VIT Chennai |
| **Sourabh Paul** *(Corresponding)* | Associate Professor, SENSE, VIT Chennai — [sourabhpaul26@gmail.com](mailto:sourabhpaul26@gmail.com) |

**Dr. Sourabh Paul** received his Ph.D. from NIT Rourkela (2019) and his M.Tech from NIT Agartala (2014). His research spans image registration, remote sensing image processing, and edge detection.

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@article{mohanty2024qse,
  title     = {QSE-Enhanced CNN Architecture for Efficient Indian Sign Language Recognition},
  author    = {Mohanty, Aditya and Mishra, Amrit Kumar and Singh, Nikhil and Paul, Sourabh},
  journal   = {IEEE Access},
  year      = {2024},
  doi       = {10.1109/ACCESS.2022.Doi Number},
  note      = {School of Electronics Engineering (SENSE), VIT Chennai}
}
```

---

## 🔗 References

Key works this paper builds on:

- Hu et al., "Squeeze-and-Excitation Networks," CVPR 2018
- Woo et al., "CBAM: Convolutional Block Attention Module," ECCV 2018
- Wang et al., "ECA-Net: Efficient Channel Attention," CVPR 2020
- Hou et al., "Coordinate Attention for Efficient Mobile Network Design," CVPR 2021
- Misra et al., "Rotate to Attend: Convolutional Triplet Attention Module," WACV 2021
- Peng et al., "Quantum Squeeze-and-Excitation Networks," IEEE QCE 2024

---

## 📄 License

This project is released for academic and research purposes. Please contact the corresponding author for commercial use inquiries.
