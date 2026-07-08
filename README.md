# multimodal-rental-price-prediction

**Multimodal Deep Learning · Regression · Real Estate · TensorFlow / Keras**

Predicting monthly apartment rental prices in Istanbul using a **late-fusion multimodal neural network** that combines property listing photos (EfficientNetB0) with structured tabular features.

---

## Results

| Model | MAE ↓ | RMSE ↓ | MAPE ↓ | R² ↑ |
|-------|------:|-------:|-------:|-----:|
| M1: Tabular MLP (baseline) | 7,247 TL | 9,340 TL | 24.01% | 0.3726 |
| **M2: Late Fusion - EfficientNetB0 ★** | **6,626 TL** | **8,654 TL** | **21.42%** | **0.4614** |

**Multimodal model vs. tabular baseline: −8.6% MAE · −10.8% MAPE · +0.089 R²**

---

## Architecture

### M1 - Tabular MLP (Baseline)

```
Input(41) → Dense(256, BN, ReLU, Dropout)
          → Dense(128, BN, ReLU, Dropout)
          → Dense(64,  BN, ReLU, Dropout)
          → Dense(1)
```

### M2 - Late Fusion Multimodal

```
Image Branch:
  ImageInput(224×224×3) → EfficientNetB0 (ImageNet weights)
                        → GlobalAveragePooling2D
                        → Dropout(0.3) → Dense(256, BN, ReLU)

Tabular Branch:
  TabInput(41) → Dense(128, BN, ReLU) → Dropout(0.3) → Dense(64, BN, ReLU)

Fusion Head:
  Concatenate([image, tabular]) → Dense(128, BN, ReLU) → Dropout(0.3) → Dense(1)
```

**Two-phase training strategy:**

| Phase | Backbone | LR Schedule | Epochs |
|-------|----------|-------------|--------|
| Phase 1 - warm-up | Frozen | Adam(1e-3) | 15 |
| Phase 2 - fine-tune | Unfrozen | CosineDecay(5e-4 → 0) | 15 |

---

## Experiments

### M2: Architecture & Training Strategy Ablation

| Config | Backbone | Fine-tune | MAE ↓ | MAPE ↓ | R² ↑ |
|--------|----------|:---------:|------:|-------:|-----:|
| A | MobileNetV2 | ✓ | 6,903 TL | 22.64% | 0.4190 |
| B | EfficientNetB0 | ✗ | 7,485 TL | 22.76% | 0.2968 |
| **C ★** | **EfficientNetB0** | **✓** | **6,626 TL** | **21.42%** | **0.4614** |

> Config B (no fine-tuning) underperforms the tabular-only baseline on R² → Phase 2 is critical.

### M1: Dropout Hyperparameter Sweep

| Dropout | MAE ↓ | MAPE ↓ | R² ↑ |
|--------:|------:|-------:|-----:|
| 0.2 | 7,418 TL | 24.72% | 0.3499 |
| **0.3 ★** | **7,247 TL** | **24.01%** | **0.3726** |
| 0.4 | 7,333 TL | 24.20% | 0.3604 |

All configurations within <2.4% MAE - model is robust to dropout in the [0.2, 0.4] range. Config 0.3 selected as default.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | sahibinden.com (web-scraped, May 2026) |
| Size | 22,543 apartment rental listings |
| Location | Istanbul, Turkey |
| Modalities | 1 photo per listing + structured features |
| Split | 70 / 15 / 15 - Train: 15,780 · Val: 3,381 · Test: 3,382 |
| Features | 41 total: scaled numeric (area, rooms) + OHE for 39 İstanbul ilçe |
| Target | `log1p(price_TL)` - log-transform handles right-skewed distribution |

**Tabular data** (`data_clean.csv`) is included in this repository.
**Property photos** (`images.zip`) are available via [Google Drive](https://drive.google.com/file/d/1BB_nAuOnFCTSEVzSRdwIrTl8QjMyjR7L/view?usp=sharing) - not redistributed on GitHub due to size and ToS constraints.

> *Data collected for academic research purposes only.*

---

## Getting Started

### Run on Google Colab (recommended - GPU required for Notebook 3)

| Notebook | Description |
|----------|-------------|
| [`1_eda_preprocessing.ipynb`](notebooks/1_eda_preprocessing.ipynb) | EDA, data cleaning, feature engineering |
| [`2_baseline_mlp.ipynb`](notebooks/2_baseline_mlp.ipynb) | M1: Tabular MLP, dropout sweep |
| [`3_multimodal.ipynb`](notebooks/3_multimodal.ipynb) | M2: Late Fusion, 3-config ablation |


---

## Project Structure

```
multimodal-rental-price-prediction/
├── notebooks/
│   ├── 1_eda_preprocessing.ipynb
│   ├── 2_baseline_mlp.ipynb
│   └── 3_multimodal.ipynb
├── data/
│   └── data_clean.csv                       # processed tabular data used for training
├── report/
│   └── DeepLearning_FinalReport.pdf
├── requirements.txt
└── README.md
```

---

## Key Techniques

- **Late Fusion** multimodal architecture - image and tabular branches trained independently, fused before the output head
- **Two-phase training** - frozen backbone, CosineDecay fine-tuning (prevents catastrophic forgetting)
- **Huber Loss** (δ=1.0) - robust regression loss for heavy-tailed price distribution
- **`log1p` target transform** - stabilizes training on right-skewed rental prices
- `use_bias=False` + **BatchNormalization** pattern (BN absorbs the bias term)
- **OHE district encoding** with rare-category grouping (< 10 samples → `Other`)
- Custom `keras.utils.Sequence` for multi-modal batch generation (dict keys: `image_input`, `tabular_input`)

---

## Author

Aleksandra Kolmogorova · Özyeğin University · Deep Learning course, 2025–2026

