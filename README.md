# Transfer Learning for Binary Forest Segmentation

Comparative study: training U-Net (ResNet50 encoder) **from scratch** vs **transfer learning** for forest / non-forest segmentation.

## Dataset
- **Source:** [Augmented Forest Segmentation (Kaggle)](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation)
- **Task:** Binary segmentation (Forest = 1, Non-Forest = 0)
- **Input:** RGB satellite image
- **Output:** Binary mask
- **Structure:** metadata CSV (`image`, `mask` columns), `images/` folder, `masks/` folder

## Model
- U-Net with ResNet50 encoder (`segmentation_models`)
- Output: 1 channel, sigmoid activation

## Experiments

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Encoder weights | Random (`None`) | ImageNet |
| Phase 1 | – | Encoder frozen, 50 epochs |
| Phase 2 | – | Full model unfrozen, 50 epochs |
| Total epochs | 100 | 100 |

## Setup
| Item | Choice |
|---|---|
| Image size | 128×128 |
| Normalization | Pixels scaled to [0, 1] |
| Loss | BCE + Dice |
| Optimizer | Adam (lr = `<LR>`) |
| Batch size | `<BATCH_SIZE>` |
| Train / Val / Test split | `<e.g. 70 / 15 / 15>` |
| Seed | `<SEED>` |
| Platform | Google Colab (`<GPU/TPU>`) |

## Results (Test Set)

| Model | IoU | F1 | Accuracy | Precision | Recall | Training Time |
|---|---|---|---|---|---|---|
| Exp 1: Scratch | `<>` | `<>` | `<>` | `<>` | `<>` | `<>` |
| Exp 2: Transfer Learning | `<>` | `<>` | `<>` | `<>` | `<>` | `<>` |

## Analysis

**Performance:** `<Which model scored higher on IoU and F1, and why. E.g. pretrained ImageNet features give better low-level edge/texture filters, faster convergence, less overfitting.>`

**Efficiency:** `<Compare total training time of both runs. Note the frozen phase is faster per epoch (fewer trainable params). Discuss gain vs compute.>`

## Visual Results

Side-by-side: Original Image | Ground Truth | Scratch Prediction | Transfer Learning Prediction

![Visual results](results/visual_comparison.png)

## Files
```
├── notebook.ipynb          # All experiments with outputs
├── README.md
└── results/
    └── visual_comparison.png
```

## How to Run
1. Open `notebook.ipynb` in Google Colab.
2. Download the dataset from Kaggle (or mount it) and set the data path.
3. Install dependencies:
   ```bash
   pip install segmentation-models
   ```
4. Run all cells in order.

## Requirements
- Python 3.x
- TensorFlow / Keras
- segmentation_models
- NumPy, Pandas, OpenCV, Matplotlib, scikit-learn
