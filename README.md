# Transfer Learning for Binary Forest Segmentation

Comparative study of U-Net (ResNet50 encoder) trained **from scratch** vs **with ImageNet transfer learning** for forest / non-forest segmentation.

## Dataset
- **Source:** [Augmented Forest Segmentation (Kaggle)](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation)
- **Task:** Binary segmentation (Forest = 1, Non-Forest = 0)
- **Input:** RGB image (256×256, resized to 128×128)
- **Output:** Binary mask
- **Size:** 5,108 image–mask pairs from 190 original pictures

## Model
- U-Net, ResNet50 encoder (`segmentation_models`), 1 output channel, sigmoid activation

## Experiments

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Encoder weights | Random (`None`) | ImageNet |
| Phase 1 | – | Encoder frozen, 50 epochs (LR 0.001) |
| Phase 2 | – | Full model unfrozen, 50 epochs (LR 0.0001) |
| Total epochs | 100 (LR 0.001) | 100 |

## Design Choices

| Item | Choice |
|---|---|
| Input size | 128 × 128 × 3 |
| Image normalization | Pixels ÷ 255 → [0, 1] |
| Mask processing | Grayscale, resized (nearest), pixel > 127 → 1 else 0 |
| Split | 70% train / 15% val / 15% test (3,632 / 851 / 625 images) |
| Split method | By original picture ID, so augmented copies of one picture never cross splits (no leakage) |
| Loss | Binary Cross-Entropy + Dice Loss |
| Optimizer | Adam |
| Batch size | 16 |
| Prediction threshold | 0.5 |
| Seed | 42 |
| Platform | Google Colab, T4 GPU, TensorFlow 2.20 |

## Results (Test Set)

| Model | IoU | F1 | Accuracy | Precision | Recall | Training Time |
|---|---|---|---|---|---|---|
| Exp 1: Scratch | 0.7148 | 0.8337 | 0.7900 | 0.8022 | 0.8677 | 49.6 min |
| Exp 2: Transfer Learning | **0.7349** | **0.8472** | **0.8079** | **0.8185** | **0.8779** | **45.0 min** |

## Analysis

### Performance
Transfer learning scored higher on every metric: IoU +0.0201, F1 +0.0135, accuracy +0.0179, precision +0.0163, recall +0.0102.

- ImageNet weights already detect edges, textures and colors that help separate forest from non-forest.
- The scratch model must learn all features from the training images alone. It reached ~0.98 train IoU but only ~0.71 validation IoU, which shows overfitting.
- The frozen phase lets the new random decoder train without damaging the pretrained encoder. Fine-tuning at a lower LR then adapts the encoder to forest imagery.

### Efficiency

| | Time |
|---|---|
| Scratch (100 epochs) | 49.6 min (~29 s/epoch) |
| Transfer, Phase 1 (frozen) | 20.3 min (~24 s/epoch) |
| Transfer, Phase 2 (fine-tune) | 24.7 min (~29 s/epoch) |
| Transfer total | 45.0 min |

- Transfer learning was **4.6 min faster** and **more accurate**.
- Frozen epochs are cheaper because no gradients are computed for the encoder.
- Fine-tune epochs cost the same as scratch epochs.
- Result: better performance at lower compute, so no real trade-off here.

## Visual Results
Side-by-side comparison on test samples: Original Image | Ground Truth | Scratch Prediction | Transfer Prediction (see Section 10 of the notebook).

## Files
```
├── 23L_2519_DLP_Assignment1.ipynb   # All experiments with outputs
└── README.md
```

## How to Run
1. Open the notebook in Google Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Run all cells in order. The dataset downloads automatically via `kagglehub`.

## Requirements
- `segmentation-models`, `tf-keras`, `kagglehub`
- TensorFlow, NumPy, Pandas, OpenCV, Matplotlib
