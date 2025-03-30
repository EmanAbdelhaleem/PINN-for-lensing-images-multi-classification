![image](https://github.com/user-attachments/assets/31c65cd5-840d-4924-b40d-011f23c9553d)


# DeepLense Classification Task

Welcome! This repository contains my work on the ML4SCI DeepLense classification challenge, where I classify gravitational lensing images into three categories:

- **No Substructure**
- **Subhalo**
- **Vortex**

I tackled two tasks:
1. **BaselineCNN** – A standard CNN-based model without any physics enhancements.
2. **LensVIT_PINN** – A physics-informed neural network (PINN) that integrates the gravitational lensing equation to improve classification accuracy.

## Overview
This repository includes two key models:

- **`BaselineCNN`**: A ResNet18-based classifier designed for traditional image classification.
- **`LensVIT_PINN`**: A hybrid model that combines Vision Transformers (ViT) and ResNet, incorporating the lensing equation for better physics-aware predictions.

Both models were trained on a dataset of **37,500 simulated images** (30,000 for training, 7,500 for validation) with the goal of achieving state-of-the-art performance.

## Repository Contents
- **`LensVitPINN.ipynb`** – The main notebook containing training and evaluation code for both models.
- **`BaselineCNN_Weights.pth`** – Pre-trained weights for the best BaselineCNN model.
- **`LensVIT_PINN_Weights.pth`** – Pre-trained weights for the best LensVIT_PINN model.

## Setup & Requirements
1. **Download the Dataset**: Get `dataset.zip` from ML4SCI and extract it into `train/` and `val/` directories.
2. **Install Dependencies**:
   ```bash
   pip install torch torchvision numpy scikit-learn matplotlib
   ```
3. **Run the Notebook**:
   - Open `LensVitPINN.ipynb` in Jupyter Notebook.
   - A GPU is recommended for training, but CPU will work (slowly).
   - The notebook allows training from scratch or loading pre-trained weights for evaluation.

---

## Model Architectures
### 1. BaselineCNN
- **Backbone**: ResNet18, pre-trained on ImageNet.
- **Modifications**:
  - Input adapted for single-channel images (`conv1` modified to `nn.Conv2d(1, 64, ...)`).
  - Output layer changed to classify three categories (`nn.Linear(512, 3)`).
- **Purpose**: A benchmark model to assess performance without physics-based enhancements.

### 2. LensVIT_PINN
A hybrid model that incorporates physics-based lensing information.

- **ViT Encoder**:
  - Uses DINO ViT-S/16, pre-trained.
  - Modified to accept single-channel input (`patch_embed.proj` adjusted to `nn.Conv2d(1, 384, ...)`).
  - Outputs 384-dimensional patch embeddings.
- **Physics Module**:
  - A linear layer (`nn.Linear(384, 150*150*2)`) predicts deflection angles (`alpha`).
  - Applies the **gravitational lensing equation** to compute source positions (`theta_S`).
  - Equation: `theta_S = theta_I - alpha_sis`, where:
    - `theta_E` (Einstein radius) is estimated as the mean of `alpha`.
    - `theta` (norm of `theta_I`) is clamped between `[0, 149]` to avoid out-of-bounds errors.
- **ResNet Classifier**:
  - Modified ResNet18 with a 2-channel input (`conv1` updated to `nn.Conv2d(2, 64, ...)`).
  - Takes both observed (`theta_I`) and reconstructed (`theta_S`) images as input.
  - Outputs final classification logits.

---

## Training Details
### BaselineCNN Training
- **Loss Function**: Cross-entropy loss (`nn.CrossEntropyLoss()`)
- **Optimizer**: AdamW (`lr=1e-4`)
- **Data Augmentation**: Random rotation (±10°)
- **Training Setup**:
  - **Epochs**: 50
  - **Batch Size**: 32
  - **Best Weights** saved based on validation AUC.

### LensVIT_PINN Training
- **Loss Function**: Combined cross-entropy (`ce_loss`) + physics loss (`nn.MSELoss(theta_S, images - alpha_reshaped)`, weighted 0.01)
- **Optimizer**: AdamW (`lr=2e-4`)
- **Data Augmentation**: Random rotation (±10°)
- **Training Setup**:
  - **Epochs**: 50
  - **Batch Size**: 32
  - **Best Weights** saved based on validation AUC.

---

## Results
| Model           | No Substructure (AUC) | Subhalo (AUC) | Vortex (AUC) | Mean AUC |
|-----------------|-----------------------|---------------|--------------|----------|
| **BaselineCNN** | 0.9926                | 0.9810        | 0.9933       | 0.9890   |
| **LensVIT_PINN**| 0.9902                | 0.9804        | 0.9932       | 0.9879   |

- **BaselineCNN**: Performs exceptionally well, setting a strong benchmark.
- **LensVIT_PINN**: Matches closely, with minor trade-offs in AUC scores.

#### Physics-Informed Trials
- **Initial Attempt**: Used additional gradient preprocessing (`physics_preprocess`), but this **reduced AUC to 0.8786** due to excessive noise.
- **Final Approach**: Removed preprocessing and relied purely on the gravitational lensing equation, leading to more stable results.

---

## Running the Models
- **Hardware**: Models were trained on a GPU (CUDA), but they can run on a CPU (slower).
- **Dataset Split**: 80:20 (Train:Validation) from 37,500 images.
- **To Run**:
  - Open `LensVitPINN.ipynb` and execute all cells.
  - Load pre-trained weights (`BaselineCNN_Weights.pth`, `LensVIT_PINN_Weights.pth`) for evaluation.

---

## Conclusion
This challenge showed that traditional CNNs still perform exceptionally well for lensing classification. While integrating physics into the model via PINNs didn’t significantly boost AUC, it remains an exciting direction for future work. Fine-tuning the physics loss or adjusting ViT parameters might unlock more potential!

**Big thanks to ML4SCI for this amazing challenge!** 🚀

