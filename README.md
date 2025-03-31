![image](https://github.com/user-attachments/assets/31c65cd5-840d-4924-b40d-011f23c9553d)


# DeepLense Classification Task

This repository contains my work on the ML4SCI DeepLense classification challenge, where I classify gravitational lensing images into three categories:

- **No Substructure**
- **Subhalo**
- **Vortex**

## Overview
This repository includes two key models:

- **`BaselineCNN`**: A ResNet18-based classifier designed for traditional image classification.
- **`LensVIT_PINN`**: A hybrid model that combines Vision Transformers (ViT) and ResNet, incorporating the lensing equation for better physics-aware predictions.

Both models were trained on a dataset of **37,500 images** (30,000 for training, 7,500 for validation).

**Hardware**: Models were trained on my RTX 4060 GPU (CUDA)


## Repository main Contents
- **`CommonTest.ipynb`** – The notebook for the baselineCNN model
- **`SpecificTask_PINN.ipynb`** – The notebook for the LensVIT_PINN model


## Model Architectures
### 1. BaselineCNN
- **Backbone**: ResNet18, pre-trained on ImageNet.
- **Modifications**:
  - Input adapted for single-channel images (`conv1` modified to `nn.Conv2d(1, 64, ...)`).
  - Output layer changed to classify three categories (`nn.Linear(512, 3)`).
- **Purpose**: A model to assess performance without physics-based enhancements.

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
- **Initial Attempt**: Used additional gradient preprocessing (`physics_preprocess`) using the following formula:
$$
\text{distortion} = \left| \tanh \left( \nabla_x \nabla_y \left( \log \left( \frac{I_{\max}}{I} \right) \right)^2 \right) \right|
$$
, but this **reduced AUC to 0.8786** due to excessive noise.
- **Final Approach**: Removed preprocessing and relied purely on the gravitational lensing equation, leading to more stable results.



