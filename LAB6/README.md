# Lab 6 — Generative Models  
Deep Learning @ NYCU (Spring 2025, TAICA)

This project implements a **conditional Denoising Diffusion Probabilistic Model (DDPM)** to generate synthetic images according to **multi-label conditions** (e.g., *“red sphere”*, *“yellow cube”*, *“gray cylinder”*).

---

## 📌 Overview
- **Task**: Conditional image generation using DDPM  
- **Dataset**: Provided in `file.zip` (train.json, test.json, new_test.json, object.json, evaluator, checkpoint)  
- **Goal**:  
  1. Train a conditional DDPM with custom embedding and noise schedule  
  2. Generate synthetic images from conditions  
  3. Evaluate generated images with the pre-trained ResNet18 evaluator  
  4. Show results: image grids & denoising process  

---

## ⚙️ Environment
Training environment: **Google Colab Pro+ (NVIDIA A100)**  

| Package       | Version   |
|---------------|-----------|
| torch         | ≥ 2.6.0   |
| torchvision   | ≥ 0.21.0  |
| numpy         | ≥ 2.0.0   |
| matplotlib    | ≥ 3.10.0  |
| tqdm          | ≥ 4.67.0  |

---

## 📂 File Structure

```


Lab6/
  ├─ report.pdf # Experiment report
  ├─ requirements.txt # Dependencies
  ├─ src/ # Source code
  │ ├─ train.py # Training script
  │ ├─ inference.py # Inference and denoising
  │ ├─ evaluate.py # Evaluation function
  │ ├─ models/ # Model definitions
  │ └─ utils.py # Helper functions
  ├─ dataset/ # Provided data
  │ ├─ train.json
  │ ├─ test.json
  │ ├─ new_test.json
  │ ├─ object.json
  │ ├─ evaluator.py
  │ └─ checkpoint.pth
  └─ images/ # Generated images
  ├─ test/ # Results for test.json
  └─ new_test/ # Results for new_test.json
```

---



## 🚀 How to Run

### Training
```bash
python src/train.py --epochs 200 --batch_size 64 --lr 1e-4
```

---

Inference
```bash

python src/inference.py --model checkpoint.pth --labels "red sphere, cyan cylinder, cyan cube"
```

---

Evaluation
```bash
python src/evaluate.py --images images/test --labels dataset/test.json
```

---

##📊 Results

Synthetic Image Grids:

test.json → 8×4 grid

new_test.json → 8×4 grid

Denoising Process:

["red sphere", "cyan cylinder", "cyan cube"]

(Figures and detailed discussion are in report.pdf
)
