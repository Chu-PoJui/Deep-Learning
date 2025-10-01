# Lab 6 — i-CLEVR Dataset Specification  

This folder contains the **specification files** for training and testing the conditional DDPM model on the **i-CLEVR dataset**.

---

## 📂 Files

### 1. `objects.json`
- Dictionary file that defines the objects and their indexes.  
- The i-CLEVR dataset includes **24 objects** in total:  
  - **Shapes**: 3 (sphere, cube, cylinder)  
  - **Colors**: 8 (e.g., red, yellow, gray, purple, etc.)  

---

### 2. `train.json`
- Training set specification file with **18,009 samples**.  
- Each entry is a dictionary where keys are **image filenames** and values are **lists of objects**.  
- Example:  
  ```json
  {
    "CLEVR_train_001032_0.png": ["yellow sphere"],
    "CLEVR_train_001032_1.png": ["yellow sphere", "gray cylinder"],
    "CLEVR_train_001032_2.png": ["yellow sphere", "gray cylinder", "purple cube"]
  }
