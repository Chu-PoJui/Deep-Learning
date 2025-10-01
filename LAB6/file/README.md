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
* Each image can contain 1 to 3 objects.

---

### 3. `test.json` & `new_test.json`
- Testing set specification files, each containing **32 samples**.  
- Both files are **lists**; each element is a list of target objects to be generated.  
- Example:
  ```json
  [
    ["gray cube"],
    ["red cube"],
    ["blue cube"],
    ["blue cube", "green cube"]
  ]
  ```

---

### 4. `evaluator.py` & `checkpoint.pth`
- Pretrained classifier used to **evaluate synthetic images**.  
- `evaluator.py`: Python script that loads the classifier and computes evaluation scores.  
- `checkpoint.pth`: pretrained model weights (ResNet18).  
- See the script for detailed usage.

---

## 📝 Notes
- These files are the **specifications for conditional DDPM training and evaluation** on i-CLEVR.  
- Usage summary:
  - `objects.json` → object/index mapping.  
  - `train.json` → training samples (filename → objects).  
  - `test.json`, `new_test.json` → test conditions (list of objects).  
  - `evaluator.py`, `checkpoint.pth` → pretrained classifier for evaluating results.  


