# 3D-CNNs-and-Model-Quantization
This assignment builds a **3D Convolutional Neural Network** for classification on NoduleMNIST3D and applies **TensorFlow Lite model compression** on ChestMNIST.

---

## 📌 Part A — 3D CNN (NoduleMNIST3D)
Key steps:
- Load the 3D medical imaging dataset  
- Normalize and reshape input volumes  
- Build a 3D CNN with:
  - Conv3D → BatchNorm → ReLU → MaxPooling  
  - Dense layers with Dropout  
- Train with EarlyStopping (monitoring AUC)
- Evaluate using:
  - Loss & accuracy  
  - ROC-AUC  
  - Learning curves

---

## 📌 Part B — Model Quantization (ChestMNIST)
Performed two types of TensorFlow Lite quantization:

### **1. Dynamic Range Quantization**
- Greatly reduces model size  
- Minimal accuracy loss  

### **2. Full Integer Quantization**
- Requires representative dataset  
- Produces even smaller `.tflite` model  
- Tested with TFLite Interpreter  

---

## 📊 Key Results
- Original model size: **~4.6 MB**  
- Dynamic range quantized: **~392 KB**  
- Full integer quantized: **~395 KB**  
- Maintains **~94% accuracy** after quantization  

---

## 🛠️ Libraries
TensorFlow • MedMNIST • NumPy • Matplotlib • scikit-learn • TensorFlow Lite
