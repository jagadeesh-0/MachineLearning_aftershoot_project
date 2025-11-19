# 📸 AI-Based White Balance Prediction

### Hybrid Deep Learning Model (CNN + Metadata MLP)

This project implements a **hybrid deep-learning system** combining **ResNet-18 image features** with **camera metadata** to predict **Temperature** and **Tint** values—similar to AfterShoot/Lightroom auto white balance correction.

The solution includes:

* 🧠 **ResNet-18 CNN** for image feature extraction
* 🟦 **MLP network** for metadata (EXIF sliders)
* 🔗 **Fusion model** concatenating CNN + metadata embeddings
* 📊 **LightGBM baseline (optional)**
* 🎯 **Final inference pipeline** generating predictions for the Validation set
* 💾 **Supports local filesystem (C:/Users/...) or Google Colab**

---

## 🚀 Features

* Hybrid Deep Learning Model (Image + Metadata)
* Automatic feature scaling & label encoding
* End-to-end pipeline:
  **Training → Validation → Inference → Submission File**
* Clean PyTorch implementation
* Works with your local dataset:

  ```
  C:/Users/jagad/Downloads/14648881b93c11f0/dataset/
  ```
* Generates:

  ```
  submission_hybrid.csv
  ```

---

## 📂 Project Structure

```
📦 MachineLearning_aftershoot_project
│
├── dataset/
│   ├── Train/
│   │   ├── images/
│   │   └── sliders.csv
│   └── Validation/
│       ├── images/
│       └── sliders_input.csv
│
├── hybrid_model_final.pt
├── submission_hybrid.csv
└── main.py  (full training + inference pipeline)
```

---

## 🧠 Model Architecture

### **🔹 Image Path (CNN)**

* Pretrained **ResNet-18 (IMAGENET1K_V1)**
* Output: 512-dim vector

### **🔹 Metadata Path (MLP)**

* StandardScaler normalization
* Separate encoders for categorical metadata
* Two-layer MLP → 64-dim vector

### **🔹 Fusion Layer**

```
cat([image_features, metadata_features])
▼
Deep Regression Head
▼
[Temperature, Tint]
```

---

## ▶️ How to Run

### **1. Install Requirements**

```bash
pip install torch torchvision lightgbm pandas numpy pillow scikit-learn tqdm
```

### **2. Place Dataset**

```
dataset/Train/images/*.tiff
dataset/Train/sliders.csv
dataset/Validation/images/*.tiff
dataset/Validation/sliders_input.csv
```

### **3. Run the Script**

```bash
python main.py
```

### **4. Output**

```
hybrid_model_final.pt
submission_hybrid.csv
```

---

## 📊 Example Output

```
id_global,Temperature,Tint
EB5BEE31...,6248,11
DE666E1F...,5996,7
...
```

---

## 🧪 Validation Metrics

* Mean Absolute Error (MAE)
* Separate scores for Temperature and Tint
* Custom printout per epoch

---

## 🤝 Contributions

Pull requests are welcome!
If you'd like enhancements such as:

* ONNX export
* Mobile/Python-only inference
* Colab notebook
  Feel free to open an issue.

---

## 📜 License

MIT License – free for personal and commercial use.

---

## 🌟 Author

**Jagadeesh Kumar**
Machine Learning Developer
GitHub: *jagadeesh-0*
