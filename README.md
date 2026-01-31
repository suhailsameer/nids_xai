# Explainable AI-based Network Intrusion Detection System (XAI-NIDS)

This project implements an **Explainable AI (XAI)–based Network Intrusion Detection System** using the **NSL-KDD dataset**, a **Deep Neural Network (DNN)** classifier, and two state-of-the-art explainability techniques: **SHAP** and **LIME**.

The goal is not only to achieve high detection performance, but also to **provide transparent, interpretable explanations** for model decisions, which is critical for security-sensitive domains such as intrusion detection.

---

## 📌 Project Objectives

- Train a **Deep Neural Network (DNN)** to detect network intrusions
- Use the **NSL-KDD benchmark dataset**
- Perform **binary classification** (Normal vs Attack)
- Apply **Explainable AI (XAI)** techniques to interpret predictions:
  - **SHAP** for global explanations
  - **LIME** for local (instance-level) explanations
- Provide human-understandable insights into model behavior

---

## 🗂 Dataset

**NSL-KDD** is an improved version of the KDD’99 dataset and is widely used for evaluating intrusion detection systems.

- Eliminates redundant records
- Contains both normal and attack traffic
- Includes 41 network traffic features

Dataset files used:
- `KDDTrain+.txt`
- `KDDTest+.txt`

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Assign column names to raw NSL-KDD files
- Convert labels into binary format:
  - `0` → Normal
  - `1` → Attack
- Encode categorical features:
  - `protocol_type`
  - `service`
  - `flag`
- Apply **StandardScaler** for feature normalization
- Remove non-informative fields (e.g., `difficulty`)

---

### 2️⃣ Deep Neural Network (DNN)
A fully connected neural network is trained using TensorFlow/Keras.

**Architecture:**
- Input layer (41 features)
- Dense layer (128 units, ReLU)
- Dropout (0.3)
- Dense layer (64 units, ReLU)
- Dropout (0.3)
- Output layer (1 unit, Sigmoid)

**Training setup:**
- Optimizer: Adam
- Loss function: Binary Cross-Entropy
- Validation split: 20%
- Batch size: 256

---

### 3️⃣ Model Evaluation
- Confusion Matrix
- Precision, Recall, and F1-score
- Analysis of false positives and false negatives

This evaluation highlights trade-offs commonly encountered in intrusion detection systems, such as minimizing false alarms while detecting attacks.

---

## 🔍 Explainability (XAI)

### 🔹 SHAP (Global Explainability)
**SHAP (SHapley Additive exPlanations)** is used to:
- Identify globally important features
- Understand how each feature influences predictions across the dataset
- Provide consistent, theoretically grounded explanations

The **PermutationExplainer** is used for compatibility with modern TensorFlow versions.

---

### 🔹 LIME (Local Explainability)
**LIME (Local Interpretable Model-agnostic Explanations)** is applied to:
- Explain individual predictions
- Highlight why a specific network connection was classified as normal or attack
- Provide instance-level feature contributions

LIME explanations are exported as **HTML files** for easy visualization and reporting.

---

## 📊 Key Insights

- The DNN achieves high classification performance on NSL-KDD
- Authentication-related features (e.g., `logged_in`, `num_compromised`) strongly influence decisions
- SHAP reveals global feature importance patterns
- LIME provides intuitive explanations for individual predictions, improving trust and transparency

---

## 🧪 Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- SHAP
- LIME
- NumPy, Pandas, Matplotlib