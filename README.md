# 🗣️ Accent Detection from Audio using MFCCs & Classical ML

This project predicts the **accent of a speaker** from a `.wav` audio sample using a complete **machine learning pipeline**.

---

## 🚀 Pipeline Overview

- 🎧 **Audio Preprocessing & Augmentation**
- 🔍 **Feature Extraction** (MFCCs, ZCR, RMSE)
- 🤖 **Model Training & Evaluation**
- 🧪 **Data Versioning** with DVC
- 🌐 **Streamlit App** for real-time accent prediction

---

## 📊 Features Extracted

From each audio file, the following features are extracted:

- 🎼 **MFCCs (13 Coefficients)** — Mel Frequency Cepstral Coefficients
- 📉 **ZCR** — Zero Crossing Rate
- 🔋 **RMSE** — Root Mean Square Energy

---

## 🎛️ Audio Augmentation Techniques

To make the model robust and generalizable, these augmentations are applied:

- ⏩ **Time Stretching**
- 🔁 **Pitch Shifting**
- 🔊 **Noise Addition**

---

## 🧠 Model Training & Evaluation

Two models were trained and evaluated:

| Model                | Accuracy | Status         |
|---------------------|----------|----------------|
| Random Forest        | ~92%     | ❌ Not selected |
| Logistic Regression  | **~96%** | ✅ Final model  |

**Evaluation Metrics:**

- ✅ Accuracy
- 🔍 Confusion Matrix
- 📈 Learning Curves

---

## 📦 Version Control with DVC

[DVC](https://dvc.org/) is used to version:

- 📁 Raw & Intermediate Datasets  
- 📐 Feature Files  
- 🧠 Trained Models  

---

## 🌐 Live App with Streamlit

A **Streamlit web app** is included to test audio samples in real-time and get predictions instantly.

---

## 📁 Folder Structure

