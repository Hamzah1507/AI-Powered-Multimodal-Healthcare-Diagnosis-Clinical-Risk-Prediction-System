<div align="center">

# 🏥 MediAI Diagnostics
### AI-Powered Multimodal Healthcare Diagnosis & Clinical Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)](https://reactjs.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-6.0-47A248?style=for-the-badge&logo=mongodb)](https://mongodb.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

![MediAI Banner](https://img.shields.io/badge/X--Ray%20Accuracy-98%25-1E3A8A?style=flat-square) 
![Brain MRI](https://img.shields.io/badge/Brain%20MRI%20Accuracy-94.75%25-7C3AED?style=flat-square)
![Diabetes](https://img.shields.io/badge/Diabetes%20Accuracy-78.57%25-059669?style=flat-square)
![Explainability](https://img.shields.io/badge/Explainability-Grad--CAM-F59E0B?style=flat-square)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [AI Models](#-ai-models--accuracy)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation--setup)
- [Usage](#-usage)
- [Datasets](#-datasets)
- [Screenshots](#-screenshots)

---

## 🔬 Overview

**MediAI Diagnostics** is a full-stack AI-powered clinical decision support system that assists medical professionals in diagnosing three critical conditions using deep learning models:

| Module | Task | Model | Accuracy |
|--------|------|-------|----------|
| 🫁 Chest X-Ray | Pneumonia Detection | ResNet-50 | **98%** |
| 🧠 Brain MRI | Tumor Classification (4 types) | EfficientNet-B3 | **94.75%** |
| 🩸 Patient Vitals | Diabetes Risk Prediction | MLP | **78.57%** |

The system goes beyond basic prediction by providing:
- 🔥 **Grad-CAM heatmaps** — highlights the exact disease region on the medical image
- 📄 **PDF diagnostic reports** — professional medical reports with patient info and heatmap
- 🗄️ **MongoDB storage** — all predictions and patient data stored securely
- 🔐 **Authentication** — user registration, login, and session management

---

## ✨ Key Features

- ✅ **Multimodal AI** — combines image analysis (X-Ray, MRI) and tabular data (vitals) in one platform
- ✅ **Explainable AI (XAI)** — Grad-CAM visualizes exactly where the AI detected abnormality
- ✅ **Focused Heatmaps** — red overlay only on high-risk regions, not the whole image
- ✅ **Disease Labels** — AI diagnosis name and confidence % overlaid on heatmap image
- ✅ **PDF Report Generation** — 2-page professional clinical report using ReportLab
- ✅ **Patient Information Capture** — name, ID, age, gender stored with each diagnosis
- ✅ **Brain MRI Validation** — automatically rejects non-MRI images with error messages
- ✅ **Professional Medical UI** — clean white dashboard resembling real hospital software
- ✅ **User Authentication** — MongoDB-backed register/login/logout system
- ✅ **Responsive Design** — works on desktop and laptop screens

---

## 🤖 AI Models & Accuracy

### 1. 🫁 Chest X-Ray — Pneumonia Detection
- **Architecture:** ResNet-50 (Transfer Learning, ImageNet pretrained)
- **Dataset:** Kaggle Chest X-Ray Images (Pneumonia) — 5,863 images
- **Classes:** Normal, Pneumonia
- **Test Accuracy:** 98%
- **Input:** 224×224 RGB image

### 2. 🧠 Brain MRI — Tumor Classification
- **Architecture:** EfficientNet-B3 (Transfer Learning, trained on Google Colab T4 GPU)
- **Dataset:** Kaggle Brain Tumor MRI Dataset — 7,023 images
- **Classes:** Glioma, Meningioma, No Tumor, Pituitary
- **Test Accuracy:** 94.75%
- **Input:** 224×224 RGB image
- **Validation:** Automatically rejects chest X-rays submitted to this module

### 3. 🩸 Diabetes Risk — Patient Vitals
- **Architecture:** Multi-Layer Perceptron (MLP)
- **Dataset:** Pima Indians Diabetes Dataset — 768 records
- **Classes:** Diabetic, Non-Diabetic
- **Test Accuracy:** 78.57%
- **Input:** 8 numerical features (glucose, BMI, age, etc.)

---

## 🛠 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React.js 18, Axios, Vite |
| **Backend** | Python 3.10, FastAPI, Uvicorn |
| **AI / ML** | PyTorch 2.0, TorchVision, EfficientNet-B3, ResNet-50 |
| **Explainability** | Grad-CAM (custom implementation with OpenCV) |
| **PDF Generation** | ReportLab |
| **Database** | MongoDB (pymongo) |
| **Image Processing** | OpenCV, Pillow |
| **Data Processing** | NumPy, scikit-learn, joblib |
| **Training** | Google Colab (T4 GPU) |

---

## 📁 Project Structure

```
capstone-healthcare-ai/
│
├── backend/
│   ├── main.py              # FastAPI app + all endpoints
│   ├── predict.py           # AI model prediction functions
│   ├── gradcam.py           # Grad-CAM heatmap generation
│   ├── report.py            # PDF report generation (ReportLab)
│   ├── database.py          # MongoDB connection + operations
│   └── auth.py              # User authentication
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component (complete UI)
│   │   ├── App.css          # Styles
│   │   └── index.css        # Global reset styles
│   ├── index.html
│   └── package.json
│
├── models/
│   ├── image_model.pth          # ResNet-50 (Chest X-Ray, 92MB)
│   ├── brain_tumor_model.pth    # EfficientNet-B3 (Brain MRI, 42MB)
│   ├── vitals_model.pth         # MLP (Diabetes, 49KB)
│   └── vitals_scaler.pkl        # StandardScaler for vitals
│
├── notebooks/
│   ├── xray_training.ipynb      # ResNet-50 training notebook
│   ├── brain_training.ipynb     # EfficientNet-B3 training notebook
│   └── diabetes_training.ipynb  # MLP training notebook
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- MongoDB (running locally on port 27017)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Hamzah1507/AI-Powered-Multimodal-Healthcare-Diagnosis-Clinical-Risk-Prediction-System.git
cd AI-Powered-Multimodal-Healthcare-Diagnosis-Clinical-Risk-Prediction-System
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install fastapi uvicorn torch torchvision
pip install opencv-python pillow numpy scikit-learn
pip install pymongo reportlab grad-cam
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

### 4. Add AI Models
Place the following model files in the `models/` directory:
- `image_model.pth` — ResNet-50 chest X-ray model
- `brain_tumor_model.pth` — EfficientNet-B3 brain MRI model
- `vitals_model.pth` — MLP diabetes model
- `vitals_scaler.pkl` — StandardScaler for vitals normalization

### 5. Start MongoDB
```bash
# Make sure MongoDB is running on localhost:27017
mongod
```

---

## 🚀 Usage

### Start Backend (Terminal 1)
```bash
cd backend
uvicorn main:app --reload
# API running at: http://127.0.0.1:8000
# Docs at: http://127.0.0.1:8000/docs
```

### Start Frontend (Terminal 2)
```bash
cd frontend
npm run dev
# App running at: http://localhost:5173
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict-xray` | Chest X-Ray pneumonia detection |
| `POST` | `/predict-brain` | Brain MRI tumor classification |
| `POST` | `/predict-vitals` | Diabetes risk from patient vitals |
| `POST` | `/gradcam-xray` | Generate Grad-CAM heatmap for X-Ray |
| `POST` | `/gradcam-brain` | Generate Grad-CAM heatmap for Brain MRI |
| `POST` | `/generate-report` | Generate PDF diagnostic report |
| `POST` | `/register` | Register new user |
| `POST` | `/login` | User login |

---

## 📊 Datasets

| Dataset | Source | Size | Classes |
|---------|--------|------|---------|
| Chest X-Ray (Pneumonia) | [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) | 5,863 images | Normal, Pneumonia |
| Brain Tumor MRI | [Kaggle](https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset) | 7,023 images | Glioma, Meningioma, No Tumor, Pituitary |
| Pima Indians Diabetes | [Kaggle/UCI](https://www.kaggle.com/uciml/pima-indians-diabetes-database) | 768 records | Diabetic, Non-Diabetic |

---

## 📸 Screenshots

### Landing Page & Authentication
> Clean dark landing page with Sign In / Create Account

### Chest X-Ray + Diabetes Dashboard
> Professional medical UI with patient info, X-ray upload, vitals input

### Brain MRI Tumor Detection
> Upload MRI scan, AI classifies tumor type with 94.75% accuracy

### Grad-CAM Heatmap
> Red overlay highlights exact tumor/pneumonia location on the scan

### PDF Diagnostic Report
> 2-page professional report with patient info, diagnosis, probabilities, and heatmap

---

## 🎯 Project Highlights

```
✅ 3 Deep Learning Models trained and deployed
✅ Explainable AI (Grad-CAM) — rare in student projects
✅ Focused heatmaps — red overlay only on disease region
✅ Professional PDF reports with embedded heatmap images
✅ Full authentication system with MongoDB
✅ React.js frontend with professional medical-grade UI
✅ FastAPI backend with 7 working endpoints
✅ Input validation — rejects wrong image types
✅ Patient data storage and session management
```

---

## ⚠️ Medical Disclaimer

> This system is developed for **research and educational purposes only**. It is **NOT intended for actual clinical use**. Always consult a qualified medical professional for medical diagnosis and treatment decisions.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**MediAI Diagnostics** — AI-Powered Healthcare Diagnosis System

⭐ Star this repo if you found it useful!

</div>
