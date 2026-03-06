# 🏥 MediAI Diagnostics

An AI-powered multimodal clinical diagnosis web application for detecting pneumonia from chest X-rays, classifying brain tumors from MRI scans, and predicting diabetes risk from patient vitals.
Built with **React.js + FastAPI + PyTorch** with Grad-CAM explainability, PDF report generation, and MongoDB storage.

---

## 🤖 AI Models

- **Chest X-Ray** — ResNet-50 (Transfer Learning) → **98% accuracy** — Detects Normal vs Pneumonia
- **Brain MRI** — EfficientNet-B3 (Transfer Learning) → **94.75% accuracy** — Classifies Glioma, Meningioma, Pituitary, No Tumor
- **Diabetes Risk** — Multi-Layer Perceptron (MLP) → **78.57% accuracy** — Predicts diabetic risk from 8 patient vitals

---

## ✨ Features

- 🫁 **Chest X-Ray Analysis** — Upload X-ray image for AI-powered pneumonia detection
- 🧠 **Brain MRI Analysis** — Upload MRI scan for tumor type classification (4 classes)
- 🩸 **Diabetes Risk Assessment** — Enter patient vitals for diabetes prediction
- 🔥 **Grad-CAM Heatmaps** — Visualizes exact disease region with focused red overlay on the scan
- 🏷️ **Disease Labels** — AI diagnosis name and confidence % drawn directly on heatmap
- 📄 **PDF Report Generation** — 2-page professional clinical report with patient info and heatmap
- 🔐 **User Authentication** — Register, login, logout with MongoDB-backed sessions
- 👤 **Patient Information Capture** — Name, ID, age, gender stored with every diagnosis
- 🗄️ **MongoDB Storage** — All predictions and patient records stored securely
- 🚫 **Image Validation** — Automatically rejects wrong image types (e.g. X-ray in Brain MRI module)
- 📊 **Probability Breakdown** — Shows confidence % for all classes with progress bars
- 🎨 **Professional Medical UI** — Clean white dashboard built for clinical use

---

## 🛠 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React.js 18, Axios, Vite |
| **Backend** | Python 3.10, FastAPI, Uvicorn |
| **AI / ML** | PyTorch 2.0, TorchVision, ResNet-50, EfficientNet-B3 |
| **Explainability** | Grad-CAM (custom implementation with OpenCV + PyTorch hooks) |
| **PDF Generation** | ReportLab |
| **Database** | MongoDB (pymongo) |
| **Image Processing** | OpenCV, Pillow |
| **Data Processing** | NumPy, scikit-learn, joblib |
| **Model Training** | Google Colab (T4 GPU) |

---

## 📁 Project Structure

```
capstone-healthcare-ai/
│
├── backend/
│   ├── main.py              # FastAPI app + all API endpoints
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
│   │   └── index.css        # Global styles
│   ├── index.html
│   └── package.json
│
├── models/
│   ├── image_model.pth          # ResNet-50 weights (Chest X-Ray)
│   ├── brain_tumor_model.pth    # EfficientNet-B3 weights (Brain MRI)
│   ├── vitals_model.pth         # MLP weights (Diabetes)
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
- MongoDB running on localhost:27017

### 1. Clone the Repository
```bash
git clone https://github.com/Hamzah1507/AI-Powered-Multimodal-Healthcare-Diagnosis-Clinical-Risk-Prediction-System.git
cd AI-Powered-Multimodal-Healthcare-Diagnosis-Clinical-Risk-Prediction-System
```

### 2. Backend Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install fastapi uvicorn torch torchvision
pip install opencv-python pillow numpy scikit-learn
pip install pymongo reportlab grad-cam
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

### 4. Place Model Files
Add the following into the `models/` folder:
- `image_model.pth`
- `brain_tumor_model.pth`
- `vitals_model.pth`
- `vitals_scaler.pkl`

---

## 🚀 Running the App

### Terminal 1 — Backend
```bash
cd backend
uvicorn main:app --reload
# Running at: http://127.0.0.1:8000
```

### Terminal 2 — Frontend
```bash
cd frontend
npm run dev
# Running at: http://localhost:5173
```

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/register` | Register new user |
| `POST` | `/login` | User login |
| `POST` | `/predict-xray` | Pneumonia detection from X-Ray |
| `POST` | `/predict-brain` | Tumor classification from Brain MRI |
| `POST` | `/predict-vitals` | Diabetes risk from patient vitals |
| `POST` | `/gradcam-xray` | Grad-CAM heatmap for X-Ray |
| `POST` | `/gradcam-brain` | Grad-CAM heatmap for Brain MRI |
| `POST` | `/generate-report` | Generate PDF diagnostic report |

---

## 📊 Datasets

| Dataset | Source | Size | Classes |
|---------|--------|------|---------|
| Chest X-Ray (Pneumonia) | [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) | 5,863 images | Normal, Pneumonia |
| Brain Tumor MRI | [Kaggle](https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset) | 7,023 images | Glioma, Meningioma, No Tumor, Pituitary |
| Pima Indians Diabetes | [Kaggle/UCI](https://www.kaggle.com/uciml/pima-indians-diabetes-database) | 768 records | Diabetic, Non-Diabetic |

---

## ⚠️ Disclaimer

This project is built for **research and educational purposes only**. It is **not intended for real clinical use**. Always consult a qualified medical professional for diagnosis and treatment.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
