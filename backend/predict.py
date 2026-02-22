import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import pickle
import numpy as np
import warnings
import transformers
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print("Loading X-Ray model...")
xray_model = models.resnet50(weights=None)
xray_model.fc = nn.Linear(xray_model.fc.in_features, 2)
xray_model.load_state_dict(torch.load('../models/image_model.pth', map_location=device))
xray_model = xray_model.to(device)
xray_model.eval()
print("X-Ray model ready! ✅")

class VitalsModel(nn.Module):
    def __init__(self):
        super(VitalsModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.network(x)

print("Loading Vitals model...")
vitals_model = VitalsModel().to(device)
vitals_model.load_state_dict(torch.load('../models/vitals_model.pth', map_location=device))
vitals_model.eval()
print("Vitals model ready! ✅")

print("Loading scaler...")
with open('../models/vitals_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Scaler ready! ✅")

print("Loading Brain Tumor model...")
brain_model = models.efficientnet_b3(weights=None)
brain_model.classifier[1] = nn.Linear(brain_model.classifier[1].in_features, 4)
brain_model.load_state_dict(torch.load('../models/brain_tumor_model.pth', map_location=device))
brain_model = brain_model.to(device)
brain_model.eval()
print("Brain Tumor model ready! ✅")

def predict_xray(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = xray_model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    classes = ['Normal', 'Pneumonia']
    pred = torch.argmax(probs).item()
    return {
        "diagnosis": classes[pred],
        "risk_score": int(probs[1].item() * 100),
        "probabilities": {
            "Normal":    round(probs[0].item() * 100, 2),
            "Pneumonia": round(probs[1].item() * 100, 2)
        }
    }

def predict_vitals(vitals: list):
    vitals_scaled = scaler.transform([vitals])
    tensor = torch.FloatTensor(vitals_scaled).to(device)
    with torch.no_grad():
        output = vitals_model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    classes = ['No Diabetes', 'Diabetes']
    pred = torch.argmax(probs).item()
    return {
        "diagnosis": classes[pred],
        "risk_score": int(probs[1].item() * 100),
        "probabilities": {
            "No Diabetes": round(probs[0].item() * 100, 2),
            "Diabetes":    round(probs[1].item() * 100, 2)
        }
    }

def predict_brain(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Validate if image looks like MRI (grayscale-like)
    img_array = np.array(img)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    rg_diff = np.mean(np.abs(r.astype(int) - g.astype(int)))
    rb_diff = np.mean(np.abs(r.astype(int) - b.astype(int)))
    gb_diff = np.mean(np.abs(g.astype(int) - b.astype(int)))
    avg_color_diff = (rg_diff + rb_diff + gb_diff) / 3

    if avg_color_diff > 20:
        return {
            "error": True,
            "message": "⚠️ Invalid image! Please upload a Brain MRI scan. Colorful or non-medical images are not accepted."
        }

    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = brain_model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    max_prob = probs.max().item()
    if max_prob < 0.5:
        return {
            "error": True,
            "message": "⚠️ Image doesn't look like a Brain MRI scan. Please upload a proper MRI image."
        }

    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    pred = torch.argmax(probs).item()
    risk_score = int((1 - probs[2].item()) * 100)
    return {
        "error": False,
        "diagnosis": classes[pred],
        "risk_score": risk_score,
        "probabilities": {
            "Glioma":     round(probs[0].item() * 100, 2),
            "Meningioma": round(probs[1].item() * 100, 2),
            "No Tumor":   round(probs[2].item() * 100, 2),
            "Pituitary":  round(probs[3].item() * 100, 2)
        }
    }