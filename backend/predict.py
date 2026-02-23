import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import warnings
import transformers
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

from model_loader import xray_model, vitals_model, brain_model, scaler, device

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_xray(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
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
    with torch.inference_mode():
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
    img_array = np.array(img)

    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    rg_diff = np.mean(np.abs(r.astype(int) - g.astype(int)))
    rb_diff = np.mean(np.abs(r.astype(int) - b.astype(int)))
    gb_diff = np.mean(np.abs(g.astype(int) - b.astype(int)))
    avg_color_diff = (rg_diff + rb_diff + gb_diff) / 3

    if avg_color_diff > 20:
        return {"error": True, "message": "⚠️ Invalid image! Please upload a Brain MRI scan only."}

    h, w = img_array.shape[:2]
    aspect_ratio = w / h
    if aspect_ratio < 0.7 or aspect_ratio > 1.5:
        return {"error": True, "message": "⚠️ Image format doesn't match a Brain MRI scan."}

    overall_brightness = np.mean(img_array)
    if overall_brightness > 160:
        return {"error": True, "message": "⚠️ This looks like a Chest X-Ray, not a Brain MRI."}

    tensor = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = brain_model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    max_prob = probs.max().item()
    if max_prob < 0.5:
        return {"error": True, "message": "⚠️ Image doesn't look like a Brain MRI scan."}

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