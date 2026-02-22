import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import warnings
import transformers
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Image Transform ───────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Load Image Model (ResNet50) ───────────────────────
class XRayModel(nn.Module):
    def __init__(self, num_classes):
        super(XRayModel, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

print("Loading X-Ray model...")
xray_model = XRayModel(num_classes=2).to(device)
xray_model.load_state_dict(torch.load('../models/image_model.pth', map_location=device))
xray_model.eval()
print("X-Ray model ready! ✅")

# ── Load Vitals Model ─────────────────────────────────
class VitalsModel(nn.Module):
    def __init__(self):
        super(VitalsModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.network(x)

print("Loading Vitals model...")
vitals_model = VitalsModel().to(device)
vitals_model.load_state_dict(torch.load('../models/vitals_model.pth', map_location=device))
vitals_model.eval()
print("Vitals model ready! ✅")

# ── Predict X-Ray ─────────────────────────────────────
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

# ── Predict Vitals ────────────────────────────────────
def predict_vitals(vitals: list):
    tensor = torch.FloatTensor([vitals]).to(device)
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