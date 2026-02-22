import logging
import warnings
warnings.filterwarnings('ignore')
logging.set_verbosity_error() if hasattr(logging, 'set_verbosity_error') else None

import transformers
transformers.logging.set_verbosity_error()
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Rebuild Fusion Model Architecture ────────────────
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        base = models.resnet50(weights=None)
        base.fc = nn.Linear(base.fc.in_features, 2)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(2048, 128)
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class VitalsEncoder(nn.Module):
    def __init__(self):
        super(VitalsEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 128)
        )
    def forward(self, x):
        return self.network(x)

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.fc = nn.Linear(768, 128)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.fc(cls)

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.image_encoder  = ImageEncoder()
        self.vitals_encoder = VitalsEncoder()
        self.text_encoder   = TextEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, image, vitals, input_ids, attention_mask):
        img_emb    = self.image_encoder(image)
        vitals_emb = self.vitals_encoder(vitals)
        text_emb   = self.text_encoder(input_ids, attention_mask)
        fused = torch.cat([img_emb, vitals_emb, text_emb], dim=1)
        return self.classifier(fused)

# ── Load Fusion Model ─────────────────────────────────
fusion = FusionModel().to(device)
fusion.load_state_dict(torch.load('models/fusion_model.pth', map_location=device))
fusion.eval()
print("Fusion model loaded! ✅")

# ── Real Inputs ───────────────────────────────────────
# 1. Real X-ray image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img_path = 'data/images/chest_xray/chest_xray/test/PNEUMONIA'
sample = [f for f in os.listdir(img_path) if f.endswith(('.jpg','.jpeg','.png'))][0]
img = Image.open(f'{img_path}/{sample}').convert('RGB')
image_tensor = transform(img).unsqueeze(0).to(device)

# 2. Real vitals
vitals = torch.FloatTensor([[6, 148, 72, 35, 0, 33.6, 0.627, 50]]).to(device)

# 3. Symptom text
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
symptom = ["Patient has fever, cough, difficulty breathing and chest pain"]
tokens = tokenizer(symptom, padding=True, truncation=True,
                   max_length=128, return_tensors='pt')
input_ids = tokens['input_ids'].to(device)
attention_mask = tokens['attention_mask'].to(device)

# ── Predict ───────────────────────────────────────────
with torch.no_grad():
    output = fusion(image_tensor, vitals, input_ids, attention_mask)
    probs = torch.softmax(output, dim=1)[0]

classes = ['Pneumonia', 'Diabetes', 'Normal']
print("\n── Multimodal Prediction Results ──")
for i, cls in enumerate(classes):
    print(f"{cls}: {probs[i].item()*100:.2f}%")

pred = torch.argmax(probs).item()
risk_score = int((1 - probs[2].item()) * 100)
print(f"\nFinal Diagnosis: {classes[pred]}")
print(f"Clinical Risk Score: {risk_score}/100")
print("\nFusion model test complete! ✅")