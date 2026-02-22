import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import pandas as pd
import numpy as np
import pickle

# ── Device ──────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Load Image Model ─────────────────────────────────
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

# ── Load Vitals Model ─────────────────────────────────
class VitalsEncoder(nn.Module):
    def __init__(self):
        super(VitalsEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 128)
        )

    def forward(self, x):
        return self.network(x)

# ── Load Text Model ───────────────────────────────────
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.fc = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.fc(cls)

# ── Fusion Model ──────────────────────────────────────
class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.image_encoder  = ImageEncoder()
        self.vitals_encoder = VitalsEncoder()
        self.text_encoder   = TextEncoder()

        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 classes: Normal, Pneumonia, Diabetes
        )

    def forward(self, image, vitals, input_ids, attention_mask):
        img_emb    = self.image_encoder(image)
        vitals_emb = self.vitals_encoder(vitals)
        text_emb   = self.text_encoder(input_ids, attention_mask)

        fused = torch.cat([img_emb, vitals_emb, text_emb], dim=1)
        return self.classifier(fused)

# ── Test Fusion Model ─────────────────────────────────
print("\nBuilding Fusion Model...")
fusion = FusionModel().to(device)

# Dummy inputs to test
dummy_image   = torch.randn(1, 3, 224, 224).to(device)
dummy_vitals  = torch.randn(1, 8).to(device)
dummy_ids     = torch.randint(0, 1000, (1, 128)).to(device)
dummy_mask    = torch.ones(1, 128, dtype=torch.long).to(device)

output = fusion(dummy_image, dummy_vitals, dummy_ids, dummy_mask)
print(f"Fusion output shape: {output.shape}")
print(f"Class scores: {output.detach().cpu().numpy()}")
print("\nFusion Model built successfully! ✅")

# Save
torch.save(fusion.state_dict(), 'models/fusion_model.pth')
print("Fusion Model saved! ✅")