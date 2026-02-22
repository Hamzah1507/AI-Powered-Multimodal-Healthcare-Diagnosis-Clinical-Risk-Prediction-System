import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer, AutoModel
import warnings
import transformers
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Encoders ──────────────────────────────────────────
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

# ── Load Model & Tokenizer ────────────────────────────
print("Loading Fusion Model...")
fusion_model = FusionModel().to(device)
fusion_model.load_state_dict(torch.load('../models/fusion_model.pth', map_location=device))
fusion_model.eval()

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
print("Model ready! ✅")