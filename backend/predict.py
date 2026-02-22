import torch
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
import io
import warnings
import transformers
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

from model_loader import fusion_model, tokenizer, device

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

classes = ['Pneumonia', 'Diabetes', 'Normal']

def predict(image_bytes: bytes, vitals: list, symptom_text: str):
    # 1. Process image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(img).unsqueeze(0).to(device)

    # 2. Process vitals
    vitals_tensor = torch.FloatTensor([vitals]).to(device)

    # 3. Process text
    tokens = tokenizer(
        [symptom_text],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    # 4. Predict
    with torch.no_grad():
        output = fusion_model(image_tensor, vitals_tensor, input_ids, attention_mask)
        probs = torch.softmax(output, dim=1)[0]

    pred = torch.argmax(probs).item()
    risk_score = int((1 - probs[2].item()) * 100)

    return {
        "diagnosis": classes[pred],
        "risk_score": risk_score,
        "probabilities": {
            "Pneumonia": round(probs[0].item() * 100, 2),
            "Diabetes":  round(probs[1].item() * 100, 2),
            "Normal":    round(probs[2].item() * 100, 2)
        }
    }