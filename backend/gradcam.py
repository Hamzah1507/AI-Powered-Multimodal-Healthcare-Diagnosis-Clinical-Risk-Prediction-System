import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Load X-Ray Model ──────────────────────────────────
xray_model = models.resnet50(weights=None)
xray_model.fc = nn.Linear(xray_model.fc.in_features, 2)
xray_model.load_state_dict(torch.load('../models/image_model.pth', map_location=device))
xray_model = xray_model.to(device)
xray_model.eval()

# ── Load Brain Model ──────────────────────────────────
brain_model = models.efficientnet_b3(weights=None)
brain_model.classifier[1] = nn.Linear(brain_model.classifier[1].in_features, 4)
brain_model.load_state_dict(torch.load('../models/brain_tumor_model.pth', map_location=device))
brain_model = brain_model.to(device)
brain_model.eval()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        loss.backward()
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx

def apply_focused_heatmap(original_img, cam, label, confidence):
    """Apply focused heatmap only on high-attention regions with disease label"""
    original = np.array(original_img.resize((224, 224)))
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    elif original.shape[2] == 4:
        original = original[:, :, :3]

    original = original.copy()

    # Only highlight TOP 30% attention areas — focused not whole image
    threshold = 0.70
    mask = cam > threshold

    # Create red overlay ONLY on high-attention areas
    overlay = original.copy()
    red_layer = np.zeros_like(original)
    red_layer[mask] = [220, 50, 50]  # Red color for disease areas

    # Blend only in the masked region
    alpha = 0.55
    overlay[mask] = cv2.addWeighted(
        original[mask].reshape(-1, 3), 1 - alpha,
        red_layer[mask].reshape(-1, 3), alpha, 0
    ).reshape(-1, 3)

    # Draw contour around the disease area
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 80, 80), 2)

    # Convert to PIL for text drawing
    pil_img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil_img)

    # Draw disease label box at top
    box_color = (220, 30, 30)
    text_color = (255, 255, 255)

    # Background box for label
    draw.rectangle([0, 0, 224, 36], fill=box_color)
    draw.text((8, 8), f"AI: {label}  ({confidence:.1f}%)", fill=text_color)

    # Draw small legend at bottom
    draw.rectangle([0, 196, 224, 224], fill=(20, 20, 20))
    draw.rectangle([6, 204, 20, 218], fill=(220, 50, 50))
    draw.text((24, 205), "High Risk Region", fill=(255, 255, 255))

    result = np.array(pil_img)
    return result

def generate_gradcam_xray(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    target_layer = xray_model.layer4[-1]
    gradcam = GradCAM(xray_model, target_layer)
    cam, pred_class = gradcam.generate(tensor)

    classes = ['Normal', 'Pneumonia']
    label = classes[pred_class]

    # Get confidence
    with torch.no_grad():
        output = xray_model(transform(img).unsqueeze(0).to(device))
        probs = torch.softmax(output, dim=1)[0]
        confidence = probs[pred_class].item() * 100

    # Only show heatmap if Pneumonia detected
    if pred_class == 1:
        overlay = apply_focused_heatmap(img, cam, label, confidence)
    else:
        # Normal — just show original with green label
        original = np.array(img.resize((224, 224)))
        pil_img = Image.fromarray(original)
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([0, 0, 224, 36], fill=(22, 163, 74))
        draw.text((8, 8), f"AI: {label}  ({confidence:.1f}%)", fill=(255, 255, 255))
        draw.rectangle([0, 196, 224, 224], fill=(20, 20, 20))
        draw.rectangle([6, 204, 20, 218], fill=(22, 163, 74))
        draw.text((24, 205), "No Abnormality Detected", fill=(255, 255, 255))
        overlay = np.array(pil_img)

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return {"heatmap": img_base64, "predicted_class": label}

def generate_gradcam_brain(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    target_layer = brain_model.features[-1]
    gradcam = GradCAM(brain_model, target_layer)
    cam, pred_class = gradcam.generate(tensor)

    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    label = classes[pred_class]

    # Get confidence
    with torch.no_grad():
        output = brain_model(transform(img).unsqueeze(0).to(device))
        probs = torch.softmax(output, dim=1)[0]
        confidence = probs[pred_class].item() * 100

    # Only show red heatmap if tumor detected
    if pred_class != 2:  # Not "No Tumor"
        overlay = apply_focused_heatmap(img, cam, label, confidence)
    else:
        original = np.array(img.resize((224, 224)))
        pil_img = Image.fromarray(original)
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([0, 0, 224, 36], fill=(22, 163, 74))
        draw.text((8, 8), f"AI: {label}  ({confidence:.1f}%)", fill=(255, 255, 255))
        draw.rectangle([0, 196, 224, 224], fill=(20, 20, 20))
        draw.rectangle([6, 204, 20, 218], fill=(22, 163, 74))
        draw.text((24, 205), "No Tumor Detected", fill=(255, 255, 255))
        overlay = np.array(pil_img)

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return {"heatmap": img_base64, "predicted_class": label}