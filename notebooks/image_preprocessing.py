import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img)
    return tensor

# Test it on one image
train_normal_path = r'data/images/chest_xray/train/NORMAL'
sample_image = os.listdir(train_normal_path)[0]
full_path = os.path.join(train_normal_path, sample_image)

tensor = preprocess_image(full_path)
print(f'Image: {sample_image}')
print(f'Tensor shape: {tensor.shape}')
print(f'Min value: {tensor.min():.4f}')
print(f'Max value: {tensor.max():.4f}')
print('Image preprocessing working! ✅')