from transformers import AutoTokenizer
import torch

# Load ClinicalBERT tokenizer
print("Loading ClinicalBERT tokenizer... (first time may take a few minutes)")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
print("Tokenizer loaded! ✅")

# Sample symptoms (simulating what a doctor would type)
sample_symptoms = [
    "Patient has fever, cough, and difficulty breathing",
    "Patient reports chest pain and shortness of breath",
    "Patient has high blood sugar and frequent urination"
]

# Tokenize symptoms
def preprocess_text(symptoms_list):
    tokens = tokenizer(
        symptoms_list,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return tokens

tokens = preprocess_text(sample_symptoms)

print("\nTokenization Results:")
print("Input IDs shape:", tokens['input_ids'].shape)
print("Attention Mask shape:", tokens['attention_mask'].shape)
print("\nSample token IDs for first symptom:")
print(tokens['input_ids'][0])
print("\nText preprocessing working! ✅")