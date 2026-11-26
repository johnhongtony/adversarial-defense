import torch
from src.models.simple_cnn import SimpleCNN

model = SimpleCNN()

x = torch.randn(4, 3, 32, 32)

# Feature Extraction
features = model.features(x)

print("Feature Shape:", features.shape)

# Classification
output = model.classifier(features)

print("Output Shape:", output.shape)