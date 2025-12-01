import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

trained_model = None
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']

class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x

def predict(image_path):
    global trained_model

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Initialize and load model if not already loaded
    if trained_model is None:
        trained_model = CarClassifierResNet()

        # Detect CPU or GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build absolute path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))  # folder of model_helper.py
        model_path = os.path.join(current_dir, "model", "saved_model.pth")

        # Load model safely
        trained_model.load_state_dict(torch.load(model_path, map_location=device))
        trained_model.to(device)
        trained_model.eval()

    # Move input tensor to the same device as the model
    device = next(trained_model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]
