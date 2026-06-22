from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError


trained_model = None
vehicle_model = None
vehicle_weights = None

MODEL_PATH = Path(__file__).resolve().parent / "model" / "saved_model.pth"
IMAGE_SIZE = (224, 224)
DAMAGE_CONFIDENCE_THRESHOLD = 0.70
VEHICLE_CONFIDENCE_THRESHOLD = 0.15

# Must match torchvision.datasets.ImageFolder's alphabetical folder order used in training.
class_names = [
    "Front Breakage",
    "Front Crushed",
    "Front Normal",
    "Rear Breakage",
    "Rear Crushed",
    "Rear Normal",
]

VEHICLE_KEYWORDS = (
    "car",
    "truck",
    "van",
    "jeep",
    "limousine",
    "ambulance",
    "cab",
    "taxi",
    "racer",
    "convertible",
    "wagon",
    "pickup",
    "bus",
    "minibus",
    "fire engine",
    "tow truck",
    "trailer truck",
    "police van",
    "car wheel",
    "car mirror",
    "grille",
)

class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights=None)
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

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _open_image(image_source):
    try:
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        return Image.open(image_source).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


def _damage_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_damage_model():
    global trained_model

    if trained_model is not None:
        return trained_model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Please add model/saved_model.pth."
        )

    device = _get_device()
    model = CarClassifierResNet(num_classes=len(class_names))

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError("Could not load the trained damage detection model.") from exc

    model.to(device)
    model.eval()
    trained_model = model
    return trained_model


def load_vehicle_model():
    global vehicle_model, vehicle_weights

    if vehicle_model is not None and vehicle_weights is not None:
        return vehicle_model, vehicle_weights

    try:
        vehicle_weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=vehicle_weights)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the vehicle validation model. Check your internet connection "
            "on first run so torchvision can download pretrained weights."
        ) from exc

    model.to(_get_device())
    model.eval()
    vehicle_model = model
    return vehicle_model, vehicle_weights


def validate_vehicle_image(image_source, threshold=VEHICLE_CONFIDENCE_THRESHOLD):
    image = _open_image(image_source)
    model, weights = load_vehicle_model()
    transform = weights.transforms()
    image_tensor = transform(image).unsqueeze(0).to(_get_device())

    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]

    categories = weights.meta["categories"]
    vehicle_indices = [
        index
        for index, label in enumerate(categories)
        if any(keyword in label.lower() for keyword in VEHICLE_KEYWORDS)
    ]
    vehicle_score = probabilities[vehicle_indices].sum().item()
    top_confidence, top_index = torch.max(probabilities, 0)
    top_predictions = torch.topk(probabilities, 5)
    top_labels = [
        {
            "label": categories[index],
            "confidence": probabilities[index].item(),
        }
        for index in top_predictions.indices.tolist()
    ]

    return {
        "is_vehicle": vehicle_score >= threshold,
        "vehicle_confidence": vehicle_score,
        "top_label": categories[top_index.item()],
        "top_confidence": top_confidence.item(),
        "top_predictions": top_labels,
        "threshold": threshold,
    }


def predict(image_source, confidence_threshold=DAMAGE_CONFIDENCE_THRESHOLD):
    image = _open_image(image_source)
    transform = _damage_transform()
    image_tensor = transform(image).unsqueeze(0)
    model = load_damage_model()

    # Move input tensor to the same device as the model
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, 0)

    confidence_value = confidence.item()
    predicted_index = predicted_class.item()

    return {
        "class_name": class_names[predicted_index],
        "confidence": confidence_value,
        "is_confident": confidence_value >= confidence_threshold,
        "threshold": confidence_threshold,
        "probabilities": {
            class_names[index]: probabilities[index].item()
            for index in range(len(class_names))
        },
    }
