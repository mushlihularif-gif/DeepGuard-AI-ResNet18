"""Shared, explicit contract for training and inference. 0=REAL, 1=FAKE."""
import io
import random
import torch
from PIL import Image, ImageOps
from torchvision import models, transforms

CLASS_TO_IDX = {"REAL": 0, "FAKE": 1}
PREPROCESS = {"size": 224, "resize": "bilinear_stretch", "exif_transpose": True,
              "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def label_mapping(names):
    aliases = {"real": 0, "realart": 0, "authentic": 0,
               "fake": 1, "aiartdata": 1, "ai": 1, "synthetic": 1}
    try:
        mapping = {i: aliases[name.lower()] for i, name in enumerate(names)}
    except KeyError as exc:
        raise ValueError(f"Unknown class name; explicit mapping required: {exc}") from exc
    if set(mapping.values()) != {0, 1}:
        raise ValueError("Dataset must contain both REAL and FAKE classes")
    return mapping


def rgb_image(image):
    return ImageOps.exif_transpose(image).convert("RGB")


class JPEGCompression:
    def __call__(self, image):
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=random.randint(55, 95))
        stream.seek(0)
        with Image.open(stream) as decoded:
            return decoded.convert("RGB")


def preprocessing(training=False):
    steps = [transforms.Resize((224, 224), antialias=True)]
    if training:
        steps += [transforms.RandomHorizontalFlip(),
                  transforms.RandomApply([JPEGCompression()], p=0.5),
                  transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 0.8))], p=0.1)]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(
        PREPROCESS["mean"], PREPROCESS["std"])])


def make_model(pretrained=False):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Checkpoint lacks validated metadata; run train.py and evaluate it first")
    if checkpoint.get("class_to_idx") != CLASS_TO_IDX or checkpoint.get("preprocess") != PREPROCESS:
        raise ValueError("Checkpoint label/preprocessing contract does not match this application")
    temperature = checkpoint.get("temperature", 0)
    if not isinstance(temperature, (int, float)) or not 0.05 <= temperature <= 20:
        raise ValueError("Invalid calibration temperature")
    model = make_model()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint
