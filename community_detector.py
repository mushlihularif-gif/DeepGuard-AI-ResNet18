"""Adapter for the authors' Community Forensics ViT-S/16 (384) checkpoint.

Weights: OwensLab/commfor-model-384, MIT, revision pinned below.
Preprocessing follows JeongsooP/Community-Forensics/dataloader.py test mode.
The checkpoint is never trained or modified by this application.
"""
import hashlib
from pathlib import Path
import torch
from torchvision import transforms

REPOSITORY = "OwensLab/commfor-model-384"
REVISION = "6076002bf0d9dd37537f965ee2f06f826c333b61"
SHA256 = "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387"
FILE_SIZE = 87262324


def load_community(path, device="cpu"):
    import timm
    from safetensors.torch import load_file
    path = Path(path)
    if path.stat().st_size != FILE_SIZE or hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Community Forensics checkpoint size/hash mismatch")

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k",
                                         pretrained=False, num_classes=1)

        def forward(self, tensor):
            # softmax([0, z])[1] == sigmoid(z); retain the API's REAL/FAKE order.
            logits = self.vit(tensor)
            return torch.cat((torch.zeros_like(logits), logits), dim=1)

    model = Classifier()
    model.load_state_dict(load_file(str(path)), strict=True)
    transform = transforms.Compose([
        transforms.Resize(440, antialias=True), transforms.CenterCrop(384),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    metadata = {"architecture": "Community Forensics ViT-S/16", "input_size": 384,
                "class_to_idx": {"REAL": 0, "FAKE": 1}, "temperature": 1.0,
                "score_type": "uncalibrated_sigmoid", "repository": REPOSITORY,
                "revision": REVISION, "partial_edit_detection_validated": False}
    return model.to(device).eval(), transform, metadata
