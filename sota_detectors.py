"""State-of-the-art 2026 AI-generated image detector adapters for DeepGuard AI.

Includes:
1. SmaugDetector (`abacusai/Smaug-AI-Image-Detector`, Apache 2.0):
   EmbeddingGemma-2 vision-only encoder (439M params) using 3 native-resolution
   528x528 crops. #1 open-source benchmark (0.982 avg AUC across OpenFake Reddit,
   OpenFake Core, and CommunityForensics) covering GPT-Image 1/2, Gemini / Nano
   Banana, Midjourney 6/7, FLUX.1/2, Seedream, Ideogram, DALL-E 3, and Firefly.

2. DetectraOpenFakeDetector (`s-gehring/detectra-openfake`, ONNX Int8):
   Multi-scale ViT-S/16 (384px, 440/576/768px crops + Platt scaling) fine-tuned
   on 2026 generators (FLUX.2, GPT-Image 2, Qwen-Image 2.1, Gemini 3.1 Flash)
   plus ChartQA and museum art negatives for ultra-low false positive rates.
"""
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


# ==============================================================================
# 1. SMAUG-AI-IMAGE-DETECTOR (EmbeddingGemma-2 Vision, 3 x 528px native crops)
# ==============================================================================

SMAUG_CROP = 528
SMAUG_SIDE = 576
SMAUG_BUDGET = 140


def smaug_views(img, crops=3):
    """Official Smaug preprocessing: shorter side -> 576 px, 3x 528x528 crops."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    if min(w, h) > SMAUG_SIDE:
        s = SMAUG_SIDE / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.Resampling.LANCZOS)
    elif min(w, h) < SMAUG_CROP:
        s = SMAUG_CROP / min(w, h)
        img = img.resize((math.ceil(w * s), math.ceil(h * s)), Image.Resampling.BICUBIC)
    w, h = img.size
    cx, cy = (w - SMAUG_CROP) // 2, (h - SMAUG_CROP) // 2
    if crops == 1:
        return [img.crop((cx, cy, cx + SMAUG_CROP, cy + SMAUG_CROP))]
    if w >= h:
        return [img.crop((x, cy, x + SMAUG_CROP, cy + SMAUG_CROP)) for x in (0, cx, w - SMAUG_CROP)]
    return [img.crop((cx, y, cx + SMAUG_CROP, cy + SMAUG_CROP)) for y in (0, cy, h - SMAUG_CROP)]


class SmaugDetector:
    def __init__(self, model_dir, device=None):
        from sentence_transformers import SentenceTransformer

        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.st = SentenceTransformer(
            str(self.model_dir),
            device=self.device,
            config_kwargs={"audio_config": None},
            model_kwargs={"torch_dtype": dtype},
        )
        ip = self.st[0].processor.image_processor
        ip.max_soft_tokens = ip.image_seq_length = SMAUG_BUDGET

    @torch.inference_mode()
    def predict_prob_fake(self, rgb_image, crops=3):
        views = smaug_views(rgb_image, crops=crops)
        flat = [{"image": v} for v in views]
        logits = self.st.encode(
            flat,
            batch_size=len(flat),
            convert_to_tensor=True,
            show_progress_bar=False,
        ).float().view(1, len(views))
        prob = float(torch.sigmoid(logits.mean(1)).item())
        return max(0.0, min(1.0, prob))


# ==============================================================================
# 2. DETECTRA-OPENFAKE (ONNX Multi-Scale ViT-S/16 384px + Platt Scaling)
# ==============================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _detectra_preprocess(rgb_image, short_edge=440, crop_size=384):
    w, h = rgb_image.size
    if w <= h:
        new_w = short_edge
        new_h = max(crop_size, int(round(h * (short_edge / w))))
    else:
        new_h = short_edge
        new_w = max(crop_size, int(round(w * (short_edge / h))))
    resized = rgb_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = max(0, (new_w - crop_size) // 2)
    top = max(0, (new_h - crop_size) // 2)
    cropped = resized.crop((left, top, left + crop_size, top + crop_size))
    arr = np.asarray(cropped, dtype=np.float32) / 255.0  # HWC
    chw = np.transpose(arr, (2, 0, 1))
    norm = (chw - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(norm, axis=0).astype(np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(min(float(x), 30.0), -30.0)))


class DetectraOpenFakeDetector:
    def __init__(self, onnx_path):
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def _run_scale(self, rgb_image, short_edge):
        inp = _detectra_preprocess(rgb_image, short_edge=short_edge, crop_size=384)
        logits = self.session.run(None, {self.input_name: inp})[0]
        logit = float(np.ravel(logits)[0])
        # Calibrated to 0.50 binary operating point (shifts raw logit from 0.95 add-on bias to 0.50 balanced FPR<=3%)
        calibrated = _sigmoid(0.552 * (logit - 2.5))
        return calibrated

    def predict_prob_fake(self, rgb_image):
        p440 = self._run_scale(rgb_image, short_edge=440)
        best = p440
        # Multi-scale native detail zoom when score is in borderline zone on high-res images
        if 0.30 <= p440 < 0.92 and min(rgb_image.size) >= 560:
            p576 = self._run_scale(rgb_image, short_edge=576)
            best = max(best, p576)
            if min(rgb_image.size) >= 740 and best < 0.92:
                p768 = self._run_scale(rgb_image, short_edge=768)
                best = max(best, p768)
        return max(0.0, min(1.0, best))


# ==============================================================================
# 3. DYNAMIC ADMIN-IMPORTED ALGORITHMS (HUGGINGFACE HUB & GENERIC ONNX)
# ==============================================================================

AI_LABEL_KEYWORDS = ("ai", "fake", "artificial", "synthetic", "generated", "deepfake", "sdxl", "flux")


def detect_ai_label_index(id2label):
    """Automatically infer which class index corresponds to AI/Fake from HF config.id2label."""
    if not id2label:
        return 1
    for k, v in id2label.items():
        name = str(v).lower().strip()
        if any(tok == name or tok in name.replace("-", "_").split("_") for tok in AI_LABEL_KEYWORDS):
            return int(k)
    return 1


class HFImageClassifierAdapter:
    """Generic HuggingFace Transformers AutoModelForImageClassification detector."""

    def __init__(self, repo_or_dir, device=None, ai_label_idx=None):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.repo_or_dir = str(repo_or_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoImageProcessor.from_pretrained(self.repo_or_dir)
        self.model = AutoModelForImageClassification.from_pretrained(self.repo_or_dir).to(self.device).eval()
        id2label = getattr(self.model.config, "id2label", {}) or {}
        self.ai_idx = int(ai_label_idx) if ai_label_idx is not None else detect_ai_label_index(id2label)
        self.id2label = id2label

    @torch.inference_mode()
    def predict_prob_fake(self, rgb_image):
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits[0].float()
        if logits.numel() == 1:
            prob = float(torch.sigmoid(logits[0]).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            idx = min(max(self.ai_idx, 0), probs.numel() - 1)
            prob = float(probs[idx].item())
        return max(0.0, min(1.0, prob))


class GenericONNXDetector:
    """Generic ONNX Runtime image classifier adapter for admin-uploaded .onnx models."""

    def __init__(self, onnx_path, input_size=224, ai_label_idx=1):
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        self.input_size = int(input_size)
        self.ai_idx = int(ai_label_idx)
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict_prob_fake(self, rgb_image):
        resized = rgb_image.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        norm = (chw - IMAGENET_MEAN) / IMAGENET_STD
        inp = np.expand_dims(norm, axis=0).astype(np.float32)
        out = np.ravel(self.session.run(None, {self.input_name: inp})[0]).astype(np.float64)
        if out.size == 1:
            return float(_sigmoid(out[0]))
        shifted = out - np.max(out)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps)
        idx = min(max(self.ai_idx, 0), probs.size - 1)
        return float(max(0.0, min(1.0, probs[idx])))
