"""Flask image classifier. Startup loads a checkpoint; it never trains one."""
import hashlib
import io
import json
import os
import time
import warnings
from pathlib import Path
from threading import Lock
from flask import Flask, Request, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

class InMemoryRequest(Request):
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        # Werkzeug otherwise spools uploads larger than 500 KB to a temp file.
        # MAX_CONTENT_LENGTH bounds the whole multipart request before processing.
        return io.BytesIO()


app = Flask(__name__)
app.request_class = InMemoryRequest
MAX_FILE_BYTES = 10 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_BYTES + 64 * 1024
Image.MAX_IMAGE_PIXELS = 25_000_000
ROOT = Path(__file__).resolve().parent
ENGINE = os.environ.get("DEEPGUARD_ENGINE", "community_forensics")
MODEL_PATH = Path(os.environ.get("DEEPGUARD_MODEL_PATH", ROOT / "models" / (
    "resnet18_model.pt" if ENGINE == "resnet18" else "community_forensics.safetensors")))
detector_model = None
preprocess_pipeline = None
model_metadata = {}
model_init_error = None
model_sha256 = None
inference_lock = Lock()
device = None


def init_model_on_startup():
    global detector_model, preprocess_pipeline, model_metadata, model_init_error, device, model_sha256
    detector_model, preprocess_pipeline, model_metadata, model_sha256 = None, None, {}, None
    try:
        import torch
        from modeling import load_checkpoint, preprocessing
        torch.set_num_threads(4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if ENGINE == "community_forensics":
            from community_detector import load_community
            model, transform, metadata = load_community(MODEL_PATH, device)
        elif ENGINE == "resnet18":
            model, metadata = load_checkpoint(MODEL_PATH, device)
            transform = preprocessing()
            metadata.update(architecture="ResNet18", input_size=224, score_type="temperature_scaled_softmax")
        else:
            raise ValueError("DEEPGUARD_ENGINE must be community_forensics or resnet18")
        digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        # Only show an external result if it belongs to this exact checkpoint.
        metadata.pop("evaluation", None)
        report_path = ROOT / "reports/cross_domain.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["model_sha256"].get(ENGINE) == digest:
                metadata["evaluation"] = {"all": report["overall"][ENGINE]}
        detector_model, model_metadata = model, metadata
        preprocess_pipeline, model_sha256 = transform, digest
        model_init_error = None
    except Exception:
        model_init_error = "Model belum siap. Periksa checkpoint dan dependensi pada log server."
        app.logger.exception("Failed to load selected detector: %s", ENGINE)


def model_status():
    return {"loaded": detector_model is not None, "engine": ENGINE,
            "architecture": model_metadata.get("architecture", "Model belum siap"),
            "input_size": model_metadata.get("input_size", 384 if ENGINE == "community_forensics" else 224),
            "score_type": model_metadata.get("score_type"),
            "score_description": ("Sigmoid belum dikalibrasi" if ENGINE == "community_forensics" else "Softmax terkalibrasi"),
            "model_sha256": model_sha256, "device": str(device), "error": model_init_error,
            "class_to_idx": model_metadata.get("class_to_idx"),
            "temperature": model_metadata.get("temperature"),
            "evaluation": model_metadata.get("evaluation"),
            "partial_edit_detection_validated": False}


@app.context_processor
def page_model_info():
    return {"model_info": model_status()}


@app.get("/")
@app.get("/index.html")
def beranda():
    return render_template("index.html")


@app.get("/deteksi")
@app.get("/deteksi.html")
def deteksi():
    return render_template("deteksi.html")


@app.get("/panduan")
@app.get("/panduan.html")
def panduan():
    return render_template("panduan.html")


@app.get("/tentang")
@app.get("/tentang.html")
def tentang():
    return render_template("tentang.html")


@app.get("/api/model-status")
def get_model_status():
    return jsonify(model_status())


def decision(prob_fake):
    if prob_fake <= 0.2:
        return "REAL", "Lebih menyerupai kelas non-AI pada data latihan; bukan bukti foto asli atau bebas edit AI."
    if prob_fake >= 0.8:
        return "FAKE", "Ada indikasi citra AI menurut model. Periksa asal gambar untuk mengonfirmasi."
    return "UNCERTAIN", "Skor berada di area ragu. Model belum bisa memberi keputusan yang cukup kuat."


def is_valid_file(filename, mimetype):
    return (Path(filename or "").suffix.lower() in {".jpg", ".jpeg", ".png"}
            and (not mimetype or mimetype.lower() in {"image/jpeg", "image/jpg", "image/png"}))


@app.post("/api/predict")
def predict_image():
    started = time.perf_counter()
    try:
        file = request.files.get("image")
        if file is None or not file.filename:
            return jsonify(error="Silakan pilih berkas gambar terlebih dahulu."), 400
        if not is_valid_file(file.filename, file.mimetype):
            return jsonify(error="Format tidak didukung. Unggah JPG atau PNG."), 400
        raw = file.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise RequestEntityTooLarge()
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    return jsonify(error="Isi berkas harus benar-benar JPEG atau PNG."), 400
                if getattr(image, "n_frames", 1) > 1:
                    return jsonify(error="Gunakan gambar diam, bukan PNG animasi."), 400
                if min(image.size) < 32:
                    return jsonify(error="Gambar terlalu kecil; minimum 32 piksel per sisi."), 400
                image.load()
                rgb = ImageOps.exif_transpose(image).convert("RGB")
    except RequestEntityTooLarge:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return jsonify(error="Gambar rusak, tidak valid, atau resolusinya terlalu besar (maksimum 25 megapiksel)."), 400

    if detector_model is None or preprocess_pipeline is None:
        return jsonify(error=model_init_error or "Model belum siap."), 503
    try:
        import torch
        with inference_lock, torch.inference_mode():
            logits = detector_model(preprocess_pipeline(rgb).unsqueeze(0).to(device))
            probabilities = torch.softmax(logits / model_metadata["temperature"], dim=1)[0]
            if not torch.isfinite(probabilities).all():
                raise ValueError("Non-finite model output")
            prob_fake = float(probabilities[1])
        label, summary = decision(prob_fake)
        notes = ["Skor model bukan persentase area gambar yang diedit AI.",
                 "Edit lokal seperti ganti baju, wajah, atau latar belum tervalidasi; hasil REAL tidak menyingkirkannya."]
        if model_metadata.get("score_type") == "uncalibrated_sigmoid":
            notes.append("Skor sigmoid belum dikalibrasi: angka tinggi bukan jaminan tingkat keakuratan.")
            notes.append("Model memakai potongan tengah gambar; edit di tepi bisa tidak teramati.")
        if min(rgb.size) < 128:
            notes.append("Resolusi rendah membatasi detail yang bisa diperiksa.")
        return jsonify(label=label, prob_real=1.0-prob_fake, prob_fake=prob_fake,
                       inference_ms=round((time.perf_counter()-started)*1000, 1),
                       summary=summary, warnings=notes, model_sha256=model_sha256,
                       architecture=model_metadata.get("architecture", "ResNet18"),
                       input_size=model_metadata.get("input_size", 224),
                       score_type=model_metadata.get("score_type", "temperature_scaled_softmax"),
                       partial_edit_detection_validated=False)
    except Exception:
        app.logger.exception("Inference failed")
        return jsonify(error="Inferensi gagal. Periksa log server."), 500


@app.errorhandler(413)
def too_large(error):
    return jsonify(error="Ukuran berkas melebihi batas maksimal 10 MB."), 413


init_model_on_startup()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
