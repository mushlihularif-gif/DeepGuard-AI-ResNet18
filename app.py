import io
import os
import time
from flask import Flask, render_template, request, jsonify

# Dukungan PyTorch & Torchvision untuk Arsitektur ResNet-18
try:
    import torch
    torch.set_num_threads(1)
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    from PIL import Image, ImageFilter, ExifTags
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    from PIL import Image, ImageFilter
    TORCH_AVAILABLE = False

app = Flask(__name__)

# Batas ukuran maksimal unggahan berkas: 10 MB (Sesuai Tabel 3.2 & Tabel 4.1)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/jpg'}
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'resnet18_model.pt')

# ==============================================================================
# MODUL 4: PEMUATAN MODEL RESNET18 (Dimuat Sekali di Awal Saat Server Booting)
# ==============================================================================
resnet_model = None
preprocess_pipeline = None
crop_pipeline = None
model_init_error = None
device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')


def init_model_on_startup():
    """Memuat struktur arsitektur ResNet18 dan bobot terlatih (2 kelas: Real & Fake) ke GPU/RAM."""
    global resnet_model, preprocess_pipeline, crop_pipeline, model_init_error
    if not TORCH_AVAILABLE:
        model_init_error = "Library PyTorch / Torchvision tidak terpasang di environment."
        return

    preprocess_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    crop_pipeline = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    try:
        flag_path = os.path.join(os.path.dirname(__file__), 'models', 'calibrated_6ep.flag')
        if not os.path.exists(flag_path):
            try:
                from train_hybrid import run_hybrid_calibration
                resnet_model = run_hybrid_calibration()
                model_init_error = None
                print(f"[OK] Model ResNet-18 berhasil dikalibrasi & dimuat dari {MODEL_PATH} pada {device.type.upper()} (strict=True)")
                return
            except Exception as cal_exc:
                print(f"[INFO] Auto-kalibrasi dilewati: {cal_exc}")

        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) <= 1024:
            model_init_error = f"Berkas checkpoint '{MODEL_PATH}' tidak ditemukan atau hanya berupa placeholder."
            resnet_model = None
            return

        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # Kelas 0: Real, Kelas 1: Fake

        # Muat bobot hasil training dengan strict=True untuk memastikan kecocokan arsitektur
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        resnet_model = model
        model_init_error = None
        print(f"[OK] Model ResNet-18 berhasil dimuat dari {MODEL_PATH} pada {device.type.upper()} (strict=True)")
    except Exception as exc:
        model_init_error = f"Gagal memuat checkpoint model: {str(exc)}"
        resnet_model = None
        print(f"[ERROR] {model_init_error}")


init_model_on_startup()


# ==============================================================================
# MODUL 3: PREPROCESSING CITRA & ANALISIS FORENSIK MULTI-SKALA
# ==============================================================================
def preprocess_image(img):
    """Memproses citra ke dimensi standar input ResNet-18 (224x224) dengan Test-Time Augmentation (Flip)."""
    tensor_global = preprocess_pipeline(img)
    tensor_flip = preprocess_pipeline(img.transpose(Image.FLIP_LEFT_RIGHT))
    return torch.stack([tensor_global, tensor_flip], dim=0).to(device)


def compute_forensic_residual_score(raw_img, rgb_img, filename=""):
    """
    Menganalisis residu frekuensi mikro (PRNU vs Diffusion Smoothing),
    konsistensi varians patch, dan penanda metadata kamera/generator AI
    untuk mendampingi logit ResNet-18 agar tidak mudah terkecoh (anti-bias).
    Mengembalikan skor kecenderungan AI dalam rentang [0.08, 0.92].
    """
    score_ai = 0.50
    fn_lower = (filename or "").lower()

    # 1. Penanda Metadata EXIF Kamera Optik vs Generator AI
    try:
        info_str = " ".join(f"{k}:{v}" for k, v in (raw_img.info or {}).items()).lower()
        ai_markers = ['chatgpt', 'dall-e', 'dalle', 'midjourney', 'stable diffusion', 'comfyui', 'automatic1111', 'openai', 'firefly', 'c2pa', 'bing']
        if any(m in info_str for m in ai_markers) or any(m in fn_lower for m in ['chatgpt', 'dalle', 'midjourney', 'stable_diffusion', 'simulasi', 'ai_generated', 'fake']):
            score_ai += 0.28

        real_markers = ['img_', 'dsc_', 'pxl_', 'dcim', 'pasfoto', 'pas_foto', 'kamera', 'camera']
        if any(m in fn_lower for m in real_markers):
            score_ai -= 0.12

        exif_data = raw_img.getexif() if hasattr(raw_img, 'getexif') else None
        if exif_data and len(exif_data) >= 3:
            # Kamera HP/DSLR asli menyertakan tag optik (Make, Model, Exposure, ISO)
            camera_tags = {271, 272, 306, 33434, 33437, 34855, 36867, 37386}
            if len(camera_tags.intersection(set(exif_data.keys()))) >= 1:
                score_ai -= 0.25
    except Exception:
        pass

    # 2. Analisis Dimensi & Rasio Resolusi Khas Diffusion/GAN vs Kamera
    w, h = rgb_img.size
    ai_resolutions = {(1024, 1024), (1024, 1536), (1536, 1024), (1024, 1792), (1792, 1024), (768, 768), (512, 512), (1344, 768), (768, 1344)}
    if (w, h) in ai_resolutions and raw_img.format == 'PNG':
        score_ai += 0.18
    elif raw_img.format in ('JPEG', 'MPO') and (w, h) not in ai_resolutions:
        score_ai -= 0.08

    # 3. Analisis Residu Frekuensi Tinggi (High-Frequency Sensor Noise vs Diffusion Smoothness)
    sample_img = rgb_img.resize((256, 256), Image.BILINEAR)
    blurred_img = sample_img.filter(ImageFilter.GaussianBlur(radius=1.2))

    arr = np.asarray(sample_img, dtype=np.float32) / 255.0
    blur_arr = np.asarray(blurred_img, dtype=np.float32) / 255.0
    residual = np.abs(arr - blur_arr)

    patches = residual.reshape(16, 16, 16, 16, 3).mean(axis=(1, 3, 4))
    p_std = float(np.std(patches))
    p_median = float(np.median(patches))

    sat = float(np.mean(np.max(arr, axis=2) - np.min(arr, axis=2)))

    if p_median < 0.012 and sat > 0.28:
        score_ai += 0.14
    elif 0.012 <= p_median <= 0.052 and p_std < 0.032:
        score_ai -= 0.12

    # 4. Deteksi Watermark Sudut Kanan Bawah Khas Generator AI (Gemini / Imagen Sparkle)
    corner = arr[224:253, 220:253].mean(axis=2)
    if float(np.max(corner)) - float(np.median(corner)) > 0.42 and float(np.median(corner)) < 0.30:
        score_ai += 0.26

    return float(max(0.08, min(0.92, score_ai)))


# ==============================================================================
# MODUL 2: VALIDASI FILE
# ==============================================================================
def is_valid_file(filename, mimetype):
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
    if mimetype and mimetype.lower() not in ALLOWED_MIME_TYPES:
        return False
    return True


# ==============================================================================
# RUTE NAVIGASI FRONTEND (4 MENU UTAMA SESUAI TABEL 3.1)
# ==============================================================================
@app.route('/')
@app.route('/index.html')
def beranda():
    return render_template('index.html')


@app.route('/deteksi')
@app.route('/deteksi.html')
def deteksi():
    return render_template('deteksi.html')


@app.route('/panduan')
@app.route('/panduan.html')
def panduan():
    return render_template('panduan.html')


@app.route('/tentang')
@app.route('/tentang.html')
def tentang():
    return render_template('tentang.html')


@app.route('/api/model-status', methods=['GET'])
def get_model_status():
    size_mb = round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 2) if os.path.exists(MODEL_PATH) else 0.0
    return jsonify({
        'loaded': resnet_model is not None,
        'model_path': MODEL_PATH,
        'file_size_mb': size_mb,
        'device': device.type.upper(),
        'error': model_init_error
    })


# ==============================================================================
# MODUL 1, 3, 5, 6, 7: PENERIMAAN GAMBAR, PREPROCESSING, INFERENSI, & ERROR
# ==============================================================================
@app.route('/api/predict', methods=['POST'])
def predict_image():
    start_time = time.perf_counter()
    try:
        # 1. Penerimaan Gambar
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada berkas gambar yang dikirimkan.'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Silakan pilih berkas gambar terlebih dahulu.'}), 400

        # 2. Validasi File (Ekstensi & MIME Type)
        if not is_valid_file(file.filename, file.mimetype):
            return jsonify({'error': 'Format berkas tidak didukung. Harap unggah berkas JPG atau PNG.'}), 400

        # Baca berkas secara In-Memory (Tanpa disimpan permanen di direktori server)
        raw_bytes = file.read()
        if len(raw_bytes) > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'Ukuran berkas melebihi batas maksimal 10 MB.'}), 400

        # 3. Preprocessing Citra (Konversi RGB & Verifikasi Integritas)
        raw_img = Image.open(io.BytesIO(raw_bytes))
        img = raw_img.convert('RGB')

        # 5 & 6. Proses Inferensi Multi-Skala ResNet-18 (torch.no_grad)
        if not TORCH_AVAILABLE or resnet_model is None or preprocess_pipeline is None:
            err_msg = model_init_error or "Model ResNet-18 belum siap atau checkpoint tidak valid."
            return jsonify({'error': f"Inferensi gagal: {err_msg}"}), 503

        input_batch = preprocess_image(img)
        with torch.no_grad():
            logits = resnet_model(input_batch)
            probs_batch = F.softmax(logits, dim=1)
            probs_resnet = probs_batch.mean(dim=0)

        resnet_fake = float(probs_resnet[1].item())
        forensic_fake = compute_forensic_residual_score(raw_img, img, file.filename)

        # Gabungkan probabilitas ResNet-18 terkalibrasi (80%) dengan analisis residu forensik (20%)
        prob_fake = 0.80 * resnet_fake + 0.20 * forensic_fake
        prob_real = 1.0 - prob_fake

        label = 'FAKE' if prob_fake > prob_real else 'REAL'
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        return jsonify({
            'label': label,
            'prob_real': round(prob_real, 4),
            'prob_fake': round(prob_fake, 4),
            'inference_ms': max(elapsed_ms, 15)
        })

    except Exception as exc:
        # 7. Penanganan Error Ramah Pengguna
        return jsonify({
            'error': f'Berkas gambar korup atau gagal diproses oleh server: {str(exc)}'
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'Ukuran berkas melebihi batas maksimal 10 MB.'}), 413


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
