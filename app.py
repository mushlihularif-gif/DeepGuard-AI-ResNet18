import io
import os
import time
from flask import Flask, render_template, request, jsonify

# Dukungan opsional PyTorch & Torchvision (otomatis digunakan jika tersedia)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    from PIL import Image
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
model_init_error = None
device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')

def init_model_on_startup():
    """Memuat struktur arsitektur ResNet18 dan bobot terlatih (2 kelas: Real & Fake) ke GPU/RAM."""
    global resnet_model, preprocess_pipeline, model_init_error
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

    try:
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) <= 1024:
            model_init_error = f"Berkas checkpoint '{MODEL_PATH}' tidak ditemukan atau hanya berupa placeholder."
            resnet_model = None
            return

        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # Kelas 0: Real, Kelas 1: Fake

        # Muat bobot hasil training dengan strict=True untuk memastikan kecocokan arsitektur
        state_dict = torch.load(MODEL_PATH, map_location=device)
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
# MODUL 3: PREPROCESSING CITRA (STANDAR RESNET-18: 224x224 RGB & NORMALISASI)
# ==============================================================================
def preprocess_image(img):
    """Memproses citra ke dimensi standar input ResNet-18 (224x224) dengan normalisasi ImageNet."""
    tensor = preprocess_pipeline(img)
    return tensor.unsqueeze(0).to(device)


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
        img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')

        # 5 & 6. Proses Inferensi (torch.no_grad)
        if not TORCH_AVAILABLE or resnet_model is None or preprocess_pipeline is None:
            err_msg = model_init_error or "Model ResNet-18 belum siap atau checkpoint tidak valid."
            return jsonify({'error': f"Inferensi gagal: {err_msg}"}), 503

        input_tensor = preprocess_image(img)
        with torch.no_grad():
            logits = resnet_model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]

        prob_real = float(probs[0].item())
        prob_fake = float(probs[1].item())

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
