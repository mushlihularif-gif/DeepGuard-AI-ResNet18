import io
import os
from PIL import Image
from app import app
from train_hybrid import generate_realistic_camera_sample, generate_synthetic_ai_sample

client = app.test_client()

print("=" * 72)
print(" PENGUJIAN AKHIR END-TO-END FLASK API (/api/predict) - DEEPGUARD AI")
print("=" * 72)

# Uji pada sampel foto kamera asli (REAL) dan sampel citra sintetis AI (FAKE)
samples = [
    ("REAL", "foto_kamera_asli_1.jpg", generate_realistic_camera_sample(101), "JPEG"),
    ("REAL", "pasfoto_kamera_2.jpg", generate_realistic_camera_sample(202), "JPEG"),
    ("REAL", "whatsapp_image_3.jpg", generate_realistic_camera_sample(303), "JPEG"),
    ("FAKE", "chatgpt_image_ai_1.png", generate_synthetic_ai_sample(404), "PNG"),
    ("FAKE", "midjourney_v6_render_2.png", generate_synthetic_ai_sample(505), "PNG"),
    ("FAKE", "stable_diffusion_ai_3.png", generate_synthetic_ai_sample(606), "PNG"),
]

correct = 0
for expected, fname, pil_img, fmt in samples:
    buf = io.BytesIO()
    if fmt == "PNG":
        pil_img = pil_img.resize((1024, 1024))
    pil_img.save(buf, format=fmt)
    buf.seek(0)

    resp = client.post(
        '/api/predict',
        data={'image': (buf, fname)},
        content_type='multipart/form-data'
    )
    res = resp.get_json()
    ok = res['label'] == expected
    if ok:
        correct += 1
    status = "[BENAR]" if ok else "[SALAH]"
    print(f" {status} Target: {expected:4s} | Prediksi: {res['label']:4s} | Real: {res['prob_real']*100:5.1f}% | Fake: {res['prob_fake']*100:5.1f}% | {fname} ({res['inference_ms']}ms)")

print("-" * 72)
print(f" HASIL AKHIR: {correct}/{len(samples)} Sampel Uji Tepat ({correct/len(samples)*100:.1f}%)")
print("=" * 72)
