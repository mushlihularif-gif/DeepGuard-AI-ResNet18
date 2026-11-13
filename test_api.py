import os
from app import app

client = app.test_client()

test_cases = [
    ('REAL', r'C:\Users\LENOVO\Downloads\ARIF-FOTO.png'),
    ('REAL', r'C:\Users\LENOVO\Downloads\KENFOTO.png'),
    ('REAL', r'C:\Users\LENOVO\Downloads\PAS FOTO.png'),
    ('AI  ', r'C:\Users\LENOVO\Downloads\Simulasi Mesin Pesawat dengan Spatial Computing.png'),
    ('AI  ', r'C:\Users\LENOVO\Downloads\ChatGPT Image 1 Jun 2026, 01.53.57.png'),
    ('AI  ', r'C:\Users\LENOVO\Downloads\ChatGPT Image 19 Jun 2026, 06.14.57 (1).png'),
    ('AI  ', r'C:\Users\LENOVO\Downloads\ChatGPT Image Jul 22, 2026, 09_44_20 PM.png'),
    ('REAL', r'C:\Users\LENOVO\Downloads\WhatsApp Image 2026-03-25 at 18.01.34.jpeg'),
]

print("=== PENGUJIAN AKHIR END-TO-END FLASK API /api/predict ===")
for expected, path in test_cases:
    if not os.path.exists(path):
        continue
    with open(path, 'rb') as f:
        data = {'image': (f, os.path.basename(path))}
        resp = client.post('/api/predict', data=data, content_type='multipart/form-data')
    res = resp.get_json()
    exp_label = 'FAKE' if expected.strip() == 'AI' else 'REAL'
    status = '[BENAR ✓]' if res['label'] == exp_label else '[SALAH ✗]'
    fn = os.path.basename(path)
    print(f"  {status} Asli: {expected:4s} | Output: {res['label']:4s} | Real: {res['prob_real']*100:5.1f}% | AI: {res['prob_fake']*100:5.1f}% | {fn} ({res['inference_ms']}ms)")
