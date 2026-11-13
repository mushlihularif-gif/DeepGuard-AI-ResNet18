"""
TRAIN V3: PELATIHAN CERDAS UNIVERSAL - ANTI-OVERFITTING
========================================================
Strategi:
1. Gunakan SELURUH 28.000 data Tiny-GenImage (bukan subset 4.000)
2. Download dataset modern: DALL-E 3, Midjourney v6, SD-XL dari Hugging Face
3. Gambar lokal hanya sebagai VALIDASI AKHIR, BUKAN training (cegah menghafal)
4. Augmentasi robustness: JPEG compression, blur, resize (paksa belajar fitur forensik universal)
5. Validasi pada 7.000 data val_resmi_saved yang TIDAK PERNAH disentuh model
6. Test akhir pada gambar laptop yang TIDAK PERNAH dipakai untuk training
"""
import io
import os
import sys
import time
import random
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import models, transforms
from PIL import Image, ImageFilter
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] {device.type.upper()}: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

PROJECT_DIR = r"E:\Teknik Informatika\Semester 5\Artificial Intelligence for Security\GACOR DETEKSI CITRA HASIL GENERATIVE AI MENGGUNAKAN RESNET18"
SAVE_PATH = os.path.join(PROJECT_DIR, "models", "resnet18_model.pt")
BACKUP_PATH = os.path.join(PROJECT_DIR, "models", "resnet18_hybrid_backup.pt")

# Backup model hybrid sebelumnya
import shutil
if os.path.exists(SAVE_PATH):
    shutil.copy2(SAVE_PATH, BACKUP_PATH)
    print(f"[Backup] Model hybrid sebelumnya disimpan ke: {BACKUP_PATH}")

# ================================================================
# BAGIAN 1: DOWNLOAD DATASET MODERN DARI HUGGING FACE
# ================================================================
print("\n" + "=" * 70)
print("[1/6] MEMUAT DATASET MODERN (Tiny-GenImage + HuggingFace Modern)")
print("=" * 70)

from datasets import load_from_disk, load_dataset

# Load full Tiny-GenImage
ds_train = load_from_disk(r"E:\train_clean_saved")
ds_val = load_from_disk(r"E:\val_resmi_saved")
print(f"  Tiny-GenImage Train : {len(ds_train):,} citra")
print(f"  Tiny-GenImage Val   : {len(ds_val):,} citra")

# Try to load a modern dataset from HuggingFace
modern_ds = None
try:
    print("\n  [Downloading] Dataset modern dari HuggingFace...")
    print("  Mencoba: Hemg/AI-Generated-vs-Real-Images-Datasets ...")
    modern_ds = load_dataset(
        "Hemg/AI-Generated-vs-Real-Images-Datasets",
        split="train",
        streaming=False,
        trust_remote_code=True
    )
    print(f"  [OK] Dataset modern berhasil dimuat: {len(modern_ds):,} sampel")
    print(f"  Features: {modern_ds.features}")
except Exception as e1:
    print(f"  [Info] Dataset 1 gagal: {e1}")
    try:
        print("  Mencoba: julienlucas/midjourney-dalle-sd-dataset ...")
        modern_ds = load_dataset(
            "julienlucas/midjourney-dalle-sd-dataset",
            split="train",
            streaming=False,
            trust_remote_code=True
        )
        print(f"  [OK] Dataset modern berhasil dimuat: {len(modern_ds):,} sampel")
    except Exception as e2:
        print(f"  [Info] Dataset 2 gagal: {e2}")
        try:
            print("  Mencoba: Shanmuk4622/ai-image-detection-dataset ...")
            modern_ds = load_dataset(
                "Shanmuk4622/ai-image-detection-dataset",
                split="train",
                streaming=False,
                trust_remote_code=True
            )
            print(f"  [OK] Dataset modern berhasil dimuat: {len(modern_ds):,} sampel")
        except Exception as e3:
            print(f"  [Info] Dataset 3 gagal: {e3}")
            print("  [FALLBACK] Hanya menggunakan Tiny-GenImage penuh (28k).")

# ================================================================
# BAGIAN 2: AUGMENTASI CERDAS ANTI-OVERFITTING
# ================================================================
print("\n" + "=" * 70)
print("[2/6] AUGMENTASI ROBUSTNESS (JPEG Compression, Blur, Resize)")
print("=" * 70)

class JPEGCompression:
    """Simulasi kompresi JPEG yang menghancurkan artefak superfisial."""
    def __init__(self, quality_range=(30, 95)):
        self.quality_range = quality_range

    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')

class GaussianBlur:
    """Simulasi blur kamera/transmisi."""
    def __init__(self, radius_range=(0.5, 2.0)):
        self.radius_range = radius_range

    def __call__(self, img):
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class RandomDownscaleUpscale:
    """Simulasi resize yang terjadi saat share lewat WhatsApp/Telegram."""
    def __init__(self, scale_range=(0.3, 0.8)):
        self.scale_range = scale_range

    def __call__(self, img):
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        new_w, new_h = max(32, int(w * scale)), max(32, int(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        return img

# Augmentasi kuat yang memaksa model belajar fitur forensik UNIVERSAL
# bukan menghafal tekstur spesifik gambar tertentu
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([JPEGCompression(quality_range=(30, 95))], p=0.4),
    transforms.RandomApply([GaussianBlur(radius_range=(0.5, 2.0))], p=0.3),
    transforms.RandomApply([RandomDownscaleUpscale(scale_range=(0.3, 0.7))], p=0.2),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("  Augmentasi aktif: RandomCrop, HFlip, JPEG(30-95), Blur, Downscale, ColorJitter, Grayscale")

# ================================================================
# BAGIAN 3: DATASET WRAPPERS
# ================================================================
class HfDatasetWrapper(Dataset):
    """Wrapper untuk dataset HuggingFace."""
    def __init__(self, hf_ds, transform, label_key='label', image_key='image', max_items=None):
        self.ds = hf_ds
        self.transform = transform
        self.label_key = label_key
        self.image_key = image_key
        self.max_items = min(len(hf_ds), max_items) if max_items else len(hf_ds)

    def __len__(self):
        return self.max_items

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item[self.image_key]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert('RGB')
        tensor = self.transform(img)
        label = int(item[self.label_key])
        return tensor, label

class ModernDatasetWrapper(Dataset):
    """Wrapper untuk dataset modern HuggingFace yang mungkin punya format berbeda."""
    def __init__(self, hf_ds, transform, max_items=None):
        self.ds = hf_ds
        self.transform = transform
        self.max_items = min(len(hf_ds), max_items) if max_items else len(hf_ds)

        # Auto-detect column names
        cols = hf_ds.column_names if hasattr(hf_ds, 'column_names') else list(hf_ds.features.keys())
        self.image_key = None
        self.label_key = None
        for c in cols:
            cl = c.lower()
            if cl in ['image', 'img', 'photo', 'picture']:
                self.image_key = c
            elif cl in ['label', 'labels', 'class', 'target', 'is_ai', 'is_fake']:
                self.label_key = c

        if self.image_key is None:
            for c in cols:
                ft = str(hf_ds.features[c])
                if 'Image' in ft:
                    self.image_key = c
                    break

        print(f"    ModernDS: image_key='{self.image_key}', label_key='{self.label_key}', cols={cols}")

        # Determine label mapping
        self.label_map = None
        if self.label_key:
            feat = hf_ds.features[self.label_key]
            if hasattr(feat, 'names'):
                print(f"    Label names: {feat.names}")
                # Map to 0=Real, 1=Fake
                names_lower = [n.lower() for n in feat.names]
                self.label_map = {}
                for i, n in enumerate(names_lower):
                    if n in ['real', 'authentic', 'human', 'camera', 'natural', 'nature']:
                        self.label_map[i] = 0
                    else:
                        self.label_map[i] = 1
                print(f"    Label mapping: {self.label_map}")

    def __len__(self):
        return self.max_items

    def __getitem__(self, idx):
        try:
            item = self.ds[idx]
            img = item[self.image_key]
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            img = img.convert('RGB')
            tensor = self.transform(img)

            raw_label = int(item[self.label_key])
            if self.label_map:
                label = self.label_map.get(raw_label, raw_label)
            else:
                label = raw_label
            return tensor, label
        except Exception:
            return torch.zeros(3, 224, 224), 0

# ================================================================
# BAGIAN 4: BUILD TRAINING DATA
# ================================================================
print("\n" + "=" * 70)
print("[3/6] MENYUSUN DATA TRAINING GABUNGAN")
print("=" * 70)

datasets_list = []

# 1. Full Tiny-GenImage (28k)
tiny_ds = HfDatasetWrapper(ds_train, train_transform)
datasets_list.append(tiny_ds)
print(f"  [A] Tiny-GenImage (Full)     : {len(tiny_ds):,} sampel")

# 2. Modern dataset jika berhasil di-download
if modern_ds is not None:
    max_modern = min(len(modern_ds), 20000)  # Cap at 20k to keep training time reasonable
    mod_ds = ModernDatasetWrapper(modern_ds, train_transform, max_items=max_modern)
    datasets_list.append(mod_ds)
    print(f"  [B] Dataset Modern (HF)      : {len(mod_ds):,} sampel")

combined = ConcatDataset(datasets_list)
print(f"\n  TOTAL DATA TRAINING          : {len(combined):,} sampel")
print(f"  (Gambar lokal TIDAK dimasukkan ke training - hanya untuk test akhir)")

# Validation set
val_ds = HfDatasetWrapper(ds_val, val_transform)
print(f"  DATA VALIDASI (Tiny-GenImage) : {len(val_ds):,} sampel")

BATCH_SIZE = 128 if device.type == 'cuda' else 16
train_loader = DataLoader(combined, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))

# ================================================================
# BAGIAN 5: MODEL & TRAINING
# ================================================================
print("\n" + "=" * 70)
print("[4/6] MEMBANGUN & MELATIH MODEL RESNET-18")
print("=" * 70)

# Start from ImageNet pretrained (fresh start, no previous fine-tune bias)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 2)

# Unfreeze ALL layers
for param in model.parameters():
    param.requires_grad = True

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total Parameter      : {total_params:,}")
print(f"  Semua parameter di-unfreeze (100% fine-tuning)")

model = model.to(device)

# Differential learning rate
backbone_params = [p for n, p in model.named_parameters() if not n.startswith('fc')]
fc_params = [p for n, p in model.named_parameters() if n.startswith('fc')]

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-4, 'weight_decay': 1e-4},
    {'params': fc_params, 'lr': 1e-3, 'weight_decay': 1e-4}
])

EPOCHS = 8
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

print(f"  Epochs               : {EPOCHS}")
print(f"  Batch Size           : {BATCH_SIZE}")
print(f"  Backbone LR          : 1e-4")
print(f"  FC Head LR           : 1e-3")
print(f"  Scheduler            : Cosine Annealing")

print(f"\n{'=' * 70}")
print(f"[5/6] MULAI TRAINING...")
print(f"{'=' * 70}\n")

best_val_acc = 0.0
best_val_f1 = 0.0
best_epoch = 0

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if step % 30 == 0 or step == len(train_loader):
            acc = 100.0 * correct / total
            avg_loss = running_loss / step
            elapsed = time.time() - epoch_start
            print(f"  [Epoch {epoch}/{EPOCHS}] Step {step}/{len(train_loader)} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | {elapsed:.0f}s")

    scheduler.step()

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            for p_item, l_item in zip(predicted.view(-1), labels.view(-1)):
                pi, li = p_item.item(), l_item.item()
                if li == 1 and pi == 1: tp += 1
                elif li == 0 and pi == 1: fp += 1
                elif li == 0 and pi == 0: tn += 1
                elif li == 1 and pi == 0: fn += 1

    val_acc = 100.0 * val_correct / val_total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    duration = time.time() - epoch_start

    print(f"\n  ========== EPOCH {epoch}/{EPOCHS} ({duration:.1f}s) ==========")
    print(f"  Val Accuracy   : {val_acc:.2f}%")
    print(f"  Recall Fake    : {recall*100:.2f}%")
    print(f"  Precision Fake : {precision*100:.2f}%")
    print(f"  F1-Score       : {f1*100:.2f}%")
    print(f"  Confusion: TP={tp:,} FP={fp:,} TN={tn:,} FN={fn:,}")

    # Save best model based on F1 score (balanced metric)
    if f1 > best_val_f1:
        best_val_f1 = f1
        best_val_acc = val_acc
        best_epoch = epoch
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state, SAVE_PATH)
        print(f"  >>> [SAVED] Best F1={f1*100:.2f}% at Epoch {epoch}")
    print()

print(f"\n{'=' * 70}")
print(f"[6/6] VERIFIKASI PADA GAMBAR YANG BELUM PERNAH DILIHAT MODEL")
print(f"{'=' * 70}")
print(f"  Model terbaik: Epoch {best_epoch} (Val Acc: {best_val_acc:.2f}%, F1: {best_val_f1*100:.2f}%)")

# Reload best checkpoint
model_eval = models.resnet18(weights=None)
model_eval.fc = nn.Linear(512, 2)
model_eval.load_state_dict(torch.load(SAVE_PATH, map_location=device), strict=True)
model_eval.to(device).eval()

# Test on LOCAL images (NEVER seen during training!)
test_targets = [
    ('REAL', r'C:\Users\LENOVO\Downloads\ARIF-FOTO.png'),
    ('REAL', r'C:\Users\LENOVO\Downloads\KENFOTO.png'),
    ('REAL', r'C:\Users\LENOVO\Downloads\PAS FOTO.png'),
    ('REAL', r'C:\Users\LENOVO\Downloads\WhatsApp Image 2026-03-25 at 18.01.34.jpeg'),
    ('REAL', r'C:\Users\LENOVO\Downloads\WhatsApp Image 2026-07-06 at 17.11.32.jpeg'),
    ('REAL', r'C:\Users\LENOVO\Downloads\patrick.jpg'),
    ('REAL', r'C:\Users\LENOVO\Downloads\buah.jpg'),
    ('REAL', r'C:\Users\LENOVO\Downloads\Bagus_Santoso,_Wakil_Bupati_Bengkalis.jpg'),
    ('FAKE', r'C:\Users\LENOVO\Downloads\Simulasi Mesin Pesawat dengan Spatial Computing.png'),
    ('FAKE', r'C:\Users\LENOVO\Downloads\ChatGPT Image 1 Jun 2026, 01.53.57.png'),
    ('FAKE', r'C:\Users\LENOVO\Downloads\ChatGPT Image 19 Jun 2026, 06.14.57 (1).png'),
    ('FAKE', r'C:\Users\LENOVO\Downloads\ChatGPT Image Jul 22, 2026, 09_44_20 PM.png'),
    ('FAKE', r'C:\Users\LENOVO\Downloads\ChatGPT Image 5 Jun 2026, 12.35.31.png'),
    ('FAKE', r'C:\Users\LENOVO\Downloads\ChatGPT Image 5 Jun 2026, 18.03.28.png'),
    ('FAKE', r'C:\Users\LENOVO\Downloads\ChatGPT Image Jun 5, 2026, 05_04_13 AM.png'),
]

n_correct = 0
n_total = 0
for expected, path in test_targets:
    if not os.path.exists(path):
        continue
    n_total += 1
    img = Image.open(path).convert('RGB')
    tensor = val_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model_eval(tensor)
        probs = F.softmax(out, dim=1)[0]
    pred = 'FAKE' if probs[1] > probs[0] else 'REAL'
    ok = pred == expected
    if ok: n_correct += 1
    status = '[BENAR ✓]' if ok else '[SALAH ✗]'
    fn = os.path.basename(path)
    print(f"  {status} Asli: {expected:4s} | Pred: {pred:4s} | Real: {probs[0]*100:5.1f}% | AI: {probs[1]*100:5.1f}% | {fn}")

print(f"\n  Skor: {n_correct}/{n_total} gambar benar ({100*n_correct/max(n_total,1):.0f}%)")
print(f"  (Gambar-gambar ini TIDAK PERNAH dipakai untuk training)")
print(f"\n  Bobot model tersimpan di: {SAVE_PATH}")
