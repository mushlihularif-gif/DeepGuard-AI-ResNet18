import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

# 1. Periksa Akselerasi GPU CUDA
if not torch.cuda.is_available():
    print("[PERINGATAN] CUDA tidak terdeteksi, fallback ke CPU.")
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
    print(f"[OK] CUDA Aktif! Menggunakan GPU: {torch.cuda.get_device_name(0)}")
    print(f"     VRAM Tersedia: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# 2. Pemuatan Dataset dari Disk
from datasets import load_from_disk

TRAIN_PATH = r"E:\train_clean_saved"
VAL_PATH = r"E:\val_resmi_saved"
PROJECT_DIR = r"E:\Teknik Informatika\Semester 5\Artificial Intelligence for Security\GACOR DETEKSI CITRA HASIL GENERATIVE AI MENGGUNAKAN RESNET18"
SAVE_PATH = os.path.join(PROJECT_DIR, "models", "resnet18_model.pt")

print("\n[1/4] Memuat Dataset Tiny-GenImage...")
ds_train = load_from_disk(TRAIN_PATH)
ds_val = load_from_disk(VAL_PATH)
print(f"      - Data Training  : {len(ds_train):,} citra")
print(f"      - Data Validasi  : {len(ds_val):,} citra")

# 3. Augmentasi & Data Loader
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FastGenImageDataset(Dataset):
    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert("RGB")
        tensor = self.transform(img)
        label = int(item["label"])  # 0=Real, 1=Fake
        return tensor, label

train_dataset = FastGenImageDataset(ds_train, train_transform)
val_dataset = FastGenImageDataset(ds_val, val_transform)

BATCH_SIZE = 128 if device.type == "cuda" else 16
NUM_WORKERS = 0  # 0 paling aman dan stabil di Windows

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)

# 4. Bangun Model ResNet-18 dengan 100% Parameter Terbuka (Fine-Tuning Penuh)
print("\n[2/4] Menginisialisasi Model ResNet-18...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Ganti layer akhir menjadi 2 kelas: Real (0) & Fake (1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Unfreeze seluruh parameter agar model pintar secara komprehensif!
for param in model.parameters():
    param.requires_grad = True

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"      - Total Parameter : {total_params:,}")
print(f"      - Parameter Dilatih: {trainable_params:,} (100% DILATIH!)")

model = model.to(device)

# Differential Learning Rate (Backbone lebih kecil, FC head lebih besar)
backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
fc_params = [p for n, p in model.named_parameters() if n.startswith("fc")]

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 3e-5, "weight_decay": 1e-4},
    {"params": fc_params, "lr": 3e-4, "weight_decay": 1e-4}
])

EPOCHS = 3
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

print(f"\n[3/4] Mulai Fine-Tuning ({EPOCHS} Epoch, Batch Size={BATCH_SIZE}, Device={device.type.upper()})...\n")

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(train_loader, start=1):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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

        if step % 25 == 0 or step == len(train_loader):
            acc = 100.0 * correct / total
            avg_loss = running_loss / step
            elapsed = time.time() - epoch_start
            print(f"  [Epoch {epoch}/{EPOCHS}] Step {step}/{len(train_loader)} | Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}% | {elapsed:.0f}s")

    scheduler.step()

    # Evaluasi Validasi
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    tp = fp = tn = fn = 0  # 1 = Fake (Positif), 0 = Real (Negatif)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            for p, l in zip(predicted.view(-1), labels.view(-1)):
                p_item = p.item()
                l_item = l.item()
                if l_item == 1 and p_item == 1: tp += 1
                elif l_item == 0 and p_item == 1: fp += 1
                elif l_item == 0 and p_item == 0: tn += 1
                elif l_item == 1 and p_item == 0: fn += 1

    val_acc = 100.0 * val_correct / val_total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    duration = time.time() - epoch_start

    print(f"\n==================== HASIL EPOCH {epoch}/{EPOCHS} ({duration:.1f}s) ====================")
    print(f"  Akurasi Validasi : {val_acc:.2f}% (Sebelumnya: 72.00%)")
    print(f"  Recall Fake (AI) : {recall*100:.2f}% (Sebelumnya: 62.00%)")
    print(f"  Presisi Fake (AI): {precision*100:.2f}%")
    print(f"  F1-Score         : {f1*100:.2f}%")
    print(f"  Confusion Matrix : TP={tp:,} | FP={fp:,} | TN={tn:,} | FN={fn:,}")
    print(f"====================================================================\n")

    # Simpan bobot terbaik (selalu dalam format CPU tensor agar kompatibel)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state, SAVE_PATH)
        print(f"  >>> [MODEL DISIMPAN] Bobot terbaik diperbarui ke {SAVE_PATH} (Akurasi: {val_acc:.2f}%)\n")

print(f"\n[4/4] Selesai! Model ResNet-18 yang telah di-fine-tune siap digunakan di Flask!")
print(f"      File tersimpan di: {SAVE_PATH}")
