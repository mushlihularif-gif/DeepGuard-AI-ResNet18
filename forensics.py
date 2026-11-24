"""Real mathematical & visual image forensics module for DeepGuard AI.

Every value, heatmap, and spectrum returned here is directly computed from the
uploaded image pixels and bytes — never hardcoded or fabricated.
"""
import base64
import io
import math
import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter, ExifTags


# Tactical dark-to-amber-to-crimson heatmap LUT (256 x 3 uint8) for clean cyber visuals
def _build_heatmap_lut():
    # 0 -> deep obsidian (#0B0E14), 85 -> cobalt (#0EA5E9), 170 -> amber (#F59E0B), 255 -> crimson (#EF4444)
    anchors = [
        (0, (11, 14, 20)),
        (64, (14, 74, 138)),
        (128, (14, 165, 233)),
        (192, (245, 158, 11)),
        (255, (239, 68, 68)),
    ]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for idx in range(len(anchors) - 1):
        x0, c0 = anchors[idx]
        x1, c1 = anchors[idx + 1]
        for x in range(x0, x1 + 1):
            t = (x - x0) / max(x1 - x0, 1)
            lut[x] = [int(c0[k] + t * (c1[k] - c0[k])) for k in range(3)]
    return lut


HEATMAP_LUT = _build_heatmap_lut()


def _pil_to_data_uri(pil_image, fmt="PNG", quality=85):
    buf = io.BytesIO()
    if fmt == "JPEG":
        pil_image.convert("RGB").save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        pil_image.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _fit_display_size(image, max_side=512):
    w, h = image.size
    if max(w, h) <= max_side:
        return image.copy()
    scale = max_side / float(max(w, h))
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)


def extract_exif_metadata(raw_bytes):
    """Extract informational EXIF fields from raw bytes (never used to alter neural scores)."""
    info = {
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "datetime": None,
        "field_count": 0,
    }
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            exif = img.getexif()
            if not exif:
                return info
            tag_map = ExifTags.TAGS
            decoded = {}
            for k, v in exif.items():
                name = tag_map.get(k, str(k))
                if isinstance(v, bytes):
                    try:
                        v = v.decode("utf-8", errors="ignore").strip("\x00").strip()
                    except Exception:
                        continue
                if isinstance(v, (str, int, float)):
                    decoded[name] = str(v).strip()
            info["field_count"] = len(decoded)
            info["has_exif"] = len(decoded) > 0
            info["camera_make"] = decoded.get("Make") or None
            info["camera_model"] = decoded.get("Model") or None
            info["software"] = decoded.get("Software") or None
            info["datetime"] = decoded.get("DateTimeOriginal") or decoded.get("DateTime") or None
    except Exception:
        pass
    return info


def compute_ela_analysis(rgb_image, include_visual=True):
    """Real Error Level Analysis (JPEG quality 90 resave residual)."""
    work = _fit_display_size(rgb_image, max_side=512)
    buf = io.BytesIO()
    work.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    with Image.open(buf) as resaved:
        diff = ImageChops.difference(work, resaved.convert("RGB"))
        diff_arr = np.asarray(diff, dtype=np.float32)

    ela_mean = float(np.mean(diff_arr))
    ela_std = float(np.std(diff_arr))
    ela_max = float(np.max(diff_arr))

    # Block-wise ELA dispersion (8x8 patches) to detect localized inconsistencies
    gray_diff = np.mean(diff_arr, axis=2)
    h, w = gray_diff.shape
    bh, bw = max(1, h // 8), max(1, w // 8)
    trimmed = gray_diff[: bh * 8, : bw * 8]
    if trimmed.size >= 64:
        blocks = trimmed.reshape(bh, 8, bw, 8).mean(axis=(1, 3))
        block_cv = float(np.std(blocks) / (np.mean(blocks) + 1e-5))
    else:
        block_cv = 0.0

    visual_uri = None
    if include_visual:
        # Scale residual dynamically so visual inspection is clear without clipping everything
        scale = 255.0 / max(float(np.percentile(gray_diff, 99.2)), 12.0)
        norm = np.clip(gray_diff * scale, 0, 255).astype(np.uint8)
        colored = HEATMAP_LUT[norm]
        visual_uri = _pil_to_data_uri(Image.fromarray(colored, mode="RGB"), fmt="JPEG", quality=84)

    return {
        "mean": round(ela_mean, 2),
        "std": round(ela_std, 2),
        "max": round(ela_max, 1),
        "block_cv": round(block_cv, 3),
        "map_data_uri": visual_uri,
    }


def compute_fft_and_noise(rgb_image, include_visual=True):
    """Real 2D Fast Fourier Transform frequency spectrum + high-pass noise residual + Laplacian sharpness."""
    work = _fit_display_size(rgb_image, max_side=512)
    gray_pil = work.convert("L")
    gray = np.asarray(gray_pil, dtype=np.float32)

    # 1. High-pass noise residual (approx sensor noise floor via median/box residual)
    smooth = np.asarray(gray_pil.filter(ImageFilter.MedianFilter(size=3)), dtype=np.float32)
    residual = gray - smooth
    noise_sigma = float(np.std(residual))

    # 2. Edge sharpness via discrete 3x3 Laplacian kernel
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        lap = (
            -4.0 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        laplacian_var = float(np.var(lap))
    else:
        laplacian_var = 0.0

    # 3. 2D-FFT on 256x256 grayscale
    fft_gray = np.asarray(rgb_image.convert("L").resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
    # Apply Hann window to eliminate border leakage
    wy = np.hanning(256)
    wx = np.hanning(256)
    window = np.outer(wy, wx)
    f = np.fft.fft2((fft_gray - np.mean(fft_gray)) * window)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    # Radial frequency bands
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    low_band = float(np.mean(mag[(dist >= 4) & (dist < 28)]))
    mid_band = float(np.mean(mag[(dist >= 28) & (dist < 72)]))
    high_band = float(np.mean(mag[(dist >= 72) & (dist <= 120)]))
    high_to_low = high_band / max(low_band, 1e-5)
    high_to_mid = high_band / max(mid_band, 1e-5)

    # Azimuthal peakiness in high frequencies (captures periodic upsampling spikes)
    hf_vals = mag[(dist >= 48) & (dist <= 115)]
    hf_peak_z = float((np.percentile(hf_vals, 99.5) - np.mean(hf_vals)) / (np.std(hf_vals) + 1e-5))

    visual_uri = None
    if include_visual:
        p_low, p_high = float(np.percentile(mag, 5)), float(np.percentile(mag, 99.5))
        norm = np.clip((mag - p_low) / max(p_high - p_low, 1e-5) * 255.0, 0, 255).astype(np.uint8)
        colored = HEATMAP_LUT[norm]
        visual_uri = _pil_to_data_uri(Image.fromarray(colored, mode="RGB"), fmt="JPEG", quality=85)

    return {
        "noise_sigma": round(noise_sigma, 2),
        "laplacian_var": round(laplacian_var, 1),
        "high_to_low_ratio": round(high_to_low, 3),
        "high_to_mid_ratio": round(high_to_mid, 3),
        "hf_peak_z": round(hf_peak_z, 2),
        "spectrum_data_uri": visual_uri,
    }


def score_ela_fft_detector(ela_metrics, fft_metrics):
    """Deterministic physics/frequency-domain forensic score (no neural network).

    Combines 4 measurable signal properties:
    - Spectral high-frequency anomaly / peakiness (hf_peak_z)
    - High-to-mid spectral roll-off deviation (high_to_mid_ratio)
    - High-pass noise floor smoothness (noise_sigma)
    - ELA compression residual uniformity (mean & block_cv)
    """
    # Calibrated logistic combination over physically interpretable signal features
    z = 0.0
    # Very smooth noise floor (< 3.5) is common in diffusion outputs; very high (> 8) leans natural sensor
    z += (4.5 - min(fft_metrics["noise_sigma"], 12.0)) * 0.22
    # Sharp periodic spikes in high-frequency FFT indicate transposed-conv / upsampling grid artifacts
    z += (fft_metrics["hf_peak_z"] - 2.65) * 0.95
    # Unusual high-to-mid spectral flatness or sharp drop
    z += (fft_metrics["high_to_mid_ratio"] - 0.84) * 3.2
    # Low ELA mean (< 2.5) on clean PNGs/re-rendered synthetic images vs higher texture error on natural photos
    z += (3.0 - min(ela_metrics["mean"], 8.0)) * 0.30
    # Localized block CV spike (> 0.65) can indicate spliced/edited regions
    if ela_metrics["block_cv"] > 0.65:
        z += (ela_metrics["block_cv"] - 0.65) * 0.8

    prob_fake = 1.0 / (1.0 + math.exp(-max(min(z, 6.0), -6.0)))
    prob_fake = float(max(0.02, min(0.98, prob_fake)))
    return {
        "prob_fake": prob_fake,
        "prob_real": 1.0 - prob_fake,
    }


def _find_last_conv_layer(model):
    """Locate the final convolutional/feature stage for Grad-CAM."""
    if hasattr(model, "layer4"):
        return model.layer4
    if hasattr(model, "features"):
        return model.features[-1]
    return None


def compute_gradcam(model, transform, rgb_image, target_device, target_class=1):
    """Compute a real Grad-CAM spatial activation overlay for CNN backbones (ResNet-18, MobileNet, EfficientNet)."""
    target_layer = _find_last_conv_layer(model)
    if target_layer is None:
        return None

    activations = []
    gradients = []

    def fwd_hook(_module, _inp, output):
        activations.append(output.detach())

    def bwd_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            tensor = transform(rgb_image).unsqueeze(0).to(target_device)
            tensor.requires_grad_(True)
            logits = model(tensor)
            score = logits[0, target_class]
            score.backward()

        if not activations or not gradients:
            return None

        acts = activations[0][0]   # [C, H, W]
        grads = gradients[0][0]    # [C, H, W]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=0))
        cam_np = cam.cpu().numpy()
        max_val = float(cam_np.max())
        if max_val <= 1e-8:
            return None
        cam_np = cam_np / max_val

        # Identify which region had highest activation for plain-language explanation
        gh, gw = cam_np.shape
        peak_y, peak_x = np.unravel_index(int(np.argmax(cam_np)), cam_np.shape)
        vert = "atas" if peak_y < gh / 3 else ("bawah" if peak_y > 2 * gh / 3 else "tengah")
        horiz = "kiri" if peak_x < gw / 3 else ("kanan" if peak_x > 2 * gw / 3 else "tengah")
        region_label = "tengah" if (vert == "tengah" and horiz == "tengah") else f"{vert}-{horiz}"
        hot_area_pct = round(float((cam_np >= 0.55).mean() * 100.0), 1)

        # Blend Grad-CAM onto fitted display image
        display = _fit_display_size(rgb_image, max_side=512).convert("RGB")
        cam_resized = np.asarray(
            Image.fromarray((cam_np * 255).astype(np.uint8)).resize(display.size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        heat_rgb = HEATMAP_LUT[(cam_resized * 255).astype(np.uint8)].astype(np.float32)
        base_rgb = np.asarray(display, dtype=np.float32)
        # Adaptive alpha: untouched dark pixels stay crisp, hot activation areas highlight strongly
        alpha = np.clip(cam_resized * 0.68, 0.08, 0.68)[:, :, None]
        blended = np.clip(base_rgb * (1.0 - alpha) + heat_rgb * alpha, 0, 255).astype(np.uint8)
        overlay_uri = _pil_to_data_uri(Image.fromarray(blended, mode="RGB"), fmt="JPEG", quality=85)

        return {
            "overlay_data_uri": overlay_uri,
            "peak_region": region_label,
            "hot_area_pct": hot_area_pct,
        }
    except Exception:
        return None
    finally:
        h_fwd.remove()
        h_bwd.remove()
        model.zero_grad(set_to_none=True)


def build_key_findings(prob_fake, label, active_algo_name, comparison, ela_metrics, fft_metrics, exif_info, provenance, gradcam_info):
    """Build 3-4 concise, high-signal Indonesian forensic bullet points grounded 100% in real numbers."""
    findings = []

    # 1. Model verdict & cross-engine consensus
    prob_ai_pct = round(prob_fake * 100.0, 1)
    prob_real_pct = round((1.0 - prob_fake) * 100.0, 1)
    peer_engines = [v for k, v in (comparison or {}).items() if k != "ensemble"]
    ai_votes = sum(1 for v in peer_engines if v["prob_fake"] >= 0.5)
    real_votes = len(peer_engines) - ai_votes

    if label == "AI_DISCLOSED":
        signer = provenance.get("signer") or "penandatangan tepercaya"
        findings.append({
            "category": "Kredensial C2PA",
            "status": "alert",
            "title": "Bukti Kriptografis AI Terverifikasi",
            "metric": "C2PA Valid",
            "detail": f"Manifest digital yang ditandatangani oleh {signer} secara eksplisit mencatat penggunaan AI dan cocok dengan hash berkas. Skor model visual ({prob_ai_pct}% AI) ditampilkan terpisah."
        })
    else:
        if len(peer_engines) >= 2:
            if ai_votes == len(peer_engines):
                consensus_note = f"Seluruh {len(peer_engines)} mesin deteksi sepakat mengindikasikan AI."
            elif real_votes == len(peer_engines):
                consensus_note = f"Seluruh {len(peer_engines)} mesin deteksi sepakat mengindikasikan citra non-AI."
            else:
                consensus_note = f"Terdapat perbedaan antar mesin ({ai_votes} AI vs {real_votes} Asli) — periksa matriks perbandingan."
        else:
            consensus_note = ""

        is_ai = prob_fake >= 0.5
        findings.append({
            "category": "Klasifikasi Visual",
            "status": "alert" if is_ai else "ok",
            "title": f"{active_algo_name}: {prob_ai_pct}% AI / {prob_real_pct}% Asli",
            "metric": f"{prob_ai_pct if is_ai else prob_real_pct}%",
            "detail": f"Model memetakan fitur spasial citra ke kelas {'Generative AI' if is_ai else 'Non-AI / Kamera'}. {consensus_note}".strip()
        })

    # 2. Frequency & Noise Signal (REAL measured values)
    sigma = fft_metrics["noise_sigma"]
    hf_z = fft_metrics["hf_peak_z"]
    ela_mean = ela_metrics["mean"]
    signal_flag = (hf_z >= 3.0) or (sigma < 3.0)
    if signal_flag:
        sig_desc = f"Noise residual rendah (σ = {sigma}) atau ditemukan puncak spektrum 2D-FFT (z = {hf_z}), pola yang sering muncul pada upsampling generator AI. Mean ELA Δ = {ela_mean}."
    else:
        sig_desc = f"Noise residual alami (σ = {sigma}), puncak spektrum 2D-FFT terkendali (z = {hf_z}), dan selisih kompresi ELA rata-rata Δ = {ela_mean}."
    findings.append({
        "category": "Frekuensi & ELA",
        "status": "warn" if signal_flag else "ok",
        "title": "Analisis Spektrum 2D-FFT & Kompresi",
        "metric": f"σ {sigma} · Δ {ela_mean}",
        "detail": sig_desc,
    })

    # 3. Spatial Activation (Grad-CAM) if available, otherwise Sharpness
    if gradcam_info:
        findings.append({
            "category": "Fokus Spasial",
            "status": "info",
            "title": f"Konsentrasi Aktivasi: Area {gradcam_info['peak_region'].title()}",
            "metric": f"{gradcam_info['hot_area_pct']}% area",
            "detail": f"Peta Grad-CAM CNN menyoroti {gradcam_info['hot_area_pct']}% piksel paling berpengaruh di area {gradcam_info['peak_region']}. Buka tab Grad-CAM di panel kiri untuk melihat titiknya.",
        })
    else:
        lap = fft_metrics["laplacian_var"]
        findings.append({
            "category": "Ketajaman Tepi",
            "status": "info",
            "title": "Variansi Gradien Tepi (Laplacian)",
            "metric": f"Var {lap}",
            "detail": f"Variansi gradien piksel terukur {lap} dengan rasio energi frekuensi tinggi-menengah {fft_metrics['high_to_mid_ratio']}.",
        })

    # 4. EXIF & C2PA Provenance (Informational & Real)
    c2pa_status = (provenance or {}).get("status", "absent")
    cam_parts = [p for p in [exif_info.get("camera_make"), exif_info.get("camera_model")] if p]
    cam_str = " ".join(cam_parts)
    if cam_str:
        meta_detail = f"Header EXIF mencatat perangkat {cam_str}" + (f" ({exif_info['software']})" if exif_info.get("software") else "") + ". Status C2PA: " + c2pa_status.replace("_", " ") + "."
        meta_metric = "EXIF Ada"
        meta_st = "ok"
    elif exif_info.get("software"):
        meta_detail = f"Header EXIF mencatat perangkat lunak: {exif_info['software']}. Tidak ada identifikasi kamera fisik. Status C2PA: {c2pa_status.replace('_', ' ')}."
        meta_metric = "Software Tag"
        meta_st = "warn"
    else:
        meta_detail = f"Tidak ada metadata kamera EXIF tertanam pada berkas (umum pada citra AI maupun foto yang dikompresi ulang aplikasi pesan). Status C2PA: {c2pa_status.replace('_', ' ')}."
        meta_metric = "Tanpa EXIF"
        meta_st = "info"

    findings.append({
        "category": "Metadata & Asal",
        "status": meta_st,
        "title": "Pemeriksaan EXIF & Kredensial C2PA",
        "metric": meta_metric,
        "detail": meta_detail,
    })

    return findings
