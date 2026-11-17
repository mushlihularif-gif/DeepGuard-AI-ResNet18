"""Validate embedded Content Credentials locally; never infer from filenames/EXIF.

Only trusted, intact active-manifest AI disclosures can affect the final verdict.
No remote manifests, identity lookups, or online revocation requests are made.
"""
import hashlib
import io
import json
from pathlib import Path

TRUST_PATH = Path(__file__).resolve().parent / "trust/c2pa-official.pem"
TRUST_SHA256 = "148c1c167e98dc909a0d5793d2feee1022aa60fef5c83a0d6b08f67f4446bc37"
AI_TYPES = {
    f"{scheme}://cv.iptc.org/newscodes/digitalsourcetype/{kind}"
    for scheme in ("http", "https")
    for kind in ("trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia")
}
MESSAGES = {
    "absent": "Tidak ditemukan kredensial C2PA tertanam. Ini bukan bukti bahwa gambar bebas AI.",
    "unavailable": "Pemeriksaan C2PA tidak tersedia; hasil hanya berasal dari model visual.",
    "invalid": "Kredensial C2PA tidak dapat divalidasi; tidak dipakai sebagai bukti penggunaan AI.",
    "untrusted": "Kredensial ditemukan, tetapi kepercayaan penandatangan belum terverifikasi; tidak dipakai untuk menentukan AI.",
    "trusted_no_ai_disclosure": "Kredensial valid tidak mencatat jenis sumber AI yang didukung pada manifest aktif. Ini tidak membuktikan bebas edit AI.",
    "trusted_ai": "Kredensial C2PA valid dan tepercaya mencatat penggunaan AI pada gambar ini.",
}


def result(status, **details):
    return {"status": status, "ai_disclosed": status == "trusted_ai",
            "summary": MESSAGES[status], "verification_mode": "offline",
            "trust_list_sha256": TRUST_SHA256, **details}


def summarize_manifest(report):
    """Policy over SDK-validated JSON, never over arbitrary image text."""
    active = report.get("manifests", {}).get(report.get("active_manifest"))
    if not active:
        return result("invalid")
    validation = report.get("validation_results", {}).get("activeManifest", {})
    failures = {entry.get("code") for entry in validation.get("failure", [])}
    if report.get("validation_state") == "Invalid" or failures - {"signingCredential.untrusted"}:
        return result("invalid")
    successes = {entry.get("code") for entry in validation.get("success", [])}
    required = {"signingCredential.trusted", "claimSignature.validated"}
    binding = {"assertion.dataHash.match", "assertion.boxesHash.match", "assertion.bmffHash.match"}
    if failures or report.get("validation_state") != "Trusted" or not required <= successes or not successes & binding:
        return result("untrusted")
    source_types = []
    for assertion in active.get("assertions", []):
        if assertion.get("label") not in {"c2pa.actions", "c2pa.actions.v2"}:
            continue
        for action in assertion.get("data", {}).get("actions", []):
            if action.get("digitalSourceType") in AI_TYPES:
                source_types.append(action["digitalSourceType"])
    signature = active.get("signature_info", {})
    return result("trusted_ai" if source_types else "trusted_no_ai_disclosure",
                  signer=signature.get("common_name"), issuer=signature.get("issuer"),
                  source_types=sorted(set(source_types)),
                  informational_codes=[entry.get("code") for entry in validation.get("informational", [])])


def inspect_provenance(raw, mime):
    try:
        from c2pa import Context, Reader, C2paError
        anchors = TRUST_PATH.read_bytes()
        if hashlib.sha256(anchors).hexdigest() != TRUST_SHA256:
            return result("unavailable")
    except (ImportError, OSError):
        return result("unavailable")
    settings = {
        "verify": {"verify_cert_anchors": True, "remote_manifest_fetch": False, "ocsp_fetch": False},
        "core": {"allowed_network_hosts": [], "decode_identity_assertions": False},
        "trust": {"trust_anchors": anchors.decode("utf-8")},
    }
    try:
        with Context.from_dict(settings) as context, io.BytesIO(raw) as stream:
            with Reader(mime, stream, context=context) as reader:
                report = json.loads(reader.json())
        return summarize_manifest(report)
    except C2paError.ManifestNotFound:
        return result("absent")
    except Exception:
        # Invalid/unsupported credentials must never turn an image into verified AI.
        return result("invalid")
