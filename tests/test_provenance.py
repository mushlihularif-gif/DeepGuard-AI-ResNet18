import copy
import hashlib
import io
import json
import struct
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from c2pa import Builder, Context, Signer, C2paSignerInfo, C2paSigningAlg
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
import provenance


def png(color="blue"):
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(stream, "PNG")
    return stream.getvalue()


def transplant_credentials(signed, target):
    """Transfer actual signed PNG chunks to different pixels, without signing."""
    cursor, credentials = 8, b""
    while cursor < len(signed):
        length = struct.unpack(">I", signed[cursor:cursor+4])[0] + 12
        if signed[cursor+4:cursor+8] == b"caBX":
            credentials += signed[cursor:cursor+length]
        cursor += length
    assert credentials
    return target[:33] + credentials + target[33:]


class CryptographicProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ephemeral test-only CA. No private keys or user pictures are stored.
        now = datetime.now(timezone.utc)
        ca_key, leaf_key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "DeepGuard TEST ONLY Root")])
        leaf_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "DeepGuard TEST ONLY Signer")])
        ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
              .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
              .not_valid_before(now-timedelta(days=1)).not_valid_after(now+timedelta(days=30))
              .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
              .add_extension(x509.KeyUsage(True, False, False, False, False, True, True, None, None), critical=True)
              .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
              .sign(ca_key, hashes.SHA256()))
        leaf = (x509.CertificateBuilder().subject_name(leaf_name).issuer_name(ca_name)
                .public_key(leaf_key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now-timedelta(days=1)).not_valid_after(now+timedelta(days=10))
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(x509.KeyUsage(True, True, False, False, False, False, False, None, None), critical=True)
                .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
                .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
                .sign(ca_key, hashes.SHA256()))
        cls.anchors = ca.public_bytes(serialization.Encoding.PEM)
        cls.signer_info = C2paSignerInfo(C2paSigningAlg.ES256,
            leaf.public_bytes(serialization.Encoding.PEM) + cls.anchors,
            leaf_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                   serialization.NoEncryption()), None)
        cls.signed = cls.sign("trainedAlgorithmicMedia")
        cls.no_ai = cls.sign("digitalCapture")
        cls.edited = cls.sign("compositeWithTrainedAlgorithmicMedia")

    @classmethod
    def sign(cls, source_type):
        manifest = {"claim_generator": "deepguard_test/1.0", "assertions": [{
            "label": "c2pa.actions", "data": {"actions": [{"action": "c2pa.created",
                "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/"+source_type}]}}]}
        with Context.from_dict({"core": {"allowed_network_hosts": []}}) as ctx, \
             Signer.from_info(cls.signer_info) as signer, Builder(json.dumps(manifest), ctx) as builder, \
             io.BytesIO(png()) as source, io.BytesIO() as target:
            builder.sign(signer, "image/png", source, target)
            return target.getvalue()

    def inspect_with_test_ca(self, raw):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test-ca.pem"
            path.write_bytes(self.anchors)
            with patch.object(provenance, "TRUST_PATH", path), \
                 patch.object(provenance, "TRUST_SHA256", hashlib.sha256(self.anchors).hexdigest()):
                return provenance.inspect_provenance(raw, "image/png")

    def test_signed_generated_and_composite_ai_are_verified(self):
        for raw in (self.signed, self.edited):
            result = self.inspect_with_test_ca(raw)
            self.assertEqual(result["status"], "trusted_ai", result)

    def test_same_signature_on_other_pixels_is_rejected(self):
        raw = transplant_credentials(self.signed, png("red"))
        self.assertEqual(self.inspect_with_test_ca(raw)["status"], "invalid")

    def test_self_signed_unknown_issuer_is_not_trusted_in_production(self):
        result = provenance.inspect_provenance(self.signed, "image/png")
        self.assertFalse(result["ai_disclosed"])
        self.assertEqual(result["status"], "untrusted")

    def test_valid_camera_declaration_never_proves_no_ai(self):
        result = self.inspect_with_test_ca(self.no_ai)
        self.assertEqual(result["status"], "trusted_no_ai_disclosure")
        self.assertFalse(result["ai_disclosed"])

    def test_missing_or_stripped_credentials_are_not_ai_evidence(self):
        with Image.open(io.BytesIO(self.signed)) as image:
            target = io.BytesIO()
            image.convert("RGB").save(target, "PNG")
        self.assertEqual(self.inspect_with_test_ca(target.getvalue())["status"], "absent")

    def test_corrupt_trust_bundle_does_not_disable_validation(self):
        with patch.object(provenance, "TRUST_SHA256", "wrong"):
            self.assertEqual(provenance.inspect_provenance(self.signed, "image/png")["status"], "unavailable")


class ManifestPolicyTests(unittest.TestCase):
    def test_unrelated_ingredient_ai_is_not_active_image_ai(self):
        report = {"active_manifest": "current", "validation_state": "Trusted",
                  "validation_results": {"activeManifest": {"failure": [], "success": [
                      {"code": c} for c in ("signingCredential.trusted", "claimSignature.validated", "assertion.dataHash.match")]}},
                  "manifests": {"current": {"assertions": []}, "unrelated": {"assertions": [{
                      "label": "c2pa.actions", "data": {"actions": [{"digitalSourceType": next(iter(provenance.AI_TYPES))}]}}]}}}
        self.assertFalse(provenance.summarize_manifest(report)["ai_disclosed"])
        bad = copy.deepcopy(report)
        bad["validation_results"]["activeManifest"]["failure"] = [{"code": "assertion.dataHash.mismatch"}]
        self.assertEqual(provenance.summarize_manifest(bad)["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
