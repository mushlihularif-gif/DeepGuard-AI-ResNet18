import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image, ImageOps
import app as server
from modeling import label_mapping, load_checkpoint, rgb_image, preprocessing


class FixedModel:
    def __init__(self, logits):
        self.logits = logits

    def __call__(self, tensor):
        return torch.tensor([self.logits], device=tensor.device)


def png_bytes(size=(128, 128)):
    stream = io.BytesIO()
    Image.new("RGB", size, (40, 80, 120)).save(stream, "PNG")
    return stream.getvalue()


class ContractTests(unittest.TestCase):
    def test_realart_does_not_become_fake(self):
        self.assertEqual(label_mapping(["AiArtData", "RealArt"]), {0: 1, 1: 0})
        self.assertEqual(label_mapping(["real", "fake"]), {0: 0, 1: 1})
        with self.assertRaises(ValueError):
            label_mapping(["unknown", "fake"])

    def test_reject_legacy_checkpoint_without_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"fc.weight": torch.ones(2, 512)}, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_exif_orientation_matches_explicit_rotation(self):
        image = Image.new("RGB", (64, 128))
        image.paste("red", (0, 0, 32, 128))
        image.getexif()[274] = 6
        actual = preprocessing()(rgb_image(image))
        expected = preprocessing()(ImageOps.exif_transpose(image).convert("RGB"))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(rgb_image(image).size, (128, 64))

    def test_uncertainty_boundaries(self):
        for prob, expected in [(0.19, "REAL"), (0.2, "REAL"), (0.21, "UNCERTAIN"),
                               (0.5, "UNCERTAIN"), (0.79, "UNCERTAIN"), (0.8, "FAKE"), (0.81, "FAKE")]:
            self.assertEqual(server.decision(prob)[0], expected)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        self.patches = [patch.object(server, "resnet_model", FixedModel([0.0, 0.0])),
                        patch.object(server, "preprocess_pipeline", preprocessing()),
                        patch.object(server, "device", torch.device("cpu")),
                        patch.object(server, "model_metadata", {"temperature": 1.0})]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def upload(self, raw=None, name="sample.png", mime="image/png"):
        response = self.client.post("/api/predict", data={"image": (io.BytesIO(raw if raw is not None else png_bytes()), name, mime)})
        response.request.input_stream.close()
        return response

    def test_class_order_and_uncertain_response(self):
        for logits, expected in [([5., 0.], "REAL"), ([0., 5.], "FAKE"), ([0., 0.], "UNCERTAIN")]:
            with patch.object(server, "resnet_model", FixedModel(logits)):
                response = self.upload()
                self.assertEqual(response.status_code, 200)
                data = response.get_json()
                self.assertEqual(data["label"], expected)
                self.assertAlmostEqual(data["prob_real"] + data["prob_fake"], 1)
                self.assertFalse(data["partial_edit_detection_validated"])

    def test_renaming_cannot_change_prediction(self):
        scores = [self.upload(name=name).get_json()["prob_fake"] for name in
                  ["IMG_1234.png", "chatgpt_fake.png", "simulasi.png", "camera.png"]]
        self.assertEqual(len(set(scores)), 1)

    def test_invalid_uploads_are_client_errors(self):
        self.assertEqual(self.client.post("/api/predict").status_code, 400)
        self.assertEqual(self.upload(b"not an image").status_code, 400)
        self.assertEqual(self.upload(name="sample.pdf").status_code, 400)
        stream = io.BytesIO()
        Image.new("RGB", (64, 64)).save(stream, "GIF")
        self.assertEqual(self.upload(stream.getvalue()).status_code, 400)
        self.assertEqual(self.upload(png_bytes((16, 16))).status_code, 400)

    def test_size_limit_returns_json_413(self):
        response = self.upload(b"x" * (server.MAX_FILE_BYTES + 1))
        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.get_json())

    def test_missing_model_does_not_guess(self):
        with patch.object(server, "resnet_model", None):
            self.assertEqual(self.upload().status_code, 503)

    def test_server_upload_storage_stays_in_memory(self):
        request = server.InMemoryRequest.from_values()
        stream = request._get_file_stream(1000000, "image/png", "example.png", 900000)
        self.assertIsInstance(stream, io.BytesIO)
        stream.close()
        request.close()

    def test_nonfinite_output_is_rejected(self):
        with patch.object(server, "resnet_model", FixedModel([float("nan"), 0])):
            self.assertEqual(self.upload().status_code, 500)

    def test_temperature_is_applied_without_reversing_labels(self):
        with patch.object(server, "resnet_model", FixedModel([0., 2.])), \
             patch.object(server, "model_metadata", {"temperature": 2.0}):
            result = self.upload().get_json()
            self.assertAlmostEqual(result["prob_fake"], 0.7310586, places=5)
            self.assertEqual(result["label"], "UNCERTAIN")

    def test_oversized_request_preserves_413_status(self):
        response = self.upload(b"x" * (server.app.config["MAX_CONTENT_LENGTH"] + 1))
        self.assertEqual(response.status_code, 413)

    def test_camera_exif_cannot_override_pixels(self):
        image = Image.new("RGB", (128, 128), (40, 80, 120))
        exif = Image.Exif()
        exif[271], exif[272], exif[306] = "Camera", "Model", "2026:11:17 12:00:00"
        plain, tagged = io.BytesIO(), io.BytesIO()
        image.save(plain, "JPEG")
        image.save(tagged, "JPEG", exif=exif)
        first = self.upload(plain.getvalue(), "photo.jpg", "image/jpeg").get_json()
        second = self.upload(tagged.getvalue(), "photo.jpg", "image/jpeg").get_json()
        self.assertEqual(first["prob_fake"], second["prob_fake"])

    def test_pages_render_even_when_model_unavailable(self):
        for path in ["/", "/deteksi", "/panduan", "/tentang"]:
            self.assertEqual(self.client.get(path).status_code, 200, path)


class SplitTests(unittest.TestCase):
    def test_committed_manifest_has_no_hash_overlap_and_balanced_classes(self):
        path = Path(__file__).parents[1] / "reports/data_manifest.json"
        if not path.exists():
            self.skipTest("Manifest not yet generated")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        seen = set()
        for name, rows in manifest["splits"].items():
            hashes = {r["content_sha256_64rgb"] for r in rows}
            self.assertEqual(len(hashes), len(rows), name)
            self.assertFalse(seen & hashes, name)
            seen.update(hashes)
            self.assertEqual(sum(r["label"] == 0 for r in rows), sum(r["label"] == 1 for r in rows))


if __name__ == "__main__":
    unittest.main()
