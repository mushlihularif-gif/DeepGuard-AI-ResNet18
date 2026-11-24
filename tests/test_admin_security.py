"""Unit & security tests for isolated Admin Portal authentication, dataset ZIP/relabel, and dynamic algorithms."""
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

import app as server
from modeling import CLASS_TO_IDX, PREPROCESS
from storage import ForensicsStorage


def _make_png(color=(30, 90, 150), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


class AdminSecurityAndFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.test_storage = ForensicsStorage(Path(self.tmpdir.name) / "test_deepguard.db")
        self.storage_patch = patch.object(server, "storage", self.test_storage)
        self.storage_patch.start()
        self.addCleanup(self.storage_patch.stop)

        server._login_attempts.clear()
        self.addCleanup(server._login_attempts.clear)
        self.client = server.app.test_client()

    def _login(self, username="admin", password="deepguard2026"):
        return self.client.post(
            "/api/admin/login",
            json={"username": username, "password": password},
        )

    def test_admin_pages_redirect_to_login_when_unauthenticated(self):
        for route in ("/admin", "/admin.html"):
            resp = self.client.get(route, follow_redirects=False)
            self.assertEqual(resp.status_code, 302, route)
            self.assertTrue(resp.headers["Location"].endswith("/admin/login"))

        login_page = self.client.get("/admin/login")
        self.assertEqual(login_page.status_code, 200)
        html = login_page.get_data(as_text=True)
        # Never leak default password on public login HTML
        self.assertNotIn("deepguard2026", html)

    def test_public_pages_hide_admin_navbar_button(self):
        for route in ("/", "/deteksi", "/panduan", "/tentang"):
            resp = self.client.get(route)
            self.assertEqual(resp.status_code, 200, route)
            html = resp.get_data(as_text=True)
            self.assertNotIn("Konsol Admin</a>", html, route)
            self.assertIn("/admin/login", html, route)

    def test_admin_apis_require_authentication(self):
        protected = [
            ("GET", "/api/admin/stats"),
            ("GET", "/api/admin/logs"),
            ("GET", "/api/admin/me"),
            ("GET", "/api/admin/users"),
            ("GET", "/api/admin/audit"),
            ("POST", "/api/admin/config"),
            ("POST", "/api/admin/dataset/upload-zip"),
            ("POST", "/api/admin/dataset/move"),
            ("POST", "/api/admin/dataset/autosplit"),
            ("POST", "/api/admin/algorithms/upload"),
            ("POST", "/api/admin/algorithms/import-hf"),
        ]
        for method, path in protected:
            resp = self.client.open(path, method=method)
            self.assertEqual(resp.status_code, 401, f"{method} {path}")
            self.assertIn("error", resp.get_json())

    def test_wrong_credentials_and_bruteforce_rate_limit(self):
        for i in range(server.LOGIN_MAX_FAILURES):
            resp = self._login("admin", f"wrong_{i}")
            self.assertEqual(resp.status_code, 401)

        blocked = self._login("admin", "deepguard2026")
        self.assertEqual(blocked.status_code, 429)
        self.assertIn("error", blocked.get_json())

    def test_valid_login_httponly_cookie_and_logout(self):
        resp = self._login()
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()["success"])
        cookies = resp.headers.getlist("Set-Cookie")
        self.assertTrue(any("HttpOnly" in c for c in cookies))

        self.assertEqual(self.client.get("/admin").status_code, 200)
        self.assertEqual(self.client.get("/api/admin/stats").status_code, 200)
        me = self.client.get("/api/admin/me").get_json()
        self.assertEqual(me["username"], "admin")

        out = self.client.post("/api/admin/logout")
        self.assertEqual(out.status_code, 200)
        self.assertEqual(self.client.get("/api/admin/stats").status_code, 401)
        self.assertEqual(self.client.get("/admin", follow_redirects=False).status_code, 302)

    def test_admin_user_crud_password_change_and_audit(self):
        self.assertEqual(self._login().status_code, 200)

        # Create secondary admin
        r_create = self.client.post(
            "/api/admin/users",
            json={"username": "analyst_qa", "password": "SecurePassword123!", "role": "admin"},
        )
        self.assertEqual(r_create.status_code, 200)
        users = [u["username"] for u in self.client.get("/api/admin/users").get_json()["users"]]
        self.assertIn("analyst_qa", users)

        # Cannot delete currently logged-in account
        self.assertEqual(self.client.delete("/api/admin/users/admin").status_code, 400)

        # Change password
        r_pw = self.client.post(
            "/api/admin/password",
            json={"old_password": "deepguard2026", "new_password": "NewStrongPassword99!"},
        )
        self.assertEqual(r_pw.status_code, 200)

        # Delete secondary admin
        self.assertEqual(self.client.delete("/api/admin/users/analyst_qa").status_code, 200)

        # Audit log should have captured events
        events = self.client.get("/api/admin/audit").get_json()["events"]
        actions = {e["action"] for e in events}
        self.assertTrue({"LOGIN", "CREATE_ADMIN", "PASSWORD_CHANGE", "DELETE_ADMIN"}.issubset(actions))

        # Re-login verifies new password works and old password fails
        self.client.post("/api/admin/logout")
        self.assertEqual(self._login("admin", "deepguard2026").status_code, 401)
        self.assertEqual(self._login("admin", "NewStrongPassword99!").status_code, 200)

    def test_dataset_zip_upload_zipslip_protection_and_relabel(self):
        self.assertEqual(self._login().status_code, 200)

        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w") as zf:
            zf.writestr("batch/real/qa_sec_real_01.png", _make_png((20, 120, 60)))
            zf.writestr("batch/fake/qa_sec_fake_01.png", _make_png((180, 40, 60)))
            zf.writestr("../evil_escape_qa.png", _make_png((255, 0, 0)))
        zbuf.seek(0)

        created_paths = []
        try:
            resp = self.client.post(
                "/api/admin/dataset/upload-zip",
                data={
                    "archive": (zbuf, "qa_dataset.zip", "application/zip"),
                    "split": "val",
                    "category": "real",
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
            body = resp.get_json()
            self.assertEqual(body["saved_real"], 1)
            self.assertEqual(body["saved_fake"], 1)
            self.assertEqual(body["skipped"], 1)

            real_file = server.ROOT / "dataset" / "val" / "real" / "qa_sec_real_01.png"
            fake_file = server.ROOT / "dataset" / "val" / "fake" / "qa_sec_fake_01.png"
            evil_file = server.ROOT / "dataset" / "evil_escape_qa.png"
            created_paths.extend([real_file, fake_file])
            self.assertTrue(real_file.exists())
            self.assertTrue(fake_file.exists())
            self.assertFalse(evil_file.exists())

            # Test relabel move real -> fake
            mv = self.client.post(
                "/api/admin/dataset/move",
                json={
                    "split": "val",
                    "category": "real",
                    "target_split": "val",
                    "target_category": "fake",
                    "filename": "qa_sec_real_01.png",
                },
            )
            self.assertEqual(mv.status_code, 200)
            moved_file = server.ROOT / "dataset" / "val" / "fake" / "qa_sec_real_01.png"
            created_paths.append(moved_file)
            self.assertFalse(real_file.exists())
            self.assertTrue(moved_file.exists())
        finally:
            for p in created_paths:
                p.unlink(missing_ok=True)

    def test_custom_pt_algorithm_upload_and_delete_protects_builtins(self):
        self.assertEqual(self._login().status_code, 200)

        # Built-in algorithm cannot be deleted
        r_builtin = self.client.delete("/api/admin/algorithms/resnet18")
        self.assertEqual(r_builtin.status_code, 400)

        # Invalid HuggingFace repo_id rejected cleanly
        r_bad_hf = self.client.post("/api/admin/algorithms/import-hf", json={"repo_id": "../../invalid"})
        self.assertEqual(r_bad_hf.status_code, 400)

        # Build a valid small ResNet-18 checkpoint in memory
        model = server.build_cnn_architecture("resnet18", pretrained=False)
        pt_buf = io.BytesIO()
        torch.save(
            {
                "schema_version": 1,
                "architecture": "resnet18",
                "class_to_idx": CLASS_TO_IDX,
                "preprocess": PREPROCESS,
                "temperature": 1.0,
                "state_dict": model.state_dict(),
            },
            pt_buf,
        )
        pt_buf.seek(0)

        algo_id = None
        try:
            resp = self.client.post(
                "/api/admin/algorithms/upload",
                data={
                    "model_file": (pt_buf, "qa_custom_cnn.pt", "application/octet-stream"),
                    "name": "QA Custom CNN",
                    "algo_type": "Custom CNN",
                    "input_size": "224",
                    "ai_label_idx": "1",
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
            algo_id = resp.get_json()["algo_id"]
            self.assertIn(algo_id, server.algorithms_registry)

            # Confirm it is immediately visible to public /api/algorithms
            pub_ids = [a["id"] for a in self.client.get("/api/algorithms").get_json()["algorithms"]]
            self.assertIn(algo_id, pub_ids)

            disk_path = server.ROOT / "models" / f"{algo_id}.pt"
            self.assertTrue(disk_path.exists())

            # Delete custom algorithm & verify cleanup
            r_del = self.client.delete(f"/api/admin/algorithms/{algo_id}")
            self.assertEqual(r_del.status_code, 200)
            self.assertNotIn(algo_id, server.algorithms_registry)
            self.assertFalse(disk_path.exists())
            algo_id = None
        finally:
            if algo_id:
                server.algorithms_registry.pop(algo_id, None)
                (server.ROOT / "models" / f"{algo_id}.pt").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
