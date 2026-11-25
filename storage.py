"""SQLite persistence layer for DeepGuard AI Admin SOC auth, audit logs, telemetry, config, and custom algorithms."""
import csv
import datetime
import io
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from werkzeug.security import check_password_hash, generate_password_hash

DEFAULT_CONFIG = {
    "default_algorithm": "ensemble",
    "decision_threshold": 0.50,
    "enabled_algorithms": ["ensemble", "smaug", "detectra", "vit", "convnext_tiny", "resnet18", "ela"],
    "ensemble_mode": "noisy_or",
    "ensemble_weights": {"smaug": 0.96, "vit": 0.92, "detectra": 0.65, "convnext_tiny": 0.0, "resnet18": 0.0, "ela": 0.0},
    "active_resnet_checkpoint": "models/resnet18_model.pt",
}


class ForensicsStorage:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS admin_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS admin_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'superadmin',
                    created_at TEXT NOT NULL,
                    last_login TEXT
                );

                CREATE TABLE IF NOT EXISTS admin_audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    username TEXT NOT NULL,
                    ip_addr TEXT NOT NULL,
                    action TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS custom_algorithms (
                    algo_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    subtitle TEXT NOT NULL,
                    algo_type TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    input_size TEXT NOT NULL,
                    ai_label_idx INTEGER,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scan_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_sha256 TEXT NOT NULL,
                    dimensions TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    algorithm TEXT NOT NULL,
                    algorithm_name TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    prob_real REAL NOT NULL,
                    prob_fake REAL NOT NULL,
                    c2pa_status TEXT NOT NULL,
                    latency_ms REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    architecture TEXT NOT NULL,
                    base_mode TEXT NOT NULL,
                    epochs INTEGER NOT NULL,
                    batch_size INTEGER NOT NULL,
                    lr REAL NOT NULL,
                    train_samples INTEGER NOT NULL,
                    val_samples INTEGER NOT NULL,
                    best_val_acc REAL,
                    best_val_loss REAL,
                    final_train_acc REAL,
                    checkpoint_path TEXT,
                    status TEXT NOT NULL,
                    history_json TEXT
                );
                """
            )
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for k, v in DEFAULT_CONFIG.items():
                conn.execute(
                    "INSERT OR IGNORE INTO admin_config (key, value, updated_at) VALUES (?, ?, ?)",
                    (k, json.dumps(v), now),
                )
            # Seed default admin if none exists yet
            count = conn.execute("SELECT COUNT(*) AS c FROM admin_users").fetchone()["c"]
            if count == 0:
                default_user = os.environ.get("DEEPGUARD_ADMIN_USER", "admin").strip() or "admin"
                default_pass = os.environ.get("DEEPGUARD_ADMIN_PASS", "deepguard2026")
                conn.execute(
                    "INSERT INTO admin_users (username, password_hash, role, created_at) VALUES (?, ?, 'superadmin', ?)",
                    (default_user, generate_password_hash(default_pass), now),
                )
            conn.commit()

    # ==========================================================================
    # ADMIN AUTHENTICATION & AUDIT TRAIL
    # ==========================================================================

    def verify_admin(self, username, password):
        uname = (username or "").strip()
        if not uname or not password:
            return None
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM admin_users WHERE username = ?", (uname,)).fetchone()
            if not row or not check_password_hash(row["password_hash"], password):
                return None
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute("UPDATE admin_users SET last_login = ? WHERE id = ?", (now, row["id"]))
            conn.commit()
            return {"id": row["id"], "username": row["username"], "role": row["role"], "last_login": now}

    def change_admin_password(self, username, old_password, new_password):
        if len(new_password or "") < 6:
            return False, "Kata sandi baru minimal 6 karakter."
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM admin_users WHERE username = ?", (username,)).fetchone()
            if not row or not check_password_hash(row["password_hash"], old_password or ""):
                return False, "Kata sandi saat ini tidak cocok."
            conn.execute(
                "UPDATE admin_users SET password_hash = ? WHERE id = ?",
                (generate_password_hash(new_password), row["id"]),
            )
            conn.commit()
        return True, "Kata sandi berhasil diperbarui."

    def create_admin(self, username, password, role="admin"):
        uname = (username or "").strip().lower()
        if not uname or len(uname) < 3 or not uname.replace("_", "").replace("-", "").isalnum():
            return False, "Username minimal 3 karakter alfanumerik."
        if len(password or "") < 6:
            return False, "Kata sandi minimal 6 karakter."
        if role not in ("admin", "superadmin", "analyst"):
            role = "admin"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO admin_users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                    (uname, generate_password_hash(password), role, now),
                )
                conn.commit()
            return True, f"Akun administrator '{uname}' berhasil dibuat."
        except sqlite3.IntegrityError:
            return False, f"Username '{uname}' sudah terdaftar."

    def list_admins(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, username, role, created_at, last_login FROM admin_users ORDER BY id ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_admin(self, username, current_username):
        if username == current_username:
            return False, "Tidak dapat menghapus akun yang sedang digunakan."
        with self._lock, self._connect() as conn:
            cnt = conn.execute("SELECT COUNT(*) AS c FROM admin_users").fetchone()["c"]
            if cnt <= 1:
                return False, "Minimal satu akun administrator harus tersisa."
            cur = conn.execute("DELETE FROM admin_users WHERE username = ?", (username,))
            conn.commit()
            if cur.rowcount == 0:
                return False, "Akun tidak ditemukan."
        return True, f"Akun '{username}' telah dihapus."

    def record_admin_event(self, username, ip_addr, action, detail, status="OK"):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO admin_audit_logs (created_at, username, ip_addr, action, detail, status) VALUES (?, ?, ?, ?, ?, ?)",
                (now, username or "anonymous", ip_addr or "-", action, detail, status),
            )
            conn.commit()

    def get_admin_events(self, limit=50):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM admin_audit_logs ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(r) for r in rows]

    # ==========================================================================
    # CUSTOM ALGORITHMS REGISTRY (HF / LOCAL .PT / LOCAL .ONNX)
    # ==========================================================================

    def save_custom_algorithm(self, *, algo_id, name, subtitle, algo_type, source_kind,
                              source_ref, input_size, ai_label_idx=None, description=""):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO custom_algorithms (
                    algo_id, name, subtitle, algo_type, source_kind,
                    source_ref, input_size, ai_label_idx, description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(algo_id) DO UPDATE SET
                    name=excluded.name, subtitle=excluded.subtitle, algo_type=excluded.algo_type,
                    source_kind=excluded.source_kind, source_ref=excluded.source_ref,
                    input_size=excluded.input_size, ai_label_idx=excluded.ai_label_idx,
                    description=excluded.description
                """,
                (
                    algo_id, name, subtitle, algo_type, source_kind,
                    source_ref, str(input_size), ai_label_idx, description, now,
                ),
            )
            conn.commit()

    def list_custom_algorithms(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM custom_algorithms ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def delete_custom_algorithm(self, algo_id):
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM custom_algorithms WHERE algo_id = ?", (algo_id,))
            conn.commit()
            return cur.rowcount > 0

    # ==========================================================================
    # CONFIG, SCAN LOGS & TRAINING RUNS
    # ==========================================================================

    def get_config(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM admin_config").fetchall()
        cfg = dict(DEFAULT_CONFIG)
        for row in rows:
            try:
                cfg[row["key"]] = json.loads(row["value"])
            except Exception:
                pass
        return cfg

    def update_config(self, updates):
        now = datetime.datetime.now().isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            for k, v in updates.items():
                if k in DEFAULT_CONFIG or k.startswith("custom_"):
                    conn.execute(
                        "INSERT INTO admin_config (key, value, updated_at) VALUES (?, ?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                        (k, json.dumps(v), now),
                    )
            conn.commit()
        return self.get_config()

    def record_scan(self, *, filename, file_sha256, dimensions, file_size, algorithm,
                    algorithm_name, verdict, prob_real, prob_fake, c2pa_status, latency_ms):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scan_logs (
                    created_at, filename, file_sha256, dimensions, file_size,
                    algorithm, algorithm_name, verdict, prob_real, prob_fake, c2pa_status, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now, filename, file_sha256, dimensions, int(file_size),
                    algorithm, algorithm_name, verdict, float(prob_real), float(prob_fake),
                    c2pa_status, float(latency_ms),
                ),
            )
            conn.commit()

    def get_scan_logs(self, limit=50, verdict_filter=None, algo_filter=None, search=None):
        clauses = []
        params = []
        if verdict_filter and verdict_filter != "ALL":
            clauses.append("verdict = ?")
            params.append(verdict_filter)
        if algo_filter and algo_filter != "ALL":
            clauses.append("algorithm = ?")
            params.append(algo_filter)
        if search:
            clauses.append("(filename LIKE ? OR file_sha256 LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM scan_logs {where} ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_scan_stats(self):
        with self._lock, self._connect() as conn:
            agg = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_scans,
                    SUM(CASE WHEN verdict = 'REAL' THEN 1 ELSE 0 END) AS real_scans,
                    SUM(CASE WHEN verdict IN ('FAKE', 'AI_DISCLOSED') THEN 1 ELSE 0 END) AS fake_scans,
                    SUM(CASE WHEN verdict = 'UNCERTAIN' THEN 1 ELSE 0 END) AS uncertain_scans,
                    AVG(latency_ms) AS avg_latency_ms
                FROM scan_logs
                """
            ).fetchone()
            algo_rows = conn.execute(
                "SELECT algorithm, COUNT(*) AS cnt FROM scan_logs GROUP BY algorithm ORDER BY cnt DESC"
            ).fetchall()
        return {
            "total_scans": int(agg["total_scans"] or 0),
            "real_scans": int(agg["real_scans"] or 0),
            "fake_scans": int(agg["fake_scans"] or 0),
            "uncertain_scans": int(agg["uncertain_scans"] or 0),
            "avg_latency_ms": round(float(agg["avg_latency_ms"] or 0.0), 1),
            "by_algorithm": {r["algorithm"]: int(r["cnt"]) for r in algo_rows},
        }

    def clear_scan_logs(self):
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM scan_logs")
            conn.commit()

    def export_scan_logs_csv(self):
        rows = self.get_scan_logs(limit=1000)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "ID", "Timestamp", "Filename", "SHA256", "Dimensions", "SizeBytes",
            "Algorithm", "Verdict", "ProbReal", "ProbFake", "C2PAStatus", "LatencyMs"
        ])
        for r in rows:
            writer.writerow([
                r["id"], r["created_at"], r["filename"], r["file_sha256"], r["dimensions"],
                r["file_size"], r["algorithm"], r["verdict"], f"{r['prob_real']:.4f}",
                f"{r['prob_fake']:.4f}", r["c2pa_status"], r["latency_ms"],
            ])
        return buf.getvalue()

    def create_training_run(self, *, architecture, base_mode, epochs, batch_size, lr, train_samples, val_samples):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO training_runs (
                    started_at, architecture, base_mode, epochs, batch_size, lr,
                    train_samples, val_samples, status, history_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'TRAINING', '[]')
                """,
                (now, architecture, base_mode, int(epochs), int(batch_size), float(lr),
                 int(train_samples), int(val_samples)),
            )
            conn.commit()
            return cur.lastrowid

    def update_training_run(self, run_id, **fields):
        allowed = {
            "finished_at", "best_val_acc", "best_val_loss", "final_train_acc",
            "checkpoint_path", "status", "history_json",
        }
        parts = []
        params = []
        for k, v in fields.items():
            if k in allowed:
                parts.append(f"{k} = ?")
                params.append(json.dumps(v) if k == "history_json" and not isinstance(v, str) else v)
        if not parts:
            return
        params.append(int(run_id))
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE training_runs SET {', '.join(parts)} WHERE id = ?", params)
            conn.commit()

    def list_training_runs(self, limit=20):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["history"] = json.loads(d.get("history_json") or "[]")
            except Exception:
                d["history"] = []
            d.pop("history_json", None)
            out.append(d)
        return out
