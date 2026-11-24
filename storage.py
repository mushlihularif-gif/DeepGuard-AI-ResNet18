"""SQLite persistence layer for DeepGuard AI Admin SOC telemetry, config, and training runs."""
import csv
import datetime
import io
import json
import sqlite3
from pathlib import Path
from threading import Lock

DEFAULT_CONFIG = {
    "default_algorithm": "ensemble",
    "decision_threshold": 0.50,
    "enabled_algorithms": ["ensemble", "resnet18", "vit", "ela"],
    "ensemble_weights": {"vit": 0.55, "resnet18": 0.30, "ela": 0.15},
    "active_resnet_checkpoint": "models/resnet18_model.pt",
}


class ForensicsStorage:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS admin_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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
            now = datetime.datetime.now().isoformat(timespec="seconds")
            for k, v in DEFAULT_CONFIG.items():
                conn.execute(
                    "INSERT OR IGNORE INTO admin_config (key, value, updated_at) VALUES (?, ?, ?)",
                    (k, json.dumps(v), now),
                )
            conn.commit()

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
