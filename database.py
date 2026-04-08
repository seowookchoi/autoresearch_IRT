"""
database.py — SQLite persistence layer for compliance_bank.db.

Tables
------
tasks                : raw generated items (context, question, gold_standard)
calibration_results  : per-item solver responses, pass/fail flags, and 2PL params
"""

import sqlite3
from datetime import datetime, timezone
from typing import Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT    UNIQUE NOT NULL,
    domain       TEXT    NOT NULL,
    context      TEXT    NOT NULL,
    question     TEXT    NOT NULL,
    gold_standard TEXT   NOT NULL,
    created_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS calibration_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT    NOT NULL REFERENCES tasks(task_id),
    vanilla_response    TEXT,
    augmented_response  TEXT,
    vanilla_pass        INTEGER NOT NULL CHECK(vanilla_pass   IN (0,1)),
    augmented_pass      INTEGER NOT NULL CHECK(augmented_pass IN (0,1)),
    is_retained         INTEGER NOT NULL CHECK(is_retained    IN (0,1)),
    retention_reason    TEXT,
    irt_a               REAL,
    irt_b               REAL,
    calibrated_at       TEXT    NOT NULL
);
"""


class Database:
    """Thin wrapper around a SQLite connection for the compliance item bank."""

    def __init__(self, db_path: str = "compliance_bank.db") -> None:
        self.db_path = db_path
        # Keep a single persistent connection so that :memory: databases
        # work correctly in tests and each operation sees the same schema.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return the persistent connection (context-manager safe)."""
        return self._conn

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def insert_task(self, task: dict) -> None:
        """Insert a generated task; silently skips duplicates."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO tasks
                    (task_id, domain, context, question, gold_standard, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task["task_id"],
                    task.get("domain", ""),
                    task["context"],
                    task["question"],
                    task["gold_standard"],
                    self._now(),
                ),
            )

    def insert_calibration_result(self, result: dict) -> None:
        """Insert a calibration result row."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO calibration_results
                    (task_id, vanilla_response, augmented_response,
                     vanilla_pass, augmented_pass, is_retained, retention_reason,
                     irt_a, irt_b, calibrated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result["task_id"],
                    result.get("vanilla_response"),
                    result.get("augmented_response"),
                    int(result["vanilla_pass"]),
                    int(result["augmented_pass"]),
                    int(result["is_retained"]),
                    result.get("retention_reason", ""),
                    result.get("irt_a"),
                    result.get("irt_b"),
                    self._now(),
                ),
            )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def fetch_all_tasks(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM tasks ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def fetch_retained(self) -> list[dict]:
        """Return retained (hard) items joined with their calibration data."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    t.task_id, t.domain, t.context, t.question, t.gold_standard,
                    c.vanilla_response, c.augmented_response,
                    c.vanilla_pass, c.augmented_pass,
                    c.is_retained, c.retention_reason,
                    c.irt_a, c.irt_b, c.calibrated_at
                FROM tasks t
                JOIN calibration_results c ON t.task_id = c.task_id
                WHERE c.is_retained = 1
                ORDER BY c.irt_b DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def fetch_summary(self) -> dict:
        """Return high-level counts for reporting."""
        with self._connect() as conn:
            total_tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            total_cal   = conn.execute("SELECT COUNT(*) FROM calibration_results").fetchone()[0]
            retained    = conn.execute(
                "SELECT COUNT(*) FROM calibration_results WHERE is_retained = 1"
            ).fetchone()[0]
            both_pass   = conn.execute(
                "SELECT COUNT(*) FROM calibration_results "
                "WHERE vanilla_pass = 1 AND augmented_pass = 1"
            ).fetchone()[0]
            both_fail   = conn.execute(
                "SELECT COUNT(*) FROM calibration_results "
                "WHERE vanilla_pass = 0 AND augmented_pass = 0"
            ).fetchone()[0]
        return {
            "total_tasks": total_tasks,
            "total_calibrations": total_cal,
            "retained": retained,
            "discarded_too_easy": both_pass,
            "discarded_too_hard": both_fail,
        }
