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
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id       TEXT    UNIQUE NOT NULL,
    domain        TEXT    NOT NULL,
    context       TEXT    NOT NULL,
    question      TEXT    NOT NULL,
    gold_standard TEXT    NOT NULL,
    source_type   TEXT    NOT NULL DEFAULT 'synthetic',
    source_ref    TEXT,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS calibration_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT    NOT NULL REFERENCES tasks(task_id),
    vanilla_response    TEXT,
    augmented_response  TEXT,
    vanilla_pass        INTEGER NOT NULL CHECK(vanilla_pass   IN (0,1)),
    augmented_pass      INTEGER NOT NULL CHECK(augmented_pass IN (0,1)),
    p_vanilla           REAL,
    p_augmented         REAL,
    pass_rate           REAL,
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
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Apply incremental schema migrations for existing databases."""
        tasks_cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(tasks)").fetchall()
        }
        if "source_type" not in tasks_cols:
            self._conn.execute(
                "ALTER TABLE tasks ADD COLUMN source_type TEXT NOT NULL DEFAULT 'synthetic'"
            )
        if "source_ref" not in tasks_cols:
            self._conn.execute(
                "ALTER TABLE tasks ADD COLUMN source_ref TEXT"
            )

        cal_cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(calibration_results)").fetchall()
        }
        if "p_vanilla" not in cal_cols:
            self._conn.execute(
                "ALTER TABLE calibration_results ADD COLUMN p_vanilla REAL"
            )
        if "p_augmented" not in cal_cols:
            self._conn.execute(
                "ALTER TABLE calibration_results ADD COLUMN p_augmented REAL"
            )
        if "pass_rate" not in cal_cols:
            self._conn.execute(
                "ALTER TABLE calibration_results ADD COLUMN pass_rate REAL"
            )

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
                    (task_id, domain, context, question, gold_standard,
                     source_type, source_ref, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task["task_id"],
                    task.get("domain", ""),
                    task["context"],
                    task["question"],
                    task["gold_standard"],
                    task.get("source_type", "synthetic"),
                    task.get("source_ref"),
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
                     vanilla_pass, augmented_pass, p_vanilla, p_augmented,
                     pass_rate, is_retained, retention_reason,
                     irt_a, irt_b, calibrated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result["task_id"],
                    result.get("vanilla_response"),
                    result.get("augmented_response"),
                    int(result["vanilla_pass"]),
                    int(result["augmented_pass"]),
                    result.get("p_vanilla"),
                    result.get("p_augmented"),
                    result.get("pass_rate"),
                    int(result["is_retained"]),
                    result.get("retention_reason", ""),
                    result.get("irt_a"),
                    result.get("irt_b"),
                    self._now(),
                ),
            )

    def delete_calibration_results_for_task(self, task_id: str) -> int:
        """
        Delete all calibration_results rows for the given task_id.
        Used by recalibration to replace stale rows with fresh ones.
        Returns the number of rows deleted.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM calibration_results WHERE task_id = ?",
                (task_id,),
            )
            return cursor.rowcount

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

    def fetch_response_vectors(self) -> dict:
        """
        Return pass/fail + b-value pairs for each solver, for all items
        that have an estimated b.

        Includes:
        - retained hard items  (vanilla FAIL / augmented PASS, b ∈ [0, 2.5])
        - retained easy items  (both PASS, b ∈ [-2.5, 0])
        - too-hard items       (both FAIL, b ∈ [2.5, 5.0])  ← Option 2

        Too-hard items add informative FAIL observations at high difficulty,
        which breaks the all-pass selection bias for the augmented θ estimate.

        Returns
        -------
        {
            "vanilla":   [(b: float, passed: bool), ...],
            "augmented": [(b: float, passed: bool), ...],
        }
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT vanilla_pass, augmented_pass, irt_b
                FROM calibration_results
                WHERE irt_b IS NOT NULL
                """
            ).fetchall()
        vanilla   = [(r["irt_b"], bool(r["vanilla_pass"]))   for r in rows]
        augmented = [(r["irt_b"], bool(r["augmented_pass"])) for r in rows]
        return {"vanilla": vanilla, "augmented": augmented}

    def fetch_b_by_source(self) -> dict:
        """
        Return b-value lists grouped by source_type, for distribution comparison.

        Only includes retained items (is_retained = 1) with a non-null irt_b.

        Returns
        -------
        {
            "synthetic": [b, ...],
            "real":      [b, ...],
        }
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT t.source_type, c.irt_b
                FROM tasks t
                JOIN calibration_results c ON t.task_id = c.task_id
                WHERE c.is_retained = 1
                  AND c.irt_b IS NOT NULL
                ORDER BY c.irt_b
                """
            ).fetchall()
        result: dict = {"synthetic": [], "real": []}
        for row in rows:
            key = row["source_type"] if row["source_type"] in result else "synthetic"
            result[key].append(row["irt_b"])
        return result

    def fetch_real_items(self) -> list[dict]:
        """Return all real-world sourced items with their calibration data."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    t.task_id, t.domain, t.context, t.question, t.gold_standard,
                    t.source_type, t.source_ref,
                    c.vanilla_pass, c.augmented_pass,
                    c.is_retained, c.retention_reason,
                    c.irt_a, c.irt_b
                FROM tasks t
                JOIN calibration_results c ON t.task_id = c.task_id
                WHERE t.source_type = 'real'
                ORDER BY c.irt_b DESC NULLS LAST
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def fetch_summary(self) -> dict:
        """Return high-level counts for reporting."""
        with self._connect() as conn:
            total_tasks    = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            total_cal      = conn.execute("SELECT COUNT(*) FROM calibration_results").fetchone()[0]
            retained_hard  = conn.execute(
                "SELECT COUNT(*) FROM calibration_results WHERE retention_reason = 'retained'"
            ).fetchone()[0]
            retained_easy  = conn.execute(
                "SELECT COUNT(*) FROM calibration_results WHERE retention_reason = 'retained_easy'"
            ).fetchone()[0]
            discarded_hard = conn.execute(
                "SELECT COUNT(*) FROM calibration_results "
                "WHERE vanilla_pass = 0 AND augmented_pass = 0"
            ).fetchone()[0]
            discarded_anom = conn.execute(
                "SELECT COUNT(*) FROM calibration_results "
                "WHERE retention_reason = 'discarded_anomalous'"
            ).fetchone()[0]
        return {
            "total_tasks":       total_tasks,
            "total_calibrations": total_cal,
            "retained_hard":     retained_hard,
            "retained_easy":     retained_easy,
            "discarded_too_hard": discarded_hard,
            "discarded_anomalous": discarded_anom,
        }
