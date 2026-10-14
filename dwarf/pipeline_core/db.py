import sqlite3
import threading

from dwarf.core.dwarf_exceptions import Dwarf_Exception
from dwarf.core.logs.logger import logger


class SQLite_DB:

    def __init__(self):
        self.conn = None
        self.lock = threading.Lock()

    def init_db(self):
        self.require_conn()
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                image_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('dispatching', 'done', 'failed')),
                result_json TEXT,
                error TEXT
            )   WITHOUT ROWID
            """
        )

        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_status ON cases (status)")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_images (
                image_id TEXT PRIMARY KEY
            ) WITHOUT ROWID
            """
        )

    def open_db(self, db_path: str):
        self.conn = sqlite3.connect(
            db_path,
            isolation_level=None,
            check_same_thread=False,
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=5000")

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            finally:
                self.conn = None

    def require_conn(self) -> None:
        if self.conn is None:
            raise Dwarf_Exception("Database is not opened")

    def parent_start(self, overwrite: bool = False):
        self.require_conn()
        if overwrite:
            with self.lock:
                self.conn.execute("BEGIN")
                try:
                    self.conn.execute("DELETE FROM cases")
                    self.conn.execute("DELETE FROM processed_images")
                    self.conn.execute("COMMIT")
                except Exception as e:
                    logger.error_with_context(e)
                    self.conn.execute("ROLLBACK")
                    raise
        else:
            return

    def try_register_case(self, case_id: str, image_id: str) -> str | None:
        self.require_conn()
        with self.lock:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO cases (case_id, image_id, status) "
                "VALUES (?, ?, 'dispatching')",
                (case_id, image_id),
            )
            if cur.rowcount == 1:
                return None
            row = self.conn.execute(
                "SELECT status FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            return row[0] if row else None

    def flush_results(self, buffer: list[tuple[str, str, str | None, str | None]]) -> None:
        self.require_conn()
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.executemany(
                    "UPDATE cases SET status = ?, result_json = ?, error = ? "
                    "WHERE case_id = ?",
                    [(status, result_json, error, case_id) for case_id, status, result_json, error in buffer],
                )
                self.conn.execute("COMMIT")
            except Exception as e:
                logger.error_with_context(e)
                self.conn.execute("ROLLBACK")
                raise

