"""Private post-thesis journal. One process owns a run; commits survive restart."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3

from .prompts import canonical_json, content_hash


class RunConflict(ValueError):
    """Changed manifest or corrupt journal; do not spend more compute."""


class RunLocked(RuntimeError):
    """Another runner owns this run directory."""


@contextmanager
def exclusive_run(directory: Path):
    """Nonblocking OS lock, released even on process death (Windows and POSIX).

    The persistent lock file must never be deleted while a runner is active.
    Requires a local filesystem with working OS locks and SQLite guarantees.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RunLocked("run is already locked; use its existing runner") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    """Derived report only; SQLite remains the authoritative journal."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class Journal:
    def __init__(self, path: Path, manifest: dict):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS run (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    manifest TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY,
                    request_key TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('started','finished','interrupted')),
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    result TEXT,
                    result_hash TEXT,
                    UNIQUE(request_key, ordinal)
                );
            """)
            existing = self.db.execute("SELECT manifest FROM run WHERE singleton=1").fetchone()
            expected = canonical_json(manifest)
            if existing is None:
                if self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]:
                    raise RunConflict("attempts exist without a manifest")
                with self.db:
                    self.db.execute("INSERT INTO run VALUES(1,?)", (expected,))
            elif existing[0] != expected:
                raise RunConflict("run manifest changed; use a new run_id for a new experiment")
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def recover(self):
        # A started record proves an attempt was reserved, not whether it reached
        # the server. Never assign it zero cost or a synthetic prediction.
        with self.db:
            self.db.execute("UPDATE attempts SET state='interrupted', finished_at=? "
                            "WHERE state='started'", (timestamp(),))

    def records(self):
        rows = []
        for stored in self.db.execute("SELECT * FROM attempts ORDER BY id"):
            row = dict(stored)
            if row["state"] == "finished":
                try:
                    value = json.loads(row["result"])
                    if content_hash(value) != row["result_hash"]:
                        raise ValueError
                except (ValueError, TypeError):
                    raise RunConflict("invalid cached result or checksum") from None
                row["result"] = value
            elif row["result"] is not None or row["result_hash"] is not None:
                raise RunConflict("unfinished attempt contains a result")
            rows.append(row)
        return rows

    def start(self, key: str, ordinal: int) -> int:
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO attempts(request_key,ordinal,state,started_at) VALUES(?,?,'started',?)",
                (key, ordinal, timestamp()),
            )
        return cursor.lastrowid

    def finish(self, attempt_id: int, result: dict):
        with self.db:
            cursor = self.db.execute(
                "UPDATE attempts SET state='finished',finished_at=?,result=?,result_hash=? "
                "WHERE id=? AND state='started'",
                (timestamp(), canonical_json(result), content_hash(result), attempt_id),
            )
            if cursor.rowcount != 1:
                raise RunConflict("attempt is not pending")
