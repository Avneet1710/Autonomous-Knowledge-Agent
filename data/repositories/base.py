"""Shared database connection manager for all repositories."""
import os
import sqlite3
from contextlib import contextmanager


@contextmanager
def get_connection(db_path: str):
    """Open a SQLite connection with automatic commit/rollback and guaranteed close."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found at '{db_path}'. "
            "Run the setup notebooks to initialise it."
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
