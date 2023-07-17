"""Module implementing database interfaces to store and access raw data."""
from pathlib import Path
from urllib.parse import urlparse

from .base import Database
from .hdf import HDFDatabase
from .sqlite import SQLiteDatabase


def connect(uri: str, mode="r") -> Database:
    """Connect to a database."""
    parsed_uri = urlparse(uri)
    db_path = Path(parsed_uri.path) or ":memory:"
    if parsed_uri.scheme in ["h5", "hdf5", "hdf"]:
        return HDFDatabase(db_path, mode=mode)
    elif parsed_uri.scheme == "sqlite":
        return SQLiteDatabase(db_path, mode=mode)
    else:
        raise ValueError("Unsupported database type")


__all__ = ["Database", "SQLiteDatabase", "HDFDatabase", "connect"]
