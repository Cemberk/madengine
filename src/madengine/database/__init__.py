"""Persistence layer for metrics storage."""

from __future__ import annotations

from typing import Any

from madengine.database.models import RunRecord


class DatabaseBackend:
    """Abstract database backend."""

    def insert(self, record: dict[str, Any]) -> str:
        raise NotImplementedError

    def query(self, filters: dict[str, Any] | None = None) -> list[dict]:
        raise NotImplementedError

    def close(self) -> None:
        pass


def get_database_backend(cfg: dict[str, Any] | Any) -> DatabaseBackend:
    """Factory for database backends."""
    if not hasattr(cfg, "get"):
        cfg = {}

    backend = cfg.get("backend", "mongodb")

    if backend == "mongodb":
        from .mongodb import MongoDBBackend

        return MongoDBBackend(cfg.get("mongodb", {}))
    elif backend == "mysql":
        from .mysql import MySQLBackend

        return MySQLBackend(cfg.get("mysql", {}))
    else:
        raise ValueError(f"Unknown database backend: {backend}")


__all__ = ["DatabaseBackend", "RunRecord", "get_database_backend"]
