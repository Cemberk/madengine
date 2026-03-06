"""MongoDB database backend."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any


class MongoDBBackend:
    """MongoDB database backend using pymongo."""

    def __init__(self, cfg: dict[str, Any] | Any):
        if not hasattr(cfg, "get"):
            cfg = {}

        uri_env = cfg.get("uri_env", "MONGODB_URI")
        self.uri = os.environ.get(uri_env) or cfg.get("uri", "mongodb://localhost:27017")
        self.database_name = cfg.get("database", "madengine")
        self.collection_name = cfg.get("collection", "runs")

        self._client = None
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            from pymongo import MongoClient

            self._client = MongoClient(self.uri)
            db = self._client[self.database_name]
            self._collection = db[self.collection_name]
        return self._collection

    def insert(self, record: dict[str, Any]) -> str:
        if "timestamp" not in record:
            record["timestamp"] = datetime.now(timezone.utc)

        result = self.collection.insert_one(record)
        return str(result.inserted_id)

    def query(self, filters: dict[str, Any] | None = None) -> list[dict]:
        filters = filters or {}
        cursor = self.collection.find(filters).sort("timestamp", -1)
        return list(cursor)

    def close(self) -> None:
        if self._client:
            self._client.close()
