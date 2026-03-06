"""Tests for database backend factory."""

from __future__ import annotations

import pytest

from madengine.database import get_database_backend
from madengine.database.mongodb import MongoDBBackend
from madengine.database.mysql import MySQLBackend


class TestGetDatabaseBackend:
    def test_mongodb_backend(self):
        backend = get_database_backend({"backend": "mongodb", "mongodb": {}})
        assert isinstance(backend, MongoDBBackend)

    def test_mysql_backend(self):
        backend = get_database_backend({"backend": "mysql", "mysql": {}})
        assert isinstance(backend, MySQLBackend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown database backend"):
            get_database_backend({"backend": "sqlite"})
