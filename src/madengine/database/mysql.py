"""MySQL database backend."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


class MySQLBackend:
    """MySQL database backend using SQLAlchemy."""

    def __init__(self, cfg: dict[str, Any] | Any):
        if not hasattr(cfg, "get"):
            cfg = {}

        self.host = cfg.get("host", "localhost")
        self.port = cfg.get("port", 3306)
        self.database = cfg.get("database", "madengine")

        user_env = cfg.get("user_env", "MYSQL_USER")
        password_env = cfg.get("password_env", "MYSQL_PASSWORD")

        self.user = os.environ.get(user_env, "root")
        self.password = os.environ.get(password_env, "")

        self._engine = None
        self._session = None

    @property
    def session(self):
        if self._session is None:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker

            url = (
                f"mysql+pymysql://{self.user}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
            self._engine = create_engine(url)
            Session = sessionmaker(bind=self._engine)
            self._session = Session()
        return self._session

    def insert(self, record: dict[str, Any]) -> str:
        from sqlalchemy import text

        config_json = json.dumps(record.get("config", {}))
        metrics_json = json.dumps(record.get("metrics", {}))

        sql = text(
            "INSERT INTO runs "
            "(timestamp, recipe, config, metrics, success, error, image, duration_seconds) "
            "VALUES "
            "(:timestamp, :recipe, :config, :metrics, :success, :error, :image, :duration_seconds)"
        )

        result = self.session.execute(
            sql,
            {
                "timestamp": record.get("timestamp", datetime.now(timezone.utc)),
                "recipe": record.get("recipe"),
                "config": config_json,
                "metrics": metrics_json,
                "success": record.get("success", True),
                "error": record.get("error"),
                "image": record.get("image"),
                "duration_seconds": record.get("duration_seconds"),
            },
        )
        self.session.commit()
        return str(result.lastrowid)

    def query(self, filters: dict[str, Any] | None = None) -> list[dict]:
        from sqlalchemy import text

        sql = "SELECT * FROM runs"
        conditions = []
        params = {}

        if filters:
            for key, value in filters.items():
                conditions.append(f"{key} = :{key}")
                params[key] = value

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY timestamp DESC"

        result = self.session.execute(text(sql), params)
        rows = []
        for row in result:
            row_dict = dict(row._mapping)
            if row_dict.get("config"):
                row_dict["config"] = json.loads(row_dict["config"])
            if row_dict.get("metrics"):
                row_dict["metrics"] = json.loads(row_dict["metrics"])
            rows.append(row_dict)

        return rows

    def close(self) -> None:
        if self._session:
            self._session.close()
