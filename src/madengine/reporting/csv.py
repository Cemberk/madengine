"""CSV export for metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVExporter:
    """Export metrics to CSV format."""

    def __init__(self, database_cfg: dict[str, Any] | None = None):
        self._database_cfg = database_cfg

    def export(
        self,
        output_path: str | Path,
        records: list[dict] | None = None,
        query: dict | None = None,
    ) -> None:
        """Export records to CSV.

        Args:
            output_path: Path to write CSV.
            records: Records to export (if not using database).
            query: Database query filters (used if records is None).
        """
        if records is None and self._database_cfg:
            from madengine.database import get_database_backend

            db = get_database_backend(self._database_cfg)
            records = db.query(query)

        if not records:
            raise ValueError("No records to export")

        output_path = Path(output_path)
        flat_records = [self._flatten(r) for r in records]

        all_keys: set[str] = set()
        for r in flat_records:
            all_keys.update(r.keys())

        fieldnames = sorted(all_keys)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_records)

    @staticmethod
    def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten nested dict for tabular export."""
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(CSVExporter._flatten(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
