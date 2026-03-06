"""HTML dashboard generation from metrics."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
    <title>madengine Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 40px;
            background: #fafafa;
            color: #333;
        }}
        h1 {{ color: #222; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #ddd; }}
        .success {{ color: green; font-weight: bold; }}
        .failure {{ color: red; font-weight: bold; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>madengine Run Report</h1>
    <p class="timestamp">Generated: {generated_at}</p>
    <table>
        <thead>
            <tr>
                <th>Timestamp</th>
                <th>Recipe</th>
                <th>Model</th>
                <th>Status</th>
                <th>Throughput</th>
                <th>Loss</th>
                <th>Duration</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</body>
</html>
"""


class HTMLExporter:
    """Generate HTML dashboard from metrics."""

    def __init__(self, database_cfg: dict[str, Any] | None = None):
        self._database_cfg = database_cfg

    def export(
        self,
        output_path: str | Path,
        records: list[dict] | None = None,
        query: dict | None = None,
        template: str | None = None,
    ) -> None:
        """Generate HTML report."""
        if records is None and self._database_cfg:
            from madengine.database import get_database_backend

            db = get_database_backend(self._database_cfg)
            records = db.query(query)

        if not records:
            raise ValueError("No records to export")

        template = template or DEFAULT_TEMPLATE

        rows = []
        for r in records:
            metrics = r.get("metrics", {})
            status_class = "success" if r.get("success") else "failure"
            status_text = "Success" if r.get("success") else "Failed"

            row = (
                "<tr>"
                f"<td>{r.get('timestamp', 'N/A')}</td>"
                f"<td>{r.get('recipe', 'N/A')}</td>"
                f"<td>{r.get('config', {}).get('model', 'N/A')}</td>"
                f'<td class="{status_class}">{status_text}</td>'
                f"<td>{metrics.get('throughput', 'N/A')}</td>"
                f"<td>{metrics.get('loss', 'N/A')}</td>"
                f"<td>{r.get('duration_seconds', 'N/A')}s</td>"
                "</tr>"
            )
            rows.append(row)

        html = template.format(
            generated_at=datetime.now(timezone.utc).isoformat(),
            rows="\n            ".join(rows),
        )

        Path(output_path).write_text(html)
