"""Email report generation and sending."""

from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from madengine.reporting.html import HTMLExporter


class EmailReporter:
    """Send HTML reports via email."""

    def __init__(self, cfg: dict[str, Any]):
        self.smtp_host = cfg.get("smtp_host")
        self.smtp_port = cfg.get("smtp_port", 587)
        self.from_address = cfg.get("from_address")
        self.to_addresses = cfg.get("to_addresses", [])
        self.use_tls = cfg.get("use_tls", True)

    def send(
        self,
        subject: str,
        records: list[dict],
        to_addresses: list[str] | None = None,
    ) -> None:
        """Generate HTML report and send via email."""
        if not self.smtp_host or not self.from_address:
            raise ValueError("SMTP host and from_address must be configured")

        recipients = to_addresses or self.to_addresses
        if not recipients:
            raise ValueError("No recipients specified")

        exporter = HTMLExporter()
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            exporter.export(output_path=f.name, records=records)
            html_content = Path(f.name).read_text()
            Path(f.name).unlink()

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_address
        msg["To"] = ", ".join(recipients)

        msg.attach(MIMEText(html_content, "html"))

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.use_tls:
                server.starttls()
            server.sendmail(self.from_address, recipients, msg.as_string())
