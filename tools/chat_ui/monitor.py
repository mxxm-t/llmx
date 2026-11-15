"""Embed the existing monitor with one collector for the workspace upstream."""

import importlib.util
from pathlib import Path


DASHBOARD = Path(__file__).resolve().parents[1] / "health_dashboard"
_spec = importlib.util.spec_from_file_location("llmx_health_collector", DASHBOARD / "server.py")
health = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(health)
ASSETS = health.ASSETS


class Monitor:
    def __init__(self, upstream, key=None):
        base = upstream.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        headers = {"Authorization": "Bearer " + key} if key else None
        self.collector = health.Collector(base + "/health", headers=headers)

    def document(self):
        html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
        html = html.replace('<html lang="en">', '<html lang="en" data-api-root="/api/monitor" data-embedded="true">', 1)
        return html.replace("</head>", '<link rel="stylesheet" href="/monitor.css"></head>', 1).encode()

    def stream(self, handler, authorized):
        health.stream_history(handler, self.collector, authorized)

    def asset(self, name):
        if name not in ASSETS:
            raise KeyError(name)
        return (DASHBOARD / name).read_bytes(), ASSETS[name]

    def stop(self):
        self.collector.stop()
