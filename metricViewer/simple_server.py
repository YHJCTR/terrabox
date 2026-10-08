from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .adapters import DEFAULT_ADAPTER, DEFAULT_SCENE


STATIC_DIR = Path(__file__).resolve().parent / "static"


class MetricViewerHandler(BaseHTTPRequestHandler):
    server_version = "MetricViewer/0.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_file(STATIC_DIR / "index.html")
                return
            if parsed.path.startswith("/static/"):
                rel = unquote(parsed.path.removeprefix("/static/"))
                target = (STATIC_DIR / rel).resolve()
                if not target.is_relative_to(STATIC_DIR.resolve()):
                    self._send_json({"detail": "invalid static path"}, status=403)
                    return
                self._send_file(target)
                return
            if parsed.path == "/api/experiments":
                scene = parse_qs(parsed.query).get("scene", [DEFAULT_SCENE])[0]
                refs = DEFAULT_ADAPTER.discover(scene)
                self._send_json(
                    {
                        "experiments": [
                            {
                                "name": ref.name,
                                "scene": ref.scene,
                                "id": ref.id,
                                "kind": ref.kind,
                                "results_dir": str(ref.source_path),
                                "prompt_path": str(ref.prompt_path) if ref.prompt_path else None,
                            }
                            for ref in refs
                        ]
                    }
                )
                return
            if parsed.path == "/api/status":
                qs = parse_qs(parsed.query)
                experiment = self._required(qs, "experiment")
                scene = qs.get("scene", [DEFAULT_SCENE])[0]
                scope = qs.get("scope", ["all"])[0]
                self._send_json(DEFAULT_ADAPTER.status(scene, experiment, scope=scope))
                return
            if parsed.path == "/api/compare":
                qs = parse_qs(parsed.query)
                current = self._required(qs, "current")
                baseline = self._required(qs, "baseline")
                scene = qs.get("scene", [DEFAULT_SCENE])[0]
                self._send_json(DEFAULT_ADAPTER.compare(scene, current, baseline))
                return
            if parsed.path == "/api/prompt":
                qs = parse_qs(parsed.query)
                experiment = self._required(qs, "experiment")
                scene = qs.get("scene", [DEFAULT_SCENE])[0]
                self._send_json(DEFAULT_ADAPTER.prompt(scene, experiment))
                return
            self._send_json({"detail": "not found"}, status=404)
        except Exception as exc:  # noqa: BLE001 - show readable UI errors
            self._send_json({"detail": str(exc)}, status=400)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _required(self, qs: dict[str, list[str]], key: str) -> str:
        value = qs.get(key, [""])[0]
        if not value:
            raise ValueError(f"missing query parameter: {key}")
        return value

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_json({"detail": "file not found"}, status=404)
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: object, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), MetricViewerHandler)
    print(f"Metric Viewer running at http://{host}:{port}")
    server.serve_forever()
