"""Lightweight healthcheck API for local and container smoke tests."""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

_START = time.time()
_VERSION = os.getenv("APP_VERSION", "unknown")
_ENV = os.getenv("APP_ENV", "dev")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}", flush=True)

    def _send(self, status: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {
                "status": "ok",
                "version": _VERSION,
                "env": _ENV,
                "uptime_seconds": round(time.time() - _START, 1),
            })

        elif self.path == "/":
            self._send(200, {
                "service": "sert-vr-training-api",
                "version": _VERSION,
                "env": _ENV,
                "endpoints": ["/health"],
            })

        else:
            self._send(404, {"error": "not found"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"[API] sert-vr-training-api v{_VERSION} ({_ENV}) listening on :{port}", flush=True)
    server.serve_forever()
