"""HTTP entrypoint shared by Flask and the standard-library fallback."""
import json
import logging
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import RLock
from typing import Any

from .brain import Strategy, empty_response

LOGGER = logging.getLogger("contestant")
ROOT = Path(__file__).resolve().parents[2]
MAX_BODY = 4 * 1024 * 1024
_LOCK = RLock()


def load_config() -> dict[str, Any]:
    filename = os.environ.get("WAR_CONFIG", str(ROOT / "config.json"))
    with open(filename, encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    if config.get("round_origin", 1) not in (0, 1):
        raise ValueError("round_origin must be 0 or 1")
    return config


_AGENT = Strategy(load_config())


def parse_json_tolerant(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
        return json.loads(cleaned)


def callback(payload: Any) -> dict[str, Any]:
    """Serialize access because the strategy keeps cross-round state."""
    with _LOCK:
        try:
            return _AGENT.callback(payload)
        except Exception:
            LOGGER.exception("policy failed")
            return empty_response()


def create_app() -> Any:
    from flask import Flask, jsonify, request

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY
    app.json.ensure_ascii = False

    @app.post("/")
    def process_request() -> Any:
        try:
            payload = parse_json_tolerant(request.get_data(as_text=True) or "")
        except (ValueError, TypeError, json.JSONDecodeError):
            payload = None
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON object required"}), 400
        response = callback(payload)
        LOGGER.info("round %s -> %s", payload.get("roundNo"), response)
        return jsonify(response)

    return app


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                self.send_error(413)
                return
            payload = parse_json_tolerant(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError
        except (ValueError, UnicodeError):
            self.send_error(400)
            return

        response = callback(payload)
        LOGGER.info("round %s -> %s", payload.get("roundNo"), response)
        body = json.dumps(
            response,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.debug(format, *args)


def serve(port: int) -> None:
    LOGGER.info("agent_version=1.3")
    LOGGER.info(
        "construction=%s",
        _AGENT.config.get("construction", {}).get("mode", "auto"),
    )
    if os.environ.get("WAR_HTTP_BACKEND") != "stdlib":
        try:
            app = create_app()
        except ModuleNotFoundError as exc:
            if exc.name != "flask":
                raise
            LOGGER.info("Flask unavailable; using stdlib HTTP server")
        else:
            app.run(
                host="0.0.0.0",
                port=port,
                threaded=False,
                debug=False,
                use_reloader=False,
            )
            return

    LOGGER.info("listening on 0.0.0.0:%d", port)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
