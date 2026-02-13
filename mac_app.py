#!/usr/bin/env python3
"""Native macOS launcher for Vocal Lens.

Starts the local Flask server on a free localhost port and opens it inside
a native webview window.
"""

from __future__ import annotations

import os
import socket
import shutil
import sys
import threading
from pathlib import Path

from werkzeug.serving import make_server

from web_app import app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_port() -> int:
    raw = os.getenv("VOCAL_LENS_PORT", "").strip()
    if raw:
        try:
            port = int(raw)
        except ValueError:
            port = 0
        if 1 <= port <= 65535:
            return port
    return _free_port()


class ServerThread(threading.Thread):
    def __init__(self, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self._server = make_server(host, port, app)
        self._context = app.app_context()
        self._context.push()

    def run(self) -> None:
        self._server.serve_forever()

    def stop(self) -> None:
        self._server.shutdown()


class DesktopApi:
    """Bridge for desktop-only UI actions exposed to JS via pywebview."""

    def __init__(self, window_getter, webview_mod) -> None:
        self._window_getter = window_getter
        self._webview = webview_mod

    def save_artifact(self, job_id: str, artifact: str) -> dict:
        try:
            from web_app import _artifact_entry, _get_job, _safe_path_from_result
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Desktop API is unavailable: {exc}"}

        job = _get_job(str(job_id))
        if job is None:
            return {"ok": False, "error": "Job not found."}
        if job.get("status") != "completed":
            return {"ok": False, "error": "Job is not completed yet."}

        path_text, error_message = _artifact_entry(job, str(artifact))
        if path_text is None:
            return {"ok": False, "error": error_message}

        try:
            source_path = _safe_path_from_result(path_text)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Invalid artifact path: {exc}"}

        if not source_path.exists() or not source_path.is_file():
            return {"ok": False, "error": "Artifact file not found."}

        window = self._window_getter()
        if window is None:
            return {"ok": False, "error": "Desktop window is not ready."}

        selected = window.create_file_dialog(
            dialog_type=self._webview.SAVE_DIALOG,
            directory=str(Path.home() / "Downloads"),
            save_filename=source_path.name,
            file_types=self._file_types_for(source_path.suffix.lower()),
        )
        if not selected:
            return {"ok": False, "cancelled": True}

        if isinstance(selected, (list, tuple)):
            destination = Path(selected[0]).expanduser().resolve()
        else:
            destination = Path(str(selected)).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_path, destination)
        return {"ok": True, "path": str(destination)}

    @staticmethod
    def _file_types_for(suffix: str) -> tuple[str, ...]:
        if suffix == ".md":
            return ("Markdown (*.md)", "All files (*.*)")
        if suffix == ".json":
            return ("JSON (*.json)", "All files (*.*)")
        if suffix == ".mp3":
            return ("MP3 (*.mp3)", "All files (*.*)")
        return ("All files (*.*)",)


def _ensure_pyobjc_file_attribute() -> None:
    """pywebview expects objc._objc.__file__ when loading Cocoa backend.

    In some py2app bundles this attribute can be missing, so patch it
    from known runtime locations before calling webview.start().
    """

    try:
        from objc import _objc
    except Exception:
        return

    if getattr(_objc, "__file__", None):
        return

    spec = getattr(_objc, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if isinstance(origin, str) and origin and origin != "built-in":
        _objc.__file__ = origin
        return

    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    frozen_candidate = (
        Path(sys.executable).resolve().parents[1]
        / "Resources"
        / "lib"
        / version
        / "lib-dynload"
        / "objc"
        / "_objc.so"
    )
    if frozen_candidate.exists():
        _objc.__file__ = str(frozen_candidate)
        return

    try:
        import objc as objc_pkg

        pkg_file = getattr(objc_pkg, "__file__", "")
        if pkg_file:
            pkg_dir = Path(pkg_file).resolve().parent
            matches = sorted(pkg_dir.glob("_objc*.so"))
            if matches:
                _objc.__file__ = str(matches[0])
                return
    except Exception:
        pass


def main() -> None:
    _ensure_pyobjc_file_attribute()
    import webview

    host = "127.0.0.1"
    port = _resolve_port()
    url = f"http://{host}:{port}"

    window_holder = {"window": None}
    desktop_api = DesktopApi(
        window_getter=lambda: window_holder.get("window"),
        webview_mod=webview,
    )

    server = ServerThread(host, port)
    server.start()

    try:
        window = webview.create_window(
            "Vocal Lens",
            url=url,
            width=1280,
            height=860,
            min_size=(980, 680),
            js_api=desktop_api,
        )
        window_holder["window"] = window
        webview.start()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
