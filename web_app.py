#!/usr/bin/env python3
"""Web UI for MP3 transcription + optional summary generation."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from transcribe_mp3 import (
    DEFAULT_MODEL,
    DEFAULT_SUMMARY_MODEL,
    MAX_CHUNK_SECONDS,
    transcribe_mp3_file,
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "web_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

TRANSCRIBE_MODELS = [
    "gpt-4o-transcribe-diarize",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
]

SUMMARY_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
]

_LANGUAGE_CATALOG: List[Tuple[str, str]] = [
    ("af", "Afrikaans"),
    ("am", "Amharic"),
    ("ar", "Arabic"),
    ("as", "Assamese"),
    ("az", "Azerbaijani"),
    ("ba", "Bashkir"),
    ("be", "Belarusian"),
    ("bg", "Bulgarian"),
    ("bn", "Bengali"),
    ("bo", "Tibetan"),
    ("br", "Breton"),
    ("bs", "Bosnian"),
    ("ca", "Catalan"),
    ("cs", "Czech"),
    ("cy", "Welsh"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("eu", "Basque"),
    ("fa", "Persian"),
    ("fi", "Finnish"),
    ("fo", "Faroese"),
    ("fr", "French"),
    ("gl", "Galician"),
    ("gu", "Gujarati"),
    ("ha", "Hausa"),
    ("haw", "Hawaiian"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hr", "Croatian"),
    ("ht", "Haitian Creole"),
    ("hu", "Hungarian"),
    ("hy", "Armenian"),
    ("id", "Indonesian"),
    ("is", "Icelandic"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("jw", "Javanese"),
    ("ka", "Georgian"),
    ("kk", "Kazakh"),
    ("km", "Khmer"),
    ("kn", "Kannada"),
    ("ko", "Korean"),
    ("la", "Latin"),
    ("lb", "Luxembourgish"),
    ("ln", "Lingala"),
    ("lo", "Lao"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("mg", "Malagasy"),
    ("mi", "Maori"),
    ("mk", "Macedonian"),
    ("ml", "Malayalam"),
    ("mn", "Mongolian"),
    ("mr", "Marathi"),
    ("ms", "Malay"),
    ("mt", "Maltese"),
    ("my", "Burmese"),
    ("ne", "Nepali"),
    ("nl", "Dutch"),
    ("nn", "Nynorsk"),
    ("no", "Norwegian"),
    ("oc", "Occitan"),
    ("pa", "Punjabi"),
    ("pl", "Polish"),
    ("ps", "Pashto"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sa", "Sanskrit"),
    ("sd", "Sindhi"),
    ("si", "Sinhala"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sn", "Shona"),
    ("so", "Somali"),
    ("sq", "Albanian"),
    ("sr", "Serbian"),
    ("su", "Sundanese"),
    ("sv", "Swedish"),
    ("sw", "Swahili"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("tg", "Tajik"),
    ("th", "Thai"),
    ("tk", "Turkmen"),
    ("tl", "Tagalog"),
    ("tr", "Turkish"),
    ("tt", "Tatar"),
    ("uk", "Ukrainian"),
    ("ur", "Urdu"),
    ("uz", "Uzbek"),
    ("vi", "Vietnamese"),
    ("yi", "Yiddish"),
    ("yo", "Yoruba"),
    ("zh", "Chinese"),
]

_PINNED_LANGUAGES = [("en", "English"), ("ko", "Korean")]
LANGUAGE_OPTIONS = _PINNED_LANGUAGES + sorted(
    [entry for entry in _LANGUAGE_CATALOG if entry[0] not in {"en", "ko"}],
    key=lambda item: item[1].lower(),
)

JOB_LOCK = threading.Lock()
JOBS: Dict[str, Dict[str, Any]] = {}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_max_chunk_seconds(raw_value: Any) -> int:
    if raw_value is None:
        return MAX_CHUNK_SECONDS
    raw = str(raw_value).strip()
    if not raw:
        return MAX_CHUNK_SECONDS
    return int(raw)


def _update_job(job_id: str, **updates: Any) -> None:
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.update(updates)
        job["updated_at"] = time.time()


def _get_job(job_id: str) -> Dict[str, Any] | None:
    with JOB_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job is not None else None


def _artifact_entry(job: Dict[str, Any], artifact: str) -> tuple[str | None, str]:
    result = job.get("result")
    if not isinstance(result, dict):
        return None, "Job result is not available."

    mapping = {
        "input_audio": "copied_audio",
        "transcript_txt": "transcript_path",
        "transcript_json": "transcript_json_path",
        "summary_txt": "summary_path",
    }
    key = mapping.get(artifact)
    if key is None:
        return None, "Unsupported artifact."

    raw = result.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return None, "Requested artifact was not generated."
    return raw, ""


def _safe_path_from_result(path_text: str) -> Path:
    path = Path(path_text).resolve()
    try:
        path.relative_to(BASE_DIR)
    except ValueError:
        raise PermissionError("Artifact path is outside workspace.")
    return path


def _run_transcription_job(
    job_id: str,
    upload_path: Path,
    job_dir: Path,
    request_data: Dict[str, Any],
) -> None:
    def on_progress(message: str, progress: float) -> None:
        _update_job(
            job_id,
            status="running",
            message=message,
            progress=progress,
        )

    try:
        result = transcribe_mp3_file(
            input_mp3=str(upload_path),
            settings_file=request_data["settings_file"],
            output_root=str(BASE_DIR),
            model=request_data["model"],
            language=request_data["language"],
            max_chunk_seconds=request_data["max_chunk_seconds"],
            generate_summary_file=request_data["generate_summary"],
            summary_model=request_data["summary_model"],
            verbose=False,
            progress_callback=on_progress,
        )
        _update_job(
            job_id,
            status="completed",
            message="Completed",
            progress=1.0,
            result=result,
            error=None,
            completed_at=time.time(),
        )
    except Exception as exc:  # noqa: BLE001
        _update_job(
            job_id,
            status="failed",
            message="Failed",
            progress=1.0,
            error=str(exc),
            completed_at=time.time(),
        )
    finally:
        try:
            if upload_path.exists():
                upload_path.unlink()
            if job_dir.exists():
                job_dir.rmdir()
        except OSError:
            pass


@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        transcribe_models=TRANSCRIBE_MODELS,
        summary_models=SUMMARY_MODELS,
        language_options=LANGUAGE_OPTIONS,
        default_model=DEFAULT_MODEL,
        default_summary_model=DEFAULT_SUMMARY_MODEL,
        default_max_chunk_seconds=MAX_CHUNK_SECONDS,
    )


@app.get("/api/health")
def health() -> Any:
    return jsonify({"ok": True})


@app.post("/api/transcribe")
def transcribe_api() -> Any:
    upload = request.files.get("audio")
    if upload is None:
        return jsonify({"ok": False, "error": "Missing file field: audio"}), 400

    filename = secure_filename(upload.filename or "")
    if not filename:
        return jsonify({"ok": False, "error": "No file selected"}), 400
    if not filename.lower().endswith(".mp3"):
        return jsonify({"ok": False, "error": "Only .mp3 files are supported"}), 400

    try:
        max_chunk_seconds = _parse_max_chunk_seconds(request.form.get("max_chunk_seconds"))
    except ValueError:
        return jsonify({"ok": False, "error": "max_chunk_seconds must be a number"}), 400

    job_id = uuid.uuid4().hex
    job_dir = UPLOAD_DIR / f"job-{job_id}"
    job_dir.mkdir(parents=True, exist_ok=False)
    upload_path = job_dir / filename
    upload.save(upload_path)

    request_data = {
        "settings_file": request.form.get("settings_file")
        or str(BASE_DIR / ".setting.json"),
        "model": request.form.get("model") or DEFAULT_MODEL,
        "summary_model": request.form.get("summary_model") or DEFAULT_SUMMARY_MODEL,
        "language": request.form.get("language") or None,
        "max_chunk_seconds": max_chunk_seconds,
        "generate_summary": _as_bool(request.form.get("generate_summary"), False),
    }

    with JOB_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "error": None,
            "result": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    worker = threading.Thread(
        target=_run_transcription_job,
        args=(job_id, upload_path, job_dir, request_data),
        daemon=True,
    )
    worker.start()

    return jsonify({"ok": True, "job_id": job_id})


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str) -> Any:
    job = _get_job(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Job not found"}), 404
    return jsonify({"ok": True, **job})


@app.get("/api/jobs/<job_id>/download/<artifact>")
def job_download(job_id: str, artifact: str) -> Any:
    job = _get_job(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Job not found"}), 404
    if job.get("status") != "completed":
        return jsonify({"ok": False, "error": "Job is not completed yet"}), 409

    artifact_path_text, error_message = _artifact_entry(job, artifact)
    if artifact_path_text is None:
        return jsonify({"ok": False, "error": error_message}), 404

    try:
        artifact_path = _safe_path_from_result(artifact_path_text)
    except PermissionError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 403

    if not artifact_path.exists() or not artifact_path.is_file():
        return jsonify({"ok": False, "error": "Artifact file not found"}), 404

    return send_file(
        artifact_path,
        as_attachment=True,
        download_name=artifact_path.name,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)
