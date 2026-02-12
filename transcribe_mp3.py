#!/usr/bin/env python3
"""Transcribe MP3 audio with speaker labels using OpenAI diarization.

Features:
- Reads settings from a `.setting.json` file.
- Splits oversized MP3 files into API-safe chunks with open-source FFmpeg.
- Produces speaker-labeled transcript text output.
- Optionally generates an AI summary from transcript text.
- Writes results to `output-<input-stem>-<unixtime>/`.
- Copies the original input MP3 into the output directory.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

MAX_API_BYTES = 25 * 1024 * 1024
TARGET_CHUNK_BYTES = 24 * 1024 * 1024
MAX_CHUNK_SECONDS = 10 * 60
DEFAULT_MODEL = "gpt-4o-transcribe-diarize"
DEFAULT_RESPONSE_FORMAT = "diarized_json"
DEFAULT_CHUNKING_STRATEGY = "auto"
DEFAULT_SUMMARY_MODEL = "gpt-4.1-mini"

# Pricing snapshot (USD) based on OpenAI pricing page as of 2026-02-12.
# Update these values over time as pricing changes.
TRANSCRIPTION_COST_PER_MINUTE_USD: Dict[str, float] = {
    "gpt-4o-transcribe-diarize": 0.006,
    "gpt-4o-transcribe": 0.006,
    "gpt-4o-mini-transcribe": 0.003,
}

SUMMARY_COST_PER_MILLION_TOKENS_USD: Dict[str, Dict[str, float]] = {
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


@dataclass
class ChunkInfo:
    path: Path
    start_seconds: float


@dataclass
class NormalizedSegment:
    start: float
    end: float
    speaker: str
    text: str


def die(message: str, code: int = 1) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(code)


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def require_binary(binary: str) -> None:
    if shutil.which(binary) is None:
        die(f"Required binary not found in PATH: {binary}")


def resolve_settings_file(input_path: Path, requested_settings_file: str) -> Path:
    requested = Path(requested_settings_file)
    candidates = [
        requested,
        Path.cwd() / requested,
        input_path.parent / requested.name,
        Path.cwd() / "input" / requested.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    die(
        "Could not find settings file. Checked: "
        + ", ".join(str(p) for p in candidates)
    )
    raise AssertionError("unreachable")


def load_settings_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        die(f"Settings file is empty: {path}")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        die(f"Invalid JSON in settings file {path}: {exc}")

    if not isinstance(payload, dict):
        die(f"Settings JSON must be an object: {path}")
    return payload


def _first_non_empty_string(data: Dict[str, Any], names: List[str]) -> Optional[str]:
    for name in names:
        value = data.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_api_key_from_settings(settings: Dict[str, Any]) -> str:
    value = _first_non_empty_string(
        settings,
        [
            "OPENAI_API_KEY",
            "openai_api_key",
            "OPENAP-API",
            "OPENAI-API-KEY",
            "api_key",
        ],
    )
    if value:
        return value

    for raw in settings.values():
        if isinstance(raw, str):
            value = raw.strip()
            if value.startswith("sk-") or value.startswith("rk-"):
                return value
    die("Could not parse OpenAI API key from settings JSON")
    raise AssertionError("unreachable")


def load_summary_prompt_from_settings(settings: Dict[str, Any]) -> Optional[str]:
    value = _first_non_empty_string(
        settings,
        [
            "SUMMARY-PROMPT",
            "SUMMARY_PROMPT",
            "summary_prompt",
            "summaryPrompt",
        ],
    )
    if value:
        return value
    nested_summary = settings.get("summary")
    if isinstance(nested_summary, dict):
        value = _first_non_empty_string(
            nested_summary,
            ["prompt", "SUMMARY-PROMPT", "SUMMARY_PROMPT"],
        )
        if value:
            return value
    return None


def ffprobe_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    out = run_cmd(cmd).stdout.strip()
    try:
        value = float(out)
    except ValueError as exc:
        raise RuntimeError(f"Failed to parse duration from ffprobe output: {out}") from exc
    if value <= 0:
        raise RuntimeError(f"Invalid duration from ffprobe: {value}")
    return value


def ffprobe_duration_seconds_safe(path: Path) -> Optional[float]:
    if shutil.which("ffprobe") is None:
        return None
    try:
        return ffprobe_duration_seconds(path)
    except Exception:
        return None


def format_usd(amount: Optional[float]) -> str:
    if amount is None:
        return "N/A"
    return f"${amount:.6f}"


def estimate_transcription_cost_usd(model: str, duration_seconds: Optional[float]) -> Optional[float]:
    if duration_seconds is None:
        return None
    per_minute = TRANSCRIPTION_COST_PER_MINUTE_USD.get(model)
    if per_minute is None:
        return None
    return (duration_seconds / 60.0) * per_minute


def estimate_summary_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> Optional[float]:
    pricing = SUMMARY_COST_PER_MILLION_TOKENS_USD.get(model)
    if pricing is None:
        return None
    return (
        (input_tokens / 1_000_000.0) * pricing["input"]
        + (output_tokens / 1_000_000.0) * pricing["output"]
    )


def estimate_chunk_seconds(
    audio_size_bytes: int,
    audio_duration_seconds: float,
    max_chunk_seconds: int,
) -> int:
    bytes_per_second = audio_size_bytes / audio_duration_seconds
    raw_seconds = int((TARGET_CHUNK_BYTES * 0.92) / bytes_per_second)
    return max(30, min(raw_seconds, max_chunk_seconds))


def split_audio_if_needed(
    input_path: Path,
    workspace_dir: Path,
    max_chunk_seconds: int,
) -> List[ChunkInfo]:
    size = input_path.stat().st_size
    if size <= MAX_API_BYTES:
        return [ChunkInfo(path=input_path, start_seconds=0.0)]

    require_binary("ffmpeg")
    require_binary("ffprobe")

    duration = ffprobe_duration_seconds(input_path)
    chunk_seconds = estimate_chunk_seconds(size, duration, max_chunk_seconds)

    chunks_dir = workspace_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Retry with smaller segment duration until all chunks fit under API limit.
    for _ in range(6):
        for existing in chunks_dir.glob("chunk_*.mp3"):
            existing.unlink()

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "48k",
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            str(chunks_dir / "chunk_%04d.mp3"),
        ]
        subprocess.run(cmd, check=True, text=True, capture_output=True)

        chunk_paths = sorted(chunks_dir.glob("chunk_*.mp3"))
        if not chunk_paths:
            die("FFmpeg chunking produced no output files.")

        oversize = [p for p in chunk_paths if p.stat().st_size > MAX_API_BYTES]
        if not oversize:
            output: List[ChunkInfo] = []
            for idx, chunk_path in enumerate(chunk_paths):
                output.append(ChunkInfo(path=chunk_path, start_seconds=idx * chunk_seconds))
            return output

        chunk_seconds = max(15, int(chunk_seconds * 0.7))

    die("Unable to split audio into chunks under API limit after multiple attempts.")
    raise AssertionError("unreachable")


def to_dict(result: Any) -> Dict[str, Any]:
    if hasattr(result, "model_dump"):
        data = result.model_dump()
        if isinstance(data, dict):
            return data
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"text": result}
    return {"text": str(result)}


def parse_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def speaker_label(raw_speaker: Any, chunk_index: int, total_chunks: int) -> str:
    text = str(raw_speaker).strip() if raw_speaker is not None else ""
    if not text:
        base = "Speaker 1"
    else:
        match = re.search(r"(\d+)", text)
        if match:
            base = f"Speaker {int(match.group(1)) + 1}"
        else:
            base = text

    if total_chunks > 1:
        return f"Chunk {chunk_index + 1} {base}"
    return base


def extract_segments(
    payload: Dict[str, Any],
    chunk_start: float,
    chunk_index: int,
    total_chunks: int,
) -> List[NormalizedSegment]:
    raw_segments = []
    for key in ("segments", "diarized_segments", "utterances"):
        value = payload.get(key)
        if isinstance(value, list):
            raw_segments = value
            break

    segments: List[NormalizedSegment] = []
    if raw_segments:
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            start = parse_float(item.get("start", item.get("start_time", 0.0)))
            end = parse_float(item.get("end", item.get("end_time", start)))
            text = str(item.get("text", item.get("transcript", ""))).strip()
            if not text:
                continue
            speaker = speaker_label(
                item.get("speaker", item.get("speaker_label", item.get("speaker_id"))),
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
            segments.append(
                NormalizedSegment(
                    start=chunk_start + start,
                    end=max(chunk_start + end, chunk_start + start),
                    speaker=speaker,
                    text=text,
                )
            )
        return segments

    fallback_text = str(payload.get("text", "")).strip()
    if fallback_text:
        segments.append(
            NormalizedSegment(
                start=chunk_start,
                end=chunk_start,
                speaker=speaker_label(None, chunk_index=chunk_index, total_chunks=total_chunks),
                text=fallback_text,
            )
        )
    return segments


def format_timestamp(seconds: float) -> str:
    whole = max(0.0, seconds)
    hours = int(whole // 3600)
    minutes = int((whole % 3600) // 60)
    secs = whole % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def write_transcript_markdown(
    path: Path,
    segments: List[NormalizedSegment],
    metadata: Dict[str, str],
) -> None:
    lines = [
        "# Transcript",
        "",
        f"- **Source:** `{metadata['source']}`",
        f"- **Model:** `{metadata['model']}`",
        f"- **Generated (UTC):** `{metadata['generated_utc']}`",
        f"- **Chunk Count:** `{metadata.get('chunk_count', 'N/A')}`",
        "",
        "---",
        "",
        "## Segments",
        "",
    ]
    if len(segments) == 0:
        lines.append("_No transcript segments returned._")
    else:
        for segment in segments:
            text = segment.text.strip().replace("\n", " ")
            lines.append(
                f"- **[{format_timestamp(segment.start)} - {format_timestamp(segment.end)}] {segment.speaker}**"
            )
            lines.append(f"  {text}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_markdown(path: Path, summary_text: str, metadata: Dict[str, str]) -> None:
    lines = [
        "# Summary",
        "",
        f"- **Source:** `{metadata['source']}`",
        f"- **Summary Model:** `{metadata['summary_model']}`",
        f"- **Generated (UTC):** `{metadata['generated_utc']}`",
        "",
        "---",
        "",
        summary_text.strip(),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_cost_markdown(path: Path, cost_data: Dict[str, Any]) -> None:
    lines = [
        "# API Cost Estimate",
        "",
        "- Estimated from local pricing snapshot and runtime usage.",
        "- Currency: `USD`",
        "",
        "## Breakdown",
        "",
        f"- **Transcription Model:** `{cost_data.get('transcription_model', 'N/A')}`",
        f"- **Transcription Duration (seconds):** `{cost_data.get('transcription_duration_seconds', 'N/A')}`",
        f"- **Estimated Transcription Cost:** `{format_usd(cost_data.get('transcription_cost_usd'))}`",
        "",
        f"- **Summary Model:** `{cost_data.get('summary_model', 'N/A')}`",
        f"- **Summary Input Tokens:** `{cost_data.get('summary_input_tokens', 0)}`",
        f"- **Summary Output Tokens:** `{cost_data.get('summary_output_tokens', 0)}`",
        f"- **Estimated Summary Cost:** `{format_usd(cost_data.get('summary_cost_usd'))}`",
        "",
        f"- **Estimated Total Cost:** `{format_usd(cost_data.get('total_cost_usd'))}`",
    ]
    if cost_data.get("is_partial"):
        lines.extend(
            [
                "",
                "> Note: estimate is partial because pricing or usage details were unavailable for one or more steps.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def import_openai_client():
    try:
        from openai import OpenAI
    except ImportError:
        die("openai SDK is not installed. Install with: python3 -m pip install openai")
    return OpenAI


def transcribe_chunk(
    client: Any,
    chunk_path: Path,
    model: str,
    language: Optional[str],
) -> Dict[str, Any]:
    with chunk_path.open("rb") as audio_file:
        payload: Dict[str, Any] = {
            "model": model,
            "file": audio_file,
            "response_format": DEFAULT_RESPONSE_FORMAT,
            "chunking_strategy": DEFAULT_CHUNKING_STRATEGY,
        }
        if language:
            payload["language"] = language
        result = client.audio.transcriptions.create(**payload)
    return to_dict(result)


def _extract_output_text(result: Any) -> str:
    output_text = getattr(result, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    payload = to_dict(result)
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return str(payload["output_text"]).strip()

    output = payload.get("output")
    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for entry in content:
                if not isinstance(entry, dict):
                    continue
                text = entry.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "\n".join(parts)
    return ""


def _extract_usage_tokens(result: Any) -> Dict[str, int]:
    usage_payload: Any = getattr(result, "usage", None)
    if hasattr(usage_payload, "model_dump"):
        usage_payload = usage_payload.model_dump()
    if not isinstance(usage_payload, dict):
        payload = to_dict(result)
        usage_payload = payload.get("usage")
    if not isinstance(usage_payload, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "available": 0,
        }

    def _as_int(value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return max(parsed, 0)

    input_tokens = _as_int(
        usage_payload.get(
            "input_tokens",
            usage_payload.get("prompt_tokens", usage_payload.get("input_token_count")),
        )
    )
    output_tokens = _as_int(
        usage_payload.get(
            "output_tokens",
            usage_payload.get(
                "completion_tokens",
                usage_payload.get("output_token_count"),
            ),
        )
    )
    total_tokens = _as_int(
        usage_payload.get(
            "total_tokens",
            input_tokens + output_tokens,
        )
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "available": 1,
    }


def generate_summary(
    client: Any,
    model: str,
    summary_prompt: str,
    transcript_text: str,
) -> tuple[str, Dict[str, int]]:
    result = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": summary_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": transcript_text},
                ],
            },
        ],
    )
    summary = _extract_output_text(result).strip()
    if not summary:
        die("Summary generation succeeded but returned empty text.")
    return summary, _extract_usage_tokens(result)


def build_output_dir(output_root: Path, input_path: Path) -> Path:
    unix_ts = int(time.time())
    out_dir = output_root / f"output-{input_path.stem}-{unix_ts}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe an MP3 with speaker labels using OpenAI diarization."
    )
    parser.add_argument("input_mp3", help="Path to input mp3 file")
    parser.add_argument(
        "--settings-file",
        default=".setting.json",
        help="Path to settings JSON containing API key and summary prompt (default: .setting.json)",
    )
    parser.add_argument("--key-file", dest="settings_file", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-root",
        default=".",
        help="Directory where output-<name>-<unixtime> folder is created",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI transcription model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language hint, e.g. en",
    )
    parser.add_argument(
        "--max-chunk-seconds",
        type=int,
        default=MAX_CHUNK_SECONDS,
        help=f"Max chunk duration in seconds (default: {MAX_CHUNK_SECONDS})",
    )
    parser.add_argument(
        "--generate-summary",
        action="store_true",
        help="Generate a summary file using SUMMARY_PROMPT from the settings file",
    )
    parser.add_argument(
        "--summary-model",
        default=DEFAULT_SUMMARY_MODEL,
        help=f"OpenAI model for summary generation (default: {DEFAULT_SUMMARY_MODEL})",
    )
    return parser.parse_args()


def transcribe_mp3_file(
    input_mp3: str,
    settings_file: str = ".setting.json",
    output_root: str = ".",
    model: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    max_chunk_seconds: int = MAX_CHUNK_SECONDS,
    generate_summary_file: bool = False,
    summary_model: str = DEFAULT_SUMMARY_MODEL,
    verbose: bool = True,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    input_path = Path(input_mp3).resolve()
    if not input_path.exists():
        die(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".mp3":
        die(f"Input must be an mp3 file: {input_path}")
    if max_chunk_seconds < 30:
        die("--max-chunk-seconds must be at least 30")

    def emit(message: str, progress: float) -> None:
        bounded = max(0.0, min(progress, 1.0))
        if verbose:
            print(f"[{int(round(bounded * 100)):>3}%] {message}", flush=True)
        if progress_callback is not None:
            progress_callback(message, bounded)

    emit("Initializing transcription pipeline...", 0.01)

    output_root_path = Path(output_root).resolve()
    output_dir = build_output_dir(output_root=output_root_path, input_path=input_path)
    working_dir = output_dir / "work"
    working_dir.mkdir(parents=True, exist_ok=True)

    settings_path = resolve_settings_file(
        input_path=input_path,
        requested_settings_file=settings_file,
    )
    settings = load_settings_file(settings_path)
    api_key = load_api_key_from_settings(settings)

    OpenAI = import_openai_client()
    client = OpenAI(api_key=api_key, timeout=600)

    copied_audio = output_dir / input_path.name
    shutil.copy2(input_path, copied_audio)
    emit("Input audio copied to output workspace.", 0.08)

    chunks = split_audio_if_needed(
        input_path=input_path,
        workspace_dir=working_dir,
        max_chunk_seconds=max_chunk_seconds,
    )
    total_chunks = len(chunks)
    chunk_durations = [ffprobe_duration_seconds_safe(chunk.path) for chunk in chunks]
    transcription_duration_seconds: Optional[float]
    if all(duration is not None for duration in chunk_durations):
        transcription_duration_seconds = sum(duration for duration in chunk_durations if duration is not None)
    else:
        transcription_duration_seconds = None
    emit(f"Prepared {total_chunks} chunk(s) for processing.", 0.12)

    total_steps = (2 * total_chunks) + 2 + (1 if generate_summary_file else 0)
    step = 0

    def advance_progress(message: str) -> None:
        nonlocal step
        step += 1
        emit(message, step / total_steps)

    all_segments: List[NormalizedSegment] = []
    raw_chunk_payloads: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        advance_progress(f"Submitting chunk {index + 1}/{total_chunks}: {chunk.path.name}")
        payload = transcribe_chunk(
            client=client,
            chunk_path=chunk.path,
            model=model,
            language=language,
        )
        raw_chunk_payloads.append(payload)
        all_segments.extend(
            extract_segments(
                payload=payload,
                chunk_start=chunk.start_seconds,
                chunk_index=index,
                total_chunks=total_chunks,
            )
        )
        advance_progress(f"Transcribed chunk {index + 1}/{total_chunks}: {chunk.path.name}")

    generated_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    transcript_path = output_dir / f"{input_path.stem}.transcript.md"
    write_transcript_markdown(
        path=transcript_path,
        segments=all_segments,
        metadata={
            "source": input_path.name,
            "model": model,
            "generated_utc": generated_utc,
            "chunk_count": str(total_chunks),
        },
    )

    merged_json = {
        "source_file": input_path.name,
        "model": model,
        "chunk_count": total_chunks,
        "segments": [
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "speaker": seg.speaker,
                "text": seg.text,
            }
            for seg in all_segments
        ],
        "raw_chunks": raw_chunk_payloads,
    }
    json_path = output_dir / f"{input_path.stem}.transcript.json"
    json_path.write_text(
        json.dumps(merged_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    advance_progress("Transcript files written.",)

    summary_path: Optional[Path] = None
    summary_text: Optional[str] = None
    summary_usage: Dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "available": 0,
    }
    if generate_summary_file:
        summary_prompt = load_summary_prompt_from_settings(settings)
        if not summary_prompt:
            die(
                "SUMMARY-PROMPT not found in settings JSON. "
                "Add SUMMARY_PROMPT (or SUMMARY-PROMPT) to the settings file."
            )
        transcript_text = transcript_path.read_text(encoding="utf-8")
        advance_progress("Generating summary from transcript...")
        summary_text, summary_usage = generate_summary(
            client=client,
            model=summary_model,
            summary_prompt=summary_prompt,
            transcript_text=transcript_text,
        )
        summary_path = output_dir / f"{input_path.stem}.summary.md"
        write_summary_markdown(
            path=summary_path,
            summary_text=summary_text,
            metadata={
                "source": input_path.name,
                "summary_model": summary_model,
                "generated_utc": generated_utc,
            },
        )

    transcription_cost_usd = estimate_transcription_cost_usd(
        model=model,
        duration_seconds=transcription_duration_seconds,
    )
    summary_cost_usd: Optional[float] = None
    if generate_summary_file:
        if summary_usage.get("available", 0) == 1:
            summary_cost_usd = estimate_summary_cost_usd(
                model=summary_model,
                input_tokens=summary_usage.get("input_tokens", 0),
                output_tokens=summary_usage.get("output_tokens", 0),
            )

    known_costs = [
        value for value in (transcription_cost_usd, summary_cost_usd) if value is not None
    ]
    total_cost_usd = sum(known_costs) if known_costs else None
    expected_components = 2 if generate_summary_file else 1
    is_partial_cost = len(known_costs) < expected_components

    cost_path = output_dir / f"{input_path.stem}.cost.md"
    write_cost_markdown(
        path=cost_path,
        cost_data={
            "transcription_model": model,
            "transcription_duration_seconds": (
                round(transcription_duration_seconds, 3)
                if transcription_duration_seconds is not None
                else "N/A"
            ),
            "transcription_cost_usd": transcription_cost_usd,
            "summary_model": summary_model if generate_summary_file else "N/A",
            "summary_input_tokens": summary_usage.get("input_tokens", 0),
            "summary_output_tokens": summary_usage.get("output_tokens", 0),
            "summary_cost_usd": summary_cost_usd,
            "total_cost_usd": total_cost_usd,
            "is_partial": is_partial_cost,
        },
    )

    advance_progress("Finalizing outputs...",)

    emit(f"Output directory: {output_dir}", 1.0)
    emit(f"Copied input audio: {copied_audio}", 1.0)
    emit(f"Transcript markdown: {transcript_path}", 1.0)
    emit(f"Transcript json: {json_path}", 1.0)
    emit(f"Cost markdown: {cost_path}", 1.0)
    emit(f"Estimated transcription cost (USD): {format_usd(transcription_cost_usd)}", 1.0)
    if generate_summary_file:
        emit(f"Estimated summary cost (USD): {format_usd(summary_cost_usd)}", 1.0)
    emit(f"Estimated total cost (USD): {format_usd(total_cost_usd)}", 1.0)
    if is_partial_cost:
        emit("Cost estimate is partial due to missing pricing or usage data.", 1.0)
    if summary_path is not None:
        emit(f"Summary markdown: {summary_path}", 1.0)

    return {
        "output_dir": str(output_dir),
        "copied_audio": str(copied_audio),
        "transcript_path": str(transcript_path),
        "transcript_json_path": str(json_path),
        "summary_path": str(summary_path) if summary_path is not None else None,
        "cost_path": str(cost_path),
        "transcript_text": transcript_path.read_text(encoding="utf-8"),
        "summary_text": summary_text,
        "chunk_count": total_chunks,
        "source_file": input_path.name,
        "transcription_duration_seconds": transcription_duration_seconds,
        "cost_estimate_usd_transcription": transcription_cost_usd,
        "cost_estimate_usd_summary": summary_cost_usd,
        "cost_estimate_usd_total": total_cost_usd,
        "cost_estimate_is_partial": is_partial_cost,
        "summary_usage": summary_usage,
    }


def main() -> None:
    args = parse_args()
    transcribe_mp3_file(
        input_mp3=args.input_mp3,
        settings_file=args.settings_file,
        output_root=args.output_root,
        model=args.model,
        language=args.language,
        max_chunk_seconds=args.max_chunk_seconds,
        generate_summary_file=args.generate_summary,
        summary_model=args.summary_model,
        verbose=True,
    )


if __name__ == "__main__":
    main()
