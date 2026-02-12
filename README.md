# Vocal Lens

Transcribe MP3 audio with speaker labels and optional AI summary generation.

## Features

- MP3 transcription with speaker annotations (diarization model).
- Automatic chunking for large files (API size-safe).
- Progress updates in CLI and web UI.
- Summary generation from transcript using a custom prompt.
- Markdown output files with structured formatting.
- Web UI with upload, live progress, tabs, and downloadable artifacts.

## Requirements

- Python 3.9+
- `ffmpeg` and `ffprobe` available in `PATH`
- OpenAI API key

## Setup

```bash
cd /Users/krha/code/transcribe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create `/Users/krha/code/transcribe/.setting.json`:

```json
{
  "OPENAI_API_KEY": "sk-...",
  "SUMMARY_PROMPT": "Your summary instructions here"
}
```

Notes:
- `.setting.json` is gitignored.
- `SUMMARY_PROMPT` is used only when summary generation is enabled.

## CLI Usage

Transcribe only:

```bash
cd /Users/krha/code/transcribe
source .venv/bin/activate
python transcribe_mp3.py input/input.mp3
```

Transcribe + summary:

```bash
python transcribe_mp3.py input/input.mp3 --generate-summary
```

Useful flags:
- `--settings-file` (default: `.setting.json`)
- `--model` (default: `gpt-4o-transcribe-diarize`)
- `--summary-model` (default: `gpt-4.1-mini`)
- `--max-chunk-seconds` (default: `600`)
- `--language` (example: `en`, `ko`)

## Web UI

Start server:

```bash
cd /Users/krha/code/transcribe
./run.sh
```

Open:
- [http://127.0.0.1:8000](http://127.0.0.1:8000)

UI includes:
- MP3 upload
- Model selectors
- Language selector
- Live progress bar
- Download buttons for generated files

## API Endpoints

- `POST /api/transcribe`  
  Starts a background job. Returns `job_id`.
- `GET /api/jobs/<job_id>`  
  Poll status/progress/result.
- `GET /api/jobs/<job_id>/download/<artifact>`  
  Download output file.

Supported `artifact` values:
- `input_audio`
- `transcript_md`
- `transcript_json`
- `summary_md`

## Output

Each run creates:

```text
output-<input-stem>-<unixtime>/
```

Typical files:
- `<name>.mp3`
- `<name>.transcript.md`
- `<name>.transcript.json`
- `<name>.summary.md` (if enabled)

## Troubleshooting

- `Port 8000 is in use`: stop old server with `pkill -f web_app.py`.
- `Required binary not found`: install `ffmpeg`/`ffprobe`.
- `Could not parse OpenAI API key`: check `.setting.json` key names/JSON format.
- Slow processing: expected for long audio and summary generation.
