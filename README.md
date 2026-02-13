# Vocal Lens

Transcribe MP3 audio with speaker labels and optional AI summary generation.

## Features

- MP3 transcription with speaker annotations (diarization model).
- Automatic chunking for large files (API size-safe).
- Progress updates in CLI and web UI.
- Summary generation from transcript using a custom prompt.
- Markdown output files with structured formatting.
- Estimated API cost breakdown per run.
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

Desktop build dependencies:

```bash
pip install -r requirements-desktop.txt
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
- For the standalone macOS app, if no local `.setting.json` is found, a template is created at:
  - `~/Library/Application Support/VocalLens/.setting.json`

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

## Standalone macOS App (.app and .dmg)

Build app bundle and DMG:

```bash
cd /Users/krha/code/transcribe
./scripts/build_macos_app.sh
```

Generated artifacts:
- `.app` bundle in `dist/`
- DMG installer at `dist/Vocal-Lens.dmg`

What the standalone app does:
- Starts the local backend server internally
- Opens a native window (webview) for the UI
- Stores output under `~/Documents/VocalLens/`

## Mac App Store Path

This repo now includes the standalone app packaging foundation (`py2app` + native launcher), but App Store publishing still requires Apple-specific release steps:

- Apple Developer account + app identifier
- Code signing with distribution certificate
- Sandbox entitlements and App Store compliance review
- Notarization / App Store upload through Apple tooling

The provided DMG flow is the direct install path; App Store submission is a separate signing/distribution pipeline.

## API Endpoints

- `POST /api/transcribe`  
  Starts a background job. Returns `job_id`.
- `GET /api/jobs/<job_id>`  
  Poll status/progress/result.
- `GET /api/jobs/<job_id>/download/<artifact>`  
  Download output file.

Supported `artifact` values:
- `input_audio`
- `cost_md`
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
- `<name>.cost.md`
- `<name>.transcript.md`
- `<name>.transcript.json`
- `<name>.summary.md` (if enabled)

## Cost Estimation

- The app estimates execution cost and writes it to `<name>.cost.md`.
- UI also shows estimated total cost.
- Cost uses:
  - transcription model minute rates
  - summary model token rates + response token usage
- Pricing is based on a local snapshot and may change over time.

## Troubleshooting

- `Port 8000 is in use`: stop old server with `pkill -f web_app.py`.
- `Required binary not found`: install `ffmpeg`/`ffprobe`.
- `Could not parse OpenAI API key`: check `.setting.json` key names/JSON format.
- Slow processing: expected for long audio and summary generation.
