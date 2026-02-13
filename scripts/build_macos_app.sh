#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
BUILD_DIR="$ROOT_DIR/build"
DMG_STAGING_DIR="$DIST_DIR/dmg-staging"
DMG_PATH="$DIST_DIR/Vocal-Lens.dmg"

cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Error: .venv not found. Create it first: python3 -m venv .venv" >&2
  exit 1
fi

source ".venv/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-desktop.txt

rm -rf "$BUILD_DIR" "$DIST_DIR"
python setup.py py2app

APP_BUNDLE="$(find "$DIST_DIR" -maxdepth 1 -name '*.app' | head -n 1)"
if [[ -z "${APP_BUNDLE:-}" ]]; then
  echo "Error: app bundle not found in $DIST_DIR" >&2
  exit 1
fi

rm -rf "$DMG_STAGING_DIR"
mkdir -p "$DMG_STAGING_DIR"
cp -R "$APP_BUNDLE" "$DMG_STAGING_DIR/"
ln -s /Applications "$DMG_STAGING_DIR/Applications"

rm -f "$DMG_PATH"
hdiutil create \
  -volname "Vocal Lens" \
  -srcfolder "$DMG_STAGING_DIR" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

rm -rf "$DMG_STAGING_DIR"

echo "App bundle: $APP_BUNDLE"
echo "DMG created: $DMG_PATH"
