from __future__ import annotations

from glob import glob

from setuptools import setup

APP = ["mac_app.py"]
DATA_FILES = [
    ("templates", glob("templates/*")),
    ("static", glob("static/*")),
    ("", [".setting.example.json"]),
]
OPTIONS = {
    "argv_emulation": False,
    "iconfile": "assets/VocalLens.icns",
    "packages": [
        "flask",
        "werkzeug",
        "jinja2",
        "openai",
        "webview",
    ],
    "plist": {
        "CFBundleName": "Vocal Lens",
        "CFBundleDisplayName": "Vocal Lens",
        "CFBundleIdentifier": "com.krha.vocallens",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleVersion": "1.0.0",
        "NSHighResolutionCapable": True,
    },
}

setup(
    app=APP,
    name="Vocal Lens",
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
