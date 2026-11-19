"""Identify the installed PDF.js text extraction assets."""

from pathlib import Path
import re


VERSION = "6.5.438"
ARCHIVE_SHA256 = "fcbe9ba21d9911843e3502adc178a7f549a524a4055f887bedc942e5490c6ba1"
READER_DIRECTORY = Path(__file__).resolve().parent / ".pdf-reader" / VERSION
INSTALL_RECORD = f"PDF.js {VERSION}\nSHA-256 {ARCHIVE_SHA256}\n".encode("ascii")
INSTALL_MARKER = "INSTALLATION"
MODULES = ("pdf.mjs", "pdf.worker.mjs")
RESOURCE_PATHS = ("cmaps/", "standard_fonts/")
ASSET = re.compile(r"(?:cmaps/[A-Za-z0-9_-]+\.bcmap|standard_fonts/[A-Za-z0-9_-]+\.(?:pfb|ttf))")
ORIGIN = re.compile(r"https?://[A-Za-z0-9.\[\]:-]+")


def asset_mime(name):
    if name in MODULES:
        return "text/javascript; charset=utf-8"
    if ASSET.fullmatch(name):
        return "font/ttf" if name.endswith(".ttf") else "application/octet-stream"
    return None


def installed(directory):
    try:
        with (directory / INSTALL_MARKER).open("rb") as record:
            if record.read(len(INSTALL_RECORD) + 1) != INSTALL_RECORD:
                return False
        return (all((directory / name).is_file() for name in MODULES)
                and all((directory / path).is_dir() for path in RESOURCE_PATHS))
    except OSError:
        return False


def asset(directory, name):
    mime = asset_mime(name)
    if not mime or not installed(directory):
        raise KeyError(name)
    file = directory / name
    try:
        resolved = file.resolve(strict=True)
        if file.is_symlink() or resolved.parent != directory.resolve() / Path(name).parent or not resolved.is_file():
            raise KeyError(name)
        return resolved.read_bytes(), mime
    except OSError as exc:
        raise KeyError(name) from exc


def worker_policy(origins):
    sources = " ".join(origin + "/pdf-reader/" + path
                       for origin in sorted(origins) if ORIGIN.fullmatch(origin)
                       for path in RESOURCE_PATHS) or "'none'"
    return f"default-src 'none'; connect-src {sources}; script-src 'none'; worker-src 'none'; base-uri 'none'"
