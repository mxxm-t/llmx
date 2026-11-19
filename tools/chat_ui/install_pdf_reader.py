"""Install the pinned optional PDF.js text reader for the companion."""

import argparse
import hashlib
import io
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from pdf_reader import (ARCHIVE_SHA256, INSTALL_MARKER, INSTALL_RECORD, MODULES,
                        READER_DIRECTORY, RESOURCE_PATHS, VERSION, asset_mime, installed)


ARCHIVE_URL = f"https://github.com/mozilla/pdf.js/releases/download/v{VERSION}/pdfjs-{VERSION}-dist.zip"
ARCHIVE_BYTES = 6540268
LICENSES = ("LICENSE", "cmaps/LICENSE", "standard_fonts/LICENSE_FOXIT", "standard_fonts/LICENSE_LIBERATION")


def read_archive(local_archive):
    if local_archive is None:
        request = Request(ARCHIVE_URL, headers={"User-Agent": "llmx-pdf-reader-install"})
        source = urlopen(request, timeout=30)
    else:
        source = local_archive.open("rb")
    with source:
        data = source.read(ARCHIVE_BYTES + 1)
    if len(data) != ARCHIVE_BYTES or hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("PDF.js archive does not match the pinned size and SHA-256")
    return data


def unpack_archive(data, destination):
    seen = set()
    with ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            original = info.filename
            if original in ("build/" + name for name in MODULES):
                name = original.removeprefix("build/")
            elif original == "LICENSE" or original.startswith(tuple("web/" + path for path in RESOURCE_PATHS)):
                name = original.removeprefix("web/")
                if name not in LICENSES and not asset_mime(name):
                    continue
            else:
                continue
            if name in seen or info.is_dir():
                raise ValueError("Unexpected duplicate PDF.js archive entry: " + original)
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
            seen.add(name)
    if not seen.issuperset((*MODULES, *LICENSES)) or any(not any(name.startswith(path) and asset_mime(name) for name in seen)
                                                       for path in RESOURCE_PATHS):
        raise ValueError("PDF.js archive is missing required text reader assets")
    (destination / INSTALL_MARKER).write_bytes(INSTALL_RECORD)


def install(directory, local_archive=None):
    if installed(directory):
        return False
    if directory.exists():
        raise ValueError(f"Incomplete PDF.js installation at {directory}. Move it aside and retry.")
    data = read_archive(local_archive)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=directory.parent) as staging:
        staged = Path(staging) / "release"
        staged.mkdir()
        unpack_archive(data, staged)
        try:
            staged.rename(directory)
        except OSError:
            if installed(directory):
                return False
            raise
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local copy of the pinned official PDF.js ZIP")
    args = parser.parse_args()
    try:
        added = install(READER_DIRECTORY, args.archive)
    except (OSError, BadZipFile, ValueError) as exc:
        parser.exit(1, f"PDF reader installation failed: {exc}\n")
    action = "Installed" if added else "Already installed"
    print(f"{action} PDF.js {VERSION} at {READER_DIRECTORY}")
    print(f"Official archive SHA-256: {ARCHIVE_SHA256}")


if __name__ == "__main__":
    main()
