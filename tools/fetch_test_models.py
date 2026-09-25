"""Download and verify the pinned HF fixtures using only the Python stdlib."""

import argparse
from email.utils import parsedate_to_datetime
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from baseline import PINNED, snapshot_path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retry_delay(headers, attempt):
    delay = 30 * 2 ** attempt
    if headers:
        value = headers.get("Retry-After", "").strip()
        if value.isascii() and value.isdigit():
            delay = max(delay, int(value))
        elif value:
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is not None:
                    delay = max(delay, math.ceil(date.timestamp() - time.time()))
            except (ValueError, TypeError, OverflowError):
                pass
        for value in re.findall(r'(?:^|[;,])\s*t\s*=\s*(\d+)\s*(?=$|[;,])',
                                headers.get("RateLimit", ""), flags=re.ASCII):
            delay = max(delay, int(value))
    if delay > 300:
        raise RuntimeError("HF requested a retry delay above 300 seconds; retry the job later")
    return delay


def download(url, destination, expected_sha256):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as f:
            temporary = Path(f.name)
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as response:
                expected_length = getattr(response, "length", None)
                received = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(block)
                    f.write(block)
                    received += len(block)
                # Bounded HTTP reads do not raise on early EOF with Content-Length.
                if expected_length is not None and received < expected_length:
                    raise http.client.IncompleteRead(b"", expected_length - received)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("SHA-256 mismatch for " + destination.name)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def fetch(spec):
    destination = snapshot_path(spec["repo"], spec["revision"], spec["file"])
    if destination.is_file() and sha256(destination) == spec["sha256"]:
        print("verified " + spec["file"], flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = "https://huggingface.co/%s/resolve/%s/%s" % (spec["repo"], spec["revision"], spec["file"])
    for attempt in range(5):
        print("downloading %s (attempt %d/5)" % (spec["file"], attempt + 1), flush=True)
        try:
            download(url, destination, spec["sha256"])
            print("verified " + spec["file"], flush=True)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead) as error:
            try:
                if isinstance(error, urllib.error.HTTPError):
                    if error.code not in (408, 429, 500, 502, 503, 504):
                        raise
                    reason = "HTTP %d" % error.code
                    headers = error.headers
                else:
                    reason = type(error).__name__
                    headers = None
                if attempt == 4:
                    raise
                delay = retry_delay(headers, attempt)
            finally:
                if isinstance(error, urllib.error.HTTPError):
                    error.close()
            print("%s downloading %s; retrying in %d seconds" %
                  (reason, spec["file"], delay), flush=True)
            time.sleep(delay)


def selected(everything=False):
    """The pinned entries of tests/data/fixtures.json a run downloads: the gate's models, or with `everything` every pinned model."""
    return [spec for spec in PINNED if everything or spec["gate"]]


def cache_key(specs):
    """A SHA-256 of `specs`, which changes when one of their downloads does and not when a check or a bound does."""
    return hashlib.sha256(json.dumps(specs, sort_keys=True).encode("utf-8")).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download the gate's models pinned in tests/data/fixtures.json into the HF cache, "
                                                 "each verified by its SHA-256 before it replaces anything there.")
    parser.add_argument("--all", action="store_true",
                        help="also download the models pinned ahead of the tensor types they hold, which the gate does not use yet")
    parser.add_argument("--key", action="store_true",
                        help="print a SHA-256 of the pins these arguments select and download nothing; the HF job's cache key")
    args = parser.parse_args(argv)
    specs = selected(args.all)
    if args.key:
        print(cache_key(specs))
        return 0
    for spec in specs:
        fetch(spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
