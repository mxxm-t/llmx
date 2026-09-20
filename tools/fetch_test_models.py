"""Download and verify the pinned HF fixtures using only the Python stdlib."""

from email.utils import parsedate_to_datetime
import hashlib
import http.client
import math
from pathlib import Path
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from baseline import BASELINE_MODELS


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
    destination = (Path.home() / ".cache" / "huggingface" / "hub" /
                   ("models--" + spec["repo"].replace("/", "--")) /
                   "snapshots" / spec["revision"] / spec["file"])
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


if __name__ == "__main__":
    for spec in BASELINE_MODELS:
        fetch(spec)
