"""Track explicitly requested installs of the companion's built-in features."""

from threading import Lock, Thread
from urllib.error import URLError

import install_pdf_reader
import pdf_reader


CATALOG = {
    "pdf-reader": {
        "name": "PDF text import",
        "description": "Choose pages with extractable PDF text to attach to a chat draft.",
        "version": pdf_reader.VERSION,
        "installed": lambda: pdf_reader.installed(pdf_reader.READER_DIRECTORY),
        "install": lambda: install_pdf_reader.install(pdf_reader.READER_DIRECTORY),
    },
}
INSTALL_ERROR = "Installation did not finish. Retry from Settings."


class OptionalFeatures:
    def __init__(self):
        self.lock = Lock()
        self.installing = set()
        self.failed = {}

    def _snapshot(self):
        features = []
        for key, feature in CATALOG.items():
            entry = dict(id=key, name=feature["name"], description=feature["description"], version=feature["version"])
            if feature["installed"]():
                entry["state"] = "installed"
            elif key in self.installing:
                entry["state"] = "installing"
            elif key in self.failed:
                entry.update(state="failed", error=self.failed[key])
            else:
                entry["state"] = "available"
            features.append(entry)
        return {"features": features}

    def snapshot(self):
        with self.lock:
            return self._snapshot()

    def _install(self, key):
        feature = CATALOG[key]
        error = None
        try:
            feature["install"]()
            if not feature["installed"]():
                error = INSTALL_ERROR
        except PermissionError:
            error = "The companion cannot write the feature files. Check its directory permissions and select Retry."
        except (URLError, TimeoutError):
            error = "The download failed. Check the companion's network connection and select Retry."
        except Exception:
            error = INSTALL_ERROR
        with self.lock:
            self.installing.remove(key)
            if error:
                self.failed[key] = error

    def start(self, key):
        feature = CATALOG[key]
        with self.lock:
            if key not in self.installing and not feature["installed"]():
                self.installing.add(key)
                self.failed.pop(key, None)
                try:
                    Thread(target=self._install, args=(key,), name=f"llmx-install-{key}", daemon=True).start()
                except RuntimeError:
                    self.installing.remove(key)
                    self.failed[key] = INSTALL_ERROR
            return self._snapshot()
