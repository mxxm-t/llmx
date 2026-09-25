"""Offline regression tests for pinned fixture download and cache behavior."""

import contextlib
from email.message import Message
from email.utils import formatdate
import hashlib
import http.client
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error

path = Path(__file__).resolve().parents[1] / "tools" / "fetch_test_models.py"
spec = importlib.util.spec_from_file_location("fetch_test_models", path)
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.home = Path(directory.name)
        self.payload = b"independent fixture payload\n"
        self.spec = {"repo": "test/model", "revision": "a" * 40,
                     "file": "fixture.gguf", "sha256": hashlib.sha256(self.payload).hexdigest()}
        self.destination = (self.home / ".cache/huggingface/hub/models--test--model/snapshots" /
                            self.spec["revision"] / "fixture.gguf")
        self.enterContext(mock.patch.object(fetcher.Path, "home", return_value=self.home))
        self.open = self.enterContext(mock.patch.object(fetcher.urllib.request, "urlopen"))
        self.sleep = self.enterContext(mock.patch.object(fetcher.time, "sleep"))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def error(self, code, **headers):
        fields = Message()
        for key, value in headers.items():
            fields[key.replace("_", "-")] = value
        return urllib.error.HTTPError("https://huggingface.co/test", code, "test", fields, io.BytesIO())

    def assert_only_destination(self):
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])

    def test_download_verifies_then_caches(self):
        self.open.return_value = io.BytesIO(self.payload)
        fetcher.fetch(self.spec)
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assert_only_destination()
        fetcher.fetch(self.spec)
        self.open.assert_called_once_with(
            "https://huggingface.co/test/model/resolve/" + "a" * 40 + "/fixture.gguf", timeout=60)
        self.sleep.assert_not_called()

    def test_corrupt_cache_is_replaced(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b"bad cache")
        self.open.return_value = io.BytesIO(self.payload)
        fetcher.fetch(self.spec)
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assert_only_destination()

    def test_429_respects_retry_after(self):
        error = self.error(429, Retry_After="90")
        self.open.side_effect = [error, io.BytesIO(self.payload)]
        fetcher.fetch(self.spec)
        self.sleep.assert_called_once_with(90)
        self.assertTrue(error.fp.closed)
        self.assert_only_destination()

    def test_429_respects_hf_reset_header(self):
        self.open.side_effect = [self.error(429, RateLimit='"resolvers";r=0;t=123'), io.BytesIO(self.payload)]
        fetcher.fetch(self.spec)
        self.sleep.assert_called_once_with(123)

    def test_retry_after_http_date(self):
        with mock.patch.object(fetcher.time, "time", return_value=1000):
            self.open.side_effect = [self.error(429, Retry_After=formatdate(1095, usegmt=True)), io.BytesIO(self.payload)]
            fetcher.fetch(self.spec)
        self.sleep.assert_called_once_with(95)

    def test_malformed_headers_use_backoff(self):
        self.open.side_effect = [self.error(429, Retry_After="invalid", RateLimit='"resolvers";t=no'),
                                 io.BytesIO(self.payload)]
        fetcher.fetch(self.spec)
        self.sleep.assert_called_once_with(30)

    def test_long_server_delay_fails_without_early_retry(self):
        self.open.side_effect = self.error(429, Retry_After="3600")
        with self.assertRaisesRegex(RuntimeError, "above 300"):
            fetcher.fetch(self.spec)
        self.open.assert_called_once()
        self.sleep.assert_not_called()
        self.assertEqual(list(self.destination.parent.iterdir()), [])

    def test_429_exhausts_bounded_attempts(self):
        self.open.side_effect = [self.error(429) for _ in range(5)]
        with self.assertRaises(urllib.error.HTTPError):
            fetcher.fetch(self.spec)
        self.assertEqual(self.open.call_count, 5)
        self.assertEqual(self.sleep.call_args_list, [mock.call(n) for n in (30, 60, 120, 240)])
        self.assertEqual(list(self.destination.parent.iterdir()), [])

    def test_transient_server_errors_retry(self):
        for code in (408, 500, 502, 503, 504):
            with self.subTest(code=code):
                self.open.side_effect = [self.error(code), io.BytesIO(self.payload)]
                fetcher.fetch(self.spec)
                self.assert_only_destination()
                self.destination.unlink()
        self.assertEqual(self.open.call_count, 10)
        self.assertEqual(self.sleep.call_count, 5)

    def test_permanent_http_errors_do_not_retry(self):
        for code in (401, 403, 404):
            with self.subTest(code=code):
                self.open.side_effect = self.error(code)
                with self.assertRaises(urllib.error.HTTPError):
                    fetcher.fetch(self.spec)
                self.assertEqual(list(self.destination.parent.iterdir()), [])
        self.assertEqual(self.open.call_count, 3)
        self.sleep.assert_not_called()

    def test_network_errors_retry(self):
        for error in (urllib.error.URLError("temporary"), TimeoutError(), ConnectionResetError()):
            with self.subTest(error=type(error).__name__):
                self.open.side_effect = [error, io.BytesIO(self.payload)]
                fetcher.fetch(self.spec)
                self.destination.unlink()
        self.assertEqual(self.open.call_count, 6)

    def test_interrupted_stream_restarts_with_clean_temporary_file(self):
        class Interrupted(io.BytesIO):
            def read(self, size):
                if self.tell():
                    raise http.client.IncompleteRead(b"partial")
                return super().read(5)
        self.open.side_effect = [Interrupted(self.payload), io.BytesIO(self.payload)]
        fetcher.fetch(self.spec)
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assert_only_destination()

    def test_fixed_length_early_eof_retries(self):
        payload = self.payload
        class Socket:
            def makefile(self, mode):
                return io.BytesIO(b"HTTP/1.1 200 OK\r\nContent-Length: " +
                                  str(len(payload)).encode("ascii") + b"\r\n\r\n" + payload[:5])
        response = http.client.HTTPResponse(Socket())
        response.begin()
        self.open.side_effect = [response, io.BytesIO(self.payload)]
        fetcher.fetch(self.spec)
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.sleep.assert_called_once_with(30)
        self.assertTrue(response.closed)
        self.assert_only_destination()

    def test_hash_mismatch_preserves_existing_file_and_fails(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b"existing invalid cache")
        self.open.return_value = io.BytesIO(b"incorrect response")
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            fetcher.fetch(self.spec)
        self.assertEqual(self.destination.read_bytes(), b"existing invalid cache")
        self.assert_only_destination()
        self.sleep.assert_not_called()

    def test_local_write_error_does_not_retry(self):
        with mock.patch.object(fetcher.tempfile, "NamedTemporaryFile", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                fetcher.fetch(self.spec)
        self.open.assert_not_called()
        self.sleep.assert_not_called()


class SelectionTests(unittest.TestCase):
    def run_main(self, argv):
        out = io.StringIO()
        with mock.patch.object(fetcher, "fetch") as fetch, contextlib.redirect_stdout(out):
            self.assertEqual(fetcher.main(argv), 0)
        return [call.args[0] for call in fetch.call_args_list], out.getvalue()

    def test_default_fetches_the_gate_and_all_every_pin(self):
        gate, _ = self.run_main([])
        everything, _ = self.run_main(["--all"])
        self.assertEqual(gate, [spec for spec in fetcher.PINNED if spec["gate"]])
        self.assertEqual(everything, fetcher.PINNED)
        self.assertTrue(gate and len(gate) < len(everything))

    def test_key_hashes_the_selected_pins_alone(self):
        fetched, out = self.run_main(["--key"])
        self.assertEqual(fetched, [])
        key = out.strip()
        self.assertRegex(key, r"^[0-9a-f]{64}$")
        self.assertEqual(key, fetcher.cache_key(fetcher.selected()))
        self.assertNotEqual(key, fetcher.cache_key(fetcher.selected(True)))
        # A model pinned ahead of the gate leaves the key as it is, and a new pin of a gate model changes it.
        later = dict(fetcher.PINNED[0], file="later.gguf", gate=False)
        with mock.patch.object(fetcher, "PINNED", fetcher.PINNED + [later]):
            self.assertEqual(fetcher.cache_key(fetcher.selected()), key)
        repinned = [dict(spec, sha256="0" * 64) if n == 0 else spec for n, spec in enumerate(fetcher.PINNED)]
        with mock.patch.object(fetcher, "PINNED", repinned):
            self.assertNotEqual(fetcher.cache_key(fetcher.selected()), key)


if __name__ == "__main__":
    unittest.main()
