import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from zipfile import ZipFile

import install_pdf_reader
import pdf_reader


class PDFReaderInstallTests(unittest.TestCase):
    def archive(self, path, exclude=None):
        entries = {
            "build/pdf.mjs": b"module",
            "build/pdf.worker.mjs": b"worker",
            "LICENSE": b"package license",
            "web/cmaps/Foo-V.bcmap": b"cmap",
            "web/cmaps/LICENSE": b"cmap license",
            "web/standard_fonts/Font.ttf": b"font",
            "web/standard_fonts/LICENSE_FOXIT": b"font license",
            "web/standard_fonts/LICENSE_LIBERATION": b"font license",
            "web/wasm/ignored.wasm": b"excluded",
            "web/viewer.html": b"excluded",
            "web/cmaps/../outside.bcmap": b"excluded",
        }
        with ZipFile(path, "w") as archive:
            for name, data in entries.items():
                if name != exclude:
                    archive.writestr(name, data)
        data = path.read_bytes()
        return dict(ARCHIVE_BYTES=len(data), ARCHIVE_SHA256=hashlib.sha256(data).hexdigest())

    def test_pdf_reader_verifies_and_installs_only_allowed_assets_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            archive, target = parent / "source.zip", parent / "reader"
            pin = self.archive(archive)
            with mock.patch.multiple(install_pdf_reader, **pin), mock.patch.object(install_pdf_reader, "urlopen") as fetch:
                self.assertTrue(install_pdf_reader.install(target, archive))
                self.assertFalse(install_pdf_reader.install(target))
                fetch.assert_not_called()
            self.assertTrue(pdf_reader.installed(target))
            files = {file.relative_to(target).as_posix() for file in target.rglob("*") if file.is_file()}
            self.assertEqual(files, {"pdf.mjs", "pdf.worker.mjs", "INSTALLATION", "cmaps/Foo-V.bcmap",
                                     "standard_fonts/Font.ttf", *install_pdf_reader.LICENSES})
            self.assertEqual(pdf_reader.asset(target, "cmaps/Foo-V.bcmap"), (b"cmap", "application/octet-stream"))
            for name in ("LICENSE", "../source.zip", "cmaps/../pdf.mjs", "cmaps/Foo-V.bcmap/", "web/viewer.html"):
                with self.subTest(name=name), self.assertRaises(KeyError):
                    pdf_reader.asset(target, name)

    def test_pdf_reader_rejects_bad_hash_and_incomplete_archive_without_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            archive, target = parent / "source.zip", parent / "reader"
            pin = self.archive(archive, exclude="build/pdf.worker.mjs")
            with mock.patch.object(install_pdf_reader, "ARCHIVE_BYTES", pin["ARCHIVE_BYTES"]):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    install_pdf_reader.install(target, archive)
            self.assertFalse(target.exists())
            with mock.patch.multiple(install_pdf_reader, **pin):
                with self.assertRaisesRegex(ValueError, "missing required"):
                    install_pdf_reader.install(target, archive)
            self.assertEqual(list(parent.iterdir()), [archive])
