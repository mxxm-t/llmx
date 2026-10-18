import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ci_annotations


class CTestAnnotationsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name) / "Testing" / "Temporary"
        self.directory.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_failed_test_list_is_reported(self):
        (self.directory / "CTestCostData.txt").write_text("cache-scenario 1 2.5\n", encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ci_annotations.main([str(self.directory.parent.parent), "--outcome", "success"])
        self.assertIn("CTest times", output.getvalue())
        self.assertNotIn("CTest failed", output.getvalue())
        self.assertNotIn("warning", output.getvalue())
        with contextlib.redirect_stdout(output := io.StringIO()):
            ci_annotations.main([str(self.directory.parent.parent), "--outcome", "failure"])
        self.assertIn("LastTestsFailed.log is missing", output.getvalue())

    def test_malformed_failed_test_list_is_reported(self):
        (self.directory / "LastTestsFailed.log").write_text("not-a-ctest-row\n", encoding="utf-8")
        lines = ci_annotations.failure_diagnostics(self.directory)
        self.assertTrue(any("no valid failed-test names" in line for line in lines))

    def test_multiple_failure_blocks_and_existing_time_notice(self):
        (self.directory / "LastTestsFailed.log").write_text("2:loader\n3:server\n", encoding="utf-8")
        (self.directory / "LastTest.log").write_text(
            "1/3 Test: pass\nOutput:\nignored pass output\n<end of output>\nTest Passed.\n"
            "2/3 Test: loader\nCommand: secret-looking command\nDirectory: private path\n"
            "Output:\nloader assertion failed\n<end of output>\nTest Failed.\n"
            "3/3 Test: server\nOutput:\nserver timed out\n<end of output>\nTest Failed.\n",
            encoding="utf-8",
        )
        (self.directory / "CTestCostData.txt").write_text("cache-scenario 1 2.5\n", encoding="utf-8")
        lines = ci_annotations.diagnostics(self.directory) + ci_annotations.failure_diagnostics(self.directory)
        joined = "\n".join(lines)
        self.assertIn("CTest failed (2)", joined)
        self.assertIn("loader assertion failed", joined)
        self.assertIn("server timed out", joined)
        self.assertNotIn("ignored pass output", joined)
        self.assertNotIn("secret-looking command", joined)
        self.assertNotIn("private path", joined)
        self.assertIn("CTest times", joined)
        self.assertIn("cache-scenario 2.5 s", joined)

    def test_all_matching_times_are_kept_until_bounded(self):
        names = ("cache-scenario", "server-resume", "arch-qwen35-fit",
                 "server-resume-cache", "arch-qwen35-fit-memory")
        (self.directory / "CTestCostData.txt").write_text(
            "".join("%s 1 2.5\n" % name for name in names), encoding="utf-8")
        lines = ci_annotations.diagnostics(self.directory)
        self.assertTrue(all(name in lines[0] for name in names))
        with mock.patch.object(ci_annotations, "MAX_TIMING_CHARS", 45):
            lines = ci_annotations.diagnostics(self.directory)
        self.assertIn("timing entries omitted", lines[0])
        self.assertLessEqual(len(lines[0]), ci_annotations.MAX_TIMING_CHARS + 100)
        with mock.patch.object(ci_annotations, "MAX_TIMING_BYTES", 10):
            lines = ci_annotations.diagnostics(self.directory)
        self.assertIn("CTestCostData.txt truncated after 10 bytes", lines[0])

    def test_missing_last_test_log_and_escaping_are_diagnostic(self):
        (self.directory / "LastTestsFailed.log").write_text("1:bad:test,name\n", encoding="utf-8")
        lines = ci_annotations.failure_diagnostics(self.directory)
        joined = "\n".join(lines)
        self.assertIn("CTest failure%3A bad%3Atest%2Cname", joined)
        self.assertIn("LastTest.log is missing", joined)
        self.assertEqual(ci_annotations.escape_data("100%\r\nnext"), "100%25%0D%0Anext")

    def test_truncated_log_keeps_a_bounded_excerpt_and_note(self):
        (self.directory / "LastTestsFailed.log").write_text("1:slow\n", encoding="utf-8")
        (self.directory / "LastTest.log").write_text(
            "1/1 Test: slow\nOutput:\nfirst useful line\n" + ("x" * 300), encoding="utf-8"
        )
        with mock.patch.object(ci_annotations, "MAX_LOG_BYTES", 70):
            lines = ci_annotations.failure_diagnostics(self.directory)
        joined = "\n".join(lines)
        self.assertIn("first useful line", joined)
        self.assertIn("LastTest.log exceeded 70 bytes", joined)
        self.assertLessEqual(len(joined), 3 * 2400 + 1000)

    def test_excerpt_keeps_final_error_after_benign_prefix(self):
        (self.directory / "LastTestsFailed.log").write_text("1:slow\n", encoding="utf-8")
        output = "ok\n" * 900 + "FATAL: final assertion failed\n"
        (self.directory / "LastTest.log").write_text(
            "1/1 Test: slow\nOutput:\n" + output + "<end of output>\nTest Failed.\n", encoding="utf-8"
        )
        with mock.patch.object(ci_annotations, "MAX_EXCERPT_CHARS", 100):
            lines = ci_annotations.failure_diagnostics(self.directory)
        joined = "\n".join(lines)
        self.assertIn("Output excerpt truncated; showing the final 100 characters.", joined)
        self.assertIn("FATAL: final assertion failed", joined)


if __name__ == "__main__":
    unittest.main()
