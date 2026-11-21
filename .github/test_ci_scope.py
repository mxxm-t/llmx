"""Exercise main-push classification against real Git trees."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("ci_scope.py").resolve()


class MainPushScope(unittest.TestCase):
    def test_push_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

            def write(name):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture\n")

            def commit():
                git("add", "-A")
                git("-c", "user.name=Fixture", "-c", "user.email=test@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
                return git("rev-parse", "HEAD")

            def check(start, end, companion):
                actual = subprocess.run([sys.executable, str(SCRIPT), start, end], cwd=root,
                                        capture_output=True, text=True, check=True).stdout.strip()
                self.assertEqual(actual, "companion=" + str(companion).lower(), (start, end))

            git("init", "-q")
            git("config", "core.autocrlf", "false")
            write("README.md")
            base = commit()
            write("docs/TESTS.md")
            docs = commit()
            check(base, docs, False)
            for number in range(305):
                write(f"tools/chat_ui/files/{number}")
            chat = commit()
            check(docs, chat, True)
            check(base, chat, True)
            check(chat, chat, False)
            check("0" * 40, chat, False)
            git("mv", "tools/chat_ui/files/0", "z-runtime.cpp")
            mixed = commit()
            check(chat, mixed, False)
            check(mixed, chat, False)
            check(base, mixed, False)
            write("tools/chat_ui_extra/file")
            write("tools/chat_ui/new")
            sibling = commit()
            check(mixed, sibling, False)


if __name__ == "__main__":
    unittest.main()
