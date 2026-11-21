"""Select the checks for a push to main using its two endpoint trees."""
import subprocess
import sys


def companion_only(before, after):
    diff = subprocess.run(
        ["git", "diff", "--no-renames", "--ignore-submodules=none", "--name-only", "-z",
         before, after, "--"], stdout=subprocess.PIPE, check=False)
    if diff.returncode or not diff.stdout.endswith(b"\0"):
        return False
    paths = diff.stdout[:-1].split(b"\0")
    docs = {b"docs/STATUS.md", b"docs/TESTS.md"}
    return (any(path.startswith(b"tools/chat_ui/") for path in paths)
            and all(path.startswith(b"tools/chat_ui/") or path in docs for path in paths))


if __name__ == "__main__":
    print("companion=true" if companion_only(*sys.argv[1:]) else "companion=false")
