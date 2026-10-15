"""Print a CTest run's failed tests and the times of the long ones as GitHub workflow annotations (stdlib only)."""

import argparse
from pathlib import Path

# Tests whose hosted times set their limits, matched by name prefix.
LONG_TESTS = ("cache-scenario", "server-resume", "arch-qwen35-fit")


def failed_tests(temporary):
    path = temporary / "LastTestsFailed.log"
    if not path.is_file():
        return []
    names = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        _, _, name = line.partition(":")
        if name.strip():
            names.append(name.strip())
    return names


def long_times(temporary):
    path = temporary / "CTestCostData.txt"
    if not path.is_file():
        return []
    times = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0].startswith(LONG_TESTS):
            try:
                times.append("%s %.1f s" % (fields[0], float(fields[2])))
            except ValueError:
                pass
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", help="the build directory CTest ran in")
    args = parser.parse_args()
    temporary = Path(args.build_dir) / "Testing" / "Temporary"
    failed = failed_tests(temporary)
    if failed:
        print("::error title=CTest failed (%d)::%s" % (len(failed), ", ".join(failed)))
    times = long_times(temporary)
    if times:
        print("::notice title=CTest times::%s" % ", ".join(times))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
