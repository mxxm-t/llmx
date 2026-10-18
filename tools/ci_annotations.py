"""Annotate bounded CTest failure details and the long tests' times (stdlib only)."""

import argparse
from pathlib import Path
import re

# Tests whose hosted times set their limits, matched by name prefix.
LONG_TESTS = ("cache-scenario", "server-resume", "arch-qwen35-fit")
MAX_FAILED_BYTES = 64 * 1024
MAX_TIMING_BYTES = 64 * 1024
MAX_LOG_BYTES = 8 * 1024 * 1024
MAX_FAILURES = 8
MAX_EXCERPT_CHARS = 2200
MAX_TIMING_CHARS = 3500
TEST_HEADER = re.compile(r"^\s*\d+/\d+\s+Test:\s*(.*?)\s*$")


def escape_data(value):
    return str(value).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_property(value):
    return (escape_data(value).replace(":", "%3A").replace(",", "%2C"))


def annotation(title, message, level="error"):
    return "::%s title=%s::%s" % (level, escape_property(title), escape_data(message))


def failed_tests(path):
    if not path.is_file():
        return [], "LastTestsFailed.log is missing."
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_FAILED_BYTES + 1)
    except OSError as exc:
        return [], "Could not read LastTestsFailed.log (%s)." % type(exc).__name__
    truncated = len(raw) > MAX_FAILED_BYTES
    text = raw[:MAX_FAILED_BYTES].decode("utf-8", "replace")
    names = []
    malformed = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        index, separator, name = line.partition(":")
        if separator and index.strip().isdigit() and name.strip():
            names.append(name.strip()[:160])
        else:
            malformed += 1
    names = list(dict.fromkeys(names))
    notes = []
    if malformed:
        notes.append("ignored %d malformed failed-test row(s)" % malformed)
    if truncated:
        notes.append("LastTestsFailed.log exceeded %d bytes and was truncated" % MAX_FAILED_BYTES)
    if not names:
        notes.append("no valid failed-test names were found")
    return names, "; ".join(notes)


def failure_output(path, wanted):
    if not path.is_file():
        return {}, "LastTest.log is missing."
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_LOG_BYTES + 1)
    except OSError as exc:
        return {}, "Could not read LastTest.log (%s)." % type(exc).__name__
    truncated = len(raw) > MAX_LOG_BYTES
    blocks = {}
    current = None
    in_output = False
    captured = ""
    output_chars = 0
    for line in raw[:MAX_LOG_BYTES].decode("utf-8", "replace").splitlines():
        match = TEST_HEADER.match(line)
        if match:
            if current is not None:
                blocks[current] = (captured, output_chars > MAX_EXCERPT_CHARS)
            current = match.group(1) if match.group(1) in wanted else None
            in_output, captured, output_chars = False, "", 0
        elif current is not None and line == "Output:":
            in_output = True
        elif current is not None and line == "<end of output>":
            in_output = False
        elif current is not None and in_output:
            captured = (captured + line + "\n")[-MAX_EXCERPT_CHARS:]
            output_chars += len(line) + 1
    if current is not None:
        blocks[current] = (captured, output_chars > MAX_EXCERPT_CHARS)
    notes = []
    if truncated:
        notes.append("LastTest.log exceeded %d bytes; excerpts may be incomplete" % MAX_LOG_BYTES)
    if not blocks:
        notes.append("no matching CTest failure block was found")
    return blocks, "; ".join(notes)


def long_times(temporary):
    path = temporary / "CTestCostData.txt"
    if not path.is_file():
        return [], False
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_TIMING_BYTES + 1)
    except OSError:
        return [], False
    truncated = len(raw) > MAX_TIMING_BYTES
    times = []
    for line in raw[:MAX_TIMING_BYTES].decode("utf-8", "replace").splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0].startswith(LONG_TESTS):
            try:
                times.append("%s %.1f s" % (fields[0], float(fields[2])))
            except ValueError:
                pass
    return times, truncated


def diagnostics(temporary):
    lines = []
    times, truncated = long_times(temporary)
    if times or truncated:
        selected = []
        size = 0
        for item in times:
            addition = len(item) + (2 if selected else 0)
            if size + addition > MAX_TIMING_CHARS:
                break
            selected.append(item)
            size += addition
        omitted = len(times) - len(selected)
        message = ", ".join(selected) or "no matching timing entries found before input limit"
        if omitted:
            message += " (+%d timing entries omitted; annotation limit)" % omitted
        if truncated:
            message += " (CTestCostData.txt truncated after %d bytes)" % MAX_TIMING_BYTES
        lines.append(annotation("CTest times", message, "notice"))
    return lines


def failure_diagnostics(temporary, failed=None, failed_note=None):
    if failed is None:
        failed, failed_note = failed_tests(temporary / "LastTestsFailed.log")
    lines = []
    if failed:
        summary = ", ".join(failed[:MAX_FAILURES])
        if len(failed) > MAX_FAILURES:
            summary += " (+%d omitted)" % (len(failed) - MAX_FAILURES)
        lines.append(annotation("CTest failed (%d)" % len(failed), summary))
        log_path = temporary / "LastTest.log"
        blocks, log_note = failure_output(log_path, set(failed))
        for name in failed[:MAX_FAILURES]:
            block = blocks.get(name)
            if block and block[0]:
                excerpt, excerpt_truncated = block
                message = ("Output excerpt truncated; showing the final %d characters.\n" % MAX_EXCERPT_CHARS if excerpt_truncated else "") + excerpt
            else:
                message = "No captured output block for this failed test."
            lines.append(annotation("CTest failure: " + name, message))
        if len(failed) > MAX_FAILURES:
            lines.append(annotation("CTest failures", "%d additional failed tests omitted." % (len(failed) - MAX_FAILURES), "warning"))
        if log_note:
            lines.append(annotation("CTest log note", log_note, "warning"))
    if failed_note:
        lines.append(annotation("CTest failed-test list note", failed_note, "warning"))
    if not failed:
        message = failed_note or "CTest failed, but LastTestsFailed.log contained no failed tests."
        lines.append(annotation("CTest failure list", message, "error"))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", help="the build directory CTest ran in")
    parser.add_argument("--outcome", choices=("success", "failure", "unknown"), default="unknown",
                        help="the CTest step outcome; missing failure files are reported only on failure")
    args = parser.parse_args(argv)
    temporary = Path(args.build_dir) / "Testing" / "Temporary"
    lines = diagnostics(temporary)
    failed, failed_note = failed_tests(temporary / "LastTestsFailed.log")
    if args.outcome == "failure" or (args.outcome == "unknown" and failed):
        lines.extend(failure_diagnostics(temporary, failed, failed_note))
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
