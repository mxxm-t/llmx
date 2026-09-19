import os
import sys
import argparse

# llmx test runner. Each test generates its own fixtures and cleans up.
# Exit code 0 = all passed.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import roundtrip
import perf
import tokenizer
import perplexity
import f32
import chat
import threads
import version
import reference_generator
import reference_consumer
import baseline
import common


def main():
    parser = argparse.ArgumentParser(description="Run llmx tests against the selected binary.")
    parser.add_argument("--exe", default=common.EXE, help="path to the built llmx executable")
    parser.add_argument("--no-perf-floor", action="store_true", help="report timings without workstation-specific floors")
    parser.add_argument("--require-baseline", action="store_true", help="fail if either real-model fixture is absent")
    args = parser.parse_args()
    common.EXE = os.path.abspath(args.exe)
    common.exe_path()
    if args.require_baseline:
        missing = [s["file"] for s in baseline.BASELINE_MODELS if not baseline.find_fixture(s)]
        if missing:
            parser.error("missing required HF fixtures: " + ", ".join(missing))
    results = []
    for name, fn in [("version", version.run),
                     ("reference-generator", reference_generator.run),
                     ("reference-consumer", reference_consumer.run),
                     ("roundtrip", roundtrip.run),
                     ("perf", lambda: perf.run(enforce_floor=not args.no_perf_floor)),
                     ("tokenizer", tokenizer.run),
                     ("perplexity", perplexity.run),
                     ("f32", f32.run),
                     ("chat", chat.run),
                     ("threads", threads.run),
                     ("baseline", baseline.run)]:
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as e:
            results.append((name, False))
            print("  error:", e)

    print()
    failed = [n for n, ok in results if not ok]
    for n, ok in results:
        print("  %-10s %s" % (n, "PASS" if ok else "FAIL"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
