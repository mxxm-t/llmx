import os
import sys

# llmx test runner. Each test generates its own fixtures and cleans up.
# Exit code 0 = all passed.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import roundtrip
import perf
import tokenizer
import baseline


def main():
    results = []
    for name, fn in [("roundtrip", roundtrip.run),
                     ("perf", perf.run),
                     ("tokenizer", tokenizer.run),
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
