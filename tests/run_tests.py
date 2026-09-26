import os
import sys
import argparse

# llmx test runner.
# Each test generates its own fixtures and cleans up.
# Exit code 0 = all passed.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import roundtrip
import raw_blocks
import perf
import tokenizer
import perplexity
import f32
import moe
import split
import shards
import server
import server_load_tool
import chat
import threads
import version
import cli
import reference_generator
import reference_consumer
import baseline
import common


def main():
    # A message can hold characters the console's code page cannot encode, as a non-ASCII path does on Windows, and printing it must not stop the run, so they print as escapes.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Run llmx tests against the selected binary.")
    parser.add_argument("--exe", default=common.EXE, help="path to the built llmx executable")
    parser.add_argument("--no-perf-floor", action="store_true", help="report timings without workstation-specific floors")
    parser.add_argument("--require-baseline", action="store_true", help="fail if any gate model pinned in tests/data/fixtures.json is absent")
    parser.add_argument("--require-tools", action="store_true",
                        help="fail, rather than skip, when a tool a component runs is not beside --exe, or numpy, which raw-blocks checks the spec decoders' numpy form with, is not installed")
    parser.add_argument("--device", default=None, help="run every command that takes --device on this backend, e.g. vulkan:0")
    parser.add_argument("--layer-shares", default=None,
                        help="with several devices in --device, their proportions of the layers, e.g. 1,1")
    parser.add_argument("--cache-type", default=None, choices=["f32", "f16"],
                        help="store both KV cache sides as this type in every command that takes --cache-type-k/-v")
    parser.add_argument("--load-mode", default=None, choices=["auto", "mapped", "direct"],
                        help="read every model's weights this way in every command that takes --load-mode")
    parser.add_argument("--only", default=None, metavar="NAMES",
                        help="run only these components, comma separated, e.g. baseline or split,server")
    args = parser.parse_args()
    common.EXE = os.path.abspath(args.exe)
    common.exe_path()
    if args.device:
        os.environ["LLMX_DEVICE"] = args.device
    if args.cache_type:
        os.environ["LLMX_CACHE_TYPE"] = args.cache_type
    if args.layer_shares:
        os.environ["LLMX_LAYER_SHARES"] = args.layer_shares
    if args.load_mode:
        os.environ["LLMX_LOAD_MODE"] = args.load_mode
    if args.require_baseline:
        missing = [s["file"] for s in baseline.BASELINE_MODELS if not baseline.find_fixture(s)]
        if missing:
            parser.error("missing required HF fixtures: " + ", ".join(missing))
    components = [("version", version.run),
                  ("cli", cli.run),
                  ("reference-generator", reference_generator.run),
                  ("reference-consumer", reference_consumer.run),
                  ("roundtrip", roundtrip.run),
                  ("raw-blocks", lambda: raw_blocks.run(require=args.require_tools)),
                  ("perf", lambda: perf.run(enforce_floor=not args.no_perf_floor)),
                  ("tokenizer", tokenizer.run),
                  ("perplexity", perplexity.run),
                  ("f32", f32.run),
                  ("moe", moe.run),
                  ("split", lambda: split.run(require=args.require_tools)),
                  ("shards", shards.run),
                  ("server", server.run),
                  ("server-load", server_load_tool.run),
                  ("chat", chat.run),
                  ("threads", threads.run),
                  ("baseline", baseline.run)]
    # An empty name, from --only "" or a stray comma, is refused like any other name the suite does not have.
    if args.only is not None:
        only = args.only.split(",")
        unknown = [name for name in only if name not in dict(components)]
        if unknown:
            parser.error("unknown component %s; the components are %s"
                         % (", ".join(repr(name) for name in unknown), ", ".join(name for name, _ in components)))
        components = [(name, fn) for name, fn in components if name in only]
    results = []
    for name, fn in components:
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
