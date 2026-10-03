import os
import sys
import argparse

# llmx test runner.
# Each test generates its own fixtures and cleans up.
# Exit code 0 = no component failed; one that compared nothing reports SKIP.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dead_code
import docs_check
import arch_boundary
import roundtrip
import raw_blocks
import perf
import tokenizer
import perplexity
import f32
import moe
import mxfp4
import iq4_nl
import qwen35
import split
import decode_probe
import shards
import server
import server_load_tool
import chat
import threads
import version
import cli
import reference_generator
import reference_consumer
import device_reference
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
    parser.add_argument("--require-device-types", default=None, metavar="NAMES",
                        help="fail instead of skipping unsupported weight types named here, comma separated, e.g. Q8_0,Q4_K; does not require a component to exercise them")
    parser.add_argument("--device", default=None, help="run every command that takes --device on this backend, e.g. vulkan:0")
    parser.add_argument("--dtype", choices=["auto", "f32", "f16", "bf16"], help="request this activation dtype on every model command")
    parser.add_argument("--layer-shares", default=None,
                        help="with several devices in --device, their proportions of the layers, e.g. 1,1")
    parser.add_argument("--cache-type", default=None, choices=["f32", "f16"],
                        help="store both KV cache sides as this type in every command that takes --cache-type-k/-v")
    parser.add_argument("--load-mode", default=None, choices=["auto", "mapped", "direct"],
                        help="read every model's weights this way in every command that takes --load-mode")
    parser.add_argument("--only", default=None, metavar="NAMES",
                        help="run only these components, comma separated, e.g. baseline or split,server")
    args = parser.parse_args()
    common.REQUIRED_DEVICE_TYPES = {}
    if args.require_device_types is not None:
        from spec_decode import TYPES
        names = {spec[0]: type_id for type_id, spec in TYPES.items()}
        required = args.require_device_types.split(",")
        unknown = [name for name in required if name not in names]
        if unknown:
            parser.error("unknown required device type %s; the types are %s"
                         % (", ".join(repr(name) for name in unknown), ", ".join(names)))
        common.REQUIRED_DEVICE_TYPES = {names[name]: name for name in required}
    common.EXE = os.path.abspath(args.exe)
    common.exe_path()
    if args.device:
        os.environ["LLMX_DEVICE"] = args.device
    if args.dtype:
        os.environ["LLMX_DTYPE"] = args.dtype
    if args.cache_type:
        os.environ["LLMX_CACHE_TYPE"] = args.cache_type
    if args.layer_shares:
        os.environ["LLMX_LAYER_SHARES"] = args.layer_shares
    if args.load_mode:
        os.environ["LLMX_LOAD_MODE"] = args.load_mode
    if args.require_baseline:
        missing = baseline.missing_gate_models()
        if missing:
            parser.error("missing required HF fixtures: " + ", ".join(missing))
    components = [("dead-code", dead_code.run),
                  ("docs", docs_check.run),
                  ("arch-boundary", arch_boundary.run),
                  ("version", version.run),
                  ("cli", cli.run),
                  ("reference-generator", reference_generator.run),
                  ("reference-consumer", reference_consumer.run),
                  ("device-reference", lambda: device_reference.run(require=args.require_tools)),
                  ("roundtrip", roundtrip.run),
                  ("raw-blocks", lambda: raw_blocks.run(require=args.require_tools)),
                  ("perf", lambda: perf.run(enforce_floor=not args.no_perf_floor)),
                  ("tokenizer", tokenizer.run),
                  ("perplexity", perplexity.run),
                  ("f32", f32.run),
                  ("moe", moe.run),
                  ("mxfp4", lambda: mxfp4.run(require=args.require_tools)),
                  ("iq4-nl", iq4_nl.run),
                  ("qwen35", lambda: qwen35.run(require=args.require_tools)),
                  ("split", lambda: split.run(require=args.require_tools)),
                  ("decode-probe", lambda: decode_probe.run(require=args.require_tools)),
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
        print("  %-10s %s" % (n, "SKIP" if ok == common.SKIPPED else "PASS" if ok else "FAIL"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
