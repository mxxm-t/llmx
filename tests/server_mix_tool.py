"""Offline checks of the many-user gate's phase isolation and refusals."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import server_mix_check as tool


class ServerMixTool(unittest.TestCase):
    def exercise(self, flags=(), state=None, fail_post=False, fail_leaver=False, change_logprob=False):
        starts, stops, visits = [], [], []
        quiet = dict(active=0, queued=0, paused=0, prefix_hits=0, prefix_tokens=0, pauses=0)
        quiet.update(state or {})
        reqs = [dict(prompt="p%d" % i, max_tokens=2, temperature=0) for i in range(4)]
        if "--logprobs" in flags:
            for request in reqs:
                request.update(logprobs=True, top_logprobs=5)

        def start(command, wait):
            port = 8000 + len(starts)
            starts.append(command)
            return port, port, port

        def post(port, body):
            visits.append((port, body["prompt"]))
            if fail_post:
                raise RuntimeError("request failed")
            reply = dict(ids=[int(body["prompt"][1:])], text=body["prompt"])
            if body.get("logprobs"):
                reply.update(logprobs=[-0.5], top_logprobs=[{body["prompt"]: -0.5}])
            return reply

        def together(port, bodies, delays=None, leavers=()):
            results = {}
            for i, body in enumerate(bodies):
                if i in leavers:
                    results[i] = "disconnect failed" if fail_leaver else None
                else:
                    reply = post(port, body)
                    if change_logprob and "logprobs" in reply:
                        reply["logprobs"][0] -= 0.1
                    results[i] = tool.answer(reply)
            return results

        with tempfile.TemporaryDirectory(prefix="llmx_mix_tool_") as directory:
            text, ids = Path(directory) / "text.txt", Path(directory) / "ids.json"
            text.write_text("a corpus", encoding="utf-8")
            argv = ["server_mix_check.py", "--model", "model.gguf", "--text", str(text),
                    "--exe", "llmx", "--requests", "4", "--cli", "0", "--ids", str(ids)] + list(flags)
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", argv))
                stack.enter_context(patch.object(tool.common, "EXE", tool.common.EXE))
                stack.enter_context(patch.object(tool.common, "start_server", side_effect=start))
                stack.enter_context(patch.object(tool.common, "stop_server", side_effect=lambda proc, log: stops.append(proc)))
                make_requests = stack.enter_context(patch.object(tool, "requests_from", return_value=reqs))
                stack.enter_context(patch.object(tool, "post", side_effect=post))
                stack.enter_context(patch.object(tool, "run_together", side_effect=together))
                stack.enter_context(patch.object(tool, "health", return_value=quiet))
                stack.enter_context(patch.object(tool.time, "monotonic", side_effect=range(0, 10000, 121)))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                if fail_post:
                    with self.assertRaisesRegex(RuntimeError, "request failed"):
                        tool.main()
                    result = None
                else:
                    result = tool.main()
                self.assertEqual(make_requests.call_args.args[3:], ("--logprobs" in flags, "--sampled" in flags))
            return result, starts, stops, visits, json.loads(ids.read_text()) if ids.exists() else None

    def test_phase_lifetimes_and_results(self):
        ordinary = self.exercise()
        fresh = self.exercise(["--fresh-phases", "--ctx-size", "8192"])
        self.assertEqual(ordinary[0], 0)
        self.assertEqual(fresh[0], 0)
        self.assertEqual(len(ordinary[1]), 1)
        self.assertEqual(ordinary[2], [8000])
        self.assertEqual(len(fresh[1]), 4)
        self.assertEqual(fresh[2], [8000, 8001, 8002, 8003])
        self.assertEqual([port for port, _ in fresh[3]], [8000] * 4 + [8001] * 4 + [8002] * 3 + [8003])
        self.assertEqual(ordinary[4], fresh[4])
        self.assertEqual(fresh[4], dict(alone=[[0], [1], [2], [3]], together=[[0], [1], [2], [3]],
                                       skewed=[[0], [1], [2], None]))
        for command in fresh[1]:
            self.assertEqual(command[command.index("--ctx-size") + 1], "8192")

    def test_fresh_phase_refuses_reuse_and_pauses(self):
        for counter in ("prefix_hits", "prefix_tokens", "pauses"):
            with self.subTest(counter=counter):
                self.assertEqual(self.exercise(["--fresh-phases"], {counter: 1})[0], 1)
                self.assertEqual(self.exercise(state={counter: 1})[0], 0)

    def test_unfinished_requests_fail_both_modes(self):
        for counter in ("active", "queued", "paused"):
            for flags in ([], ["--fresh-phases"]):
                with self.subTest(counter=counter, flags=flags):
                    self.assertEqual(self.exercise(flags, {counter: 1})[0], 1)

    def test_request_failure_stops_its_server(self):
        for flags in ([], ["--fresh-phases"]):
            result = self.exercise(flags, fail_post=True)
            self.assertEqual(len(result[1]), 1)
            self.assertEqual(result[2], [8000])

    def test_logprobs_and_passes_with_both_lifetimes(self):
        for fresh in ([], ["--fresh-phases"]):
            with self.subTest(fresh=fresh):
                flags = fresh + ["--logprobs", "--passes", "3", "--ctx-size", "8192"]
                result = self.exercise(flags)
                self.assertEqual(result[0], 0)
                self.assertEqual(result[4]["alone"][0], [[0], [-0.5], [{"p0": -0.5}]])
                self.assertEqual(result[4]["alone"], result[4]["together"])
                for command in result[1]:
                    self.assertEqual(command[command.index("--passes") + 1], "3")
                    self.assertEqual(command[command.index("--ctx-size") + 1], "8192")
                self.assertEqual(self.exercise(flags, change_logprob=True)[0], 1)

    def test_failed_disconnect_fails_both_lifetimes(self):
        for flags in ([], ["--fresh-phases"]):
            with self.subTest(flags=flags):
                self.assertEqual(self.exercise(flags, fail_leaver=True)[0], 1)

    def test_sampled_with_both_lifetimes(self):
        for flags in (["--sampled"], ["--sampled", "--fresh-phases", "--logprobs"]):
            with self.subTest(flags=flags):
                result = self.exercise(flags)
                self.assertEqual(result[0], 0)
                self.assertEqual(len(result[1]), 4 if "--fresh-phases" in flags else 1)
                self.assertEqual(len(result[2]), len(result[1]))
                self.assertEqual(result[4]["alone"], result[4]["together"])

    def test_cache_type_matches_server_and_cli(self):
        for cache in (None, "f16", "f32"):
            for fresh in ([], ["--fresh-phases"]):
                with self.subTest(cache=cache, fresh=fresh):
                    flags = fresh + ["--cli", "1"] + (["--cache-type", cache] if cache else [])
                    with patch.object(tool.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=b"reply")) as cli:
                        with patch.object(tool.common, "generate_text", return_value=b"p0"):
                            result = self.exercise(flags)
                    self.assertEqual(result[0], 0)
                    cli.assert_called_once()
                    for command in result[1] + [cli.call_args.args[0]]:
                        for name in ("--cache-type-k", "--cache-type-v"):
                            self.assertEqual(command.count(name), 1 if cache else 0)
                            if cache:
                                self.assertEqual(command[command.index(name) + 1], cache)

    def test_incompatible_modes_fail_before_start(self):
        for flag in ("--fresh-phases", "--sampled"):
            argv = ["server_mix_check.py", "--model", "absent.gguf", "--text", "absent.txt", flag, "--uncapped"]
            with self.subTest(flag=flag), patch.object(sys, "argv", argv), patch.object(tool.common, "start_server") as start:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    tool.main()
                self.assertEqual(raised.exception.code, 2)
                start.assert_not_called()


def run():
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(ServerMixTool))
    return result.wasSuccessful()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
