import io
import json
import os
import tarfile
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from workspace import (MAX_FILE, MAX_FILES, MAX_OUTPUT, MAX_PROJECT_DEFINITIONS, MAX_PROJECT_INSTRUCTIONS,
                       DockerRunner, ProjectConflict, Workspace, WorkspaceTools, checked_files, file_path)


def archive(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as tar:
        for name, content, kind in entries:
            item = tarfile.TarInfo(name)
            if kind == "symlink":
                item.type = tarfile.SYMTYPE
                item.linkname = content
                tar.addfile(item)
            else:
                data = content if isinstance(content, bytes) else content.encode("utf-8")
                item.size = len(data)
                tar.addfile(item, io.BytesIO(data))
    return output.getvalue()


class FakeRunner:
    def __init__(self, block=False, updated=None, output="ready"):
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.block = block
        self.updated = updated or {}
        self.output = output
        self.calls = []

    def run(self, files, command, emit, cancel):
        self.calls.append((dict(files), command))
        self.started.set()
        emit(self.output)
        if self.block:
            self.release.wait(3)
        self.finished.set()
        if cancel.is_set():
            return None, -1
        return dict(self.updated), 0


class FakeProcess:
    def __init__(self, output=b"", returncode=0):
        self.stdout = io.BytesIO(output)
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


class FakeRemote:
    last_error = None

    def __init__(self):
        self.calls = []

    def list_tools(self, discover=True):
        return [{"name": name, "description": "Remote echo",
                 "inputSchema": {"type": "object"}, "server": "remote", "approval": True}
                for name in ("remote__echo", "workspace__remote_echo")]

    def call_tool(self, name, arguments, cancel_event=None):
        self.calls.append((name, arguments))
        return {"content": [{"type": "text", "text": "remote"}], "isError": False}


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def test_import_batch_is_create_only_and_atomic(self):
        workspace = Workspace(self.temp.name)
        entry = {"path": "src/empty.txt", "content": ""}
        existing = {"path": "saved.txt", "content": "original"}
        baseline = workspace.write(existing["path"], existing["content"], 0)
        for entries in ([entry, existing], [entry, entry], [entry, {"path": "src", "content": "x"}],
                        [entry, {"path": "bad.bin", "content": "\0"}], [], [entry, {}]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                workspace.import_files(entries, 1)
            self.assertEqual(workspace.snapshot(), baseline)
        with self.assertRaises(ValueError):
            workspace.import_files([entry], 0)
        with patch.object(workspace, "_save", side_effect=OSError("disk full")), self.assertRaises(OSError):
            workspace.import_files([entry], 1)
        self.assertEqual(workspace.snapshot(), baseline)
        self.assertEqual(Workspace(self.temp.name).snapshot(), baseline)

    def test_checkpoint_diff_keeps_missing_newlines_and_empty_files(self):
        workspace = Workspace(self.temp.name)

        def write(path, content, delete=False):
            workspace.write(path, content, workspace.files()["revision"], delete)

        write("a.txt", "before")
        write("b.txt", "")
        write("line.txt", "one\r\ntwo\n")
        checkpoint = workspace.checkpoint("before")["checkpoints"][-1]["id"]
        self.assertEqual(workspace.diff(checkpoint)["diff"], "")
        write("a.txt", "after")
        write("b.txt", "", delete=True)
        write("c.txt", "")
        write("line.txt", "one\r\ntwo")
        self.assertEqual(workspace.diff(checkpoint)["diff"], (
            "--- checkpoint/a.txt\n+++ workspace/a.txt\n@@ -1 +1 @@\n"
            "-before\n\\ No newline at end of file\n+after\n\\ No newline at end of file\n"
            "Deleted empty file: b.txt\nAdded empty file: c.txt\n"
            "--- checkpoint/line.txt\n+++ workspace/line.txt\n@@ -1,2 +1,2 @@\n"
            " one\r\n-two\n+two\n\\ No newline at end of file\n"
        ))

    def wait_job(self, workspace, job_id, timeout=2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = workspace.job(job_id)
            if job["status"] != "running":
                return job
            threading.Event().wait(0.005)
        self.fail("workspace job did not finish")

    def test_paths_and_file_bounds(self):
        self.assertEqual(file_path("src/main.py"), "src/main.py")
        for path in ("", "/absolute", "../escape", "a/../b", "a//b", "C:/file", "a\\b", "./dot"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    file_path(path)
        self.assertEqual(checked_files({"src/a.py": "x"}), {"src/a.py": "x"})
        with self.assertRaisesRegex(ValueError, "1 MiB"):
            checked_files({"large": "x" * (MAX_FILE + 1)})
        with self.assertRaisesRegex(ValueError, "256 text files"):
            checked_files({"f%d" % i: "" for i in range(MAX_FILES + 1)})
        with self.assertRaisesRegex(ValueError, "also be a directory"):
            checked_files({"src": "file", "src/main.py": "child"})
        with self.assertRaises(ValueError):
            checked_files({"nul.txt": "bad\0text"})

    def test_total_project_size_is_bounded(self):
        file_count = 8
        each = MAX_FILE
        with self.assertRaisesRegex(ValueError, "8 MiB"):
            checked_files({"f%d" % i: "x" * each for i in range(file_count + 1)})

    def test_snapshot_checkpoint_diff_restore_and_revision_conflict(self):
        workspace = Workspace(self.temp.name, runner=FakeRunner())
        first = workspace.write("src/main.py", "old\n", 0)
        point = workspace.checkpoint("baseline")
        checkpoint_id = point["checkpoints"][-1]["id"]
        changed = workspace.write("src/main.py", "new\n", first["revision"])
        workspace.write("README.md", "note\n", changed["revision"])
        self.assertIn("-old", workspace.diff(checkpoint_id)["diff"])
        self.assertIn("+new", workspace.diff(checkpoint_id)["diff"])
        with self.assertRaisesRegex(ValueError, "changed"):
            workspace.write("stale.txt", "x", 0)
        restored = workspace.restore(checkpoint_id, workspace.snapshot()["revision"])
        self.assertEqual(workspace.read("src/main.py")["content"], "old\n")
        with self.assertRaises(KeyError):
            workspace.read("README.md")
        self.assertEqual(restored["revision"], 4)
        self.assertEqual(restored["checkpoints"][-1]["label"], "Before restore")

    def test_project_definitions_load_legacy_state_and_persist_separately(self):
        old = dict(revision=5, files={"notes.md": "old", "src/main.py": "first"}, checkpoints=[], jobs=[])
        with open(os.path.join(self.temp.name, "workspace.json"), "w", encoding="utf-8") as saved:
            json.dump(old, saved)
        workspace = Workspace(self.temp.name)
        self.assertEqual(workspace.projects(), [])
        before = workspace.snapshot()
        project = workspace.save_project(dict(name="  Release prep  ", instructions="Check imports.\n",
                                              paths=["src/main.py", "notes.md"]))
        self.assertEqual(project["name"], "Release prep")
        self.assertEqual(project["version"], 1)
        self.assertEqual(workspace.snapshot(), before)
        self.assertEqual(Workspace(self.temp.name).projects(), [project])
        context = workspace.project_context(project["id"])
        self.assertEqual(context["revision"], 5)
        self.assertEqual(context["files"], [dict(path="src/main.py", text="first"), dict(path="notes.md", text="old")])
        workspace.write("src/main.py", "next", 5)
        self.assertEqual(context["files"][0]["text"], "first")
        self.assertEqual(workspace.project_context(project["id"])["files"][0]["text"], "next")
        workspace.write("notes.md", None, 6, delete=True)
        missing = Workspace(self.temp.name)
        self.assertEqual(missing.projects(), [project])
        with self.assertRaisesRegex(ProjectConflict, "notes.md"):
            missing.project_context(project["id"])

    def test_project_definition_validation_bounds_and_failed_save(self):
        workspace = Workspace(self.temp.name)
        workspace.write("notes.md", "", 0)
        malformed = [
            (dict(name=" ", paths=[]), "project name"),
            (dict(name="x" * 121, paths=[]), "project name"),
            (dict(name="Project", instructions="\0", paths=[]), "16 KiB"),
            (dict(name="Project", instructions="\u00e9" * (MAX_PROJECT_INSTRUCTIONS // 2 + 1), paths=[]), "16 KiB"),
            (dict(name="Project", paths=["notes.md"] * 9), "up to eight"),
            (dict(name="Project", paths=["notes.md", "notes.md"]), "only once"),
            (dict(name="Project", paths=["../escape"]), "stay inside"),
            (dict(name="Project", paths=["missing.md"]), "Missing saved"),
        ]
        for body, message in malformed:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                workspace.save_project(body)
        with patch.object(workspace, "_save", side_effect=OSError("disk error")):
            with self.assertRaisesRegex(OSError, "disk error"):
                workspace.save_project(dict(name="Failed save", paths=[]))
        self.assertEqual(workspace.projects(), [])
        self.assertEqual(Workspace(self.temp.name).projects(), [])
        for index in range(MAX_PROJECT_DEFINITIONS):
            workspace.save_project(dict(name=f"Project {index}", paths=[]))
        with self.assertRaisesRegex(ValueError, "up to 32"):
            workspace.save_project(dict(name="Extra", paths=[]))
        self.assertEqual(len(workspace.projects()), MAX_PROJECT_DEFINITIONS)

    def test_fake_runner_updates_files_and_caps_output(self):
        runner = FakeRunner(updated={"result.txt": "generated"}, output="x" * (MAX_OUTPUT * 2))
        workspace = Workspace(self.temp.name, runner=runner)
        workspace.write("input.txt", "seed", 0)
        job = workspace.start("python build.py")
        self.assertTrue(runner.finished.wait(2))
        result = self.wait_job(workspace, job["id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["output"]), MAX_OUTPUT)
        self.assertEqual(workspace.read("result.txt")["content"], "generated")
        self.assertEqual(workspace.snapshot()["revision"], 2)
        self.assertEqual(runner.calls[0][1], "python build.py")

    def test_fake_runner_cancellation_discards_output_files(self):
        runner = FakeRunner(block=True, updated={"new.txt": "must not apply"})
        workspace = Workspace(self.temp.name, runner=runner)
        workspace.write("input.txt", "keep", 0)
        revision = workspace.snapshot()["revision"]
        job = workspace.start("sleep 10")
        self.assertTrue(runner.started.wait(2))
        workspace.cancel(job["id"])
        runner.release.set()
        self.assertTrue(runner.finished.wait(2))
        self.assertEqual(self.wait_job(workspace, job["id"])["status"], "cancelled")
        self.assertEqual(workspace.snapshot()["revision"], revision)
        self.assertEqual(workspace.read("input.txt")["content"], "keep")
        with self.assertRaises(KeyError):
            workspace.read("new.txt")

    def test_no_runner_never_falls_back_to_host_execution(self):
        workspace = Workspace(self.temp.name)
        with patch("workspace.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ValueError, "Host execution is never used"):
                workspace.start("echo unsafe")
            popen.assert_not_called()

    def docker_runner(self, output_tar):
        runner = DockerRunner("example/sandbox:latest")
        calls = []

        def invoke(args, payload=None, limit=32 * MAX_FILE):
            calls.append((list(args), payload, limit))
            if "-cf" in args and "tar" in args:
                return output_tar
            return b""

        runner.invoke = invoke
        return runner, calls

    def test_docker_run_uses_restricted_flags_and_no_host_mounts(self):
        runner, calls = self.docker_runner(archive([("out.txt", "ok", "file")]))
        process = FakeProcess()
        with patch("workspace.subprocess.Popen", return_value=process) as popen:
            try:
                updated, code = runner.run({"in.txt": "seed"}, "python task.py", lambda _: None, threading.Event())
            except Exception as exc:
                self.fail("fake Docker invocation failed: %s; calls=%r" % (exc, calls))
        self.assertEqual((updated, code), ({"out.txt": "ok"}, 0))
        run_args = calls[0][0]
        self.assertEqual(run_args[:2], ["run", "--detach"])
        for flag in ("--pull=never", "--read-only", "--network", "none", "--cap-drop", "ALL",
                     "--security-opt", "no-new-privileges", "--pids-limit", "64", "--memory", "256m",
                     "--memory-swap", "256m", "--cpus", "1", "--user", "65534:65534"):
            self.assertIn(flag, run_args)
        self.assertFalse(any(arg in ("-v", "--volume", "--mount") for arg in run_args))
        exec_args = popen.call_args.args[0]
        self.assertEqual(exec_args[-3:], ["/bin/sh", "-c", "python task.py"])
        self.assertEqual(calls[-1][0][:2], ["rm", "--force"])

    def test_malicious_container_archives_are_refused_without_extracting(self):
        malicious = [
            archive([("../escape", "bad", "file")]),
            archive([("/absolute", "bad", "file")]),
            archive([("link", "../../host", "symlink")]),
            archive([("duplicate", "one", "file"), ("duplicate", "two", "file")]),
        ]
        for payload in malicious:
            with self.subTest(payload=len(payload)):
                runner, _ = self.docker_runner(payload)
                with patch("workspace.subprocess.Popen", return_value=FakeProcess()):
                    with self.assertRaises(ValueError):
                        runner.run({}, "true", lambda _: None, threading.Event())

    def test_unavailable_remote_does_not_hide_workspace_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            remote = FakeRemote()
            with patch.object(remote, "list_tools", side_effect=RuntimeError("connection unavailable")):
                tools = WorkspaceTools(remote, Workspace(directory)).list_tools()
            self.assertEqual([tool["name"] for tool in tools],
                             ["workspace_list", "workspace_read", "workspace_write"])

    def test_reply_binds_writes_to_target_file_before_generation(self):
        workspace = Workspace(self.temp.name)
        workspace.write("site.html", "old", 0)
        tools = WorkspaceTools(FakeRemote(), workspace)
        spec = next(t for t in tools.list_tools() if t["name"] == "workspace_write")
        self.assertEqual(set(spec["inputSchema"]["properties"]), {"path", "content"})
        reply = tools.bind_reply({"workspace_write"})
        first = reply()
        def write(invoke, path, content, **extra):
            result = invoke("workspace_write", dict(path=path, content=content, **extra))
            return json.loads(result["content"][0]["text"])
        workspace.write("notes.txt", "unrelated", 1)
        self.assertEqual(write(first, "site.html", "new", revision=6)["status"], "saved")
        with self.assertRaisesRegex(ValueError, "workspace_read"):
            write(first, "site.html", "competing")
        self.assertEqual(write(first, "site.html", "new")["status"], "already_saved")
        self.assertEqual(write(first, "more.txt", "other")["status"], "saved")
        second = reply()
        workspace.write("site.html", "human edit", workspace.files()["revision"])
        with self.assertRaisesRegex(ValueError, "workspace_read"):
            write(second, "site.html", "overwrite")
        with self.assertRaisesRegex(ValueError, "workspace_read"):
            write(reply(), "site.html", "blind retry")
        second("workspace_read", {"path": "site.html"})
        self.assertEqual(write(reply(), "site.html", "reviewed")["status"], "saved")
        third = reply()
        workspace.write("site.html", "", workspace.files()["revision"], delete=True)
        with self.assertRaisesRegex(ValueError, "workspace_list"):
            write(third, "site.html", "resurrect")
        third("workspace_list", {})
        self.assertEqual(write(reply(), "site.html", "recreated")["status"], "saved")

    def test_repeated_agent_write_preserves_revision_and_checkpoints(self):
        workspace = Workspace(self.temp.name)
        tools = WorkspaceTools(FakeRemote(), workspace)
        def write(path, content, revision):
            result = tools.call_tool("workspace_write", dict(path=path, content=content, revision=revision))
            return json.loads(result["content"][0]["text"])

        result = write("site.html", "<main>Hello</main>", 0)
        self.assertEqual(result["revision"], 1)
        workspace.write("notes.txt", "another edit", 1)
        before = workspace.snapshot()
        for revision in (0, 1, 2):
            result = write("site.html", "<main>Hello</main>", revision)
            self.assertEqual((result["status"], result["revision"]), ("already_saved", 2))
        with self.assertRaisesRegex(ValueError, "workspace_read"):
            write("site.html", "different", 1)
        with self.assertRaisesRegex(ValueError, "workspace_list"):
            write("missing.txt", "new", 1)
        for invalid in (True, None, -1, 3):
            with self.assertRaises(ValueError):
                write("site.html", "<main>Hello</main>", invalid)
        with self.assertRaises(ValueError):
            write("new.txt", None, 2)
        self.assertEqual(workspace.snapshot(), before)
        self.assertEqual(workspace.read("site.html")["content"], "<main>Hello</main>")
        self.assertEqual(write("site.html", "new text", 2)["revision"], 3)
        self.assertEqual(len(workspace.snapshot()["checkpoints"]), 2)

    def test_workspace_tools_list_read_write_and_remote_passthrough(self):
        workspace = Workspace(self.temp.name)
        workspace.write("src/main.py", "one\ntwo\nthree", 0)
        remote = FakeRemote()
        tools = WorkspaceTools(remote, workspace)
        definitions = {item["name"]: item for item in tools.list_tools()}
        self.assertEqual(set(definitions), {"workspace_list", "workspace_read", "workspace_write",
                                            "remote__echo", "workspace__remote_echo"})
        self.assertTrue(definitions["workspace_write"]["approval"])
        self.assertNotIn("workspace_run", definitions)
        listed = tools.call_tool("workspace_list", {})
        self.assertIn("src/main.py", listed["content"][0]["text"])
        read = tools.call_tool("workspace_read", {"path": "src/main.py", "offset": 1, "limit": 1})
        self.assertEqual(json.loads(read["content"][0]["text"])["content"], "two")
        revision = workspace.snapshot()["revision"]
        written = tools.call_tool("workspace_write", {"path": "src/new.py", "content": "new", "revision": revision})
        self.assertEqual(json.loads(written["content"][0]["text"])["revision"], 2)
        remote_result = tools.call_tool("remote__echo", {"value": 1})
        self.assertEqual(remote_result["content"][0]["text"], "remote")
        prefixed_remote_result = tools.call_tool("workspace__remote_echo", {"value": 2})
        self.assertEqual(prefixed_remote_result["content"][0]["text"], "remote")
        self.assertEqual(remote.calls, [("remote__echo", {"value": 1}),
                                       ("workspace__remote_echo", {"value": 2})])
        workspace.runner = FakeRunner(updated={"ran.txt": "echo ok"})
        enabled = {item["name"]: item for item in tools.list_tools()}
        self.assertTrue(enabled["workspace_run"]["approval"])
        run_result = tools.call_tool("workspace_run", {"command": "echo ok"})
        self.assertFalse(run_result["isError"])
        self.assertIn('"status": "completed"', run_result["content"][0]["text"])
        workspace.runner = FakeRunner(updated={"ran.txt": "echo ok"})
        enabled = {item["name"]: item for item in tools.list_tools()}
        self.assertTrue(enabled["workspace_run"]["approval"])
        run_result = tools.call_tool("workspace_run", {"command": "echo ok"})
        self.assertFalse(run_result["isError"])
        self.assertIn('"status": "completed"', run_result["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
