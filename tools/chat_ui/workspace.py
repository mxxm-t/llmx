"""Managed project files, checkpoints and bounded container execution."""

from copy import deepcopy
from functools import partial
import difflib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import threading
import time
import uuid


MAX_FILE = 1024 * 1024
MAX_FILES = 256
MAX_PROJECT = 8 * MAX_FILE
MAX_OUTPUT = 128 * 1024
MAX_PROJECT_DEFINITIONS = 32
MAX_SELECTED_FILES = 8
MAX_PROJECT_INSTRUCTIONS = 16 * 1024


class ProjectConflict(ValueError):
    pass


def file_path(value):
    if not isinstance(value, str) or not value or len(value) > 240:
        raise ValueError("Use a relative file path of at most 240 characters")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/")) or ":" in value or "\\" in value or any(ord(c) < 32 for c in value):
        raise ValueError("File paths must stay inside the workspace")
    return str(path)


def _selected_paths(paths):
    if not isinstance(paths, list) or len(paths) > MAX_SELECTED_FILES:
        raise ValueError("Choose up to eight saved Workspace files")
    paths = [file_path(path) for path in paths]
    if len(paths) != len(set(paths)):
        raise ValueError("Select each Workspace file only once")
    return paths


def checked_files(files):
    if not isinstance(files, dict) or len(files) > MAX_FILES:
        raise ValueError("A workspace holds at most 256 text files")
    size = 0
    for path, content in files.items():
        file_path(path)
        if not isinstance(content, str) or "\0" in content:
            raise ValueError("Workspace files must be UTF-8 text")
        length = len(content.encode())
        if length > MAX_FILE:
            raise ValueError("Each file must be at most 1 MiB")
        size += length
        if any(parent.as_posix() in files for parent in PurePosixPath(path).parents if parent.as_posix() != "."):
            raise ValueError("A file cannot also be a directory")
    if size > MAX_PROJECT:
        raise ValueError("Workspace files must total at most 8 MiB")
    return files


def _file_diff(path, old, new, before, after):
    if old == new:
        return ""
    if not old and not new:
        action = "Added" if old is None else "Deleted"
        return f"{action} empty file: {path}\n"
    lines = difflib.unified_diff(io.StringIO(old or "").readlines(), io.StringIO(new or "").readlines(),
                                 fromfile="/dev/null" if old is None else before + "/" + path,
                                 tofile="/dev/null" if new is None else after + "/" + path)
    parts = []
    for line in lines:
        parts.append(line)
        if not line.endswith("\n"):
            parts.append("\n\\ No newline at end of file\n")
    return "".join(parts)


class DockerRunner:
    def __init__(self, image, host=None):
        self.image = image
        self.docker = ["docker"] + (["--host", host] if host else [])

    def invoke(self, args, payload=None, limit=32 * MAX_FILE):
        with subprocess.Popen(self.docker + args, stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            result, errors = bytearray(), bytearray()
            exceeded = threading.Event()

            def drain(stream, target, bound):
                while True:
                    block = stream.read(65536)
                    if not block:
                        return
                    if len(target) + len(block) > bound:
                        exceeded.set()
                        process.kill()
                        return
                    target.extend(block)

            readers = [threading.Thread(target=drain, args=(process.stdout, result, limit), daemon=True),
                       threading.Thread(target=drain, args=(process.stderr, errors, 65536), daemon=True)]
            watchdog = threading.Timer(30, lambda: process.kill() if process.poll() is None else None)
            watchdog.daemon = True
            watchdog.start()
            for thread in readers:
                thread.start()
            try:
                if payload is not None:
                    process.stdin.write(payload)
                    process.stdin.close()
                process.wait(timeout=30)
            except BaseException:
                process.kill()
                process.wait()
                raise
            finally:
                watchdog.cancel()
                for thread in readers:
                    thread.join(timeout=2)
            if any(thread.is_alive() for thread in readers):
                raise TimeoutError("Container transport did not close")
            if exceeded.is_set():
                raise ValueError("Container response exceeded its size limit")
            if process.returncode:
                raise RuntimeError(errors.decode(errors="replace")[-2000:] or "Container operation failed")
            return bytes(result)

    def run(self, files, command, emit, cancel):
        name = "llmx-workspace-" + uuid.uuid4().hex
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            directories = {p.as_posix() for path in files for p in PurePosixPath(path).parents if p.as_posix() != "."}
            for path in sorted(directories, key=lambda p: (p.count("/"), p)):
                entry = tarfile.TarInfo(path)
                entry.type, entry.mode, entry.uid, entry.gid = tarfile.DIRTYPE, 0o700, 65534, 65534
                tar.addfile(entry)
            for path, content in files.items():
                data = content.encode()
                entry = tarfile.TarInfo(path)
                entry.size, entry.mode, entry.uid, entry.gid = len(data), 0o600, 65534, 65534
                tar.addfile(entry, io.BytesIO(data))
        process = None
        try:
            self.invoke(["run", "--detach", "--rm", "--pull=never", "--name", name, "--label", "llmx.workspace=1",
                         "--init", "--read-only", "--network", "none", "--cap-drop", "ALL",
                         "--env", "PYTHONDONTWRITEBYTECODE=1",
                         "--security-opt", "no-new-privileges", "--pids-limit", "64", "--memory", "256m",
                         "--memory-swap", "256m", "--cpus", "1", "--user", "65534:65534",
                         "--tmpfs", "/workspace:rw,nosuid,nodev,size=64m,mode=1777",
                         "--tmpfs", "/tmp:rw,nosuid,nodev,size=32m,mode=1777",
                         "--workdir", "/workspace", "--entrypoint", "/bin/sh", self.image, "-c", "sleep 300"])
            self.invoke(["exec", "--interactive", "--user", "65534:65534", name, "tar", "-xf", "-", "-C", "/workspace"], archive.getvalue())
            if cancel.is_set():
                return None, -1
            process = subprocess.Popen(self.docker + ["exec", "--user", "65534:65534", name, "/bin/sh", "-c", command],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

            def output():
                import codecs
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                while True:
                    block = process.stdout.read1(4096)
                    if not block:
                        emit(decoder.decode(b"", final=True))
                        return
                    emit(decoder.decode(block))

            reader = threading.Thread(target=output, daemon=True)
            reader.start()
            deadline = time.monotonic() + 120
            while process.poll() is None:
                if cancel.wait(0.1) or time.monotonic() > deadline:
                    self.invoke(["rm", "--force", name], limit=65536)
                    process.kill()
                    process.wait(timeout=5)
                    reader.join(timeout=5)
                    if cancel.is_set():
                        return None, -1
                    raise TimeoutError("Command exceeded its 120 second limit")
            reader.join(timeout=5)
            raw = self.invoke(["exec", "--user", "65534:65534", name, "tar", "-cf", "-", "-C", "/workspace", "."])
            updated = {}
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
                count = 0
                for entry in tar:
                    count += 1
                    if count > MAX_FILES * 16:
                        raise ValueError("Container produced too many directory entries")
                    path = entry.name
                    while path.startswith("./"):
                        path = path[2:]
                    if entry.isdir():
                        continue
                    file_path(path)
                    if not entry.isfile() or entry.size > MAX_FILE or path in updated:
                        raise ValueError("Container output contains a link, duplicate or oversized file")
                    updated[path] = tar.extractfile(entry).read(MAX_FILE + 1).decode("utf-8")
                    checked_files(updated)
            return updated, process.returncode
        finally:
            if process and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            try:
                self.invoke(["rm", "--force", name], limit=65536)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                pass


class Workspace:
    def __init__(self, directory, image=None, docker_host=None, runner=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file = self.directory / "workspace.json"
        self.lock = threading.RLock()
        self.runner = runner or (DockerRunner(image, docker_host) if image else None)
        self.image = image
        self.control = None
        self.state = dict(revision=0, files={}, checkpoints=[], jobs=[], projects={})
        if self.file.exists():
            self.state = json.loads(self.file.read_text(encoding="utf-8"))
            checked_files(self.state["files"])
            self.state.setdefault("projects", {})
            self._validate_projects(self.state["projects"])
            for job in self.state["jobs"]:
                if job["status"] == "running":
                    job.update(status="failed", error="Workspace service restarted. Files remain at the saved checkpoint.")

    def _save(self, state=None):
        temporary = self.file.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state if state is None else state, ensure_ascii=True), encoding="utf-8")
        temporary.replace(self.file)

    def snapshot(self):
        with self.lock:
            return dict(self.files(),
                        checkpoints=[{k: v for k, v in point.items() if k != "files"} for point in self.state["checkpoints"]],
                        jobs=[dict(job) for job in self.state["jobs"]],
                        execution=dict(enabled=self.runner is not None, image=self.image,
                                       limits=dict(file_bytes=MAX_FILE, project_bytes=MAX_PROJECT, files=MAX_FILES, command_seconds=120, output_chars=MAX_OUTPUT, checkpoints=20),
                                       description="Isolated container, network disabled, no host mounts, 120 second command limit" if self.runner else "Configure --sandbox-image to enable isolated execution"))

    def files(self, paths=None):
        if paths is not None:
            paths = _selected_paths(paths)
            if not paths:
                raise ValueError("Choose at least one saved Workspace file")
        with self.lock:
            if paths is not None:
                return self._read_files(paths)
            return dict(revision=self.state["revision"],
                        files=[dict(path=path, size=len(text.encode())) for path, text in sorted(self.state["files"].items())])

    def _read_files(self, paths):
        return dict(revision=self.state["revision"],
                    files=[dict(path=path, text=self.state["files"][path]) for path in paths])

    def read(self, path):
        with self.lock:
            path = file_path(path)
            return dict(path=path, content=self.state["files"][path], revision=self.state["revision"])

    def preview_snapshot(self):
        with self.lock:
            return self.state["revision"], dict(self.state["files"])

    @staticmethod
    def _project_values(values):
        if set(values) - {"name", "instructions", "paths"}:
            raise ValueError("Unsupported Workspace project field")
        name, instructions, paths = values.get("name"), values.get("instructions", ""), values.get("paths")
        name = name.strip() if isinstance(name, str) else ""
        if not 1 <= len(name) <= 120 or any(ord(c) < 32 for c in name):
            raise ValueError("Use a project name of 1 to 120 characters")
        name.encode()
        if not isinstance(instructions, str) or "\0" in instructions or len(instructions.encode()) > MAX_PROJECT_INSTRUCTIONS:
            raise ValueError("Project instructions must be UTF-8 text of at most 16 KiB")
        return dict(name=name, instructions=instructions, paths=_selected_paths(paths))

    @classmethod
    def _validate_projects(cls, projects):
        if not isinstance(projects, dict) or len(projects) > MAX_PROJECT_DEFINITIONS:
            raise ValueError("Invalid saved Workspace projects")
        for key, project in projects.items():
            if (not isinstance(project, dict) or set(project) != {"id", "version", "name", "instructions", "paths"}
                    or project["id"] != key or type(project["version"]) is not int or project["version"] < 1):
                raise ValueError("Invalid saved Workspace project")
            cls._project_values({name: project[name] for name in ("name", "instructions", "paths")})

    def projects(self):
        with self.lock:
            return deepcopy(list(self.state["projects"].values()))

    def _require_project_files(self, paths):
        missing = [path for path in paths if path not in self.state["files"]]
        if missing:
            raise ProjectConflict("Missing saved Workspace files: " + ", ".join(missing) + ". Save them or remove them from the project.")

    def _project_at_version(self, project_id, version):
        if type(version) is not int or version < 1:
            raise ValueError("Expected a positive Workspace project version")
        project = self.state["projects"][project_id]
        if project["version"] != version:
            raise ProjectConflict("Workspace project changed. Reload it before saving or deleting.")
        return project

    def _save_projects(self, projects):
        state = dict(self.state, projects=projects)
        self._save(state)
        self.state = state

    def save_project(self, body, project_id=None):
        values = dict(body)
        version = values.pop("version", None) if project_id is not None else None
        values = self._project_values(values)
        with self.lock:
            projects = dict(self.state["projects"])
            if project_id is None:
                if len(projects) >= MAX_PROJECT_DEFINITIONS:
                    raise ValueError("Workspace holds up to 32 project definitions")
                project_id, next_version = uuid.uuid4().hex, 1
            else:
                next_version = self._project_at_version(project_id, version)["version"] + 1
            self._require_project_files(values["paths"])
            project = dict(id=project_id, version=next_version, **values)
            projects[project_id] = project
            self._save_projects(projects)
            return deepcopy(project)

    def delete_project(self, project_id, version):
        with self.lock:
            self._project_at_version(project_id, version)
            projects = dict(self.state["projects"])
            del projects[project_id]
            self._save_projects(projects)

    def project_context(self, project_id):
        with self.lock:
            project = self.state["projects"][project_id]
            paths = project["paths"]
            self._require_project_files(paths)
            return dict(self._read_files(paths), project=deepcopy(project))

    def _editable(self, revision, conflict="Workspace changed. Reload before saving"):
        if self.control is not None:
            raise ValueError("Wait for the active command to finish or cancel it")
        if type(revision) is not int or revision != self.state["revision"]:
            raise ValueError(conflict)

    def write(self, path, content, revision, delete=False, *, agent_edit=False, base=None):
        with self.lock:
            path = file_path(path)
            current = self.state["revision"]
            files = self.state["files"]
            if agent_edit:
                if base is not None:
                    unchanged = files.get(path) == base.get(path) or (path in files and files[path] == content)
                    revision = current if unchanged else None
                if type(revision) is int and 0 <= revision <= current and path in files and files[path] == content:
                    self._editable(current)
                    return dict(self.files(), status="already_saved")
                action = "Read the current file with workspace_read and review the changes" if path in files else "Call workspace_list to check the saved files"
                self._editable(revision, f"Workspace revision is {current}. {action} before retrying.")
            else:
                self._editable(revision)
            updated = dict(files)
            if delete:
                del updated[path]
            else:
                updated[path] = content
            state = dict(self.state, files=checked_files(updated), revision=current + 1)
            if agent_edit:
                state["checkpoints"] = self._checkpoints_with("Before agent edit: " + path[:90])
            self._save(state)
            self.state = state
            return dict(self.files(), status="saved") if agent_edit else self.snapshot()

    def _checkpoints_with(self, label):
        point = dict(id=uuid.uuid4().hex, label=label.strip(), created=time.time(), files=dict(self.state["files"]))
        return (self.state["checkpoints"] + [point])[-20:]

    def checkpoint(self, label):
        with self.lock:
            if not isinstance(label, str) or not 1 <= len(label.strip()) <= 120:
                raise ValueError("Use a checkpoint name of 1 to 120 characters")
            self.state["checkpoints"] = self._checkpoints_with(label)
            self._save()
            return self.snapshot()

    def _point(self, point_id):
        return next((p for p in self.state["checkpoints"] if p["id"] == point_id), None)

    def restore(self, point_id, revision):
        with self.lock:
            self._editable(revision)
            point = self._point(point_id)
            if not point:
                raise KeyError(point_id)
            files = dict(point["files"])
            self.checkpoint("Before restore")
            self.state["files"] = files
            self.state["revision"] += 1
            self._save()
            return self.snapshot()

    def diff(self, point_id):
        with self.lock:
            point = self._point(point_id)
            if not point:
                raise KeyError(point_id)
            before, after = point["files"], self.state["files"]
            return {"diff": "".join(_file_diff(path, before.get(path), after.get(path), "checkpoint", "workspace")
                                    for path in sorted(before.keys() | after.keys()))}

    def draft_diff(self, path, content):
        path = file_path(path)
        checked_files({path: content})
        saved = self.read(path)
        return {"diff": _file_diff(path, saved["content"], content, "saved", "draft")}

    def job(self, job_id):
        with self.lock:
            for job in self.state["jobs"]:
                if job["id"] == job_id:
                    return dict(job)
        raise KeyError(job_id)

    def start(self, command):
        with self.lock:
            self._editable(self.state["revision"])
            if not self.runner:
                raise ValueError("Isolated execution is not configured. Host execution is never used")
            if not isinstance(command, str) or not command.strip() or len(command) > 8192 or "\0" in command:
                raise ValueError("Use a command of 1 to 8192 characters")
            self.checkpoint("Before command: " + command[:90])
            job = dict(id=uuid.uuid4().hex, command=command, status="running", output="", output_truncated=False, exit_code=None, error=None, created=time.time())
            self.state["jobs"] = (self.state["jobs"] + [job])[-20:]
            self.control = threading.Event()
            self._save()
            thread = threading.Thread(target=self._run, args=(job, dict(self.state["files"]), self.control), daemon=True)
            thread.start()
            return dict(job)

    def _run(self, job, files, cancel):
        def emit(text):
            with self.lock:
                job["output_truncated"] = job["output_truncated"] or len(job["output"]) + len(text) > MAX_OUTPUT
                job["output"] = (job["output"] + text)[-MAX_OUTPUT:]
        try:
            updated, code = self.runner.run(files, job["command"], emit, cancel)
            with self.lock:
                if updated is not None and not cancel.is_set():
                    self.state["files"] = checked_files(updated)
                    self.state["revision"] += 1
                job.update(status="cancelled" if cancel.is_set() else "completed" if code == 0 else "failed", exit_code=code)
        except Exception as exc:
            with self.lock:
                job.update(status="cancelled" if cancel.is_set() else "failed", error=str(exc)[:2000])
        finally:
            with self.lock:
                self.control = None
                self._save()

    def cancel(self, job_id):
        with self.lock:
            job = self.job(job_id)
            if job["status"] == "running" and self.control:
                self.control.set()
            return job

    def close(self):
        with self.lock:
            if self.control:
                self.control.set()


class WorkspaceTools:
    def __init__(self, remote, workspace):
        self.remote, self.workspace = remote, workspace

    @property
    def last_error(self):
        return self.remote.last_error

    def list_tools(self, discover=True):
        definitions = [
            ("workspace_list", "List managed project files and the current revision.", {}, [], False),
            ("workspace_read", "Read lines of a managed file. Offset is zero based.",
             {"path": {"type": "string"}, "offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 500}}, ["path"], False),
            ("workspace_write", "Save a UTF-8 file. Use path and content. A change creates a checkpoint. If another edit conflicts, read the file before retrying.",
             {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"], True),
        ]
        if self.workspace.runner:
            definitions.append(("workspace_run", "Run a command in the isolated project container. Network is disabled. Changes return to managed files.",
                                {"command": {"type": "string"}}, ["command"], True))
        tools = [dict(name=name, description=description, inputSchema=dict(type="object", properties=properties, required=required, additionalProperties=False),
                      server="Workspace", connection="builtin:workspace", approval=approval) for name, description, properties, required, approval in definitions]
        try:
            return tools + self.remote.list_tools(discover=discover)
        except RuntimeError:
            return tools

    def connections(self):
        return [dict(id="builtin:workspace", name="Workspace", kind="builtin", status="ready",
                     tool_count=4 if self.workspace.runner else 3, approval=None,
                     checked_at=None, duration_ms=None, error=None)] + self.remote.connections()

    def test_connection(self, identity):
        self.remote.test_connection(identity)

    def bind_reply(self, selected):
        if "workspace_write" not in selected:
            return None
        _, observed = self.workspace.preview_snapshot()

        def bind_round():
            return partial(self.call_tool, base=dict(observed), observed=observed)

        return bind_round

    def call_tool(self, name, arguments, cancel_event=None, *, base=None, observed=None):
        if name not in ("workspace_list", "workspace_read", "workspace_write", "workspace_run"):
            return self.remote.call_tool(name, arguments, cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError("Cancelled")
        if name == "workspace_list":
            with self.workspace.lock:
                result = self.workspace.files()
                if observed is not None:
                    for missing in observed.keys() - {file["path"] for file in result["files"]}:
                        del observed[missing]
        elif name == "workspace_read":
            offset, limit = arguments.get("offset", 0), arguments.get("limit", 200)
            if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 500:
                raise ValueError("Invalid line range")
            with self.workspace.lock:
                result = self.workspace.read(arguments.get("path"))
                if observed is not None:
                    observed[result["path"]] = result["content"]
            lines = result["content"].splitlines()
            text = "\n".join(lines[offset:offset + limit])
            result.update(content=text[:32768], offset=offset, total_lines=len(lines),
                          truncated=len(text) > 32768 or offset + limit < len(lines))
        elif name == "workspace_write":
            with self.workspace.lock:
                path, content = arguments.get("path"), arguments.get("content")
                result = self.workspace.write(path, content, arguments.get("revision"), agent_edit=True, base=base)
                if observed is not None:
                    observed[file_path(path)] = content
        elif name == "workspace_run":
            job = self.workspace.start(arguments.get("command"))
            while job["status"] == "running":
                if cancel_event is not None and cancel_event.wait(0.1):
                    self.workspace.cancel(job["id"])
                    raise ValueError("Cancelled")
                if cancel_event is None:
                    time.sleep(0.1)
                job = self.workspace.job(job["id"])
            result = job
        else:
            raise ValueError("Unknown workspace tool")
        return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=True)}],
                "isError": name == "workspace_run" and result["status"] != "completed"}
