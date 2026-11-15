# llmx companion

The llmx companion brings Chat, Monitor, Workspace and Settings under one
address, session and shared dashboard theme. Workspace means project files,
checkpoints and command execution; conversations are listed under History.
Conversations, branches, tool results and request metrics live in a local SQLite database. The companion uses
Python's standard library and buildless HTML, CSS and JavaScript. It changes no
inference code and needs no package installation or external assets.

## Run

Start llmx's server first, then run from the repository root:

```powershell
python tools/chat_ui/server.py --upstream http://127.0.0.1:8090 --port 8183
```

Open `http://127.0.0.1:8183`. `--upstream` accepts an origin or a URL ending in
`/v1`. By default, data is in `~/.llmx-chat/chat.sqlite3`. `--data-dir PATH`
selects another directory. Settings can download an online SQLite chat backup
and restore one while replies are idle. Restore replaces conversations, and
accepts backups up to 256 MiB as a streamed binary upload. Back up the entire data
directory with the companion stopped to include workspace files and checkpoints.
Do not point two companion processes at the same database.

The upstream may run on another machine. For example, use
`--upstream http://11.10.10.13:8090` when that address is your llmx endpoint.
The companion does not discover production ports or start an inference server.

For access from another machine, supply an owner password file and the exact
browser-facing origin:

```text
python tools/chat_ui/server.py --upstream http://127.0.0.1:8090 --host 0.0.0.0 --port 8183 --password-file /run/secrets/chat-password --public-origin https://chat.example.com
```

Non-loopback listening requires an HTTPS public origin and an owner password.
Put HTTPS in front of that deployment with a reverse proxy. The proxy must pass
the browser's Host header and disable buffering for `/api/runs/*/events` and
`/api/monitor/stream`.
The password must contain 12 to 1024 characters. The owner session cookie is
HttpOnly and SameSite=Strict; an HTTPS public origin also sets Secure. An optional
`--api-key-file PATH` supplies an upstream bearer key without placing it in the
command line or browser. The files and data directory belong to the operator.

Login bodies are limited to 16 KiB before they are read. Ten failed logins in
one minute block further attempts from that socket's source address until the
failures expire. Forwarded address headers are not trusted, so clients behind
one proxy or NAT share a limit. Failure records are bounded to 1024 active
addresses, with new addresses refused while that table is full.

The service admits at most 64 simultaneous HTTP connections, including event
streams. Excess connections are closed before a handler thread is started.
This bounds resources, but does not guarantee availability against clients
that keep connections occupied. Configure request deadlines and connection
limits at the reverse proxy for a public deployment.

This is one owner workspace. Everyone who knows its password can access its
conversations and tools. There is no multi-user account or role system.

## Conversations

- New conversation reuses the current empty conversation and keeps its unsent
  text and attachments. Repeated clicks create at most one new conversation.
- Replies stream while the companion records them. Refreshing or switching
  conversations does not cancel generation. Reopening the chat replays missed
  events from the saved cursor without resubmitting the prompt.
- Stop cancels the current model request. A running external tool may already
  have performed an action, so Stop cannot promise to undo it.
- Edit a user message or retry an answer to create an alternative branch. The
  original branch stays available through the message's branch controls.
- Each sidebar row has a compact anchored actions menu for rename, pin/unpin,
  move to folder, duplicate, archive/restore, export and delete. Actions target
  that row without opening it. Rename, folder and delete use accessible dialogs.
  The sidebar also supports search. Export and import
  preserve messages and their branches. Imported chats do not restart runs.
- Settings are per conversation. System instructions and generation settings
  apply to subsequent calls, including subsequent calls after a tool result.
- Reasoning is displayed only when the upstream supplies `reasoning_content`.
  You can open or close it while the reply is being generated. Reloading
  reconnects to the same reply without cancelling or restarting generation.
  The renderer supports basic Markdown, code blocks, tables and safe links.
  It does not execute model HTML or JavaScript.
- Attach up to eight UTF-8 text/code files or PNG/JPEG images. Text is included
  in full with its file name, not retrieved from an embedding index. Limits are
  2 MiB per text file, 8 MiB per image and 10 MiB total decoded attachments.
  The upstream model must support images. PDF extraction, audio, OCR, rendered
  LaTeX, executable artifacts and knowledge-base retrieval are not implemented.
- A companion process restart marks unfinished runs interrupted. It never
  retries a tool automatically after losing its outcome.

## Temporary chats

Temporary chat in the sidebar opens a conversation held only in companion memory.
It is absent from saved history and SQLite backups. Messages, attachments, settings
and replay events share the normal conversation engine in a separate in-memory
store. Reloading the page recovers the current temporary chat through an identifier
in the tab's session storage; no transcript is saved in browser local storage.

End chat, New conversation or selecting another conversation discards the temporary
chat after confirmation if it has messages or an unsent draft. Stop an active reply
first. Switching to Monitor, Workspace or Settings leaves the chat open. Closing
the tab does not guarantee immediate deletion: abandoned idle chats expire after
one hour, checked once a minute. Active replies remain until stopped or finished.
A companion restart removes all temporary chats. At most 16 can be open.

Text entry and destructive confirmations use the shared accessible action dialog,
including ending a temporary chat. Cancelling before submission leaves the chat
and workspace unchanged. Focus returns to the invoking control when it is still
available; keyboard shortcuts such
as Ctrl+K and Ctrl+S do not run behind an open dialog. Async actions keep the dialog
busy until they finish and show failures there. The browser's native
`beforeunload` warning remains for unsaved workspace edits.

Tools are disabled in temporary chats, including workspace writes and external MCP
calls. This is a conversation-history option, not anonymous inference: the model
server can retain logs or caches, and manually copying or exporting a conversation
creates a separate copy. Workspace files remain independent of chat mode.

## Tools

The operator can attach MCP servers over HTTP. Create a private JSON file:

```json
{
  "servers": [
    {
      "name": "workspace",
      "url": "http://127.0.0.1:9100/mcp",
      "approval": true
    }
  ]
}
```

Pass it as `--tools PATH`. Optional `headers` supply server credentials. They
are never included in the browser's configuration response. The browser lists
server-prefixed tool names in Settings; select only the tools for that chat.

A model call that requests tools ends before those tools execute. Each call
has separate arguments, status, duration and results. Approval is required by
default. Remembering approval applies only to the named tool for the current
run. An operator's `approval: false` allows its selected tools automatically.
Approvals expire after 30 minutes. Up to four calls from one model reply can
execute concurrently; at most 16 tool calls per model reply and 16 model/tool
rounds per run are accepted. The companion permits at most eight saved-chat
runs and eight temporary-chat runs at once.

The activity view keeps the full bounded tool result. A result larger than
64 KiB is explicitly shortened before it enters the next model prompt; the
model sees a truncation notice. This prevents a large result silently consuming
the whole context. HTTP MCP responses are bounded at 8 MiB. This client negotiates
protocol version `2025-06-18` and refuses a different negotiated version. Stdio MCP servers,
OAuth setup and arbitrary browser-installed tool endpoints are not supported.

## Statistics and context

Input and output are server-reported tokens, not counts of SSE chunks. Cached
input is part of input, not additional tokens. Each model call's final usage is
stored once, even if an upstream repeats it in several events. One tool turn
can contain several model calls. Missing usage is unavailable, never zero.

The inspector exposes usage, final prompt/generation rates, queue time and the
effective context limit where the upstream provides them. llmx's server time to
first token is preparation plus queue plus prompt time. Its generation rate
counts tokens after the first, using the time after that first token. These are
different from end-to-end elapsed time and browser delivery speed.

The live display shows state and elapsed time. It does not invent live token
rates from chunks or join unrelated health IDs to a reply. A backend contract
for correlating live request metrics is needed before adding that information.
Context reports the server's effective limit, not the model's advertised maximum.
Raw context metadata preserves any dropped-message and dropped-token counts.

The existing health dashboard remains the view for device topology, cache tiers,
PLE placement and aggregate runtime throughput, embedded in Monitor. Its renderer,
collector and theme are reused directly. Light and dark mode apply to the monitor's
panels, charts and labels as well as the surrounding workspace. One collector polls the configured
upstream's `/v1/health` every two seconds, independent of the number of tabs.
With no valid saved theme, the workspace follows the operating system's light or
dark preference; the theme control saves an explicit choice across reloads.
The browser receives those collected updates over SSE. This does not turn llmx's
health endpoint into a push endpoint. Missing measurements stay unavailable.

## Code and workspace

Code blocks offer highlighting, copy, download and a source viewer. HTML previews
run only after an explicit Preview action in an opaque-origin sandboxed frame.
They cannot access the workspace cookie or parent DOM. The preview response's
CSP refuses network resource fetches, nested frames and forms; browser previews
are not an OS network sandbox and do not promise to block every self-navigation.
Console messages are displayed as bounded text after checking the frame and its
per-preview token. Preview documents expire after 30 minutes, with at most 16
retained in memory. External scripts, packages and fonts are not loaded.

Workspace tool results include an Open Workspace button to reach the saved files.
Workspace owns managed UTF-8 project files, a source editor, unified diffs,
command output and file checkpoints. It never opens arbitrary host paths.
Each file is limited to 1 MiB, the project to 8 MiB and 256 files. Revision checks
refuse stale edits. Up to 20 checkpoints and 20 command records are retained.
Restoring a checkpoint first saves the current files. These are file snapshots,
not process snapshots or a promise to undo actions made by external MCP tools.
Creating or deleting a file, naming or restoring a checkpoint, editing a generated
file and discarding unsaved edits use the same accessible action dialog. Cancelling before submission
does not write, delete or discard files; revision checks still reject stale writes.

Optional execution requires Docker and an existing Linux image with `/bin/sh`,
`tar` and `sleep`, plus the languages or tools the operator needs:

```text
python tools/chat_ui/server.py --sandbox-image python:3-slim
```

`--sandbox-docker-host ssh://user@host` selects a remote Docker service. The
operator's Docker or SSH access is privileged and remains outside the container.
The companion does not pull images. Each command receives a disposable container
with no host mounts, no network, no capabilities, no privilege escalation, a
read-only root filesystem, a non-root user, one CPU, 256 MiB of memory and 64
processes. Project and temporary storage are bounded tmpfs mounts. Commands time
out after 120 seconds; the output view retains the last 128 Ki characters.
The service never substitutes host execution when Docker is unavailable.

Saved files are copied in and validated UTF-8 files copied back on completion,
including nonzero command exits. A checkpoint is taken before every command.
Cancellation, timeouts, links, binary or oversized outputs refuse the file update
and retain the pre-command files. Binary build artifacts are not persisted in
this version. Commands can compile and run within the same container invocation.
Python bytecode writes are disabled so ordinary local-module imports do not
produce binary workspace files. Containers remove themselves when they exit.

The optional workspace tools use the existing model/tool loop and approval UI.
Enable `workspace_list`, `workspace_read`, `workspace_write` and, when configured,
`workspace_run` in conversation settings. Agent writes and commands require
approval and create checkpoints. Reading returns bounded lines rather than
silently inserting the whole project into the prompt. This is not a Pi/OpenCode
adapter or a general IDE, and does not claim their complete feature set.

## Ownership and checks

`store.py` owns persistence and replay. `engine.py` owns model/tool turns and
permissions. `upstream.py` owns bounded compatible HTTP transport. `tools.py`
owns the MCP connection. `server.py` owns HTTP routes and owner authentication.
`workspace.py` owns managed files and isolated execution. `monitor.py` connects
the existing dashboard owner to the workspace without another collector.
`app.js`, `render.js`, `app.css` and `index.html` own presentation. The exact
browser/service protocol is in [CONTRACT.md](CONTRACT.md).

Run the focused checks from the repository root:

```text
python -X utf8 -m unittest discover -s tools/chat_ui -v
node --check tools/chat_ui/app.js
node --check tools/chat_ui/render.js
python -X utf8 tests/run_tests.py --only docs,dead-code
```

The tests use temporary databases and local protocol fixtures, not a GPU or
production server. Browser checks use an installed Chrome/Chromium and Node
when available and explicitly report a skip otherwise. Static UI and standalone
companion changes need their focused checks, not model numerical gates.
The dedicated Chat workspace workflow runs these and the existing dashboard
tests on Windows and Linux, requiring the browser check with
`LLMX_CHAT_REQUIRE_BROWSER=1`. This is test configuration, not a runtime setting.

The real container check is optional because hosted and local machines may have
no running Docker service or installed image. It checks non-root execution,
read-only root, network refusal, absent Docker socket, returned files, checkpoint
restore, cancellation and symlink rejection:

```text
python tools/chat_ui/check_sandbox.py --image python:3-slim --docker-host ssh://user@host --output sandbox-result.json
```

## Qualification

The 2026-11-15 feature lands by fast-forward as one commit from main 929f042c6.
The exact source, tests and workflow at 53f83204a pass the dedicated
[Windows/Linux run](https://github.com/mxxm-t/llmx/actions/runs/43658421323).
The landing rebase adds main's unrelated ROCm change 929f042c6 and preserves
both STATUS blocks. Companion, dashboard and workflow inputs are identical
to that tested tree. Final docs checks cover the combined landing tree.

| Check | Result |
|---|---|
| Companion protocol, storage, workspace and browser checks | Passed on Windows and Linux |
| Existing dashboard tests | 20 passed on each hosted platform |
| Real Docker isolation, file round-trip and cancellation | Passed on the rig, no GPUs |
| Actual llmx synthetic-F32 chat protocol | Two turns completed, input/output usage 23/8 and 77/8 |
| Responsive navigation | Actual coordinate clicks at seven sizes, landscape and 200 percent CSS zoom |
| Docs and dead-code | Passed, with existing findings retained |

Original failures remain available: [run 43656889014](https://github.com/mxxm-t/llmx/actions/runs/43656889014)
exposed browser startup and document-navigation synchronization in the test;
[run 43657441535](https://github.com/mxxm-t/llmx/actions/runs/43657441535)
passed Windows and exposed a Linux-timed product race where an already-failed
snapshot omitted the error banner. A locally failing check reproduced it.
Snapshot hydration now restores the selected branch's failure reason, including
on reload without another run stream. The final hosted run proves the correction.

Local checks also caught and fixed repeated New conversation creation, a closed
mobile drawer intercepting clicks, a database filename shadowing the history
method, and restore's former small JSON-upload limit. A valid SQLite backup over
12 MiB is now restored through the streamed binary route. Original failures and
screenshots remain in the local receipts rather than being treated as passes.

The Docker image was `python:3-slim`, Linux amd64, image ID
`sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928`.
The check verified uid 65534, read-only root, blocked network, absent Docker
socket, nested-file updates, checkpoint restore, cancellation and link refusal.
All containers created by that check were removed. These are application and
isolation results, not model accuracy, throughput or absolute security claims.
