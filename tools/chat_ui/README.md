# llmx companion

The llmx companion brings Chat, Monitor, Workspace and Settings under one
address, session and shared dashboard theme. Workspace means project files,
checkpoints and command execution; conversations are listed under History.
Conversations, branches, tool results and request metrics live in a local SQLite database. The companion uses
Python's standard library and buildless HTML, CSS and JavaScript. It changes no
inference code and needs no package installation or external assets.

Tool names and arguments appear while the model prepares a call. Open the call
to inspect its incoming arguments, or refresh to resume from its saved state.
Preparing a call does not run it: approval and execution begin only after the
model stream completes. An interrupted call keeps its partial arguments and is
not retried automatically.

## Work on files

Successful Workspace tool results can open their file or command output
directly. Select a file in Workspace to edit it, or create one with New file.
Preview HTML renders the current editor buffer, including unsaved edits, in an
isolated browser frame. It uses saved CSS and classic JavaScript linked directly
from that document. The preview lists the saved Workspace revision and included
assets. Other files offer View source. Close the viewer to return to the editor.

Attach to Chat adds the current editor buffer, including unsaved edits, to the
current draft and returns focus to the composer. The attachment can be removed
before sending. A path such as `src/index.js` appears as `src:index.js` to
distinguish it from another file with the same basename. The normal text
attachment limits and temporary chat behavior apply.

To reuse files in another chat, open **Projects** in Workspace. Give the
project a name, select up to eight saved files and optionally enter instructions.
In Chat, open **Settings** and select **Use a Workspace project**. If the
picker is empty, choose **Open Workspace projects** to reach **New project**
with your draft and editor intact. For a saved project, review the file paths,
sizes and saved revision. Check the instructions box if you want them to
replace this conversation's system instructions, then choose **Add to draft**.
You can remove attachments before sending.

Each use takes a snapshot. Editing a saved file later affects the next use.
The draft and previously sent messages retain their copies. Applying the same
files twice keeps one copy. If the draft already contains a different version
of a selected file, remove that attachment before adding the new version.
Temporary chats can use a project explicitly. Their drafts stay in memory.
Project definitions live with Workspace files in the companion data directory.
The conversation database backup covers the copies in saved conversations.

Create checkpoint captures saved workspace files. Open Checkpoint history to
review differences or restore a checkpoint. The command runner starts from
saved workspace files and creates a checkpoint before a command runs. Back to
Chat returns to the current conversation and keeps the editor buffer for the
current page.

## Run

Start llmx's server first, then run from the repository root:

```powershell
python tools/chat_ui/server.py --upstream http://127.0.0.1:8090 --port 8183
```

Open `http://127.0.0.1:8183`. `--upstream` accepts an origin or a URL ending in
`/v1`. By default, data is in `~/.llmx-chat/chat.sqlite3`. `--data-dir PATH`
selects another directory. Settings can download an online SQLite chat backup
and restore one while replies are idle. Restore replaces conversations, and
accepts backups up to 256 MiB as a streamed binary upload. Broken message branches,
including cycles and references to other conversations, are refused before the
active database is replaced. Back up the entire data
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
`--api-key-file PATH` supplies one raw upstream bearer key without placing it in
the command line or browser. Use a separate file containing only one key, not an
upstream key database with multiple keys or comment lines. The file is read once
at startup; restart the companion after rotating the key. This key is sent only
to the configured upstream and does not authenticate browser users;
the companion's owner password controls browser access. The files and data
directory belong to the operator.

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

## Model selection

The selector lists model IDs reported by the connected upstream server. Chat sends
the selected ID with each request, but the current `llmx serve` process keeps
the model file loaded at startup. It does not route a request to another model
or load a different file from this selector. To use another model, connect to an
upstream serving that model. If a saved conversation refers to a model ID the
connected upstream does not report, that choice is marked unavailable and
sending stays disabled until you select an available model. Conversation
Settings describes the selector alongside the reported context capacities.

Before starting a run with an explicit model choice, the companion fetches the
upstream catalogue again. A missing choice is refused before any message or run
is stored. The browser updates its choices and keeps the unsent prompt,
attachments and generation settings. If the upstream is unavailable, the last
known choices remain so you can try sending again without reloading. This check
adds one upstream `/models` request per explicit-model run, including retries.
It does not lock the upstream model or control later server restarts.

## Conversations

Main navigation highlights the selected section. Hover uses a lighter fill, and
keyboard focus keeps its visible outline.

- Unsent text, attachments and edit targets stay with each ordinary conversation
  when you switch chats or select the same chat again. They are saved in this
  browser tab's session storage for refresh recovery, not in chat backups.
  Closing the tab ends that session. If draft storage writes fail or reach the
  browser's quota, a notice explains that affected drafts remain in this page
  only; switching chats still preserves them, but refreshing can discard them.
  Sending clears only the submitted draft, preserving newer typing. Signing out
  clears this tab's ordinary drafts. Temporary chat drafts stay in memory and
  are discarded on refresh, as the composer explains.
- New conversation reuses the current empty conversation and keeps its unsent
  text and attachments. A first blank chat keeps the settings already entered.
  Starting a new conversation from an existing populated one uses the companion
  defaults, including an empty tool selection. Repeated clicks create at most
  one new conversation.
  If a conversation is created after you have selected another chat, it stays
  in the sidebar without replacing that selection or clearing its newer draft.
- Replies stream while the companion records them. Refreshing or switching
  conversations does not cancel generation. Reopening the chat replays missed
  events from the saved cursor without resubmitting the prompt.
- Stop cancels the current model request. A running external tool may already
  have performed an action, so Stop cannot promise to undo it.
- Edit a user message or retry an answer to create an alternative branch. The
  original branch stays available through the message's branch controls. A retry
  continues in its original conversation if you navigate away while admission
  or snapshot refresh is pending; its completion does not reconnect or scroll
  the newly selected view.
- Each sidebar row has a compact anchored actions menu for rename, pin/unpin,
  move to folder, duplicate, archive/restore, export and delete. Actions target
  that row without opening it. Rename, folder and delete use accessible dialogs.
  Sidebar search matches titles, folders and visible user and assistant text in
  every saved conversation in the current Chats or Archive view, including
  alternate branches. It matches literal text without regard to case. Temporary
  chats, reasoning, attachments and tool payloads are excluded. Export and
  import preserve messages and their branches. Imported chats do not restart runs.
  Saved rows show when their latest reply is in progress, needs tool approval,
  failed or was interrupted. These cues refer to the latest attempt even if
  you select an older branch. Open the conversation to review a decision or
  see what happened. Archive and search rows show the same cues.
- Send and Retry use the current conversation settings. Acceptance saves them
  for this chat and captures them for all model calls in that reply, including
  those after tool results. **Save for this chat** saves without starting a
  reply, even if the saved model is unavailable. A blank chat offers this
  button after it is created. Switching chats or refreshing restores the last
  saved settings. You can keep editing while Save or a reply request is
  pending. Wait for that request to finish before submitting another change
  for the same chat.
- Prompts have a **You** label and a tinted bubble. Assistant prose uses shorter
  lines on wide screens, while code and tables retain the space they need.
  Consecutive model calls in a tool-assisted reply share a heading and a side
  rule. Each call keeps its own actions and expandable details. One statistics
  icon covers the reply, including its tool-related model calls.
- Reasoning is displayed only when the upstream supplies `reasoning_content`.
  You can open or close it while the reply is being generated. Reloading
  reconnects to the same reply without cancelling or restarting generation.
  The renderer supports basic Markdown, nested lists with preserved starting
  numbers, code blocks, tables and safe links. Indented paragraphs and code stay
  inside their list item. List nesting is bounded at eight levels; deeper text
  remains visible as literal content. Unchanged leading Markdown blocks remain
  connected during streaming, preserving their selection and code controls.
  The changing suffix is rebuilt, so selection inside it may be lost.
  A focused message action stays focused across message updates while its
  action and target remain available.
  It does not execute model HTML or JavaScript.
- Attach up to eight UTF-8 text/code files or PNG/JPEG images. Text is included
  in full with its file name, not retrieved from an embedding index. Limits are
  2 MiB per text file, 8 MiB per image and 10 MiB total decoded attachments.
  Hover over Attach or focus it with the keyboard for the file types. The
  Attachments section in Conversation Settings shows the limits on any device.
  A read that finishes after switching conversations is discarded with a notice.
  The upstream model and server must support images. Image-bearing prompts are
  checked through llmx tokenization before a reply starts, including images in
  conversation history. A rejection keeps your draft and attachments.
  Older saved image turns refused by the chat template before any reply or tool
  call stay visible but are excluded from later prompts. Retrying that image turn
  still includes its images. Accepted images and partial replies remain in context. PDF extraction, audio, OCR, rendered
  LaTeX, executable artifacts and knowledge-base retrieval are not implemented.
- A companion process restart marks unfinished runs interrupted. It never
  retries a tool automatically after losing its outcome.

## Reply limits and context

New conversations use No additional cap for replies. The companion omits
`max_tokens`, letting the model stop normally or reach the server's remaining
request capacity. Choose Custom token limit to set a positive output-token cap.
Reasoning tokens count toward output. Existing saved numeric caps remain in
place until changed. Changing reply mode does not resize the server's context.

The settings panel distinguishes the model's advertised context from the
running server's request limit. Conversation history, tool results and the reply
share the request limit; these capacities are not a count of remaining tokens.
Request limits come from the same shared monitor stream used for live rates,
while Settings is visible. Stale, unavailable or differently named runtime data
shows Unavailable instead of falling back to the model's larger capacity.

Python owns the defaults returned as `settings_defaults` in `/api/config`.
The browser uses those values for new chats, preserving saved settings.
Other sampling defaults stay temperature 0.7, top P 0.95 and top K 40.

## Reply statistics and activity

Use the Conversation details button in the header to open or close Activity,
Stats and Settings. On wide screens the panel starts closed and the browser
remembers your docked-panel choice. On narrow screens, Details opens as a drawer
when needed. The composer's Settings shortcut opens directly to Settings.
On narrow screens, opening either drawer moves focus inside and disables
background controls. Tab and Shift+Tab stay inside the drawer. Escape, Close
or selecting the shade returns focus to the opener. Navigation moves focus to
the destination instead. The docked Details panel leaves the page keyboard
order available.

Hover over a reply's statistics icon to preview reported token counts and timings.
Click the icon, or use Enter or Space, to keep the panel open for text selection
and Copy values. Touch users tap the icon. Close, Escape or a click outside the
panel dismisses it. Values are a snapshot when opened, so streamed updates do
not disturb selection; reopen to refresh.

A reply that uses tools includes all its model calls in the token totals and
shows timing details for each call. Rates are not averaged. Missing metrics stay
unavailable, and cached input remains part of input. Reported source data is
available inside each call's details.

Activity shows the latest reply and its tool progress or errors. Earlier replies
are grouped in collapsed history. View reply and tool entries navigate to the
matching conversation content. The Stats tab holds conversation totals for the
selected branch, with other branches available separately when present.

## Temporary chats

Temporary chat in the conversation header opens a conversation held only in
companion memory. In temporary mode, the same button is labeled Normal chat and
returns to a blank normal conversation, saved only when you send a message. On
narrow screens the labels are Temporary and Normal. The button keeps its width
when toggled, with a dashed Temporary action and a solid Normal action.
It is absent from saved history and SQLite backups. Messages, attachments, settings
and replay events share the normal conversation engine in a separate in-memory
store. Reloading the page recovers the current temporary chat through an identifier
in the tab's session storage; no transcript is saved in browser local storage.

Normal chat, End chat, New conversation or selecting another conversation discards the temporary
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
are never included in the browser's configuration response. Settings groups tools
by connection and shows its discovery status, available tool count, last check,
duration and error. Built-in Workspace tools stay available when an MCP server
fails. Endpoint URLs, credential headers and protocol session IDs are not exposed.

Test connection repeats initialization when needed and refreshes that connection's
tool list. It does not call a tool or probe other connections. Only one discovery
check runs at a time, while cached tools remain readable. The existing HTTP
request timeout, response-size and pagination limits apply. A failed page refuses
the connection's entire new catalogue. Automatic retries keep their 30-second
backoff, and the button permits an explicit retry. Status is the last discovery
result, not a continuous health guarantee.

Tool selection is saved per conversation. Each tool shows whether approval is
required. Refreshing connections preserves unsaved chat settings, draft text and
selected tools that became unavailable. Unavailable selections can be removed.
Approval policy comes from the operator configuration and is not editable in the
browser. Workspace reads run directly, while writes and commands need approval.
Changing configured endpoints or credentials still requires a companion restart.
The catalogue exposes at most 128 remote tools. A bounded history of 4096 remote
names keeps aliases stable across refreshes in one process; reaching that limit
requires restarting the companion before discovering renamed tools.

A model call that requests tools ends before those tools execute. Each call
has separate arguments, status, duration and results. Activity entries open the
matching call's existing arguments and result in the conversation. Pending
approval opens the arguments for review. **Allow once** approves the displayed
call. The optional checkbox changes the action to **Allow for this reply**,
which also allows later calls to the same tool with potentially different
arguments. **Deny this call** declines just the displayed call. Approval is
required by default. An operator's `approval: false` allows its selected tools
automatically.
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

The live display shows state and elapsed time. When the upstream uses the
opening completion `id` for its health row, Chat matches that exact ID in the existing Monitor
feed to show prompt work and measured prompt or generation tokens per second.
Both views share `request_rates.js` and one collector. No extra upstream polling
is started. Rates need two distinct samples and include time between scheduler
snapshots, not just kernel execution. Repeated, missing, failed or stale samples
show no current rate. A measured zero remains zero. Leaving Chat, hiding the tab
or finishing the model call closes Chat's subscription. Older upstreams keep
state and elapsed time without live rates. No rates are inferred from text chunks.
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
retained in memory. Remote scripts, packages and fonts are blocked.

Workspace previews resolve relative stylesheet links and classic `.js` scripts
against the HTML file's path. For example, `pages/index.html` can link
`../style/site.css` or `./boot.js`. Save those assets before previewing. A
missing eligible asset yields an error with its resolved path. The assembled
document, including the encoded assets, is limited to 1 MiB. This covers direct
CSS and JavaScript links. Module imports, CSS imports, secondary assets,
root-relative links and routing need a feature beyond this preview.

Built-in Workspace read and write results can open their saved file directly.
Command results can open the matching retained job output. File links use the
same unsaved-edit confirmation as the file list. If a file was deleted or a
job is no longer retained, Workspace shows its unavailability without
replacing the active editor or choosing a different job for the link.
List results and results without target metadata still open Workspace.
These actions only navigate. Select Workspace limits to see storage and
execution limits.
Workspace owns managed UTF-8 project files, a source editor, unified diffs,
command output and file checkpoints. It never opens arbitrary host paths.
From the file editor, Tab moves focus to the next control and Shift+Tab moves
it to the previous control. Use Ctrl+S or Cmd+S to save.
Each file is limited to 1 MiB, the project to 8 MiB and 256 files. Revision checks
refuse stale edits. You can keep typing while Save is pending; later edits stay
marked Unsaved. One save runs at a time. Switching files does not let an older
save mark the new editor content as saved. A successful write retains its
accepted revision even if refreshing the file list fails.
Up to 20 checkpoints and 20 command records are retained.
Restoring a checkpoint first saves the current files. These are file snapshots,
not process snapshots or a promise to undo actions made by external MCP tools.
Creating or deleting a file, naming or restoring a checkpoint, editing a generated
file and discarding unsaved edits use the same accessible action dialog. Cancelling before submission
does not write, delete or discard files; revision checks still reject stale writes.
Delayed file reads never replace a newer selection. If you type while another
file loads, opening it asks again before discarding those new edits. Saving or
starting another editor transition invalidates the older pending open.

The Command runner starts one command at a time and shows its output and history.
It is not an interactive shell. Sign-in in App settings describes access to the
companion, separate from the Workspace execution configuration.

Optional execution requires Docker and an existing Linux image with `/bin/sh`,
`tar` and `sleep`, plus the languages or tools the operator needs:

```text
python tools/chat_ui/server.py --sandbox-image python:3-slim
```

`--sandbox-docker-host ssh://user@host` selects a remote Docker service. The
Docker CLI uses the operating-system account running the companion; for an SSH
endpoint, SSH uses that account's configured identity and access to the remote
daemon ([SSH socket access](https://docs.docker.com/engine/security/protect-access/)).
Keep those credentials outside the workspace and treat the daemon as a separate
trust boundary. Docker warns that control of a rootful daemon can grant
host-root-equivalent access ([daemon attack surface](https://docs.docker.com/engine/security/));
a compromised companion process could use whatever daemon authority its account
has. Workspace commands use the fixed container options described below.

Each workspace command receives a disposable container with no host mounts, no
network, no capabilities, no privilege escalation, a read-only root filesystem,
a non-root user, one CPU, 256 MiB of memory and 64 processes. Project and
temporary storage are bounded tmpfs mounts. These controls constrain those
command containers; they do not restrict daemon requests from a compromised
companion process. The companion does not pull images. Commands time out after
120 seconds; the output view retains the last 128 Ki characters. The service
never substitutes host execution when Docker is unavailable.

For deployments that enable workspace execution, prefer a dedicated daemon host
to limit the impact of daemon access. Docker rootless mode can reduce daemon and
container privileges to those of its unprivileged host user ([rootless mode](https://docs.docker.com/engine/security/rootless/));
it is a mitigation, not a substitute for protecting companion access. The exact
rootless configuration has not been qualified for this companion.

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
silently inserting the whole project into the prompt.

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

## Identity

Chat and Monitor share the lowercase llmx wordmark in `tools/health_dashboard/`.
The companion sidebar uses a compact 76 by 30 px slot for the shared lettering.
`logo.svg` contains outlined lettering, with no font dependency or gradients.
Both interfaces use it as a CSS mask with an accessible label, so the letters
follow the selected theme. `favicon.svg` uses the wordmark's x, from mxxm, on a flat
charcoal tile for the browser tab. In-page headers use the wordmark alone. `icon-192.png` and `icon-512.png` export the standalone icon at app sizes.
Both servers expose these fixed asset paths.
