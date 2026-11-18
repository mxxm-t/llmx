# Chat companion contract

The Python standard-library service owns persisted conversations, branches, runs and tools. The static browser application owns presentation. All JSON APIs are same-origin under `/api`. Mutations require `X-Chat-Request: 1`, and JSON bodies use `Content-Type: application/json`. Errors use `{error: string}` with an HTTP error status.

## Routes

- `GET /api/session`: `{authenticated, password_required}`. All other APIs require a session when a password is configured.
- `POST /api/login` with `{password}` sets an HttpOnly session cookie. `POST /api/logout` clears it.
- `GET /api/config`: `{models: [{id, context_length?}], settings_defaults, tools: [{name, description, inputSchema, server, approval}], upstream_available, upstream_error, tools_error}`. No credentials are returned. A tool discovery failure does not disable a reachable model.
- `GET /api/chats`: `{chats: [{id,title,pinned,archived,folder,created,updated,active_leaf,settings}]}`.
- `POST /api/chats/search` with `{query,archived}` returns `{chats: [{id,title,folder,pinned,archived,updated}]}` in pinned-first, newest-first order. The query is trimmed and must contain 1 to 256 characters. `archived` must be a boolean. Search matches a case-folded literal substring of the title, folder or user and assistant message content across saved branches in that archive view. It omits temporary chats, reasoning, attachments, tool payloads, settings and events. A search holds a read-only snapshot on its own WAL connection. Another search receives HTTP 429 while that reader is active.
- `POST /api/chats` with optional `{title,settings,folder,temporary}` creates and returns a chat. `temporary` defaults to false and must be a boolean.
- `GET /api/chats/{id}`: `{chat, messages: [...], runs: [...]}`. Messages include all branches. Follow `parent_id` backward from `chat.active_leaf` for the selected path.
- `PATCH /api/chats/{id}` accepts `title,pinned,archived,folder,active_leaf,settings`. Selecting a branch while a run is active is refused.
- `DELETE /api/chats/{id}` deletes an idle chat.
- `GET /api/chats/{id}/export`: versioned JSON download.
- `POST /api/import` with exported JSON creates a new chat with fresh identifiers.
- `POST /api/chats/{id}/runs` with `{content,attachments:[],parent_id,settings}` adds a user message then starts a run. Parent may be null for a root branch. With `{regenerate:true,parent_id: USER_MESSAGE_ID,settings}`, reuse that user message and start an alternative response. Returns `{run}`.
- `GET /api/runs/{id}/events?after=N`: SSE with monotonically increasing `id`, `event: update`, and JSON `{seq,kind,data,time}`. Reconnect sends `Last-Event-ID`. Persisted events replay after the cursor. The stream closes after its terminal state has been delivered. A browser disconnect does not cancel generation.
- `POST /api/runs/{id}/cancel` requests cancellation. It does not promise to undo an external tool action.
- `POST /api/runs/{id}/approve` with `{call_id,allow,remember:false}` resolves a pending tool call. Remembered allowance covers later calls to the same tool in this run, including different arguments. Denial resolves the specified call.
- `POST /api/tokenize` with `{text}` returns the upstream response, for optional draft estimates only.

An SSE `delta` carries `message_id` and appended `content`/`reasoning` text. It may
also carry `tool_calls`: fragments with stable internal `id`, upstream `index`,
and appended `name`/`arguments` strings. The store applies each fragment to the
message and advances the run cursor in the same transaction, so a snapshot plus
events after its cursor restores preparation without duplicated text. The browser
shows these calls as `preparing`; they cannot be approved or executed yet.
The final assistant `message` replaces them with complete calls before the normal
approval/execution flow. Cancellation, broken streams and service restart retain
the received arguments with a terminal status. Fragments share the existing text
coalescer and journal only newly received bytes; saved message snapshots still
use the existing whole-message persistence path.

The prompt-history owner omits an older image user turn and its adjacent empty
failed assistant only when the run records a chat-template refusal. The saved
messages and branches remain unchanged. Accepted images, partial replies,
reasoning and tool calls stay in prompt history. Regenerating from the image user
message still submits its images and requires image admission.

## Tool connections

`GET /api/tools` returns cached `{tools, tool_servers, tools_error}` without
probing remote endpoints. Initial `/api/config` discovery includes the same fields.
Each tool has a `connection` string. A server record has `id`, `name`, `kind`
(`builtin` or `mcp`), `status` (`unchecked`, `checking`, `ready` or `error`),
`tool_count`, `approval`, `checked_at`, `duration_ms` and `error`. Times are Unix
seconds and milliseconds. Unmeasured values are null. Workspace has mixed approval
policy (`approval: null`), with the effective policy on each tool. Counts refer to
tools exposed by this companion, not an unlimited remote inventory.

`POST /api/tools/test` accepts only `{server: configured_id}` and returns the same
snapshot after discovery. Unknown IDs return 404, malformed bodies 400, and a
concurrent discovery check 409. The normal authentication, origin and mutation
header checks apply. Built-in tools have no remote connection to test. This route
accepts no URL or credential fields and never invokes `tools/call`.
A discovered remote failure is a 200 response with that connection's error state.
Request-level errors do not imply that the remote server itself failed.
Other connections and the built-in catalogue remain available. Refreshing tool
metadata does not modify conversation settings, stored messages or running tools.

## Admission

For a run with a nonempty `settings.model`, Application checks the current
upstream `/models` catalogue before starting the engine or storing a message.
Model catalogues are fetched fresh for both configuration and admission. The
ten-second cache covers the other configuration fields. Admission neither waits
for that cache nor rediscovers tools. A missing choice returns 409, and a failed
catalogue fetch returns 503. Both replies include `code: "model_unavailable"`, `error`,
`models`, `upstream_available` and `upstream_error`. A successful checked run
returns those three catalogue fields alongside `run`, without an error code.
Empty default-model calls retain the ordinary `{run}` response.

The browser consumes these snapshots without resetting the composer. A fresh
catalogue marks a missing selection unavailable. On a catalogue failure it
retains the last known choices for retry. This is a check at admission, not an
atomic model-routing guarantee across an upstream restart.

Login JSON is limited to 16 KiB, checked from Content-Length before reading the
body. Other JSON routes retain their 16 MiB limit. Failed-login records use the
socket peer address, never forwarded headers, with ten failures per minute and
at most 1024 active addresses. A full record table refuses a new address rather
than evicting an address whose failures still count.

The transport admits 64 handler connections, including SSE, before starting
threads. Excess sockets close without an HTTP response. Slots are released
when handlers finish or thread creation fails. Non-loopback deployment requires
a password and an HTTPS public origin, validated before binding the socket.

## Temporary conversation lifetime

`config.temporary` advertises `enabled`, `ttl_seconds`, `max_chats` and `tools`.
Temporary chats and runs use `tmp_` identifiers and the same conversation routes,
store implementation and generation engine. `TemporaryChats` owns their separate
in-memory SQLite store, capacity and expiry; no temporary records enter the durable
store, history listing or backup. The browser keeps only the current identifier in
session storage. A missing or expired identifier returns 404.

Temporary chats return `temporary: true`. Creation, settings changes and runs refuse
a nonempty tool selection; the temporary engine has no callable tools. Explicit
export is supported. Importing an export intentionally creates a normal saved chat.
Idle records expire after 3600 seconds without API access, checked every minute;
active runs refresh that lifetime. DELETE requires an idle chat. Server shutdown
cancels active temporary runs and closes their memory store.

## Workspace and monitor routes

These routes share the chat session and mutation-header rules.

- `GET /monitor/view`: existing dashboard renderer with shared theme and embedded navigation.
- `GET /theme.css`: public, static visual tokens shared with the standalone dashboard.
- `GET /api/monitor/history?after=N` and `GET /api/monitor/stream`: existing collector snapshot and SSE history protocol. One collector serves all clients. A stream ends when the session expires or is revoked.
- `POST /api/previews` with `{html}` returns `{url}` for an authenticated opaque-origin preview document. Documents are at most 1 MiB and expire after 30 minutes; at most 16 are retained. They never execute on the host.
- `GET /api/backup`: online SQLite chat backup download. It does not include workspace files.
- `POST /api/restore` with a binary SQLite body and `Content-Type: application/vnd.sqlite3`: validates and replaces the idle chat database. The upload is streamed to a temporary file and limited to 256 MiB, advertised as `config.backup_max_bytes`. The caller reloads its view afterward. This binary route still requires the normal mutation header.
- `GET /api/workspace`: `{revision,files:[{path,size}],checkpoints:[{id,label,created}],jobs:[...],execution:{enabled,image,description,limits}}`. Execution enabled means configured, not a successful Docker health check. A failed container start is reported as a failed job.
- `GET /api/workspace/file?path=PATH`: `{path,content,revision}`.
- `POST /api/workspace/file` with `{path,content,revision}` or `/api/workspace/delete` with `{path,revision}`: revision-checked edit returning the new snapshot.
- `POST /api/workspace/checkpoints` with `{label}`: checkpoint and return snapshot. `POST /api/workspace/checkpoints/ID/restore` with `{revision}` restores after first checkpointing the current files.
- `GET /api/workspace/diff?checkpoint=ID`: `{diff}` with a unified text diff.
- `POST /api/workspace/jobs` with `{command}`: `{job}`. `GET /api/workspace/jobs/ID` reads it; `POST /api/workspace/jobs/ID/cancel` with `{}` requests cancellation.

A job contains `id,command,status,output,output_truncated,exit_code,error,created`.
States are `running`, `completed`, `failed`, `cancelled`. Output is bounded tail
text; the truncation flag must be shown. Browser job polling runs only while a
job is active and the Workspace view is open. One command runs at a time per
managed workspace. It blocks file edits until its files return or it is cancelled.
Container files return only after all entries pass validation. Cancellation or
an invalid returned file leaves the saved project unchanged.

## Data

Settings: `{model,system,temperature,top_p,top_k,max_tokens,tools:[]}`. Omitted settings use the companion defaults returned in `settings_defaults`. `max_tokens` defaults to null, meaning no additional client cap; the engine omits that field from the upstream request. A positive integer sets an explicit reply cap. Null is accepted only for this numeric setting, and existing numeric chat settings stay unchanged. Tools are selected by the exposed tool name. A model name selects only what the configured upstream actually serves.

The optional positive integer `context_length` is forwarded from `/models`.
The browser also reads `latest.server.request_tokens` through the existing
monitor stream while Settings is visible, requiring a successful recent sample
whose model matches the selection. It presents model capacity and request
capacity separately, and does not use stale metadata to clamp saved values.
No extra upstream health fetch is added to model admission or configuration.

Attachment: `{name,type,text}` for UTF-8 text, or `{name,type,data}` for PNG/JPEG base64 data URLs. The browser accepts text up to 2 MiB, images up to 8 MiB and 10 MiB total across at most eight files. It shows each file size and the combined size before send. The service validates the same decoded byte limits and stores attachments in the message. Other formats are refused explicitly. No URL fetching or hidden retrieval occurs.

Message: `{id,chat_id,parent_id,role,content,reasoning,attachments,tool_calls,tool_call_id,run_id,status,created,usage,timings,context,request}`. Optional fields may be absent. Tool call records have `{id,name,arguments,status,result,duration_ms,error}`. Status can be `pending`, `approval`, `running`, `completed`, `denied`, `failed`, `cancelled` or `interrupted`. Tool outputs remain independent messages for model history.

The renderer opens tool arguments when the call first requires approval or a
pending approval is loaded. When a streamed tool update changes a focused call,
the renderer keeps its details and summary connected while reconciling approval
controls to the new state. Completion removes the approval controls.

Run: `{id,chat_id,status,message_id,created,updated,error,last_seq}`. Run statuses are `running`, `approval`, `completed`, `cancelled`, `failed` and `interrupted`.

Events:
- `message`: complete message record, insert or replace by id.
- `delta`: `{message_id,content,reasoning}`, append supplied text only.
- `metrics`: `{message_id,usage,timings,context,request}`, merge available objects.
- `tool`: `{message_id,call}`, replace the matching call by id.
- `state`: full run record. Terminal states stop streaming.

A snapshot contains persisted text and the matching run cursor. Subscribe from `last_seq`, not zero, after loading that snapshot. Usage is stored once per model call, not added for each event. A turn can contain several model calls. Cached input is a subset of input. Live chunks are never counted as tokens. The UI shows final rates from server timings and derives live rates only from matching health counters. It shows elapsed time while the run is active.

When the opening upstream event provides `id`, the companion stores
`request: {id}` on that model call's assistant message. The identity is an opaque
string of 1 to 128 visible ASCII characters, without spaces. Missing or malformed
metadata leaves generation usable without correlation. Later events cannot set
or replace it. Reload and event replay preserve the identity. Chat imports drop
it and model history never includes it. An upstream must use the same opaque key
in health for correlation. Chat joins it exactly against the existing monitor SSE
feed, using the shared dashboard counter-rate implementation. Repeated snapshots
do not provide a new rate. Missing rows, collection errors and stale data clear
the visible rates. Subscription lifetime follows the visible active model call,
including tool follow-ups and browser page restoration. Final metrics remain
separate from live rates.

## UI behavior

The Markdown owner tracks source line ranges for generated blocks against one
current source array per root. Streaming keeps the identical leading blocks in
place and rebuilds the changed suffix, including within reasoning. Retained
blocks keep selection, focused code controls and copy feedback. Code controls
read their rendered code text; they do not retain old parser buffers.

Sidebar: new chat, search across saved conversations in the selected archive view, compact anchored grouped conversation actions, pin/archive, folders, import/export and duplicate. Rename, folder and delete actions use accessible dialogs. Center: branch-aware messages, reasoning disclosure, safe Markdown/code/tables, tool rows with Open Workspace links for workspace tools, composer with text/image attachments. Inspector: latest reply activity, collapsed earlier replies, conversation totals and effective settings. A reply statistics icon previews grouped call metrics on hover or keyboard focus; clicking pins a snapshot for text selection and copying, with timing and source-data breakdowns per call. Activity tool entries navigate by message and call identity to the existing tool details, opening and focusing the summary; narrow layouts close the inspector drawer first. Edit and retry preserve old branches. Message actions use outlined SVG icon controls with accessible names, hover and keyboard tooltips, copy confirmation and touch sizing on mobile. Reasoning disclosure nodes stay connected while SSE deltas update streamed content, preserving pointer interaction; reload resumes the same run through persisted SSE replay. Mobile uses drawers and stable controls. Errors and empty states are explicit. Use ASCII source and SVG icons, no external CDN, packages or fonts.

Adjacent assistant messages on the selected branch share a visual header only
when their nonempty run identities match. Continuations retain their own article
label, actions, reasoning, tool details and per-call metrics. Branch changes
recompute this presentation even when message contents are unchanged. Imports
without run identities and separate replies keep separate headers.

## Backup branch validation

Restore validates message parents and active branches before replacing the live
database. JSON and stored parent references must agree, parents and active leaves
must belong to the same conversation, and every message branch must be acyclic.
Valid branches may be stored in any row order. A refused backup leaves the current
conversations and event journal unchanged.

## Workspace editor

The file editor uses native Tab and Shift+Tab focus movement and preserves the
buffer when focus leaves. Ctrl+S or Cmd+S invokes the Workspace Save action
when available and no dialog is open.

## Workspace snapshots

File reads belong to a selection generation owned by `workspace.js`. A newer
editor transition invalidates an older read, including its errors. Input typed
during a read requires fresh discard confirmation before the result is applied.
Metadata refresh alone does not replace the editor or invalidate its selection.

Save captures its path, content and base revision. Only a matching editor base
receives the submitted content and returned revision as its saved baseline.
Later typing remains dirty, and only one save may be pending. A metadata refresh
does not advance the editor revision past the accepted write or erase its
success if that refresh fails.

Workspace refresh callers wait for an older refresh to finish before starting
their own. A completed mutation therefore receives a subsequent snapshot rather
than silently reusing a response requested before that mutation. Failure of the
older request does not cancel a queued refresh; its own caller reports the error.

## Image admission

Before persisting a reply, the engine sends an image-bearing prompt to the
upstream `/v1/tokenize` route with the same message content, selected tools and
template settings as generation. This includes images in follow-up and retry
history. A refusal leaves the conversation unchanged and keeps the browser draft
and attachments. The UI offers an image-capable server or removal of the images,
rather than exposing a template type error. Connection failures ask the user to
retry. Images are never dropped or converted into invented descriptions.

Text-only prompts do not make this extra request. Image admission requires the
llmx tokenize route; an upstream without that route cannot validate image input.
This checks input processing, not whether the model will describe an image well.

Saved template failures on an image-bearing branch also receive an actionable
banner when the chat is reopened. The original error remains in Activity for
diagnosis; stored history is not rewritten. Other failures retain their specific
messages.

## Pending Send requests

Send captures the selected conversation, branch parent, settings, text and files
before the request starts. Later composer edits stay in place when it completes.
Changing conversations prevents the old completion from clearing that view,
replacing its snapshot or subscribing to its stream again. A failure arriving
after navigation uses a toast rather than replacing the new conversation's banner.
The browser owns compose state per ordinary conversation, including text,
attachments and the edited message ID. Navigation preserves it; tab session
storage restores it after refresh. This is separate from the server database
and backups. Temporary chat content never enters that draft store. A draft
write failure keeps the in-memory value and shows a reload-loss notice rather
than claiming persistence. Signing out clears the tab's draft store.

Run admission saves its captured settings in both the run and the conversation.
The browser uses the live controls for Send and Retry. The explicit settings
PATCH saves the conversation without starting a run and can preserve an
unavailable model ID. Save is hidden before a conversation exists. The browser
allows one pending settings save or admission per chat, with the controls still
editable. A delayed Save updates only its target and leaves the current form
values in place. Opening a conversation reloads the server's saved settings.
Creating a new chat from an existing populated one uses the server defaults.

The accepted send removes only its matching draft record. Returning to the
originating conversation before admission completes still clears that unchanged
draft and reconnects the visible run; edits made meanwhile retain their own
record. Navigation requests replace the view only if they are still current,
and the previous conversation's stream stays connected until replacement.

## Pending Retry requests

Retry keeps its original conversation across admission. If that conversation
is visible when admission completes, including after leaving and returning,
the current view refreshes and reconnects to its reply. Navigation during that
refresh prevents the completion from replacing or scrolling the new view.
An offscreen retry continues in its own conversation without replacing another
conversation's snapshot or stream. Errors use the existing action notification.
