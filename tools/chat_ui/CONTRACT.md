# Chat companion contract

The Python standard-library service owns persisted conversations, branches, runs and tools. The static browser application owns presentation. All JSON APIs are same-origin under `/api`. Mutations require `X-Chat-Request: 1`, and JSON bodies use `Content-Type: application/json`. Errors use `{error: string}` with an HTTP error status.

## Routes

- `GET /api/session`: `{authenticated, password_required}`. All other APIs require a session when a password is configured.
- `POST /api/login` with `{password}` sets an HttpOnly session cookie. `POST /api/logout` clears it.
- `GET /api/config`: `{models: [{id}], tools: [{name, description, inputSchema, server, approval}], upstream_available, upstream_error, tools_error}`. No credentials are returned. A tool discovery failure does not disable a reachable model.
- `GET /api/chats`: `{chats: [{id,title,pinned,archived,folder,created,updated,active_leaf,settings}]}`.
- `POST /api/chats` with optional `{title,settings,folder,temporary}` creates and returns a chat. `temporary` defaults to false and must be a boolean.
- `GET /api/chats/{id}`: `{chat, messages: [...], runs: [...]}`. Messages include all branches. Follow `parent_id` backward from `chat.active_leaf` for the selected path.
- `PATCH /api/chats/{id}` accepts `title,pinned,archived,folder,active_leaf,settings`. Selecting a branch while a run is active is refused.
- `DELETE /api/chats/{id}` deletes an idle chat.
- `GET /api/chats/{id}/export`: versioned JSON download.
- `POST /api/import` with exported JSON creates a new chat with fresh identifiers.
- `POST /api/chats/{id}/runs` with `{content,attachments:[],parent_id,settings}` adds a user message then starts a run. Parent may be null for a root branch. With `{regenerate:true,parent_id: USER_MESSAGE_ID,settings}`, reuse that user message and start an alternative response. Returns `{run}`.
- `GET /api/runs/{id}/events?after=N`: SSE with monotonically increasing `id`, `event: update`, and JSON `{seq,kind,data,time}`. Reconnect sends `Last-Event-ID`. Persisted events replay after the cursor. The stream closes after its terminal state has been delivered. A browser disconnect does not cancel generation.
- `POST /api/runs/{id}/cancel` requests cancellation. It does not promise to undo an external tool action.
- `POST /api/runs/{id}/approve` with `{call_id,allow,remember:false}` resolves a pending tool call. Remember applies only to this run and tool.
- `POST /api/tokenize` with `{text}` returns the upstream response, for optional draft estimates only.

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

Settings: `{model,system,temperature,top_p,top_k,max_tokens,tools:[]}`. Omitted settings use documented companion defaults. Tools are selected by the exposed tool name. A model name selects only what the configured upstream actually serves.

Attachment: `{name,type,text}` for UTF-8 text, or `{name,type,data}` for PNG/JPEG base64 data URLs. The browser accepts text up to 2 MiB, images up to 8 MiB and 10 MiB total across at most eight files. It shows each file size and the combined size before send. The service validates the same decoded byte limits and stores attachments in the message. Other formats are refused explicitly. No URL fetching or hidden retrieval occurs.

Message: `{id,chat_id,parent_id,role,content,reasoning,attachments,tool_calls,tool_call_id,run_id,status,created,usage,timings,context}`. Optional fields may be absent. Tool call records have `{id,name,arguments,status,result,duration_ms,error}`. Status can be `pending`, `approval`, `running`, `completed`, `denied`, `failed`, `cancelled` or `interrupted`. Tool outputs remain independent messages for model history.

Run: `{id,chat_id,status,message_id,created,updated,error,last_seq}`. Run statuses are `running`, `approval`, `completed`, `cancelled`, `failed` and `interrupted`.

Events:
- `message`: complete message record, insert or replace by id.
- `delta`: `{message_id,content,reasoning}`, append supplied text only.
- `metrics`: `{message_id,usage,timings,context}`, merge available objects.
- `tool`: `{message_id,call}`, replace the matching call by id.
- `state`: full run record. Terminal states stop streaming.

A snapshot contains persisted text and the matching run cursor. Subscribe from `last_seq`, not zero, after loading that snapshot. Usage is stored once per model call, not added for each event. A turn can contain several model calls. Cached input is a subset of input. Live chunks are never counted as tokens. The UI shows measured server rates only when provided, and an elapsed timer while live. It must not invent per-request health rates.

## UI behavior

Sidebar: new chat, search across loaded conversations, compact anchored grouped conversation actions, pin/archive, folders, import/export and duplicate. Rename, folder and delete actions use accessible dialogs. Center: branch-aware messages, reasoning disclosure, safe Markdown/code/tables, tool rows with Open Workspace links for workspace tools, composer with text/image attachments. Inspector: activity, statistics and effective settings. Activity tool entries navigate by message and call identity to the existing tool details, opening and focusing the summary; narrow layouts close the inspector drawer first. Edit and retry preserve old branches. Message actions use outlined SVG icon controls with accessible names, hover and keyboard tooltips, copy confirmation and touch sizing on mobile. Reasoning disclosure nodes stay connected while SSE deltas update streamed content, preserving pointer interaction; reload resumes the same run through persisted SSE replay. Mobile uses drawers and stable controls. Errors and empty states are explicit. Use ASCII source and SVG icons, no external CDN, packages or fonts.

## Backup branch validation

Restore validates message parents and active branches before replacing the live
database. JSON and stored parent references must agree, parents and active leaves
must belong to the same conversation, and every message branch must be acyclic.
Valid branches may be stored in any row order. A refused backup leaves the current
conversations and event journal unchanged.

## Workspace snapshots

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
This does not provide persistent drafts when leaving a conversation.
