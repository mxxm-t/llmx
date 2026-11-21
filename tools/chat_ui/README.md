# llmx companion

The llmx companion brings Chat, Saved files, Monitor and Settings under one
address and session. Open **Saved files** to edit files, reuse projects or run
commands.
Conversations, branches, tool results and request metrics live in a local SQLite database. The companion uses
Python's standard library and buildless HTML, CSS and JavaScript. The base
companion runs without additional packages. PDF text import is an optional
browser add-on installed by the operator.

## Run

From the repository root, start llmx with a chat model. Replace `model.gguf`
with its path. This local example listens on `127.0.0.1:8080`.

```sh
./build/llmx serve model.gguf --no-auth
```

On Windows, a `build.bat` build uses `.\llmx.exe serve model.gguf --no-auth`.
Keep that terminal open and start the companion in a second terminal.

```sh
python tools/chat_ui/server.py --upstream http://127.0.0.1:8080
```

Open `http://127.0.0.1:8183` and send a message. If your model server is
already running, use its address for `--upstream` and skip the first command.
For a server that requires a key, add `--api-key-file PATH` to the companion
command. The file must contain one upstream key.

To try a tool with a model that supports tool calls, open **Conversation
settings** beside the model selector, enable `workspace_list` under **Tools
for this conversation**, and ask the model to list your saved files.
Tool names and arguments appear while the model prepares a call. Open the call
to inspect its incoming arguments, or refresh to resume from its saved state.
Preparing a call does not run it: approval and execution begin only after the
model stream completes. An interrupted call keeps its partial arguments and is
not retried automatically.

## Work on files

When a tool reads or writes a workspace file, its path appears in the conversation.
Choose **Attach to draft** to include the current saved text in your follow-up.
**Open file** on a read takes you to the editor. On a write it opens the saved file for inspection.
HTML opens in Preview. Choose **Source** to read the saved text, **Open in editor** to work on the same path, or **Download** to keep a copy. The editor opens the latest saved version or returns to unsaved edits you already have for that file.
If a write hits a revision conflict, **Open saved copy** opens the version currently in the workspace.
If that path has no saved file, the conversation shows a message beside the filename.

The **Project** button beside the paperclip opens a saved project directly.
In Chat, open **Attach** and choose **From this device**, **Saved files** or
**Saved project**. You can find files by name or folder and select several at
once. Adding them keeps you in the current draft. Sending the message shares
a copy of their saved text.

Use **Browse saved files** in the conversation settings to open the same file picker.
Open **Saved files** in the sidebar for a working copy you can edit and reuse
across conversations. This is the companion's workspace.
Open a code block and choose **Edit in workspace** to name the file. If the path already contains different text, choose **Edit replacement** to open the new text in the editor. Choose **Review changes** to compare an edited draft with the current saved file. Close the comparison to keep editing. Only **Save** writes the draft.
For example, save a generated web page there, edit it, preview it with its
saved stylesheet, then attach the edited HTML to a draft.

To bring a local project into **Saved files**, choose **Import**. Pick individual
text files or a folder and review the paths before confirming. File picks start
checked. For a folder, check the files to include. Hidden entries and common
dependency directories are omitted from folder picks. Folder imports keep paths relative to the selected folder, so HTML, CSS and JavaScript can
keep their existing relative links. Checked files are saved together. A name
collision or a stale workspace revision leaves the whole batch unsaved.
Importing files does not attach them to a conversation.

Start in **Files** to create or edit a file. **Find a file** filters saved paths
by filename or folder while keeping your editor open. Use **Projects** to group files
with instructions, **Commands** to run against saved files in a configured
sandbox, or **History** to compare and restore checkpoints.
Switching between these sections keeps the current editor buffer.

Successful Workspace tool results can open their file or command output
directly. Select a file in Workspace to edit it, or create one with New file.
Preview HTML renders the current editor buffer, including unsaved edits, in an
isolated browser frame. It uses saved CSS and classic JavaScript linked directly
from that document. The preview lists the saved Workspace revision and included
assets. Other files offer View source. Close the viewer to return to the editor.

Attach to Chat adds the current editor buffer, including unsaved edits, to the
current draft and returns focus to the composer. The attachment can be removed
before sending. Editor attachments preserve the text, including a leading UTF-8 byte-order mark and existing line endings. A workspace file named `notes.pdf` attaches as text. Pick a PDF from your device to select document pages. A path such as `src/index.js` appears as `src:index.js` to
distinguish it from another file with the same basename. The normal text
attachment limits and temporary chat behavior apply.

To reuse files in another chat, open **Projects** in Workspace. Give the
project a name, select up to eight saved files and optionally enter instructions.
In Chat, choose **Attach > Saved project**. If the
picker is empty, choose **Manage saved projects** to reach **New project**
with your draft and editor intact. For a saved project, review the file paths,
sizes and saved revision. Check the instructions box if you want them to
replace this conversation's system instructions, then choose **Add to draft**.
You can remove attachments before sending.

In **Saved files > Projects**, choose **Download ZIP** beside a project to keep
a local copy. The archive contains its selected files at their saved relative
paths. Save your editor changes first if you want them in the download.
Project instructions stay in the companion. If a selected file has been deleted,
restore it or remove its path from the project before downloading.

Each use takes a snapshot. Editing a saved file later affects the next use.
The draft and previously sent messages retain their copies. Attaching the same workspace file twice keeps one copy, including when you attach directly from the editor. If the draft already contains a different version
of a selected file, the editor offers **Replace in draft**. This updates the existing attachment in place, including when all eight slots are in use. It uses your current edit and keeps the saved file intact. In the file or project picker, remove the old attachment before adding a different version.
Temporary chats can use a project explicitly. Their drafts stay in memory.
Project definitions live with Workspace files in the companion data directory.
The conversation database backup covers the copies in saved conversations.

Create checkpoint captures saved workspace files. Open Checkpoint history to
review differences or restore a checkpoint. The command runner starts from
saved workspace files and creates a checkpoint before a command runs. Back to
Chat focuses the composer for an unsent draft or the title for an open conversation. It keeps the editor buffer for the
current page.

## Storage and deployment

`--upstream` accepts an origin or a URL ending in
`/v1`. By default, data is in `~/.llmx-chat/chat.sqlite3`. `--data-dir PATH`
selects another directory. Settings can download an online SQLite chat backup
and restore one while replies are idle. Restore replaces conversations, and
accepts backups up to 256 MiB as a streamed binary upload. Broken message branches,
including cycles and references to other conversations, are refused before the
active database is replaced. Back up the entire data
directory with the companion stopped to include workspace files and checkpoints.
Do not point two companion processes at the same database.

For access from another machine, supply an owner password file and the exact
browser-facing origin:

```text
python tools/chat_ui/server.py --upstream http://127.0.0.1:8080 --host 0.0.0.0 --port 8183 --password-file /run/secrets/chat-password --public-origin https://chat.example.com
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

## Optional PDF text reader

Open **Settings > Optional features** in the sidebar. Select **Install** next
to **PDF text import**. The companion downloads the pinned release, verifies
its SHA-256 and enables PDF attachments when installation finishes. You can
return to Chat during the download. Your current draft stays in place.

The operator can also install from the repository root:

```sh
python tools/chat_ui/install_pdf_reader.py
```

Both methods install the same files under `tools/chat_ui/.pdf-reader/` in
this checkout. The companion service needs write access there for a Settings
installation. For an offline installation, pass `--archive PATH` to the
command with the official release ZIP. Refresh Chat after a command-line
installation. The regular configuration response tells the
attachment picker whether the reader is present. Opening Chat or using plain
text does not fetch PDF scripts, fonts or character maps.

Choose a PDF with **Attach**, select page numbers such as `1, 3-5`, then
choose **Review selected pages**. **Add text to draft** attaches the displayed
text with page numbers. You can remove it or edit your message before sending.
The browser reads the PDF locally using assets from this companion. Only the
reviewed text enters the conversation under the PDF's original filename. In a sent
message, select the filename to read or copy the exact excerpt and its page numbers.
Download saves the excerpt as a text file. Image-only scans need OCR before import.

Limits per import are 20 MiB, 200 document pages, 12 selected pages and 2 MiB
of extracted text. Password-protected documents need an unlocked copy.

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

Select the llmx wordmark to return to the Chat landing page. This keeps the
draft of the conversation you left and adds no empty row to chat history.
Ending a temporary chat still uses its existing confirmation.

**Unsent drafts** lists other saved conversations with unfinished messages in
this browser tab. Each row shows the conversation title and a text excerpt or
attachment name. Select a row to resume typing with its attachments and edit
target restored. The draft in **New conversation** stays available when you
return. The list appears only when a draft is available and excludes archived
and temporary conversations.

The sidebar places Chat, Saved files, Monitor and Settings under the llmx
wordmark. The current view is highlighted. Below New conversation, saved chats
are grouped by pin and folder. Select a group heading to collapse or expand
it. The archive keeps its own open groups.
Search shows matches in one list, including conversations in collapsed groups. On a phone, the chat landing
page also lists up to three recently updated conversations with messages. Archived
conversations stay in the archive. Select a recent conversation to reopen its
active branch. **Attach from workspace** sits beside the composer on wide
screens and below it on narrow screens. It lists up to three saved files or
projects by name. Select a file to open the picker with that path
checked, or a project to review its current files and optional instructions.
**Browse saved files** and **Browse projects** open the full lists. When the
workspace has no projects, **Create a project** opens the project editor.
Reviewing a choice leaves your draft intact until you confirm. Temporary chat
keeps **Browse saved files** available and loads the list when you open it.

- Unsent text, attachments and edit targets stay with each ordinary conversation
  when you switch chats or select the same chat again. They are saved in this
  browser tab's session storage for refresh recovery, not in chat backups.
  Closing the tab ends that session. If draft storage writes fail or reach the
  browser's quota, a notice explains that affected drafts remain in this page
  only; switching chats still preserves them, but refreshing can discard them.
  Sending clears only the submitted draft, preserving newer typing. Signing out
  clears this tab's ordinary drafts. Temporary chat drafts stay in memory and
  are discarded on refresh, as the composer explains.
- Ctrl+K (Command+K on macOS) activates **New conversation** from the companion navigation and focuses the composer.
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
- Scroll up to read an earlier turn. Once chat stops following the reply, **Jump to latest** appears below the message box. Select it to resume following.
- Select a passage in an answer and choose **Quote in reply** below it. With no selection, the action quotes the whole answer. The quote goes at the end of your draft, ready for a follow-up question.
- The reading bar shows which turn is in view. Choose **Contents** for the current
  branch. **Prompts** lists your messages in order and shows filenames for file-only
  prompts. **Reply headings** lists the sections in formatted answers. Repeated
  headings within a turn show their occurrence and a short excerpt where available. Select a
  heading to jump directly to that section.
- In either index, Up and Down move between entries, Home and End reach the first
  or last entry, and Enter jumps. Longer indexes have pages of 25 entries.
  **Conversation actions > Conversation outline** opens the same index.
- Choose **Find** in the reading bar or press Ctrl+Shift+F (Command+Shift+F on
  macOS). Search message text and attachment names on the current branch. Choosing
  a result highlights its first visible match. Use **Previous match** or **Next match**
  to step through occurrences in reading order, including repeated text in the same
  answer. F3 advances, Shift+F3 goes back, and Escape closes the search. Select
  the search term to edit it. The bar updates as replies stream. A filename match
  on an image outlines the image. Browsers without custom text highlights outline
  the matching block. A result marked **Markdown** jumps to the message whose source matches.
- **Conversation actions > Download Markdown** saves the current branch's prompt and reply text in reading order. It leaves out tool payloads, reasoning, attachments and unsent drafts. Use **Export JSON** for a backup with branches.
- The header actions menu appears when a saved or temporary conversation is open.
  Each sidebar row has a compact anchored actions menu for rename, pin/unpin,
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
- Wide conversations have a speaker and time column beside the messages. On a
  narrow screen those labels appear above the text. An inset rule marks each
  prompt. Copy and Edit appear below it. Their icons share the styling, keyboard focus and touch targets
  used by Retry, Statistics and the branch arrows under a reply. Statistics
  stays highlighted while its panel is open.
  Assistant prose uses shorter lines on wide screens, while code and tables
  retain the space they need. Attached files appear inside the draft box with their sizes.
  The draft grows as its text wraps when the available width changes.
  Consecutive model calls in a tool-assisted reply share a heading and a side
  rule. Each call keeps its own actions and expandable details. One statistics
  icon covers the reply, including its tool-related model calls.
- Reasoning is displayed only when the upstream supplies `reasoning_content`.
  You can open or close it while the reply is being generated. Reloading
  reconnects to the same reply without cancelling or restarting generation.
  If a completed reply contains reasoning and no answer, Chat shows a notice
  next to Retry. The notice uses the model's finish reason when available.
  The renderer supports basic Markdown, nested lists with preserved starting
  numbers, code blocks, tables and safe links. Tables honor left, center and
  right column alignment. Use `\|` for a literal pipe inside a cell. Open a code block and choose
  **Wrap lines** to read long lines without scrolling sideways. An image in a Markdown reply
  appears as a link with its description, or **Open image** if empty. Open the
  link to view the image in a new tab. Indented paragraphs and code stay
  inside their list item. List nesting is bounded at eight levels; deeper text
  remains visible as literal content. Unchanged leading Markdown blocks remain
  connected during streaming, preserving their selection and code controls.
  Later text can stream after a completed code sample without highlighting the
  sample again when a blank line separates them.
  The changing suffix is rebuilt, so selection inside it may be lost.
  A focused message action stays focused across message updates while its
  action and target remain available.
  It does not execute model HTML or JavaScript.
- Drop files onto the conversation or choose them from the attachment menu.
  Attach up to eight files, including UTF-8 text/code, PNG/JPEG images and
  selected PDF text when the [PDF reader](#optional-pdf-text-reader) is installed. Text is included
  in full with its file name. Select a filename in your draft to inspect the copy that will be sent. Text opens as read-only source. A PNG or JPEG opens fitted to the viewer, with its dimensions and a download of the original image. If the image was scaled down, choose **Actual size** to inspect its detail. Scroll or swipe to pan. Choose **Actual size** again to fit the image. Choose **Attach > From this conversation** to find an earlier attachment by filename and turn on the current branch. Open a file there or in a sent message and choose **Add to draft** to reuse that exact version in a follow-up. Adding the same file twice keeps one copy. If your draft contains a different version with the same name, remove it first. Close it to return to the file, or choose Remove beside a draft file to leave it out. Limits are
  2 MiB per text file, 8 MiB per image and 10 MiB total decoded attachments.
  Hover over Attach or focus it with the keyboard for the file types. The
  Attachments section in Conversation Settings shows the limits on any device.
  A read that finishes after switching conversations is discarded with a notice.
  The upstream model and server must support images. Image-bearing prompts are
  checked through llmx tokenization before a reply starts, including images in
  conversation history. A rejection keeps your draft and attachments.
  Older saved image turns refused by the chat template before any reply or tool
  call stay visible but are excluded from later prompts. Retrying that image turn
  still includes its images. Accepted images and partial replies remain in context.
  Image-only PDFs need OCR before import. The companion has no built-in OCR,
  audio input, LaTeX renderer, executable chat artifacts or knowledge-base retrieval.
- A companion process restart marks unfinished runs interrupted. It never
  retries a tool automatically after losing its outcome.

## Reply limits and context

New conversations use No additional cap for replies. The companion omits
`max_tokens`, letting the model stop normally or reach the server's remaining
request capacity. Choose Custom token limit to set a positive output-token cap.
Reasoning tokens count toward output. Existing saved numeric caps remain in
place until changed. Changing reply mode does not resize the server's context.

After a tool result, llmx can refuse a continuation whose full prompt fits but whose reply reservation exceeds the request limit. The companion then retries once with exactly the remaining reply room, keeping the prompt and tool results intact and retaining the saved reply setting for future calls. If the prompt leaves no room or the upstream has already streamed output, the failure is reported normally.

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
when needed. The sliders beside the model selector open conversation settings.
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

Code blocks offer highlighting, copy, download and a source viewer. To show
a Markdown example that contains a triple-backtick block, wrap the example
in four backticks or use a tilde fence. The inner fence stays in the copied
source. HTML previews
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
Opening a saved file preserves its line endings. Edits to LF, CRLF and CR files keep that convention. In mixed files, newly inserted lines use the first line ending. Review, download, attach and save all use the same editor text.

Before the first save, **Discard** closes the unsaved file. After a file is saved, **Delete** removes its saved copy. Each action asks for confirmation.
From the file editor, Tab moves focus to the next control and Shift+Tab moves
it to the previous control. Use Ctrl+S or Cmd+S to save.
Each file is limited to 1 MiB, the project to 8 MiB and 256 files. Revision checks
refuse stale edits. You can keep typing while Save is pending; later edits stay
marked Unsaved. One save runs at a time. Switching files does not let an older
save mark the new editor content as saved. A successful write retains its
accepted revision even if refreshing the file list fails.
In checkpoint history, **View diff** compares the checkpoint with your current saved files. It lists added and deleted empty files by name and marks lines that have no final newline.
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
approval. A write takes a path and the new text. The companion tracks the saved file through the reply, including the wait for approval. Editing another file does not invalidate the write. A write that changes the file creates a checkpoint. An exact repeat
returns `already_saved` with the current revision. For a conflicting edit,
read the current file before retrying. Reading returns bounded lines rather than
silently inserting the whole project into the prompt.

## Ownership and checks

`store.py` owns persistence and replay. `engine.py` owns model/tool turns and
permissions. `upstream.py` owns bounded compatible HTTP transport. `tools.py`
owns the MCP connection. `server.py` owns HTTP routes and owner authentication.
`workspace.py` owns managed files and isolated execution. `monitor.py` connects
the existing dashboard owner to the workspace without another collector.
`pdf_reader.py` owns the optional asset layout. `pdf_attachment.js` owns
page selection and extraction in the browser.
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
The browser harness gives Chrome 60 seconds on Windows and 30 seconds elsewhere to start and connect. A page-discovery timeout reports its local endpoint, last result and process diagnostics.
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
