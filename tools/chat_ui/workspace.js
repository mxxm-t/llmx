/* The managed workspace API owns files, revision conflicts, jobs and history. */
(() => {
  'use strict';
  const {node, number} = window.ChatRender;
  const {api} = window.ChatShell;
  const view = document.getElementById('view-workbench');
  const state = {snapshot: null, path: '', original: '', revision: 0, activeJob: null, timer: null, loading: null, isNew: false, selection: 0, saving: false, jobSelection: 0};
  let viewLoad;
  const terminal = job => ['completed', 'failed', 'cancelled'].includes(job.status);
  function control(label, cls, listener) { const button = node('button', cls || 'secondary-button', label); button.type = 'button'; if (listener) button.addEventListener('click', () => Promise.resolve().then(listener).catch(showError)); return button; }
  const toolbar = node('div', 'view-toolbar'); const heading = node('div'); heading.append(node('h2', '', 'Managed workspace'), node('p', '', 'Files, checkpoints and isolated command execution.'));
  const headingActions = node('div', 'workspace-actions'); const revisionLabel = node('span', 'workspace-revision', 'Not loaded'); const refresh = control('Refresh', '', () => load()); const newFile = control('+ New file', 'primary-button', createFile); headingActions.append(revisionLabel, refresh, newFile); toolbar.append(heading, headingActions);
  const error = node('div', 'view-notice workspace-error'); error.setAttribute('role', 'alert'); error.tabIndex = -1; error.hidden = true;
  const limits = node('details', 'workspace-limits'), limitsText = node('p');
  limits.append(node('summary', '', 'Workspace limits'), limitsText); limits.hidden = true;
  const layout = node('div', 'workbench-layout');
  const filePanel = node('section', 'workspace-files'); filePanel.setAttribute('aria-label', 'Workspace files'); const fileHeading = node('div', 'workspace-section-heading'); fileHeading.append(node('h3', '', 'Files')); const fileCount = node('span', 'muted', '0'); fileHeading.append(fileCount); const fileList = node('nav', 'workspace-file-list'); fileList.setAttribute('aria-label', 'Files'); filePanel.append(fileHeading, fileList);
  const editorPanel = node('section', 'workspace-editor'); editorPanel.setAttribute('aria-label', 'File editor');
  const editorHeading = node('div', 'workspace-editor-heading'); const fileLabel = node('span', 'workspace-file-path', 'Select a file'); const dirtyLabel = node('span', 'unsaved-indicator', 'Unsaved'); dirtyLabel.hidden = true;
  const fileActions = node('div', 'workspace-editor-actions'); const preview = control('Open viewer', '', openArtifact); const download = control('Download', '', () => window.ChatArtifacts.download(editor.value, state.path, fileLanguage(state.path))); const save = control('Save', 'primary-button', saveFile); save.id = 'workspace-save'; const remove = control('Delete', 'text-button danger', deleteFile); fileActions.append(preview, download, remove, save); editorHeading.append(fileLabel, dirtyLabel, fileActions);
  const editor = node('textarea', 'workspace-code-editor'); editor.id = 'workspace-editor'; editor.spellcheck = false; editor.autocomplete = 'off'; editor.wrap = 'off'; editor.setAttribute('aria-label', 'File content'); editor.placeholder = 'Create a file or select one from the workspace.'; editor.disabled = true;
  const editorFooter = node('div', 'workspace-editor-footer'); const editorPosition = node('span', '', 'UTF-8'); const saveHint = node('span', '', 'Ctrl / Cmd + S to save'); editorFooter.append(editorPosition, saveHint); editorPanel.append(editorHeading, editor, editorFooter);
  const historyPanel = node('section', 'workspace-checkpoints'); historyPanel.setAttribute('aria-label', 'Checkpoints'); const checkpointHeading = node('div', 'workspace-section-heading'); checkpointHeading.append(node('h3', '', 'Checkpoints')); const checkpointAdd = control('+', 'icon', createCheckpoint); checkpointAdd.setAttribute('aria-label', 'Create checkpoint'); checkpointHeading.append(checkpointAdd); const checkpoints = node('div', 'workspace-checkpoint-list'); historyPanel.append(checkpointHeading, checkpoints);
  layout.append(filePanel, editorPanel, historyPanel);
  const terminalPanel = node('section', 'workspace-terminal'); terminalPanel.setAttribute('aria-label', 'Isolated execution');
  const terminalHeading = node('div', 'workspace-section-heading'); terminalHeading.append(node('h3', '', 'Command runner')); const executionBadge = node('span', 'badge', 'Checking execution'); terminalHeading.append(executionBadge);
  const executionNote = node('p', 'execution-description', 'Loading execution configuration.');
  const commandForm = node('form', 'workspace-command'); const command = node('input'); command.id = 'workspace-command'; command.type = 'text'; command.placeholder = 'Enter a command for the isolated container'; command.setAttribute('aria-label', 'Container command'); command.autocomplete = 'off'; command.maxLength = 8192;
  const run = node('button', 'primary-button', 'Run command'); run.type = 'submit'; run.id = 'workspace-run'; const cancel = control('Stop', 'secondary-button', cancelJob); cancel.hidden = true; commandForm.append(node('span', 'terminal-prompt', '>_'), command, run, cancel);
  const jobTabs = node('div', 'workspace-job-tabs'); jobTabs.setAttribute('aria-label', 'Command history');
  const output = node('pre', 'workspace-output', 'Command output appears here.'); output.id = 'workspace-output'; output.setAttribute('aria-label', 'Command output'); output.tabIndex = 0; const jobStatus = node('div', 'workspace-job-status'); jobStatus.setAttribute('role', 'status'); terminalPanel.append(terminalHeading, executionNote, commandForm, jobTabs, output, jobStatus);
  view.append(toolbar, error, limits, layout, terminalPanel);
  function showError(problem) { error.textContent = problem.message || String(problem); error.hidden = false; }
  function clearError() { error.hidden = true; error.textContent = ''; }
  function bytes(value) { const units = ['B', 'KiB', 'MiB', 'GiB']; let index = 0; while (value >= 1024 && index < units.length - 1) { value /= 1024; index++; } return number(value) + ' ' + units[index]; }
  function seconds(value) { return value >= 60 && value % 60 === 0 ? number(value / 60) + ' min' : number(value) + ' s'; }
  function dirty() { return !editor.disabled && (state.isNew || editor.value !== state.original); }
  async function canReplace() { return !dirty() || Boolean(await window.ChatShell.conversationForm({title: 'Discard unsaved changes?', description: 'Discard the unsaved changes in ' + state.path + '?', destructive: true, submit: 'Discard'})); }
  function fileLanguage(path) { const ext = path.split('.').pop().toLowerCase(); return ({htm: 'html', py: 'python', js: 'javascript', ts: 'typescript', sh: 'bash', rs: 'rust', md: 'markdown', patch: 'diff'})[ext] || ext; }
  function updateEditor() {
    dirtyLabel.hidden = !dirty(); const enabled = Boolean(state.path); for (const button of [preview, download, save, remove]) button.disabled = !enabled; save.disabled = !enabled || !dirty() || state.saving; save.setAttribute('aria-busy', String(state.saving));
    fileLabel.textContent = state.path || 'Select a file';
    const before = editor.value.slice(0, editor.selectionStart); const rows = before.split('\n'); editorPosition.textContent = enabled ? fileLanguage(state.path).toUpperCase() + ' / Ln ' + rows.length + ', Col ' + (rows.at(-1).length + 1) + ' / UTF-8' : 'UTF-8';
  }
  function renderFiles() {
    fileList.replaceChildren(); const files = state.snapshot?.files || []; fileCount.textContent = String(files.length);
    files.forEach(file => { const button = control('', 'workspace-file' + (file.path === state.path ? ' active' : ''), () => openFile(file.path)); button.title = file.path; button.append(node('span', 'workspace-file-icon', file.path.endsWith('.html') ? '<>' : '#'), node('span', 'workspace-file-name', file.path), node('span', 'workspace-file-size', number(file.size, ' B'))); if (file.path === state.path) button.setAttribute('aria-current', 'page'); fileList.append(button); });
    if (!files.length) fileList.append(node('p', 'workspace-empty', 'No files yet. Create a file or open a code block from a conversation.'));
  }
  function renderCheckpoints() {
    checkpoints.replaceChildren(); const list = state.snapshot?.checkpoints || [];
    list.slice().reverse().forEach(checkpoint => {
      const card = node('div', 'workspace-checkpoint'); const created = new Date(typeof checkpoint.created === 'number' ? checkpoint.created * 1000 : checkpoint.created);
      card.append(node('strong', '', checkpoint.label || 'Checkpoint'), node('small', '', isNaN(created) ? checkpoint.id.slice(0, 10) : created.toLocaleString([], {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'})));
      const actions = node('div', 'checkpoint-actions'); actions.append(control('View diff', 'text-button', () => viewDiff(checkpoint)), control('Restore', 'text-button', () => restoreCheckpoint(checkpoint))); card.append(actions); checkpoints.append(card);
    });
    if (!list.length) checkpoints.append(node('p', 'workspace-empty', 'Save a checkpoint before making changes. Commands checkpoint saved files automatically.'));
  }
  function selectJob(id) {
    state.activeJob = id; state.jobSelection++; renderJobs();
  }
  function renderJobs() {
    const jobs = state.snapshot?.jobs || []; jobTabs.replaceChildren();
    jobs.slice().reverse().forEach(job => { const button = control(job.command, 'workspace-job' + (job.id === state.activeJob ? ' active' : ''), () => selectJob(job.id)); button.title = job.command; button.append(node('span', 'badge' + (job.status === 'failed' ? ' error' : ''), job.status)); jobTabs.append(button); });
    const job = jobs.find(item => item.id === state.activeJob) || jobs.at(-1);
    if (job) {
      state.activeJob = job.id; const text = String(job.output || '') + (job.error ? '\n' + job.error : '');
      if (output.textContent !== text) { const nearBottom = output.scrollHeight - output.scrollTop - output.clientHeight < 50; output.textContent = text || (terminal(job) ? 'Command produced no output.' : 'Command is running...'); if (nearBottom) output.scrollTop = output.scrollHeight; }
      jobStatus.textContent = job.status + (typeof job.exit_code === 'number' ? ' / exit ' + job.exit_code : '') + (job.output_truncated ? ' / Showing the retained output tail; earlier output was truncated.' : ''); cancel.hidden = terminal(job);
    } else { output.textContent = 'Command output appears here.'; jobStatus.textContent = ''; cancel.hidden = true; }
    const running = jobs.some(item => !terminal(item)); run.disabled = !state.snapshot?.execution?.enabled || running; command.disabled = !state.snapshot?.execution?.enabled || running;
  }
  function applySnapshot(snapshot) {
    state.snapshot = snapshot; revisionLabel.textContent = 'Revision ' + snapshot.revision;
    executionBadge.textContent = snapshot.execution?.enabled ? 'Isolated container' : 'Execution unavailable'; executionBadge.classList.toggle('error', !snapshot.execution?.enabled);
    executionNote.textContent = snapshot.execution?.description || (snapshot.execution?.enabled ? 'Commands use a disposable container with networking disabled and no host directory mounts.' : 'Configure a sandbox image to enable isolated command execution. File editing and checkpoints remain available.');
    const policy = snapshot.execution?.limits || {}; const parts = [];
    for (const [key, label, suffix] of [['file_bytes', 'Per file', ''], ['project_bytes', 'Workspace', ''], ['files', 'Files', ''], ['command_seconds', 'Command time', ''], ['output_chars', 'Retained output', ' characters'], ['checkpoints', 'Checkpoints', '']]) if (typeof policy[key] === 'number') parts.push(label + ': ' + (key.endsWith('_bytes') ? bytes(policy[key]) : key === 'command_seconds' ? seconds(policy[key]) : number(policy[key], suffix)));
    limitsText.textContent = parts.join(' / '); limits.hidden = !parts.length;
    renderFiles(); renderCheckpoints(); renderJobs(); scheduleJobs();
  }
  async function load() {
    while (state.loading) {
      try { await state.loading; } catch { /* The older caller reports its failure; this caller still needs a refresh. */ }
    }
    const loading = api('/workspace'); state.loading = loading;
    try { const snapshot = await loading; clearError(); applySnapshot(snapshot); if (!state.path) state.revision = state.snapshot.revision; }
    finally { state.loading = null; }
  }
  async function openFile(path, ready) {
    const selection = ++state.selection;
    const isCurrent = () => selection === state.selection && (!ready || viewLoad === ready);
    if (!await canReplace() || !isCurrent()) return;
    clearError(); const content = editor.value;
    if (ready) {
      await ready;
      if (!isCurrent()) return;
    }
    let file;
    try { file = await api('/workspace/file?path=' + encodeURIComponent(path)); }
    catch (problem) {
      if (!isCurrent()) return;
      if (problem.status === 404) throw new Error('This Workspace file is no longer available: ' + path);
      throw problem;
    }
    if (!isCurrent()) return;
    if (editor.value !== content && !await canReplace()) return;
    if (!isCurrent()) return;
    state.path = file.path; state.original = file.content; state.isNew = false; state.revision = file.revision; editor.value = file.content; editor.disabled = false; updateEditor(); renderFiles(); editor.focus();
  }
  async function createFile() {
    const selection = ++state.selection;
    if (!await canReplace() || selection !== state.selection) return;
    const path = await window.ChatShell.conversationForm({title: 'New workspace file', description: 'Enter a path relative to the managed workspace.', label: 'File path', maxLength: 240, value: 'untitled.txt', submit: 'Create'});
    if (!path?.trim() || selection !== state.selection) return;
    if (state.snapshot?.files.some(file => file.path === path.trim())) return openFile(path.trim());
    state.path = path.trim(); state.original = ''; state.isNew = true; state.revision = state.snapshot?.revision || 0; editor.value = ''; editor.disabled = false; updateEditor(); renderFiles(); editor.focus();
  }
  async function saveFile() {
    if (!state.path || state.saving) return;
    ++state.selection;
    const submitted = {path: state.path, content: editor.value, revision: state.revision};
    clearError(); state.saving = true; updateEditor();
    try {
      const result = await api('/workspace/file', 'POST', submitted);
      if (state.path === submitted.path && state.revision === submitted.revision) {
        state.original = submitted.content; state.isNew = false; state.revision = result.revision;
      }
      await load();
    } finally { state.saving = false; updateEditor(); }
  }
  async function deleteFile() {
    const path = state.path, revision = state.revision;
    if (!path) return;
    const selection = ++state.selection;
    await window.ChatShell.conversationForm({title: 'Delete workspace file?', description: 'Delete ' + path + ' from the managed workspace?', destructive: true, submit: 'Delete'}, async () => {
      await api('/workspace/delete', 'POST', {path, revision});
      if (selection === state.selection) { state.path = ''; state.original = ''; state.isNew = false; editor.value = ''; editor.disabled = true; }
      await load(); updateEditor();
    });
  }
  function openArtifact() { if (state.path) window.ChatArtifacts.open({name: state.path, source: editor.value, language: fileLanguage(state.path)}); }
  async function createCheckpoint() {
    await window.ChatShell.conversationForm({title: 'Create checkpoint', description: 'Save a named checkpoint of the current saved files.', label: 'Checkpoint name', maxLength: 120, value: 'Workspace checkpoint', submit: 'Create'}, async label => {
      await api('/workspace/checkpoints', 'POST', {label}); await load();
    });
  }
  async function viewDiff(checkpoint) { const result = await api('/workspace/diff?checkpoint=' + encodeURIComponent(checkpoint.id)); window.ChatArtifacts.open({name: checkpoint.label + '.diff', source: result.diff || 'No differences from the current saved files.\n', language: 'diff'}); }
  async function restoreCheckpoint(checkpoint) {
    const revision = state.snapshot.revision, selection = ++state.selection;
    await window.ChatShell.conversationForm({title: 'Restore checkpoint?', description: 'Restore "' + checkpoint.label + '"? This replaces the managed workspace files and discards unsaved editor changes.', destructive: true, submit: 'Restore'}, async () => {
      const content = editor.value;
      await api('/workspace/checkpoints/' + encodeURIComponent(checkpoint.id) + '/restore', 'POST', {revision});
      await load(); if (selection !== state.selection) return;
      state.original = content; state.isNew = false;
      if (state.snapshot.files.some(file => file.path === state.path)) await openFile(state.path); else { state.path = ''; editor.value = ''; state.original = ''; editor.disabled = true; updateEditor(); }
    });
  }
  async function startJob(event) {
    event.preventDefault(); if (!command.value.trim() || run.disabled) return; clearError(); run.disabled = true; state.jobSelection++;
    try { const result = await api('/workspace/jobs', 'POST', {command: command.value.trim()}); state.activeJob = result.job.id; await load(); }
    catch (problem) { renderJobs(); showError(problem); }
  }
  async function cancelJob() { const job = state.snapshot?.jobs.find(item => item.id === state.activeJob); if (!job || terminal(job)) return; await api('/workspace/jobs/' + encodeURIComponent(job.id) + '/cancel', 'POST', {}); await load(); }
  function scheduleJobs() {
    clearTimeout(state.timer); state.timer = null;
    if (!state.snapshot?.jobs.some(job => !terminal(job)) || document.hidden || document.getElementById('app').dataset.view !== 'workbench') return;
    state.timer = setTimeout(async () => { try { const running = state.snapshot.jobs.filter(job => !terminal(job)); for (const job of running) { const result = await api('/workspace/jobs/' + encodeURIComponent(job.id)); const index = state.snapshot.jobs.findIndex(item => item.id === job.id); state.snapshot.jobs[index] = result.job; } renderJobs(); if (state.snapshot.jobs.every(terminal)) await load(); else scheduleJobs(); } catch (problem) { showError(problem); } }, 500);
  }
  editor.addEventListener('input', updateEditor); editor.addEventListener('keyup', updateEditor); editor.addEventListener('click', updateEditor); commandForm.addEventListener('submit', startJob);
  document.addEventListener('keydown', event => { if (document.querySelector('dialog[open]')) return; if (document.getElementById('app').dataset.view === 'workbench' && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); if (!save.disabled) saveFile().catch(showError); } });
  document.addEventListener('companion:view', event => {
    if (event.detail === 'workbench') { viewLoad = load(); viewLoad.catch(showError); }
    else { viewLoad = null; clearTimeout(state.timer); state.timer = null; }
  });
  async function openToolResult(target) {
    window.ChatShell.show('workbench');
    if (!target) return;
    const loading = viewLoad, selection = state.jobSelection;
    try {
      if (target.path) { await openFile(target.path, loading); return; }
      await loading;
      if (viewLoad !== loading || selection !== state.jobSelection) return;
      if (!state.snapshot.jobs.some(job => job.id === target.jobId)) throw new Error('This Workspace job is no longer available.');
      selectJob(target.jobId); output.focus();
    } catch (problem) {
      if (viewLoad !== loading) return;
      showError(problem); error.focus();
    }
  }
  window.ChatWorkspace = {openToolResult};
  document.addEventListener('visibilitychange', () => { if (!document.hidden && document.getElementById('app').dataset.view === 'workbench') load().catch(showError); });
  document.addEventListener('companion:artifact-edit', async event => {
    const selection = ++state.selection;
    if (!await canReplace() || selection !== state.selection) return; window.ChatShell.show('workbench');
    try { await load(); if (selection !== state.selection) return; const artifact = event.detail; const path = await window.ChatShell.conversationForm({title: 'Edit generated file', description: 'Choose where to save this generated file in the managed workspace.', label: 'File path', maxLength: 240, value: artifact.name, submit: 'Edit'}); if (!path?.trim() || selection !== state.selection) return; state.path = path.trim(); state.original = ''; state.isNew = true; state.revision = state.snapshot.revision; editor.value = artifact.source; editor.disabled = false; updateEditor(); renderFiles(); editor.focus(); }
    catch (problem) { showError(problem); }
  });
  window.addEventListener('beforeunload', event => { if (dirty()) { event.preventDefault(); event.returnValue = ''; } });
  updateEditor();
})();
