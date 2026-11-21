/* The managed workspace API owns files, revision conflicts, jobs and history. */
(() => {
  'use strict';
  const {node, icon, number} = window.ChatRender;
  const {api} = window.ChatShell;
  const view = document.getElementById('view-workbench');
  const state = {snapshot: null, path: '', original: '', revision: 0, activeJob: null, timer: null, loading: null, isNew: false, selection: 0, saving: false, attaching: false, jobSelection: 0};
  const projectState = {items: null, loading: null, editing: null, saving: false, files: []};
  let viewLoad, section = 'files';
  const terminal = job => ['completed', 'failed', 'cancelled'].includes(job.status);
  function control(label, cls, listener) { const button = node('button', cls || 'secondary-button', label); button.type = 'button'; if (listener) button.addEventListener('click', () => Promise.resolve().then(listener).catch(showError)); return button; }
  const toolbar = node('div', 'view-toolbar'); const back = control('Back to Chat', 'text-button workspace-back', returnToChat);
  const headingActions = node('div', 'workspace-actions'), revisionLabel = node('span', 'workspace-revision');
  const refresh = control('Refresh', '', refreshWorkspace), newFile = control('New file', 'primary-button', createFile);
  const importFiles = control('Import', '', openImport);
  importFiles.disabled = true; importFiles.id = 'workspace-import';
  importFiles.setAttribute('aria-haspopup', 'dialog'); importFiles.setAttribute('aria-controls', 'workspace-import-dialog');
  headingActions.append(revisionLabel, refresh, importFiles, newFile); toolbar.append(back, headingActions);
  const sections = node('nav', 'workspace-sections'); sections.id = 'workspace-sections'; sections.setAttribute('aria-label', 'Workspace sections');
  const error = node('div', 'view-notice workspace-error'); error.setAttribute('role', 'alert'); error.tabIndex = -1; error.hidden = true;
  const limits = node('details', 'workspace-limits'), limitsText = node('p');
  limits.append(node('summary', '', 'Workspace limits'), limitsText); limits.hidden = true;
  const layout = node('div', 'workbench-layout');
  const filePanel = node('section', 'workspace-files'); filePanel.setAttribute('aria-label', 'Workspace files'); const fileHeading = node('div', 'workspace-section-heading'); fileHeading.append(node('h3', '', 'Files')); const fileCount = node('span', 'muted', '0'); fileHeading.append(fileCount);
  const fileList = node('nav', 'workspace-file-list'); fileList.setAttribute('aria-label', 'Files');
  const fileFilter = node('input'); fileFilter.type = 'search'; fileFilter.id = 'workspace-file-filter';
  fileFilter.placeholder = 'Find a file'; fileFilter.setAttribute('aria-label', 'Filter saved files by name or folder');
  const fileSearch = node('label', 'search-box workspace-file-search');
  fileSearch.append(icon('M16 16l5 5M18 11a7 7 0 1 1-14 0 7 7 0 0 1 14 0Z'), fileFilter);
  fileFilter.addEventListener('input', renderFiles); filePanel.append(fileHeading, fileSearch, fileList);
  const editorPanel = node('section', 'workspace-editor'); editorPanel.setAttribute('aria-label', 'File editor');
  const editorHeading = node('div', 'workspace-editor-heading'); const fileLabel = node('span', 'workspace-file-path', 'Select a file'); const dirtyLabel = node('span', 'unsaved-indicator', 'Unsaved'); dirtyLabel.hidden = true;
  const fileActions = node('div', 'workspace-editor-actions'); const preview = control('View source', '', openArtifact); const attach = control('Attach to Chat', '', attachFile); attach.id = 'workspace-attach'; const download = control('Download', '', () => window.ChatArtifacts.download(editorText(), state.path, fileLanguage(state.path))); const save = control('Save', 'primary-button', saveFile); save.id = 'workspace-save'; const remove = control('Delete', 'text-button danger', deleteFile); const review = control('Review changes', '', reviewChanges); review.id = 'workspace-review'; fileActions.append(preview, attach, download, remove, review, save); editorHeading.append(fileLabel, dirtyLabel, fileActions);
  const editor = node('textarea', 'workspace-code-editor'); editor.id = 'workspace-editor'; editor.spellcheck = false; editor.autocomplete = 'off'; editor.wrap = 'off'; editor.setAttribute('aria-label', 'File content'); editor.placeholder = 'Create a file or select one from the workspace.'; editor.disabled = true;
  const editorFooter = node('div', 'workspace-editor-footer'); const editorPosition = node('span', '', 'UTF-8'); const saveHint = node('span', '', 'Ctrl / Cmd + S to save'); editorFooter.append(editorPosition, saveHint); editorPanel.append(editorHeading, editor, editorFooter);
  const start = node('div', 'workspace-start'); start.id = 'workspace-start';
  const startTitle = node('h3'), startNote = node('p');
  start.append(startTitle, startNote); editorPanel.prepend(start);
  layout.append(filePanel, editorPanel);
  const projects = node('section', 'workspace-projects'); projects.id = 'workspace-projects';
  const projectsIntro = node('p', 'workspace-panel-description', 'Group saved files and instructions to reuse them together in a chat.');
  const projectsPanel = node('div', 'workspace-projects-panel');
  const projectStatus = node('p', 'workspace-project-status'); projectStatus.setAttribute('role', 'status'); projectStatus.hidden = true;
  const projectList = node('ul', 'workspace-project-list'); projectList.setAttribute('aria-label', 'Saved projects');
  const projectNew = control('New project', '', () => editProject(null)); projectNew.id = 'workspace-project-new'; projectNew.disabled = true;
  const projectForm = node('form', 'workspace-project-form'); projectForm.id = 'workspace-project-form'; projectForm.hidden = true;
  const projectTitle = node('h3'); projectTitle.id = 'workspace-project-title'; projectForm.setAttribute('aria-labelledby', projectTitle.id);
  const projectFields = node('fieldset', 'workspace-project-fields');
  const projectName = node('input'); projectName.id = 'workspace-project-name'; projectName.required = true; projectName.maxLength = 240;
  const projectInstructions = node('textarea'); projectInstructions.id = 'workspace-project-instructions'; projectInstructions.rows = 3; projectInstructions.maxLength = 16 * 1024;
  const projectNameLabel = node('label', 'field', 'Project name'); projectNameLabel.append(projectName);
  const projectInstructionsLabel = node('label', 'field', 'Instructions (optional)'); projectInstructionsLabel.append(projectInstructions);
  const projectPaths = node('fieldset', 'workspace-project-paths'); projectPaths.id = 'workspace-project-paths';
  const projectPathCount = node('legend'); const projectPathChoices = node('div', 'workspace-project-path-choices');
  projectPaths.append(projectPathCount, projectPathChoices);
  projectFields.append(projectNameLabel, projectInstructionsLabel, projectPaths);
  const projectError = node('p', 'error-text'); projectError.id = 'workspace-project-error'; projectError.setAttribute('role', 'alert'); projectError.hidden = true;
  const projectActions = node('div', 'workspace-project-actions');
  const projectCancel = control('Cancel', '', () => closeProjectEditor()); projectCancel.id = 'workspace-project-cancel';
  const projectDelete = control('Delete project', 'secondary-button danger', deleteProject); projectDelete.id = 'workspace-project-delete';
  const projectSave = node('button', 'primary-button', 'Save project'); projectSave.id = 'workspace-project-save'; projectSave.type = 'submit';
  projectActions.append(projectCancel, projectDelete, projectSave);
  projectForm.append(projectTitle, projectFields, projectError, projectActions);
  projectsPanel.append(projectStatus, projectList, projectNew, projectForm); projects.append(projectsIntro, projectsPanel);
  const historyPanel = node('section', 'workspace-checkpoints'); historyPanel.setAttribute('aria-label', 'Checkpoints');
  const checkpointAdd = control('Create checkpoint', '', createCheckpoint); const history = node('details'); const historySummary = node('summary', '', 'Checkpoint history'); const checkpoints = node('div', 'workspace-checkpoint-list');
  history.append(historySummary, checkpoints); historyPanel.append(node('p', 'workspace-panel-description', 'Save a restore point for your files before a change, or compare an earlier version.'), checkpointAdd, history);
  const terminalPanel = node('section', 'workspace-terminal'); terminalPanel.setAttribute('aria-label', 'Isolated execution');
  const terminalHeading = node('div', 'workspace-section-heading'); terminalHeading.append(node('h3', '', 'Command runner')); const executionBadge = node('span', 'badge', 'Checking execution'); terminalHeading.append(executionBadge);
  const executionNote = node('p', 'execution-description', 'Loading execution configuration.');
  const commandForm = node('form', 'workspace-command'); const command = node('input'); command.id = 'workspace-command'; command.type = 'text'; command.placeholder = 'Enter a command for the isolated container'; command.setAttribute('aria-label', 'Container command'); command.autocomplete = 'off'; command.maxLength = 8192;
  const run = node('button', 'primary-button', 'Run command'); run.type = 'submit'; run.id = 'workspace-run'; const cancel = control('Stop', 'secondary-button', cancelJob); cancel.hidden = true; commandForm.append(node('span', 'terminal-prompt', '>_'), command, run, cancel);
  const jobTabs = node('div', 'workspace-job-tabs'); jobTabs.setAttribute('aria-label', 'Command history');
  const output = node('pre', 'workspace-output', 'Command output appears here.'); output.id = 'workspace-output'; output.setAttribute('aria-label', 'Command output'); output.tabIndex = 0; const jobStatus = node('div', 'workspace-job-status'); jobStatus.setAttribute('role', 'status'); terminalPanel.append(terminalHeading, executionNote, commandForm, jobTabs, output, jobStatus);
  const panels = {files: layout, projects, commands: terminalPanel, history: historyPanel};
  for (const [key, label] of [['files', 'Files'], ['projects', 'Projects'], ['commands', 'Commands'], ['history', 'History']]) {
    const button = control(label, 'workspace-section', () => selectSection(key));
    button.dataset.section = key; sections.append(button);
  }
  view.append(sections, toolbar, error, layout, projects, terminalPanel, historyPanel, limits);
  function selectSection(next) {
    if (section !== next) { state.selection++; state.jobSelection++; }
    section = next;
    for (const button of sections.children) button.setAttribute('aria-pressed', String(button.dataset.section === next));
    for (const [key, panel] of Object.entries(panels)) panel.hidden = key !== next;
    newFile.hidden = importFiles.hidden = next !== 'files';
    if (next === 'projects') loadProjects();
  }
  function returnToChat(target) {
    window.ChatShell.show('chat');
    (target || window.ChatShell.viewFocusTarget()).focus();
  }
  function showError(problem) { if (!state.snapshot) startNote.textContent = 'Select Refresh to load your saved files.'; error.textContent = problem.message || String(problem); error.hidden = false; }
  function clearError() { error.hidden = true; error.textContent = ''; }
  function setProjectError(message = '') { projectError.textContent = message; projectError.hidden = !message; }
  function setProjectStatus(message = '', failed = false) {
    projectStatus.textContent = message; projectStatus.hidden = !message; projectStatus.classList.toggle('error-text', failed);
  }
  function updateProjectControls() {
    const available = Boolean(state.snapshot && projectState.items && !projectState.loading);
    projectNew.disabled = !available;
    projectList.querySelectorAll('button').forEach(button => { button.disabled = !available; });
  }
  async function downloadProject(project, button) {
    if (button.hasAttribute('aria-busy')) return;
    button.setAttribute('aria-busy', 'true');
    button.setAttribute('aria-disabled', 'true');
    setProjectStatus('');
    try {
      const response = await fetch('/api/workspace/projects/' + encodeURIComponent(project.id) + '/archive', {credentials: 'same-origin'});
      if (!response.ok) {
        if (response.status === 401) document.dispatchEvent(new CustomEvent('companion:unauthorized'));
        const problem = await response.json().catch(() => null);
        throw new Error(problem?.error || 'Could not download the project (' + response.status + ').');
      }
      if (response.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase() !== 'application/zip')
        throw new Error('The server did not return a ZIP archive.');
      const filename = /(?:^|;)\s*filename="([^"]+)"/i.exec(response.headers.get('Content-Disposition') || '')?.[1] || 'llmx-project.zip';
      window.ChatArtifacts.downloadBlob(await response.blob(), filename);
      if (button.isConnected && !projectList.hidden) setProjectStatus('Download started for ' + filename + '.');
    } catch (problem) {
      if (button.isConnected && !projectList.hidden) setProjectStatus(problem.message || 'Could not download the project.', true);
    } finally {
      button.removeAttribute('aria-disabled');
      button.removeAttribute('aria-busy');
    }
  }
  function renderProjects() {
    projectList.replaceChildren();
    for (const project of [...(projectState.items || [])].sort((a, b) => a.name.localeCompare(b.name))) {
      const row = node('li'); const button = control('', 'workspace-project-row', () => editProject(project));
      button.dataset.project = project.id;
      button.append(node('strong', '', project.name), node('span', 'muted', number(project.paths.length) + (project.paths.length === 1 ? ' path' : ' paths')));
      row.append(button);
      if (project.paths.length) {
        const download = control('Download ZIP', 'secondary-button workspace-project-download', () => downloadProject(project, download));
        download.setAttribute('aria-label', 'Download ZIP for ' + project.name);
        row.append(download);
      }
      projectList.append(row);
    }
    updateProjectControls();
  }
  function loadProjects() {
    if (projectState.loading || projectState.editing) return projectState.loading;
    setProjectStatus('Loading projects...');
    projectState.loading = api('/workspace/projects').then(result => {
      projectState.items = result.projects; renderProjects();
      setProjectStatus(result.projects.length ? '' : 'No projects yet.');
    }).catch(problem => {
      projectState.items = null; renderProjects(); setProjectStatus(problem.message, true);
    }).finally(() => {
      projectState.loading = null; updateProjectControls();
    });
    updateProjectControls();
    return projectState.loading;
  }
  async function refreshWorkspace() {
    await load();
    if (section === 'projects') await loadProjects();
  }
  function renderProjectPaths() {
    const selection = projectState.editing.paths;
    const active = projectPathChoices.contains(document.activeElement) ? document.activeElement.value : null;
    const available = new Set(projectState.files);
    const paths = new Set([...selection, ...available]);
    projectPathChoices.replaceChildren();
    let focus;
    for (const path of paths) {
      const missing = !available.has(path);
      const field = node('label', 'workspace-project-path');
      const check = node('input'); check.type = 'checkbox'; check.checked = selection.has(path); check.value = path;
      if (active === path) focus = check;
      const label = node('span', '', path);
      if (missing) label.append(node('span', 'error-text', ' (missing)'));
      check.addEventListener('change', () => {
        if (check.checked && selection.size >= 8) { check.checked = false; setProjectError('Select up to 8 saved files.'); return; }
        if (check.checked) selection.add(path); else selection.delete(path);
        projectPathCount.textContent = 'Saved files (' + number(selection.size) + ' of 8)';
        setProjectError();
      });
      field.append(check, label); projectPathChoices.append(field);
    }
    if (!paths.size) projectPathChoices.append(node('p', 'muted', 'No saved files available.'));
    projectPathCount.textContent = 'Saved files (' + number(selection.size) + ' of 8)';
    if (focus && !projectFields.disabled) focus.focus({preventScroll: true});
  }
  function editProject(project) {
    if (projectState.editing || projectNew.disabled) return;
    projectState.editing = {project, paths: new Set(project?.paths || []), origin: document.activeElement};
    projectTitle.textContent = project ? 'Edit project' : 'New project';
    projectName.value = project?.name || ''; projectInstructions.value = project?.instructions || '';
    projectDelete.hidden = !project; projectForm.hidden = false; projectList.hidden = true; projectNew.hidden = true;
    setProjectStatus(); setProjectError(); renderProjectPaths(); projectName.focus();
  }
  function closeProjectEditor(savedId) {
    const origin = projectState.editing?.origin;
    const restoreFocus = section === 'projects' && !view.hidden && (projectForm.contains(document.activeElement) || document.activeElement === document.body);
    projectState.editing = null; projectForm.hidden = true; projectList.hidden = false; projectNew.hidden = false;
    setProjectError(); setProjectStatus(projectState.items.length ? '' : 'No projects yet.');
    const saved = savedId && projectList.querySelector('[data-project="' + CSS.escape(savedId) + '"]');
    if (restoreFocus) (saved || (origin?.isConnected && projects.contains(origin) ? origin : projectNew)).focus();
  }
  async function saveProject(event) {
    event.preventDefault();
    const editing = projectState.editing;
    if (!editing || projectState.saving) return;
    const name = projectName.value.trim(); const instructions = projectInstructions.value; const paths = [...editing.paths];
    if (!name || [...name].length > 120) { setProjectError('Enter a project name with 1 to 120 characters.'); projectName.focus(); return; }
    if (new TextEncoder().encode(instructions).byteLength > 16 * 1024) { setProjectError('Project instructions must fit within 16 KiB.'); projectInstructions.focus(); return; }
    if (paths.some(path => !projectState.files.includes(path))) { setProjectError('Remove missing files before saving this project.'); return; }
    const project = editing.project; const route = '/workspace/projects' + (project ? '/' + encodeURIComponent(project.id) : '');
    setProjectError(); projectState.saving = true; projectFields.disabled = true; projectCancel.disabled = true; projectDelete.disabled = true; projectSave.disabled = true; projectSave.setAttribute('aria-busy', 'true');
    let saved;
    try {
      const result = await api(route, project ? 'PATCH' : 'POST', {name, instructions, paths, ...(project ? {version: project.version} : {})});
      saved = result.project;
      projectState.items = [...projectState.items.filter(item => item.id !== saved.id), saved];
      renderProjects(); closeProjectEditor(saved.id);
    } catch (problem) { setProjectError(problem.message); }
    finally {
      projectState.saving = false; projectFields.disabled = false; projectCancel.disabled = false; projectDelete.disabled = false; projectSave.disabled = false;
      projectSave.removeAttribute('aria-busy');
      if (!saved && section === 'projects' && !view.hidden && (projectForm.contains(document.activeElement) || document.activeElement === document.body)) projectSave.focus({preventScroll: true});
    }
  }
  async function deleteProject() {
    const project = projectState.editing?.project;
    if (!project || projectState.saving) return;
    const removed = await window.ChatShell.conversationForm({title: 'Delete project?', description: 'Delete "' + project.name + '"? Its saved files will remain in Workspace.', destructive: true, submit: 'Delete'}, async () => {
      await api('/workspace/projects/' + encodeURIComponent(project.id), 'DELETE', {version: project.version});
    });
    if (removed !== true) return;
    projectState.items = projectState.items.filter(item => item.id !== project.id);
    renderProjects(); closeProjectEditor();
  }
  function bytes(value) { const units = ['B', 'KiB', 'MiB', 'GiB']; let index = 0; while (value >= 1024 && index < units.length - 1) { value /= 1024; index++; } return number(value) + ' ' + units[index]; }
  function seconds(value) { return value >= 60 && value % 60 === 0 ? number(value / 60) + ' min' : number(value) + ' s'; }
  let source = '', displayed = '', newline = '\n', mixedEndings = false;
  function setEditorText(text) {
    editor.value = text;
    source = text;
    displayed = editor.value;
    newline = text.match(/\r\n|[\r\n]/)?.[0] || '\n';
    mixedEndings = text.includes('\r') && /[\r\n]/.test(text.replaceAll(newline, ''));
  }
  function editorText() {
    const next = editor.value;
    if (next === displayed) return source;
    if (mixedEndings) {
      const before = displayed.split('\n'), after = next.split('\n');
      const ends = source.match(/\r\n|[\r\n]/g) || [];
      let oldTail = before.length, newTail = after.length;
      while (oldTail && newTail && before[oldTail - 1] === after[newTail - 1]) { oldTail--; newTail--; }
      source = after.map((line, i) => {
        if (i === after.length - 1) return line;
        let ending = newline;
        if (i >= newTail) ending = ends[i - newTail + oldTail];
        else if (i < oldTail) ending = ends[i];
        return line + (ending || newline);
      }).join('');
    } else {
      source = newline === '\n' ? next : next.replaceAll('\n', newline);
    }
    displayed = next;
    return source;
  }
  function dirty() { return !editor.disabled && (editorText() !== state.original || state.isNew); }
  async function canReplace() { return !dirty() || Boolean(await window.ChatShell.conversationForm({title: 'Discard unsaved changes?', description: 'Discard the unsaved changes in ' + state.path + '?', destructive: true, submit: 'Discard'})); }
  function fileLanguage(path) { const ext = path.split('.').pop().toLowerCase(); return ({htm: 'html', py: 'python', js: 'javascript', ts: 'typescript', sh: 'bash', rs: 'rust', md: 'markdown', patch: 'diff'})[ext] || ext; }
  function updateEditor() {
    dirtyLabel.hidden = !dirty(); const enabled = Boolean(state.path); for (const button of [preview, download]) button.disabled = !enabled; save.disabled = !enabled || !dirty() || state.saving; save.setAttribute('aria-busy', String(state.saving));
    review.hidden = !dirty() || state.isNew; review.disabled = state.saving;
    remove.textContent = state.isNew ? 'Discard' : 'Delete';
    remove.disabled = !enabled || state.saving;
    attach.disabled = !enabled || state.attaching; attach.setAttribute('aria-busy', String(state.attaching));
    preview.textContent = fileLanguage(state.path) === 'html' ? 'Preview HTML' : 'View source';
    fileLabel.textContent = state.path;
    for (const item of [editorHeading, editor, editorFooter]) item.hidden = !enabled;
    start.hidden = enabled;
    layout.classList.toggle('has-selection', enabled);
    const before = editor.value.slice(0, editor.selectionStart); const rows = before.split('\n'); editorPosition.textContent = enabled ? fileLanguage(state.path).toUpperCase() + ' / Ln ' + rows.length + ', Col ' + (rows.at(-1).length + 1) + ' / UTF-8' : 'UTF-8';
  }
  function renderFiles() {
    fileList.replaceChildren(); const files = state.snapshot?.files || []; fileCount.textContent = String(files.length);
    const query = fileFilter.value.trim().toLowerCase();
    const matches = files.filter(file => file.path.toLowerCase().includes(query));
    for (const file of matches) {
      const button = control('', 'workspace-file' + (file.path === state.path ? ' active' : ''), () => openFile(file.path));
      button.title = file.path;
      button.append(icon('M5 3h9l5 5v13H5Z M14 3v6h5', 'workspace-file-icon'), node('span', 'workspace-file-name', file.path), node('small', 'workspace-file-size', bytes(file.size)));
      if (file.path === state.path) button.setAttribute('aria-current', 'page');
      fileList.append(button);
    }
    if (query && !matches.length) fileList.append(node('p', 'workspace-empty', 'No matching files.'));
    newFile.disabled = importFiles.disabled = !state.snapshot;
    filePanel.hidden = !files.length;
    if (!state.snapshot) {
      startTitle.textContent = 'Workspace files'; startNote.textContent = 'Loading saved files...';
      return;
    }
    layout.classList.toggle('has-files', Boolean(files.length));
    startTitle.textContent = files.length ? 'Choose a file to edit' : 'No saved files yet';
    startNote.textContent = files.length ? 'Select a saved file to edit it, preview it or attach its current text to Chat.' : 'Import local text files or choose New file. To keep code from a reply, open its code block in Chat and choose Edit in workspace.';
  }
  function renderCheckpoints() {
    checkpoints.replaceChildren(); const list = state.snapshot?.checkpoints || [];
    historySummary.textContent = 'Checkpoint history (' + number(list.length) + ')';
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
    const executionDescription = snapshot.execution?.description || (snapshot.execution?.enabled ? 'Commands use a disposable container with networking disabled and no host directory mounts.' : 'Configure a sandbox image to enable isolated command execution. File editing and checkpoints remain available.');
    executionNote.textContent = snapshot.execution?.enabled ? 'Commands use saved workspace files. ' + executionDescription : executionDescription;
    const policy = snapshot.execution?.limits || {}; const parts = [];
    for (const [key, label, suffix] of [['file_bytes', 'Per file', ''], ['project_bytes', 'Workspace', ''], ['files', 'Files', ''], ['command_seconds', 'Command time', ''], ['output_chars', 'Retained output', ' characters'], ['checkpoints', 'Checkpoints', '']]) if (typeof policy[key] === 'number') parts.push(label + ': ' + (key.endsWith('_bytes') ? bytes(policy[key]) : key === 'command_seconds' ? seconds(policy[key]) : number(policy[key], suffix)));
    limitsText.textContent = parts.join(' / '); limits.hidden = !parts.length;
    renderFiles(); renderCheckpoints(); renderJobs(); scheduleJobs();
    const projectFiles = (snapshot.files || []).map(file => file.path);
    if (projectFiles.length !== projectState.files.length || projectFiles.some((path, index) => projectState.files[index] !== path)) {
      projectState.files = projectFiles;
      if (projectState.editing) renderProjectPaths();
    }
    updateProjectControls();
  }
  async function load() {
    while (state.loading) {
      try { await state.loading; } catch { /* The older caller reports its failure; this caller still needs a refresh. */ }
    }
    const previous = state.snapshot, loading = api('/workspace'); state.loading = loading;
    try {
      const snapshot = await loading; clearError();
      if (state.snapshot === previous || snapshot.revision >= state.snapshot.revision) applySnapshot(snapshot);
      if (!state.path) state.revision = state.snapshot.revision;
    }
    finally { state.loading = null; }
  }
  const importPart = id => document.getElementById('workspace-import-' + id);
  const importDialog = importPart('dialog'), importList = importPart('list'), filterImport = importPart('filter');
  const importStatus = importPart('summary'), importError = importPart('error'), confirmImport = importPart('submit');
  const importRefresh = importPart('refresh'), cancelImport = importPart('cancel'), importMore = importPart('more');
  const localFiles = document.getElementById('workspace-local-files'), localFolder = document.getElementById('workspace-local-folder');
  const chooseFiles = document.getElementById('workspace-choose-files'), chooseFolder = document.getElementById('workspace-choose-folder');
  const dependencyFolders = new Set(['node_modules', 'vendor', '__pycache__', 'site-packages']);
  let importing = null;

  function importProblem(text = '') { importError.textContent = text; importError.hidden = !text; }
  function openImport() {
    if (importDialog.open || !state.snapshot) return;
    importing = {snapshot: state.snapshot, entries: [], shown: 40, phase: 'review', stale: false};
    localFiles.value = localFolder.value = filterImport.value = '';
    importPart('note').textContent = 'Choose UTF-8 text files. A folder lets you select individual paths.';
    importProblem(); importList.replaceChildren(); updateImportList(); importDialog.showModal();
  }
  function updateImportList() {
    const {entries, shown} = importing;
    const term = filterImport.value.trim().toLowerCase();
    const matches = entries.filter(item => item.path.toLowerCase().includes(term));
    const rows = document.createDocumentFragment();
    for (const entry of matches.slice(0, shown)) {
      const label = node('label'), box = node('input');
      box.type = 'checkbox'; box.value = entry.path; box.checked = entry.selected;
      box.addEventListener('change', () => { entry.selected = box.checked; importProblem(); updateImportControls(); });
      label.append(box, node('span', '', entry.path), node('small', '', bytes(entry.file.size))); rows.append(label);
    }
    if (entries.length && !matches.length) rows.append(node('p', 'workspace-empty', 'No matching paths.'));
    importList.replaceChildren(rows);
    importList.hidden = !entries.length; importPart('search').hidden = !entries.length;
    importMore.hidden = matches.length <= shown;
    updateImportControls();
  }
  function updateImportControls() {
    const {entries, snapshot, phase, stale} = importing;
    const busy = phase !== 'review', selected = entries.filter(entry => entry.selected);
    chooseFiles.disabled = chooseFolder.disabled = filterImport.disabled = importList.disabled = importMore.disabled = busy;
    cancelImport.disabled = phase === 'saving'; importRefresh.hidden = !stale; importRefresh.disabled = busy;
    const policy = snapshot.execution?.limits || {}, saved = new Set(snapshot.files.map(file => file.path)), names = new Set();
    let size = 0, issue = '';
    for (const {path, file} of selected) {
      size += file.size;
      if (saved.has(path)) issue ||= path + ' already exists. Uncheck it to continue.';
      if (names.has(path)) issue ||= 'Two selected files use ' + path + '. Choose only one.';
      if (state.isNew && (state.path === path || path.startsWith(state.path + '/') || state.path.startsWith(path + '/')))
        issue ||= path + ' conflicts with the unsaved editor file.';
      if (file.size > policy.file_bytes) issue ||= path + ' exceeds the ' + bytes(policy.file_bytes) + ' file limit.';
      names.add(path);
    }
    if (selected.length + saved.size > policy.files) issue ||= 'The workspace file limit is ' + policy.files + '.';
    if (size + snapshot.files.reduce((sum, file) => sum + file.size, 0) > policy.project_bytes)
      issue ||= 'The workspace size limit is ' + bytes(policy.project_bytes) + '.';
    const count = selected.length;
    const total = count + ' selected, ' + bytes(size);
    importStatus.textContent = phase === 'reading' ? 'Reading selected files...' : phase === 'saving' ? 'Saving files...' : entries.length ? total + (issue ? '. ' + issue : '') : '';
    confirmImport.textContent = count ? 'Import ' + count + (count === 1 ? ' file' : ' files') : 'Import';
    confirmImport.disabled = !count || busy || stale || Boolean(issue);
  }
  function chooseImport(input, folder) {
    if (!importing || importing.phase !== 'review') return;
    const entries = []; let omitted = 0;
    for (const file of input.files) {
      const parts = folder ? file.webkitRelativePath.split('/') : [file.name];
      if (folder && (parts.length < 2 || parts.some(part => part.startsWith('.')) ||
          parts.slice(0,-1).some(part => dependencyFolders.has(part.toLowerCase())) || /^(thumbs\.db|desktop\.ini)$/i.test(file.name))) { omitted++; continue; }
      entries.push({file, path: folder ? parts.slice(1).join('/') : file.name, selected: !folder});
    }
    importing.entries = entries.sort((a,b) => a.path.localeCompare(b.path));
    importing.shown = 40; filterImport.value = ''; input.value = '';
    importPart('note').textContent = folder ? 'Choose files to include. Paths start inside the chosen folder.' +
      (omitted ? ' Omitted ' + omitted + ' hidden, system or dependency ' + (omitted === 1 ? 'entry.' : 'entries.') : '') : 'Review the selected files before importing.';
    if (!entries.length) importPart('note').textContent += ' No files available.';
    importProblem(); updateImportList();
  }
  chooseFiles.addEventListener('click', () => localFiles.click());
  chooseFolder.hidden = !('webkitdirectory' in localFolder);
  chooseFolder.addEventListener('click', () => localFolder.click());
  localFiles.addEventListener('change', () => chooseImport(localFiles, false));
  localFolder.addEventListener('change', () => chooseImport(localFolder, true));
  filterImport.addEventListener('input', () => { importing.shown = 40; updateImportList(); });
  importMore.addEventListener('click', () => {
    const start = importing.shown; importing.shown += 40; updateImportList(); importList.querySelectorAll('input')[start]?.focus();
  });
  cancelImport.addEventListener('click', () => importDialog.close());
  importDialog.addEventListener('cancel', event => { if (importing?.phase === 'saving') event.preventDefault(); });
  importDialog.addEventListener('close', () => {
    if (importDialog.open) return;
    importing = null; localFiles.value = localFolder.value = ''; importList.replaceChildren();
  });
  importRefresh.addEventListener('click', async () => {
    const review = importing;
    review.phase = 'refreshing'; updateImportControls();
    try {
      await load();
      if (review !== importing || !importDialog.open) return;
      review.snapshot = state.snapshot; review.stale = false; importProblem();
    } catch (problem) { if (review === importing) importProblem(problem.message); }
    finally { if (review === importing) { review.phase = 'review'; updateImportControls(); } }
  });
  importPart('form').addEventListener('submit', async event => {
    event.preventDefault();
    if (!importing || confirmImport.disabled) return;
    const review = importing, revision = review.snapshot.revision;
    const current = () => review === importing && importDialog.open;
    review.phase = 'reading'; importProblem(); updateImportControls();
    try {
      const files = [];
      for (const {file, path} of review.entries.filter(item => item.selected)) {
        const data = await file.arrayBuffer();
        if (!current()) return;
        let content;
        try { content = new TextDecoder('utf-8', {fatal:true, ignoreBOM:true}).decode(data); }
        catch { throw new Error(path + ' is not valid UTF-8 text.'); }
        if (content.includes('\0')) throw new Error(path + ' contains a NUL byte. Choose a text file.');
        files.push({path, content});
      }
      review.phase = 'saving'; updateImportControls();
      const saved = await api('/workspace/files', 'POST', {revision, files});
      if (!current()) return;
      if (state.revision === revision) state.revision = saved.revision;
      if (saved.revision >= state.snapshot.revision) applySnapshot(saved);
      clearError(); importDialog.close(); importFiles.focus();
    } catch (problem) {
      if (!current()) return;
      review.stale = problem.status === 409;
      importProblem(problem.message || 'Could not import the selected files.');
    } finally { if (current()) { review.phase = 'review'; updateImportControls(); } }
  });
  document.addEventListener('companion:unauthorized', () => { if (importDialog.open) importDialog.close(); });
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
    state.path = file.path; state.original = file.content; state.isNew = false; state.revision = file.revision; setEditorText(file.content); editor.disabled = false;
    editor.setSelectionRange(0, 0); updateEditor(); renderFiles(); editor.focus(); editor.scrollTop = 0; editor.scrollLeft = 0;
  }
  async function createFile() {
    const selection = ++state.selection;
    if (!await canReplace() || selection !== state.selection) return;
    const path = await window.ChatShell.conversationForm({title: 'New workspace file', description: 'Name the file. Include a folder if needed, such as notes/plan.md.', label: 'File path', maxLength: 240, value: 'untitled.txt', submit: 'Create'});
    if (!path?.trim() || selection !== state.selection) return;
    if (state.snapshot?.files.some(file => file.path === path.trim())) return openFile(path.trim());
    state.path = path.trim(); state.original = ''; state.isNew = true; state.revision = state.snapshot?.revision || 0; setEditorText(''); editor.disabled = false; updateEditor(); renderFiles(); editor.focus();
  }
  async function reviewChanges() {
    const path = state.path, content = editorText(), selection = state.selection;
    const result = await api('/workspace/diff', 'POST', {path, content});
    if (state.selection !== selection || editorText() !== content || view.hidden || section !== 'files') return;
    window.ChatArtifacts.open({name: path + '.diff', source: result.diff || 'The draft matches the saved file.\n', language: 'diff'}, {allowWorkspaceEdit: false});
  }
  async function saveFile() {
    if (!state.path || state.saving) return;
    ++state.selection;
    const submitted = {path: state.path, content: editorText(), revision: state.revision};
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
    const {path, revision, isNew} = state;
    if (!path || state.saving) return;
    const selection = ++state.selection;
    const action = isNew ? 'Discard' : 'Delete';
    const confirmed = await window.ChatShell.conversationForm({
      title: isNew ? 'Discard unsaved file?' : 'Delete workspace file?',
      description: isNew ? 'Discard your unsaved edits to ' + path + '?' : 'Delete ' + path + ' from the managed workspace?',
      destructive: true, submit: action
    }, async () => {
      if (selection !== state.selection) return;
      if (!isNew) await api('/workspace/delete', 'POST', {path, revision});
      if (selection === state.selection) {
        state.path = ''; state.original = ''; state.isNew = false;
        setEditorText(''); editor.disabled = true;
        clearError(); updateEditor(); renderFiles();
      }
      if (!isNew) await load();
    });
    if (confirmed && selection === state.selection && !view.hidden && section === 'files') newFile.focus();
  }
  function openArtifact() {
    if (!state.path) return;
    const language = fileLanguage(state.path);
    window.ChatArtifacts.open({name: state.path, source: editorText(), language}, {initialPreview: language === 'html', allowWorkspaceEdit: false, workspacePath: /\.html?$/i.test(state.path) ? state.path : ''});
  }
  async function attachFile() {
    if (!state.path || state.attaching) return;
    const loading = viewLoad, selection = state.selection;
    const isCurrent = () => viewLoad === loading && state.selection === selection;
    clearError(); state.attaching = true; updateEditor();
    try {
      if (/[:\\]/.test(state.path)) throw new Error('Use a Workspace path with forward slashes and no colons to attach this file.');
      const name = window.ChatComposer.workspaceName(state.path);
      await window.ChatComposer.attach([new File([editorText()], name, {type: 'text/plain'})]);
      if (isCurrent()) returnToChat(document.getElementById('draft'));
    } catch (problem) {
      if (!isCurrent()) return;
      showError(problem);
      error.focus();
    } finally {
      state.attaching = false; updateEditor();
    }
  }
  async function createCheckpoint() {
    await window.ChatShell.conversationForm({title: 'Create checkpoint', description: 'Save a named checkpoint of the current saved files.', label: 'Checkpoint name', maxLength: 120, value: 'Workspace checkpoint', submit: 'Create'}, async label => {
      await api('/workspace/checkpoints', 'POST', {label}); await load(); history.open = true;
    });
  }
  async function viewDiff(checkpoint) { const result = await api('/workspace/diff?checkpoint=' + encodeURIComponent(checkpoint.id)); window.ChatArtifacts.open({name: checkpoint.label + '.diff', source: result.diff || 'No differences from the current saved files.\n', language: 'diff'}); }
  async function restoreCheckpoint(checkpoint) {
    const revision = state.snapshot.revision, selection = ++state.selection;
    await window.ChatShell.conversationForm({title: 'Restore checkpoint?', description: 'Restore "' + checkpoint.label + '"? This replaces the managed workspace files and discards unsaved editor changes.', destructive: true, submit: 'Restore'}, async () => {
      const content = editorText();
      await api('/workspace/checkpoints/' + encodeURIComponent(checkpoint.id) + '/restore', 'POST', {revision});
      await load(); if (selection !== state.selection) return;
      state.original = content; state.isNew = false;
      if (state.snapshot.files.some(file => file.path === state.path)) await openFile(state.path); else { state.path = ''; setEditorText(''); state.original = ''; editor.disabled = true; updateEditor(); }
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
  projectForm.addEventListener('submit', saveProject);
  document.addEventListener('keydown', event => { if (document.querySelector('dialog[open]')) return; if (document.getElementById('app').dataset.view === 'workbench' && section === 'files' && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); if (!save.disabled) saveFile().catch(showError); } });
  document.addEventListener('companion:view', event => {
    if (event.detail === 'workbench') { viewLoad = load(); viewLoad.catch(showError); if (section === 'projects') loadProjects(); }
    else { viewLoad = null; clearTimeout(state.timer); state.timer = null; if (importDialog.open) importDialog.close(); }
  });
  async function openToolResult(target) {
    window.ChatShell.show('workbench');
    if (!target) { selectSection('files'); return; }
    selectSection(target.path ? 'files' : 'commands');
    const loading = viewLoad, selection = state.jobSelection;
    try {
      if (target.path) {
        if (target.path === state.path && dirty()) {
          ++state.selection;
          editor.focus();
        } else await openFile(target.path, loading);
        return;
      }
      await loading;
      if (viewLoad !== loading || selection !== state.jobSelection) return;
      if (!state.snapshot.jobs.some(job => job.id === target.jobId)) throw new Error('This Workspace job is no longer available.');
      selectJob(target.jobId); output.focus();
    } catch (problem) {
      if (viewLoad !== loading) return;
      showError(problem); error.focus();
    }
  }
  async function openProjects(create = false) {
    window.ChatShell.show('workbench');
    const loading = viewLoad;
    selectSection('projects');
    const projectsSummary = sections.querySelector('[data-section=projects]');
    if (projectState.editing && !projectFields.disabled) { projectName.focus(); return; }
    projectsSummary.focus();
    try { await Promise.all([loading, loadProjects()]); }
    catch { return; }
    if (viewLoad !== loading || section !== 'projects' || document.activeElement !== projectsSummary || projectNew.disabled || projectNew.hidden) return;
    if (create) editProject(null); else projectNew.focus();
  }
  function openFiles() {
    selectSection('files'); window.ChatShell.show('workbench');
    sections.querySelector('[data-section=files]').focus();
  }
  window.ChatWorkspace = {openToolResult, openProjects, openFiles};
  document.addEventListener('visibilitychange', () => { if (!document.hidden && document.getElementById('app').dataset.view === 'workbench') load().catch(showError); });
  document.addEventListener('companion:artifact-edit', async event => {
    if (event.detail.workspacePath) return openToolResult({path: event.detail.workspacePath});
    selectSection('files');
    const selection = ++state.selection;
    if (!await canReplace() || selection !== state.selection) return;
    const previousText = editor.value;
    window.ChatShell.show('workbench');
    const loading = viewLoad;
    const current = () => selection === state.selection && viewLoad === loading;
    try {
      await loading;
      if (!current()) return;
      const snapshot = state.snapshot;
      const artifact = event.detail;
      const path = await window.ChatShell.conversationForm({title: 'Edit generated file',
        description: 'Choose a path for this file.', label: 'File path', maxLength: 240, value: artifact.name, submit: 'Edit'});
      if (!path || !current()) return;
      let existing;
      if (snapshot.files.some(file => file.path === path)) {
        existing = await api('/workspace/file?path=' + encodeURIComponent(path));
        if (!current()) return;
        if (existing.content !== artifact.source) {
          const replace = await window.ChatShell.conversationForm({title: 'Replace saved file?',
            description: path + ' already exists. Open the generated text in the editor? Save will replace the file.',
            destructive: true, submit: 'Edit replacement'});
          if (!replace || !current()) return;
        }
      }
      if (editor.value !== previousText && !await canReplace()) return;
      if (!current()) return;
      state.path = path;
      state.original = existing?.content ?? '';
      state.isNew = !existing;
      state.revision = existing?.revision ?? snapshot.revision;
      setEditorText(artifact.source);
      editor.disabled = false;
      updateEditor();
      renderFiles();
      editor.focus();
    }
    catch (problem) { showError(problem); }
  });
  window.addEventListener('beforeunload', event => { if (dirty()) { event.preventDefault(); event.returnValue = ''; } });
  selectSection('files'); renderFiles(); updateEditor();
})();
