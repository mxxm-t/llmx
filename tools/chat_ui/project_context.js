/* Apply a reviewed Workspace snapshot through the current chat composer. */
(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const dialog = byId('project-context-dialog'), select = byId('project-context-select');
  const status = byId('project-context-status'), files = byId('project-context-files');
  const instructions = byId('project-context-instructions'), copy = byId('project-context-copy');
  const error = byId('project-context-error'), apply = byId('project-context-apply');
  const field = byId('project-context-field'), setup = byId('project-context-open-workspace');
  const {api} = window.ChatShell;
  let draft, snapshot, request = 0;

  function reset() {
    snapshot = null; apply.disabled = true; apply.hidden = false; error.hidden = true;
    field.hidden = false; setup.hidden = true;
    files.replaceChildren(); files.hidden = true;
    instructions.hidden = true; byId('project-context-text').value = ''; copy.checked = false;
  }
  function showError(problem) {
    error.textContent = problem.message || String(problem); error.hidden = false;
  }
  async function selectProject() {
    const sequence = ++request;
    reset(); status.textContent = select.value ? 'Reading saved files...' : 'Choose a project to review its files.';
    if (!select.value) return;
    try {
      const data = await api('/workspace/projects/' + encodeURIComponent(select.value) + '/context');
      if (sequence !== request || !dialog.open) return;
      let total = 0;
      for (const file of data.files) {
        const size = new TextEncoder().encode(file.text).byteLength;
        total += size;
        const item = document.createElement('li');
        const name = document.createElement('span'), bytes = document.createElement('small');
        name.textContent = file.path; bytes.textContent = window.ChatComposer.formatAttachmentSize(size);
        item.append(name, bytes);
        files.append(item);
      }
      files.hidden = !data.files.length;
      status.textContent = 'Workspace revision ' + data.revision + ' / ' + data.files.length + (data.files.length === 1 ? ' file' : ' files') + ' / ' + window.ChatComposer.formatAttachmentSize(total);
      instructions.hidden = !data.project.instructions;
      byId('project-context-text').value = data.project.instructions;
      byId('project-context-copy-label').textContent = draft.system && draft.system !== data.project.instructions
        ? 'Replace the current conversation instructions' : 'Use these instructions in this conversation';
      snapshot = data; apply.disabled = !data.files.length;
    } catch (problem) {
      if (sequence === request && dialog.open) { status.textContent = ''; showError(problem); }
    }
  }
  async function open(projectId = '') {
    if (dialog.open) return;
    const sequence = ++request;
    draft = window.ChatComposer.projectDraft();
    reset(); select.replaceChildren(); select.disabled = true; status.textContent = 'Loading projects...';
    dialog.showModal();
    try {
      const {projects} = await api('/workspace/projects');
      if (sequence !== request || !dialog.open) return;
      if (projectId && !projects.some(project => project.id === projectId)) showError(new Error('This project is no longer available.'));
      if (!projects.length) {
        status.textContent = 'Create a project to reuse saved Workspace files here.';
        field.hidden = true; apply.hidden = true; setup.hidden = false;
        return;
      }
      select.add(new Option('Choose a project', ''));
      for (const project of projects) select.add(new Option(project.name, project.id));
      select.disabled = false;
      if (projectId) select.value = projectId;
      else if (projects.length === 1) select.value = projects[0].id;
      if (!projectId || select.value) await selectProject();
      else status.textContent = 'Choose a project to review its files.';
    } catch (problem) {
      if (sequence === request && dialog.open) { status.textContent = ''; showError(problem); }
    }
  }
  select.addEventListener('change', selectProject);
  copy.addEventListener('change', () => { apply.disabled = !snapshot || (!snapshot.files.length && !copy.checked); });
  byId('project-context-form').addEventListener('submit', event => {
    event.preventDefault();
    if (!snapshot || apply.disabled) return;
    try {
      draft.apply(snapshot, copy.checked);
      dialog.close(); byId('draft').focus();
    } catch (problem) { showError(problem); }
  });
  dialog.addEventListener('close', () => { if (!dialog.open) { ++request; draft = null; reset(); } });
  document.addEventListener('companion:unauthorized', () => { if (dialog.open) dialog.close(); });
  byId('project-context-cancel').addEventListener('click', () => dialog.close());
  setup.addEventListener('click', () => {
    if (!dialog.open || setup.hidden) return;
    dialog.close();
    window.ChatWorkspace.openProjects();
  });
  const picker = byId('saved-file-dialog'), list = byId('saved-file-list'), search = byId('saved-file-filter');
  const add = byId('saved-file-add'), message = byId('saved-file-status'), problem = byId('saved-file-error');
  let fileDraft, fileRequest;

  function filterFiles() {
    const term = search.value.trim().toLowerCase();
    let count = 0;
    for (const label of list.children) {
      label.hidden = !label.querySelector('input').value.toLowerCase().includes(term);
      if (!label.hidden) count++;
    }
    const checked = list.querySelectorAll(':checked').length;
    add.disabled = !checked || checked > 8;
    message.textContent = checked > 8 ? 'Choose up to 8 files.' : !count ? 'No matching files.' : checked ? checked + ' selected' : '';
  }
  async function openFiles(path = '') {
    if (picker.open) return;
    const controller = new AbortController();
    fileRequest?.abort(); fileRequest = controller;
    fileDraft = window.ChatComposer.projectDraft();
    list.replaceChildren(); list.hidden = true; list.disabled = false; search.disabled = false;
    byId('saved-file-search').hidden = true; search.value = '';
    byId('saved-file-open').hidden = true; problem.hidden = true;
    add.disabled = true; add.hidden = false; message.textContent = 'Loading saved files...'; picker.showModal();
    try {
      const data = await api('/workspace/files', 'GET', undefined, controller.signal);
      if (controller.signal.aborted || !picker.open) return;
      if (path && !data.files.some(file => file.path === path)) {
        problem.textContent = 'This saved file is no longer available.'; problem.hidden = false;
      }
      if (!data.files.length) {
        message.textContent = 'No saved files yet. Create one in the file editor or attach a file from your device.';
        byId('saved-file-open').hidden = false; add.hidden = true; return;
      }
      for (const file of data.files) {
        const label = document.createElement('label'), checkbox = document.createElement('input');
        checkbox.type = 'checkbox'; checkbox.value = file.path; checkbox.checked = path === file.path;
        const name = document.createElement('span'); name.textContent = file.path;
        const size = document.createElement('small'); size.textContent = window.ChatComposer.formatAttachmentSize(file.size);
        label.append(checkbox, name, size); list.append(label);
      }
      list.hidden = false; byId('saved-file-search').hidden = false; filterFiles();
      if (document.activeElement === byId('saved-file-cancel') || document.activeElement === picker) search.focus();
    } catch (error) {
      if (controller.signal.aborted || !picker.open) return;
      message.textContent = ''; problem.textContent = error.message; problem.hidden = false;
    }
  }
  search.addEventListener('input', filterFiles);
  list.addEventListener('change', filterFiles);
  byId('saved-file-form').addEventListener('submit', async event => {
    event.preventDefault();
    if (add.disabled || list.disabled) return;
    const controller = fileRequest, query = new URLSearchParams();
    for (const input of list.querySelectorAll(':checked')) query.append('path', input.value);
    list.disabled = true; search.disabled = true; add.disabled = true; problem.hidden = true;
    message.textContent = 'Reading selected files...';
    try {
      const data = await api('/workspace/files?' + query, 'GET', undefined, controller.signal);
      if (controller.signal.aborted || !picker.open) return;
      fileDraft.apply(data, false); picker.close(); byId('draft').focus();
    } catch (error) {
      if (controller.signal.aborted || !picker.open) return;
      list.disabled = false; filterFiles(); problem.textContent = error.message; problem.hidden = false;
    } finally { if (fileRequest === controller) search.disabled = false; }
  });
  picker.addEventListener('close', () => {
    if (picker.open) return;
    fileRequest?.abort(); fileRequest = null; fileDraft = null; search.disabled = false; list.replaceChildren();
  });
  document.addEventListener('companion:unauthorized', () => { if (picker.open) picker.close(); });
  byId('saved-file-cancel').addEventListener('click', () => picker.close());
  byId('saved-file-open').addEventListener('click', () => { picker.close(); window.ChatWorkspace.openFiles(); });
  const sourcePanel = byId('landing-context'), rows = byId('landing-context-list');
  const notice = byId('landing-context-status'), browse = byId('landing-context-browse');
  let source = 'files', authorized = Boolean(window.ChatShell.config), active = false, load;
  let landing = byId('app').dataset.temporary === 'false' && !byId('welcome').hidden;

  async function loadSources() {
    load?.abort();
    const controller = new AbortController();
    load = controller;
    const projects = source === 'projects';
    rows.replaceChildren(); notice.hidden = false; notice.textContent = projects ? 'Loading projects...' : 'Loading saved files...';
    browse.dataset.action = projects ? 'attach-project' : 'attach-files';
    browse.setAttribute('aria-haspopup', 'dialog');
    browse.textContent = projects ? 'Browse projects' : 'Browse saved files';
    try {
      const data = await api('/workspace/' + source, 'GET', undefined, controller.signal);
      if (controller.signal.aborted) return;
      const items = data[source];
      const label = item => projects ? item.name : item.path;
      const sorted = [...items].sort((a, b) => label(a).localeCompare(label(b)));
      for (const item of sorted.slice(0, 3)) {
        const {node, icon} = window.ChatRender;
        const row = node('button', 'landing-context-row');
        row.type = 'button'; row.title = label(item);
        row.setAttribute('aria-label', 'Review ' + (projects ? 'project ' : 'file ') + label(item) + ' for this draft');
        const name = node('span', 'resource-label');
        const slash = projects ? -1 : item.path.lastIndexOf('/');
        name.append(node('span', 'resource-name', label(item).slice(slash + 1)));
        if (slash >= 0) name.append(node('span', 'resource-folder', item.path.slice(0, slash + 1)));
        const detail = projects ? item.paths.length + (item.paths.length === 1 ? ' file' : ' files') : window.ChatComposer.formatAttachmentSize(item.size);
        row.append(icon(projects ? 'M3 7V5h6l2 2h10v12H3Z' : 'M5 3h9l5 5v13H5Z M14 3v6h5'), name, node('small', '', detail));
        row.addEventListener('click', () => projects ? open(item.id) : openFiles(item.path));
        rows.append(row);
      }
      notice.hidden = Boolean(items.length);
      if (!items.length) {
        notice.textContent = projects ? 'No saved projects yet.' : 'No saved files yet.';
        if (projects) { delete browse.dataset.action; browse.removeAttribute('aria-haspopup'); browse.textContent = 'Create a project'; }
      }
    } catch (problem) {
      if (!controller.signal.aborted) notice.textContent = projects ? 'Could not load saved projects.' : 'Could not load saved files.';
    }
  }
  function syncLanding() {
    const visible = authorized && landing && byId('app').dataset.view === 'chat';
    if (active === visible) return;
    active = visible;
    if (visible) loadSources();
    else { load?.abort(); rows.replaceChildren(); notice.hidden = true; }
  }
  function setLanding(value) { if (landing !== value) { landing = value; syncLanding(); } }
  sourcePanel.addEventListener('click', event => {
    const control = event.target.closest('[data-source]');
    if (control && source !== control.dataset.source) {
      source = control.dataset.source;
      sourcePanel.querySelectorAll('[data-source]').forEach(button => button.setAttribute('aria-pressed', String(button === control)));
      if (active) loadSources();
    } else if (event.target === browse && !browse.dataset.action) window.ChatWorkspace.openProjects(true);
  });
  document.addEventListener('companion:ready', () => { authorized = true; syncLanding(); });
  document.addEventListener('companion:unauthorized', () => { authorized = false; syncLanding(); });
  document.addEventListener('companion:view', syncLanding);
  window.ChatProjectContext = {openFiles, openProjects: open, setLanding};
  syncLanding();
  byId('use-workspace-project').addEventListener('click', () => open());
})();
