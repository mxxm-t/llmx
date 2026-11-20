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
  async function open() {
    if (dialog.open) return;
    const sequence = ++request;
    draft = window.ChatComposer.projectDraft();
    reset(); select.replaceChildren(); select.disabled = true; status.textContent = 'Loading projects...';
    dialog.showModal();
    try {
      const {projects} = await api('/workspace/projects');
      if (sequence !== request || !dialog.open) return;
      if (!projects.length) {
        status.textContent = 'Create a project to reuse saved Workspace files here.';
        field.hidden = true; apply.hidden = true; setup.hidden = false;
        return;
      }
      select.add(new Option('Choose a project', ''));
      for (const project of projects) select.add(new Option(project.name, project.id));
      select.disabled = false;
      if (projects.length === 1) select.value = projects[0].id;
      await selectProject();
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
  async function openFiles() {
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
      if (!data.files.length) {
        message.textContent = 'No saved files yet. Create one in the file editor or attach a file from your device.';
        byId('saved-file-open').hidden = false; add.hidden = true; return;
      }
      for (const file of data.files) {
        const label = document.createElement('label'), checkbox = document.createElement('input');
        checkbox.type = 'checkbox'; checkbox.value = file.path;
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
  window.ChatProjectContext = {openFiles, openProjects: open};
  byId('use-workspace-project').addEventListener('click', open);
})();
