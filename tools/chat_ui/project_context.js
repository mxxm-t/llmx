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
        item.textContent = file.path + ' / ' + window.ChatComposer.formatAttachmentSize(size);
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
  byId('use-workspace-project').addEventListener('click', open);
})();
