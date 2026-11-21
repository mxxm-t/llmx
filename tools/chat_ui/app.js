/* The service owns conversation state; this file owns browser presentation. */
(() => {
  'use strict';
  const {node, icon, markdown, updateMarkdown, number, json} = window.ChatRender;
  async function api(...args) {
    try {
      const result = await window.ChatShell.api(...args);
      if (result?.run && Array.isArray(result.models)) updateModelState(result);
      return result;
    } catch (error) {
      if (error.details?.code === 'model_unavailable') updateModelState(error.details);
      throw error;
    }
  }
  const $ = id => document.getElementById(id);
  const baseFileTypes = $('file-input').accept;
  const textFileExtensions = new Set(baseFileTypes.split(',').filter(type => type.startsWith('.')));
  let pdfSelection = null;
  const systemTheme = window.matchMedia('(prefers-color-scheme: light)');
  const storedTheme = localStorage.getItem('llmx-theme');
  let dockedDetailsOpen = localStorage.getItem('llmx-conversation-details') === 'open';
  let detailsInDrawer = usesDetailsDrawer(), modalDrawer = null, drawerOpener = null;
  let themeChosen = storedTheme === 'light' || storedTheme === 'dark';
  function setTheme(theme, save = false) {
    document.documentElement.dataset.theme = theme;
    $('theme-button').textContent = theme === 'light' ? 'Dark theme' : 'Light theme';
    if (save) { localStorage.setItem('llmx-theme', theme); themeChosen = true; }
  }
  systemTheme.addEventListener('change', event => { if (!themeChosen) setTheme(event.matches ? 'light' : 'dark'); });
  const terminal = status => ['completed', 'failed', 'cancelled', 'interrupted'].includes(status);
  const S = {chats: [], chat: null, messages: [], runs: [], config: {models: [], tools: []}, attachments: [], edit: null, stream: null, cursor: 0, archive: false, tab: 'activity', scheduled: false, fingerprint: new Map(), inspectorKey: '', generation: 0, busy: false};
  let toastTimer, creatingChat = false, menuId = null, composerRevision = 0;
  const chatSearch = {request: null, timer: null, running: false, chats: [], notice: ''};
  const closedSections = new Set();
  const sidebarRuns = {ready: false, active: false, epoch: 0, request: null, poll: null, expires: null, states: new Map()};
  const sidebarRunCues = new Map([
    ['running', ['', 'Reply in progress']], ['approval', ['?', 'Waiting for tool approval']],
    ['failed', ['!', 'Latest reply failed']], ['interrupted', ['~', 'Latest reply interrupted']],
  ]);
  const drafts = new Map(), changedDrafts = new Set(), volatileDrafts = new Set(), draftPrefix = 'llmx-draft:';
  const admissions = new Map(), settingsWrites = new Map();
  let draftTimer, draftReady = false;
  const draftNotice = node('p', 'muted fine-print'); draftNotice.hidden = true; draftNotice.setAttribute('role', 'status'); draftNotice.id = 'draft-notice'; $('attachments').after(draftNotice);
  function draftKey() { return S.chat?.id || 'new'; }
  function composeValue() { return {text: $('draft').value, attachments: S.attachments.slice(), edit: S.edit?.id || null}; }
  function sameDraft(a, b) { return Boolean(a && b && a.text === b.text && a.edit === b.edit && a.attachments.length === b.attachments.length && a.attachments.every((item, index) => item === b.attachments[index])); }
  function hasDraft(value) { return Boolean(value && (value.text || value.attachments.length || value.edit)); }
  function draftWarning() {
    const temporary = S.chat?.temporary && hasDraft(composeValue());
    draftNotice.textContent = temporary ? 'This temporary draft stays in this page. Refreshing discards it.' : volatileDrafts.size ? 'Some drafts are kept in this page only because browser storage is unavailable or full. Refreshing may discard them.' : '';
    draftNotice.hidden = !draftNotice.textContent;
  }
  function forgetDraft(key) {
    drafts.delete(key); changedDrafts.delete(key); volatileDrafts.delete(key);
    try { sessionStorage.removeItem(draftPrefix + key); } catch (_) { /* Browser storage may be disabled. */ }
  }
  function flushDrafts() {
    clearTimeout(draftTimer);
    for (const key of changedDrafts) {
      const value = drafts.get(key);
      try { if (hasDraft(value)) sessionStorage.setItem(draftPrefix + key, JSON.stringify(value)); else sessionStorage.removeItem(draftPrefix + key); volatileDrafts.delete(key); }
      catch (_) { volatileDrafts.add(key); try { sessionStorage.removeItem(draftPrefix + key); } catch (_) { /* Keep the current in-memory draft. */ } }
      changedDrafts.delete(key);
    }
    draftWarning();
  }
  function rememberDraft() {
    if (!draftReady) return;
    if (!S.chat?.temporary) {
      const value = composeValue();
      if (!sameDraft(drafts.get(draftKey()), value)) {
        drafts.set(draftKey(), value); changedDrafts.add(draftKey());
        clearTimeout(draftTimer); draftTimer = setTimeout(flushDrafts, 250);
      }
    }
    draftWarning();
  }
  function readDraft(key) {
    if (drafts.has(key)) return drafts.get(key);
    try {
      const raw = sessionStorage.getItem(draftPrefix + key); if (!raw) return null;
      const value = JSON.parse(raw);
      if (typeof value.text !== 'string' || value.text.length > 1000000 || !Array.isArray(value.attachments) || value.attachments.length > 8 || value.edit !== null && typeof value.edit !== 'string' || value.attachments.some(a => !a || typeof a.name !== 'string' || typeof a.type !== 'string' || typeof a.text !== 'string' && typeof a.data !== 'string')) throw new Error('Invalid draft');
      drafts.set(key, value); return value;
    } catch (_) { forgetDraft(key); toast('The saved draft could not be restored.'); return null; }
  }
  function renderLandingDrafts(visible = draftReady && !S.messages.length && !S.chat?.temporary && !$('login-dialog').open) {
    const list = $('landing-draft-list');
    list.replaceChildren();
    if (visible) {
      const keys = new Set(drafts.keys());
      try {
        for (const key of Object.keys(sessionStorage)) if (key.startsWith(draftPrefix)) keys.add(key.slice(draftPrefix.length));
      } catch (_) { /* In-memory drafts still work when storage is unavailable. */ }
      for (const chat of S.chats) {
        if (!keys.has(chat.id) || chat.archived || chat.temporary || chat.id === S.chat?.id) continue;
        const value = readDraft(chat.id);
        if (!hasDraft(value)) continue;
        const row = node('button', 'draft-row'); row.type = 'button'; row.dataset.chat = chat.id;
        const label = node('span', 'draft-row-label');
        const excerpt = value.text.trim().slice(0, 160).replace(/\s+/g, ' ');
        label.append(node('span', 'draft-row-title', chat.title || 'Untitled conversation'),
          node('span', 'draft-row-preview', excerpt || value.attachments[0]?.name || (value.edit ? 'Editing a message' : 'Unsent message')));
        row.append(label);
        if (value.attachments.length) {
          const count = value.attachments.length;
          row.append(node('small', '', count + (count === 1 ? ' file' : ' files')));
        }
        row.append(icon('M4 12h15m-5-5 5 5-5 5'));
        list.append(row);
      }
    }
    $('landing-drafts').hidden = !list.childElementCount;
  }
  function restoreDraft() {
    draftReady = false;
    const value = S.chat?.temporary ? null : readDraft(draftKey());
    $('draft').value = value?.text || ''; S.attachments = value?.attachments.slice() || [];
    S.edit = value?.edit ? S.messages.find(m => m.id === value.edit && m.role === 'user') || null : null;
    if (value?.edit && !S.edit) toast('The edited message is no longer available. Your draft is kept as a new message.');
    composerRevision++; resizeDraft(); renderAttachments(); $('edit-note').hidden = !S.edit;
    draftReady = true; draftWarning(); renderLandingDrafts();
  }
  function clearDrafts() {
    clearTimeout(draftTimer); drafts.clear(); changedDrafts.clear(); volatileDrafts.clear();
    try { for (const key of Object.keys(sessionStorage)) if (key.startsWith(draftPrefix)) sessionStorage.removeItem(key); } catch (_) { /* Browser storage may be disabled. */ }
    draftReady = false;
  }
  async function chatSnapshot(id) {
    let epoch, data;
    do { epoch = admissions.get(id); data = await api('/chats/' + encodeURIComponent(id)); } while (epoch !== admissions.get(id));
    return data;
  }
  function admitted(id) { admissions.set(id, (admissions.get(id) || 0) + 1); }
  const liveMetrics = {id: null, stream: null, history: [], row: null, received: 0, collector: null};
  let runtimeLimits = null;
  const objectIds = new WeakMap(); let nextObjectId = 0;
  function identity(value) { if (!value || typeof value !== 'object') return value; if (!objectIds.has(value)) objectIds.set(value, ++nextObjectId); return objectIds.get(value); }
  function messageKey(m, text = false) { return [m.id, m.role, m.status, m.finish_reason, text ? m.content : undefined, text ? m.reasoning : undefined, identity(m.usage), identity(m.timings), identity(m.context), (m.tool_calls || []).map(identity), (m.attachments || []).map(identity)]; }
  function timestamp(value) { return new Date(typeof value === 'number' && value < 100000000000 ? value * 1000 : value); }
  function toast(message) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4200); }
  function banner(message) { $('banner').textContent = message || ''; $('banner').hidden = !message; }
  function configurationWarning() { return [S.config.upstream_available === false ? S.config.upstream_error || 'The model server is unavailable. Check the companion connection and reload to reconnect.' : '', S.config.tools_error ? 'Tool connection issue: ' + S.config.tools_error : ''].filter(Boolean).join(' '); }
  function runError(run) {
    const error = run.error || (run.status === 'interrupted' ? 'The response was interrupted.' : 'The response failed.');
    if (!error.includes('chat template refused the messages')) return error;
    const messages = new Map(S.messages.map(m => [m.id, m])), seen = new Set();
    let message = messages.get(run.message_id);
    while (message && !seen.has(message.id)) {
      seen.add(message.id);
      if ((message.attachments || []).some(a => (a.data || '').startsWith('data:image/')))
        return 'The server rejected this message with images. Edit the message to remove the images or use an image-capable model and server.';
      message = messages.get(message.parent_id);
    }
    return error;
  }
  function snapshotBanner() {
    const leaf = S.messages.find(message => message.id === S.chat?.active_leaf);
    const run = S.runs.find(item => item.id === leaf?.run_id);
    const failed = run && ['failed', 'interrupted'].includes(run.status);
    banner(failed ? runError(run) : configurationWarning());
  }
  function currentRun() { return S.runs.find(r => !terminal(r.status)); }
  function closeStream() { if (S.stream) S.stream.close(); S.stream = null; }
  document.addEventListener('companion:unauthorized', () => {
    sessionReady = navigationReady = false; pendingView = null; ++routeEpoch;
    closeStream(); closePassages(); closeDrawers(false); renderLandingDrafts(false);
  });
  function connection(text, offline = false) { const el = $('connection'); el.replaceChildren(node('span', 'status-dot'), document.createTextNode(text)); el.classList.toggle('offline', offline); document.querySelector('.inspector-bottom').replaceChildren(node('span', 'status-dot'), document.createTextNode(offline ? text : 'Connected to the model server')); }
  function settings() {
    const value = {model: $('model').value, system: $('setting-system').value};
    for (const key of ['temperature', 'top_p', 'top_k']) {
      const input = $('setting-' + key);
      if (!input.checkValidity() || input.value.trim() === '') throw new Error('Check the ' + key.replaceAll('_', ' ') + ' setting.');
      value[key] = Number(input.value);
    }
    const cap = $('setting-max_tokens');
    if ($('setting-reply-limit').value === 'custom' && (!cap.checkValidity() || cap.value.trim() === '')) throw new Error('Enter a positive whole number for the reply limit.');
    value.max_tokens = $('setting-reply-limit').value === 'none' ? null : Number(cap.value);
    value.tools = S.chat?.temporary ? [] : Array.from($('tool-settings').querySelectorAll('input:checked')).map(el => el.value);
    return value;
  }
  function settingsForRun() {
    const value = settings();
    if (!(S.config.models || []).some(model => model.id === value.model)) throw new Error('This model is not available on the connected server. Choose another model before sending.');
    return value;
  }
  function renderSettingsState() {
    const chat = S.chat, pending = settingsWrites.get(chat?.id), button = $('save-conversation-settings');
    button.hidden = !chat;
    button.disabled = !chat || Boolean(pending);
    const help = !chat ? 'Send a message to save these settings with your first reply.'
      : pending === 'reply' ? 'Wait for the reply request to be accepted before saving. You can keep editing.'
      : pending === 'save' ? 'Saving settings for this chat...'
      : 'Sending or retrying saves these settings. Save for this chat keeps them without starting a reply.';
    if ($('conversation-settings-help').textContent !== help) $('conversation-settings-help').textContent = help;
    $('conversation-settings-unsaved').hidden = !chat;
  }
  async function writeConversationSettings(chatId, operation, write) {
    if (settingsWrites.has(chatId)) throw new Error('Wait for the pending reply request or settings save.');
    settingsWrites.set(chatId, operation); renderSettingsState(); scheduleRender();
    try { return await write(); }
    finally { settingsWrites.delete(chatId); renderSettingsState(); scheduleRender(); }
  }
  async function admitRun(chatId, body) {
    await writeConversationSettings(chatId, 'reply', async () => {
      await api('/chats/' + encodeURIComponent(chatId) + '/runs', 'POST', body);
      admitted(chatId);
    });
  }
  async function saveConversationSettings() {
    const chat = S.chat, generation = S.generation;
    if (!chat) return;
    const value = settings();
    const saved = await writeConversationSettings(chat.id, 'save', () => api('/chats/' + encodeURIComponent(chat.id), 'PATCH', {settings: value}));
    if (S.chat?.id === chat.id && S.generation === generation) {
      S.chat.settings = saved.settings;
      toast('Settings saved for this chat.');
    }
    if (!chat.temporary) await refreshChats();
  }
  function modelChoices(selected) {
    $('model').replaceChildren();
    const models = S.config.models || [];
    models.forEach(model => { const option = node('option', '', model.id); option.value = model.id; $('model').append(option); });
    if (!models.length) { const option = node('option', '', selected ? selected + ' (unavailable)' : 'No model available'); option.value = selected || ''; option.disabled = true; $('model').append(option); }
    else if (selected && !models.some(m => m.id === selected)) { const option = node('option', '', selected + ' (unavailable)'); option.value = selected; option.disabled = true; $('model').append(option); }
    if (selected) $('model').value = selected;
  }
  function updateModelState(value) {
    if (!Array.isArray(value.models) || typeof value.upstream_available !== 'boolean') return;
    S.config = {...S.config, upstream_available: value.upstream_available, upstream_error: value.upstream_error};
    if (value.upstream_available) { S.config.models = value.models; modelChoices($('model').value); }
    document.dispatchEvent(new CustomEvent('companion:config', {detail: S.config}));
    connection(value.upstream_available ? 'Connected' : 'Unavailable', !value.upstream_available);
    scheduleRender();
  }
  function loadSettings() {
    const defaults = S.config.settings_defaults; if (!defaults) return;
    const value = {...defaults, ...S.chat?.settings};
    modelChoices(value.model);
    for (const key of ['system', 'temperature', 'top_p', 'top_k']) $('setting-' + key).value = value[key];
    $('setting-reply-limit').value = value.max_tokens === null ? 'none' : 'custom';
    $('setting-max_tokens').value = value.max_tokens === null ? '' : value.max_tokens;
    renderReplyLimit(); renderContextLimits(); renderToolSelections(value.tools || []); renderSettingsState();
  }
  function renderReplyLimit() {
    const custom = $('setting-reply-limit').value === 'custom';
    $('reply-token-limit').hidden = !custom;
    $('setting-max_tokens').disabled = !custom; $('setting-max_tokens').required = custom;
  }
  function renderContextLimits() {
    const model = (S.config.models || []).find(m => m.id === $('model').value);
    const valid = value => Number.isSafeInteger(value) && value > 0;
    const age = runtimeLimits ? Date.now() / 1000 - runtimeLimits.observed : Infinity;
    const server = runtimeLimits?.server;
    const fresh = S.config.upstream_available && age >= -5 && age < runtimeLimits?.maxAge && server?.model === model?.id;
    const modelText = model && valid(model.context_length) ? number(model.context_length) + ' tokens' : 'Unavailable';
    const requestText = fresh && valid(server.request_tokens) ? number(server.request_tokens) + ' tokens' : 'Unavailable';
    if ($('model-context-limit').textContent !== modelText) $('model-context-limit').textContent = modelText;
    if ($('request-context-limit').textContent !== requestText) $('request-context-limit').textContent = requestText;
  }
  function renderToolSelections(selected = []) {
    const out = $('tool-settings'); out.replaceChildren();
    if (S.chat?.temporary) { out.append(node('p', 'muted fine-print', 'Tools are disabled in temporary chats.')); return; }
    const tools = S.config.tools || [], groups = new Map();
    tools.forEach(tool => {
      const connectionId = typeof tool.connection === 'string' ? tool.connection : '';
      const server = (S.config.tool_servers || []).find(item => item.id === connectionId);
      const key = connectionId || tool.server || 'available';
      if (!groups.has(key)) groups.set(key, {name: server?.name || tool.server || 'Available tools', tools: []});
      groups.get(key).tools.push(tool);
    });
    groups.forEach(group => {
      const section = node('section', 'tool-selection-group');
      section.append(node('strong', 'tool-selection-title', window.ChatRender.connectionLabel(group.name)));
      group.tools.forEach(tool => {
        const label = node('label', 'tool-setting'); const input = node('input'); input.type = 'checkbox'; input.value = tool.name; input.checked = selected.includes(tool.name);
        const detail = node('span', '', tool.name);
        detail.append(node('small', '', tool.approval === true ? 'Approval required' : tool.approval === false ? 'Approval not required' : tool.connection === 'builtin:workspace' ? 'Approval varies by tool' : 'Approval policy is managed by the operator'));
        if (tool.description) detail.append(node('small', '', tool.description));
        label.append(input, detail); section.append(label);
      });
      out.append(section);
    });
    const names = new Set(tools.map(tool => tool.name));
    selected.filter(name => !names.has(name)).forEach(name => {
      const label = node('label', 'tool-setting unavailable-tool'); const input = node('input'); input.type = 'checkbox'; input.value = name; input.checked = true;
      const detail = node('span', '', name); detail.append(node('small', '', 'Unavailable. Selection preserved'));
      label.append(input, detail); out.append(label);
    });
    if (!tools.length && !selected.length) out.append(node('p', 'muted fine-print', S.config.tools_error || 'No tools available. Add MCP servers with --tools PATH when starting the companion.'));
  }
  document.addEventListener('companion:tools-updated', event => {
    const selected = S.chat?.temporary ? [] : Array.from($('tool-settings').querySelectorAll('input:checked')).map(input => input.value);
    S.config = {...S.config, ...event.detail};
    renderToolSelections(selected);
    snapshotBanner();
  });
  async function refreshChats() {
    const data = await api('/chats'); S.chats = data.chats || [];
    const recent = S.chats.filter(chat => !chat.archived && chat.active_leaf).sort((a, b) => b.updated - a.updated).slice(0, 3);
    const list = $('recent-list'), today = new Date();
    list.replaceChildren();
    for (const chat of recent) {
      const row = node('button', 'recent-chat'); row.type = 'button'; row.dataset.chat = chat.id;
      row.append(node('span', 'recent-name', chat.title || 'Untitled conversation'));
      const date = timestamp(chat.updated);
      if (!Number.isNaN(date.valueOf())) {
        const label = date.toDateString() === today.toDateString() ? date.toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'})
          : date.toLocaleDateString([], {month: 'short', day: 'numeric', year: date.getFullYear() === today.getFullYear() ? undefined : 'numeric'});
        const time = node('time', '', label); time.dateTime = date.toISOString(); time.title = date.toLocaleString(); row.append(time);
      }
      list.append(row);
    }
    $('recent-conversations').hidden = !recent.length;
    renderLandingDrafts();
    queueChatSearch();
  }
  function queueChatSearch() {
    const query = $('search').value.trim();
    chatSearch.request = query ? {query, archived: S.archive} : null;
    chatSearch.chats = [];
    chatSearch.notice = query ? 'Searching conversations...' : '';
    clearTimeout(chatSearch.timer);
    chatSearch.timer = query ? setTimeout(searchChats, 250) : null;
    renderSidebar();
  }
  async function searchChats() {
    chatSearch.timer = null;
    const request = chatSearch.request;
    if (chatSearch.running || !request) return;
    chatSearch.running = true;
    try {
      const data = await api('/chats/search', 'POST', request);
      if (request === chatSearch.request) { chatSearch.chats = data.chats; chatSearch.notice = ''; }
    } catch (error) {
      if (request === chatSearch.request) chatSearch.notice = 'Search failed. ' + error.message;
    } finally {
      chatSearch.running = false;
      if (request === chatSearch.request) renderSidebar();
      else if (chatSearch.request && chatSearch.timer === null) searchChats();
    }
  }
  function updateSidebarCue(button) {
    const cue = button.querySelector('.chat-run-cue');
    const status = sidebarRuns.states.get(button.dataset.chat) || '';
    if (cue.dataset.status === status) return;
    const [mark, label] = sidebarRunCues.get(status) || ['', ''];
    cue.dataset.status = status;
    cue.textContent = mark;
    cue.title = label;
    if (label) button.setAttribute('aria-label', button.querySelector('.chat-name').textContent + ', ' + label);
    else button.removeAttribute('aria-label');
  }
  function setSidebarStates(states = []) {
    sidebarRuns.states = new Map(states.filter(row => sidebarRunCues.has(row.status)).map(row => [row.id, row.status]));
    $('chat-list').querySelectorAll('.chat-item').forEach(updateSidebarCue);
  }
  function stopSidebarRuns() {
    sidebarRuns.active = false;
    sidebarRuns.epoch++;
    clearTimeout(sidebarRuns.poll);
    clearTimeout(sidebarRuns.expires);
    sidebarRuns.request?.abort();
    setSidebarStates();
  }
  function syncSidebarRuns() {
    const active = sidebarRuns.ready && !document.hidden && $('app').dataset.view === 'chat';
    if (active === sidebarRuns.active) return;
    if (!active) { stopSidebarRuns(); return; }
    sidebarRuns.active = true;
    pollSidebarRuns();
  }
  async function pollSidebarRuns() {
    if (!sidebarRuns.active || sidebarRuns.request) return;
    const epoch = sidebarRuns.epoch;
    const maxAge = 9000;
    const deadline = performance.now() + maxAge;
    const request = new AbortController();
    sidebarRuns.request = request;
    const timeout = setTimeout(() => request.abort(), maxAge);
    try {
      const response = await fetch('/api/chats/run-status', {credentials: 'same-origin', signal: request.signal});
      if (!sidebarRuns.active || epoch !== sidebarRuns.epoch || request.signal.aborted || performance.now() >= deadline) return;
      if (response.status === 401) document.dispatchEvent(new CustomEvent('companion:unauthorized'));
      if (!response.ok) return;
      const data = await response.json();
      const remaining = deadline - performance.now();
      if (sidebarRuns.active && epoch === sidebarRuns.epoch && remaining > 0) {
        setSidebarStates(data.chats || []);
        clearTimeout(sidebarRuns.expires);
        sidebarRuns.expires = setTimeout(setSidebarStates, remaining);
      }
    } catch (_) { /* Keep the last snapshot only until its freshness deadline. */ }
    finally {
      clearTimeout(timeout);
      sidebarRuns.request = null;
      if (sidebarRuns.active) sidebarRuns.poll = setTimeout(pollSidebarRuns, epoch === sidebarRuns.epoch ? 3000 : 0);
    }
  }
  function renderSidebar() {
    const chats = chatSearch.request ? chatSearch.chats : S.chats.filter(c => Boolean(c.archived) === S.archive);
    if (!chatSearch.request) chats.sort((a, b) => Number(Boolean(b.pinned)) - Number(Boolean(a.pinned)) || (String(b.updated).localeCompare(String(a.updated))));
    const groups = new Map();
    chats.forEach(chat => {
      const group = chatSearch.request ? '' : chat.pinned ? 'pinned' : chat.folder ? 'folder:' + chat.folder : 'recent';
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group).push(chat);
    });
    const out = $('chat-list');
    for (const section of out.querySelectorAll('.history-section')) {
      if (section.open) closedSections.delete(section.dataset.section);
      else closedSections.add(section.dataset.section);
    }
    out.replaceChildren();
    groups.forEach((items, group) => {
      let container = out;
      if (group) {
        const name = group === 'pinned' ? 'Pinned' : group === 'recent' ? 'Recent' : items[0].folder;
        const key = (S.archive ? 'archive:' : 'chats:') + group;
        const section = node('details', 'history-section');
        section.open = !closedSections.has(key);
        const heading = node('summary', 'folder-heading');
        const label = group.startsWith('folder:') && (name === 'Pinned' || name === 'Recent') ? name + ' (folder)' : name;
        heading.append(icon('m8 10 4 4 4-4'), node('span', 'section-name', label), node('span', 'section-count', String(items.length)));
        section.append(heading);
        section.dataset.section = key;
        out.append(section);
        container = section;
      }
      items.forEach(chat => {
        const button = node('button', 'chat-item' + (S.chat?.id === chat.id ? ' active' : '')); button.type = 'button'; button.dataset.chat = chat.id; button.title = chat.title;
        if (S.chat?.id === chat.id) button.setAttribute('aria-current', 'page');
        button.append(node('span', 'chat-name', chat.title || 'Untitled conversation'));
        const cue = node('span', 'chat-run-cue'); cue.setAttribute('aria-hidden', 'true'); button.append(cue);
        updateSidebarCue(button);
        if (chat.pinned) button.append(icon('M9 3h6l-1 6 4 4H6l4-4Z M12 13v8', 'pin-mark'));
        const row = node('div', 'chat-row' + (S.chat?.id === chat.id ? ' active' : ''));
        const menu = node('button', 'chat-row-menu'); menu.append(icon('M5 12h.2m6.7 0h.2m6.7 0h.2')); menu.type = 'button'; menu.dataset.action = 'chat-menu'; menu.dataset.id = chat.id;
        menu.setAttribute('aria-label', 'Actions for ' + (chat.title || 'Untitled conversation')); menu.setAttribute('aria-haspopup', 'menu'); menu.title = 'Conversation actions';
        row.append(button, menu); container.append(row);
      });
    });
    if (!chats.length) out.append(node('p', 'empty-history', chatSearch.notice || (chatSearch.request ? 'No matching conversations.' : S.archive ? 'Your archive is empty.' : 'No saved conversations yet.')));
    $('library-title').textContent = chatSearch.request ? 'Search results' : S.archive ? 'Archive' : 'Conversations';
    $('archive-toggle').textContent = S.archive ? 'Back to chats' : 'Show archive';
  }
  const viewRoutes = {chat: 'chat', workbench: 'saved-files', monitor: 'monitor', preferences: 'settings'};
  let routeEpoch = 0, sessionReady = false, navigationReady = false, shownEntry = null, pendingView = null;
  function saveRoute(replace = false) {
    const view = $('app').dataset.view, temporary = view === 'chat' && Boolean(S.chat?.temporary);
    const search = view === 'chat' && S.chat && !temporary ? '?chat=' + encodeURIComponent(S.chat.id) : '?view=' + viewRoutes[view];
    const previous = history.state?.llmxChat, same = search === location.search;
    if (same && previous?.temporary === temporary) { shownEntry = previous; return; }
    const push = !replace && !same;
    shownEntry = {position: (previous?.position ?? 0) + Number(push), temporary};
    history[push ? 'pushState' : 'replaceState']({llmxChat: shownEntry}, '', location.pathname + search);
  }
  function selectView(view) {
    if (!sessionReady) { if (!$('login-dialog').open) pendingView = view; return; }
    ++routeEpoch;
    window.ChatShell.renderView(view);
    saveRoute();
  }
  async function restoreRoute() {
    const epoch = ++routeEpoch, previous = shownEntry, entry = history.state?.llmxChat;
    const query = new URLSearchParams(location.search), linkedChat = query.get('chat');
    const legacy = !query.has('chat') && !query.has('view');
    const view = query.has('chat') ? 'chat' : Object.keys(viewRoutes).find(name => viewRoutes[name] === query.get('view')) || 'chat';
    window.ChatShell.closeMenu(false);
    if (view !== 'chat') { window.ChatShell.renderView(view); saveRoute(true); return; }
    const temporary = entry?.temporary || legacy ? sessionStorage.getItem('llmx-temporary-chat') : null;
    const lastSelected = legacy || temporary ? localStorage.getItem('llmx-chat') : null;
    const recent = S.chats.some(chat => chat.id === lastSelected) ? lastSelected : null;
    const restorePrevious = () => {
      if (epoch !== routeEpoch) return;
      const offset = previous?.position - entry?.position;
      if (offset) history.go(offset); else saveRoute(true);
    };
    const target = temporary || linkedChat || recent;
    if (S.chat?.temporary && S.chat.id === target) { window.ChatShell.renderView('chat'); saveRoute(true); return; }
    if (S.chat?.temporary) {
      try { if (!await leaveTemporary()) { restorePrevious(); return; } }
      catch (error) { toast(error.message); restorePrevious(); return; }
    }
    if (epoch !== routeEpoch) return;
    for (const id of temporary ? [temporary, recent] : [target]) {
      if (!id) continue;
      if (id === temporary || !id.startsWith('tmp_')) {
        try {
          const opened = await openChat(id, null, {epoch, record: false});
          if (opened) saveRoute(true);
          return;
        } catch (error) {
          if (epoch !== routeEpoch) return;
          if (error.status !== 404) throw error;
        }
      }
      if (id === temporary) {
        sessionStorage.removeItem('llmx-temporary-chat');
        toast('Your temporary chat has ended or expired.');
      } else {
        refreshChats().catch(error => toast(error.message));
        toast('This conversation is no longer available.');
      }
    }
    if (S.chat) { clearChat({keepDraft: true}); renderSidebar(); }
    else { restoreDraft(); render(); }
    window.ChatShell.renderView('chat'); saveRoute(true);
  }
  document.addEventListener('companion:navigate', event => selectView(event.detail));
  window.addEventListener('popstate', () => {
    if (navigationReady) restoreRoute().catch(error => toast(error.message));
  });
  async function openChat(id, snapshot = null, {epoch = ++routeEpoch, record = true} = {}) {
    if (epoch !== routeEpoch) return false;
    if (S.chat?.temporary && S.chat.id !== id && !await leaveTemporary()) return false;
    if (epoch !== routeEpoch) return false;
    rememberDraft(); flushDrafts();
    hideReplyStats();
    pdfSelection?.abort();
    const generation = ++S.generation; banner(configurationWarning());
    const isCurrent = () => generation === S.generation && epoch === routeEpoch;
    let data;
    try { data = snapshot || await chatSnapshot(id); }
    catch (error) { if (isCurrent()) throw error; return false; }
    if (!isCurrent()) return false;
    rememberDraft(); flushDrafts();
    closeStream();
    closePassages();
    S.chat = data.chat; S.messages = data.messages || []; S.runs = data.runs || []; S.edit = null; S.fingerprint.clear(); S.inspectorKey = '';
    snapshotBanner();
    restoreDraft();
    if (S.chat.temporary) sessionStorage.setItem('llmx-temporary-chat', id); else localStorage.setItem('llmx-chat', id);
    loadSettings(); renderSidebar(); render();
    window.ChatShell.renderView('chat'); closeDrawers();
    if (record) saveRoute();
    requestAnimationFrame(() => bottom());
    const run = currentRun(); if (run) subscribe(run);
    return true;
  }
  function chatCreating(value) {
    creatingChat = value;
    document.querySelectorAll('[data-action="home"],[data-action="new"],[data-action="temporary"]').forEach(control => { control.disabled = value; control.setAttribute('aria-busy', String(value)); });
  }
  async function newChat({landing = false} = {}) {
    if (creatingChat || S.busy) return;
    if (!landing && S.chat && !S.chat.temporary && !S.chat.archived && !S.messages.length && !S.runs.length) {
      selectView('chat'); closeDrawers(); $('draft').focus(); return;
    }
    const epoch = ++routeEpoch;
    chatCreating(true);
    try {
      if (S.chat?.temporary && !await leaveTemporary()) return;
      if (epoch !== routeEpoch) return;
      if (landing) {
        if (S.chat) { clearChat({keepDraft: true}); renderSidebar(); }
        window.ChatShell.renderView('chat'); saveRoute();
        closeDrawers(false); $('draft').focus();
        return;
      }
      const generation = S.generation, selectedChat = S.chat?.id;
      const chat = await api('/chats', 'POST', {settings: selectedChat ? S.config.settings_defaults : settings()}); await refreshChats();
      if (!selectedChat && S.generation === generation && !S.chat && epoch === routeEpoch) {
        rememberDraft(); const value = drafts.get('new'); forgetDraft('new');
        if (value) { drafts.set(chat.id, value); changedDrafts.add(chat.id); }
      }
      if (generation === S.generation && selectedChat === S.chat?.id && await openChat(chat.id, {chat, messages: [], runs: []}, {epoch})) {
        if (!selectedChat) forgetDraft('new');
        $('draft').focus();
      }
    } finally { chatCreating(false); }
  }
  function clearChat({keepDraft = false} = {}) {
    closePassages();
    hideReplyStats();
    pdfSelection?.abort();
    const old = S.chat;
    if (keepDraft) { rememberDraft(); flushDrafts(); } else forgetDraft(draftKey());
    draftReady = false; closeStream(); ++S.generation;
    if (old?.temporary) sessionStorage.removeItem('llmx-temporary-chat'); else localStorage.removeItem('llmx-chat');
    S.chat = null; S.messages = []; S.runs = []; S.edit = null; S.attachments = []; S.fingerprint.clear(); S.inspectorKey = '';
    restoreDraft(); loadSettings(); render();
  }
  async function leaveTemporary() {
    if (!S.chat?.temporary) return true;
    if (currentRun() || S.busy) throw new Error('Stop the response before ending this temporary chat.');
    const id = S.chat.id;
    const end = async () => {
      try { await api('/chats/' + encodeURIComponent(id), 'DELETE'); }
      catch (error) { if (error.status !== 404) throw error; }
      if (S.chat?.id === id) clearChat();
      await refreshChats();
    };
    if (S.messages.length || $('draft').value.trim() || S.attachments.length)
      return Boolean(await window.ChatShell.conversationForm({title: 'End temporary chat?', description: 'Its messages will be discarded.', destructive: true, submit: 'End chat'}, end));
    await end(); return true;
  }
  async function endTemporaryChat() {
    const epoch = ++routeEpoch;
    if (await leaveTemporary() && epoch === routeEpoch) saveRoute(true);
  }
  async function temporaryChat() {
    if (creatingChat || S.busy) return;
    if (S.chat?.temporary) {
      chatCreating(true);
      try { await endTemporaryChat(); }
      finally { chatCreating(false); }
      return;
    }
    if (!S.config.temporary?.enabled) throw new Error('Temporary chats are unavailable on this server.');
    const epoch = ++routeEpoch;
    chatCreating(true);
    try {
      const values = settings(); values.tools = [];
      const chat = await api('/chats', 'POST', {temporary: true, settings: values});
      if (await openChat(chat.id, {chat, messages: [], runs: []}, {epoch})) $('draft').focus();
      else await api('/chats/' + encodeURIComponent(chat.id), 'DELETE');
    } finally { chatCreating(false); }
  }
  function sidebarChat(id) { return chatSearch.chats.find(chat => chat.id === id) || S.chats.find(chat => chat.id === id) || (S.chat?.id === id ? S.chat : null); }
  function showChatMenu(id, anchor) {
    const chat = sidebarChat(id);
    if (!chat) throw new Error('Create a conversation to manage it.');
    menuId = chat.id;
    const items = chat.temporary ? [] : [
      {action: 'rename', label: 'Rename'},
      {action: 'pin', label: chat.pinned ? 'Unpin' : 'Pin'},
      {action: 'folder', label: 'Move to folder'},
      {action: 'duplicate', label: 'Duplicate', separator: true},
      {action: 'archive', label: chat.archived ? 'Restore from archive' : 'Archive'}
    ];
    const openTranscript = anchor.closest('.topbar') && S.messages.length;
    if (openTranscript) {
      items.unshift({action: 'find-message', label: 'Find in conversation'});
      if (path().some(message => message.role === 'user')) items.splice(1, 0, {action: 'outline', label: 'Conversation outline'});
      items.push({action: 'transcript', label: 'Download Markdown', separator: true});
    }
    items.push({action: 'export', label: 'Export JSON', separator: !openTranscript}, {action: 'delete', label: chat.temporary ? 'End temporary chat' : 'Delete conversation', danger: true, separator: true, disabled: Boolean(chat.temporary && (currentRun() || S.busy))});
    window.ChatShell.menu(anchor, chat.temporary ? 'Temporary chat' : chat.title || 'Conversation', items, name => action(name, {dataset: {id: chat.id}}).catch(error => toast(error.message)));
  }
  function path() {
    const map = new Map(S.messages.map(m => [m.id, m])); const selected = []; const seen = new Set();
    let id = S.chat?.active_leaf;
    while (id && map.has(id) && !seen.has(id)) { seen.add(id); const message = map.get(id); selected.unshift(message); id = message.parent_id; }
    return selected;
  }
  function dialogueMessages() { return path().filter(m => m.role === 'user' || m.role === 'assistant'); }
  function focusTranscript(target, block) {
    if (!target?.isConnected) return;
    closeDrawers(false);
    target.tabIndex = -1;
    target.focus({preventScroll: true});
    target.scrollIntoView({block, behavior: 'instant'});
  }
  function jumpToMessage(id) {
    focusTranscript($('messages').querySelector('[data-message="' + CSS.escape(id) + '"]'), 'center');
  }
  function readingMidpoint() {
    const region = $('scroll-region');
    return region.getBoundingClientRect().top + region.clientHeight / 2;
  }
  function readingIndex(elements, middle) {
    let low = 0, high = elements.length;
    while (low < high) {
      const index = Math.floor((low + high) / 2);
      if (elements[index].getBoundingClientRect().top <= middle) low = index + 1;
      else high = index;
    }
    return Math.max(0, low - 1);
  }
  let turnArticles = [], readingFrame = 0;
  function syncReadingPosition() {
    if (readingFrame || $('reading-navigation').hidden) return;
    readingFrame = requestAnimationFrame(() => {
      readingFrame = 0;
      if ($('reading-navigation').hidden || $('app').dataset.view !== 'chat') return;
      const count = turnArticles.length;
      const text = count ? 'Turn ' + (readingIndex(turnArticles, readingMidpoint()) + 1) + ' of ' + count : 'Conversation';
      if ($('reading-position').textContent !== text) $('reading-position').textContent = text;
    });
  }
  const answerHeadings = ':scope > .markdown :is(h2,h3,h4,h5,h6)';
  function showOutline(messages, list, earlier, later) {
    const byId = new Map(messages.map(message => [message.id, message]));
    const turns = [], headings = [], labels = new Map();
    for (const article of $('messages').children) {
      const id = article.dataset.message, message = byId.get(id);
      if (message?.role === 'user') {
        const label = message.content?.trim() || (message.attachments || []).map(file => file.name).join(', ') || 'Empty prompt';
        turns.push({id, label, element: article, number: turns.length + 1});
        labels.clear();
      } else if (message?.role === 'assistant') {
        article.querySelectorAll(answerHeadings).forEach((element, heading) => {
          const label = element.textContent.trim();
          if (!label) return;
          const repeats = labels.get(label) || {count: 0};
          repeats.count++; labels.set(label, repeats);
          headings.push({id, label, element, heading, number: turns.length, repeats, occurrence: repeats.count});
        });
      }
    }
    const prompts = $('outline-turns'), sections = $('outline-headings');
    prompts.querySelector('span').textContent = turns.length;
    sections.querySelector('span').textContent = headings.length;
    let entries, byHeading, first, current, atLocation;
    const size = 25;
    function select(index, focus = true) {
      const start = Math.floor(index / size) * size;
      if (start !== first) {
        first = start;
        const rows = document.createDocumentFragment();
        for (let i = first; i < Math.min(first + size, entries.length); i++) {
          const entry = entries[i], item = node('button', 'find-message-result');
          item.type = 'button'; item.tabIndex = -1; item.dataset.target = entry.id;
          const number = node('strong', '', byHeading ? (entry.number ? 'Turn ' + entry.number : 'Reply') : String(entry.number));
          if (byHeading) {
            item.dataset.heading = entry.heading;
            item.dataset.level = entry.element.tagName;
          } else number.setAttribute('aria-label', 'Turn ' + entry.number);
          const snippet = entry.label.length > 160 ? entry.label.slice(0, 157) + '...' : entry.label;
          const title = node('span', 'outline-label', snippet);
          if (entry.repeats?.count > 1) {
            const next = entry.element.nextElementSibling;
            const excerpt = next?.matches('p,ul,ol,blockquote') ? next.textContent.trim().slice(0, 110) : '';
            const position = entry.occurrence + ' of ' + entry.repeats.count;
            title.append(node('small', 'outline-excerpt', position + (excerpt ? ' \u00b7 ' + excerpt : '')));
          }
          item.append(number, title);
          if (i === current && atLocation) {
            item.setAttribute('aria-current', 'location');
            item.append(node('small', '', 'Current'));
          }
          item.onclick = () => {
            $('find-message-dialog').close();
            if (entry.element.isConnected && (!byHeading || entry.element.textContent.trim() === entry.label))
              return focusTranscript(entry.element, byHeading ? 'start' : 'center');
            const article = $('messages').querySelector('[data-message="' + CSS.escape(entry.id) + '"]');
            const heading = byHeading && article?.querySelectorAll(answerHeadings)[entry.heading];
            if (heading && heading.textContent.trim() === entry.label) focusTranscript(heading, 'start');
            else if (article) {
              focusTranscript(article, 'start');
              if (byHeading) toast('This heading changed. Showing its reply.');
            }
          };
          rows.append(item);
        }
        list.replaceChildren(rows);
        earlier.hidden = first === 0; later.hidden = first + size >= entries.length;
        const count = entries.length, unit = byHeading ? 'heading' : 'turn';
        $('find-message-count').textContent = !count ? (byHeading ? 'No reply headings on this branch.' : 'No prompts on this branch.')
          : count > size ? (byHeading ? 'Headings ' : 'Turns ') + (first + 1) + ' to ' + Math.min(first + size, count) + ' of ' + count
          : count + ' ' + unit + (count === 1 ? '' : 's');
      }
      const previous = list.querySelector('[tabindex="0"]'), item = list.children[index - first];
      if (previous) previous.tabIndex = -1;
      if (item) {
        item.tabIndex = 0;
        if (focus) item.focus();
        else item.scrollIntoView({block:'nearest'});
      }
    }
    function choose(headingsMode, focus) {
      byHeading = headingsMode;
      entries = byHeading ? headings : turns;
      first = -1;
      const middle = readingMidpoint();
      current = readingIndex(entries.map(entry => entry.element), middle);
      const heading = byHeading && entries[current]?.element;
      atLocation = !byHeading || (heading && heading.getBoundingClientRect().top <= middle &&
        heading.closest('.message').getBoundingClientRect().bottom >= middle);
      prompts.setAttribute('aria-pressed', String(!byHeading)); sections.setAttribute('aria-pressed', String(byHeading));
      list.setAttribute('aria-description', 'Use Up and Down to move, and Enter to jump.');
      earlier.textContent = byHeading ? 'Earlier headings' : 'Earlier turns';
      later.textContent = byHeading ? 'Later headings' : 'Later turns';
      list.onclick = null;
      select(current, focus);
      if (focus && !entries.length) (byHeading ? sections : prompts).focus();
    }
    prompts.onclick = () => choose(false, false); sections.onclick = () => choose(true, false);
    list.onkeydown = event => {
      const item = event.target.closest('button[data-target]');
      if (!item || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const index = first + Array.prototype.indexOf.call(list.children, item);
      const next = {ArrowUp: index - 1, ArrowDown: index + 1, Home: 0, End: entries.length - 1}[event.key];
      if (typeof next !== 'number') return;
      event.preventDefault(); select(Math.max(0, Math.min(next, entries.length - 1)));
    };
    earlier.onclick = () => select(Math.max(0, first - size));
    later.onclick = () => select(first + size);
    choose(!turns.length && Boolean(headings.length), true);
  }
  function literalPattern(term, flags) {
    return new RegExp(Array.from(term, ch => '^$\\.*+?()[]{}|'.includes(ch) ? '\\' + ch : ch).join(''), flags);
  }
  function searchEntries(messages, cache) {
    const data = new Map(messages.map(message => [message.id, message])), entries = [];
    for (const article of $('messages').children) {
      const message = data.get(article.dataset.message);
      if (!message) continue;
      const fingerprint = S.fingerprint.get(message.id), saved = cache.get(article);
      if (saved && saved.fingerprint === fingerprint) { entries.push({message, article, fields:saved.fields}); continue; }
      const fields = [];
      for (const block of article.querySelectorAll(':scope > .message-attachments > *, :scope > .markdown > *')) {
        let runs = cache.get(block);
        if (!runs) {
          runs = [];
          const filename = block.parentElement.classList.contains('message-attachments');
          const image = filename && block.querySelector('.attachment-image');
          if (image) runs.push({text:image.alt, element:image, nodes:[], filename});
          else {
            const walker = document.createTreeWalker(block, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
              acceptNode: item => item.classList?.contains('code-heading') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT
            });
            let field;
            for (let item = walker.nextNode(); item; item = walker.nextNode()) {
              if (item.nodeName === 'BR') { field = null; continue; }
              if (item.nodeType !== Node.TEXT_NODE || !item.length) continue;
              const element = item.parentElement.closest('p,li,td,th,pre,blockquote,h2,h3,h4,h5,h6') || block;
              if (!field || field.element !== element) {
                field = {text:'', element, nodes:[], filename};
                runs.push(field);
              }
              field.text += item.data; field.nodes.push(item);
            }
          }
          cache.set(block, runs);
        }
        for (const field of runs) fields.push(field);
      }
      cache.set(article, {fingerprint, fields});
      entries.push({message, article, fields});
    }
    return entries;
  }

  let passage = null, passageTimer = null;
  function clearPassageMark() {
    window.CSS.highlights?.delete('chat-passage');
    passage?.marked?.classList.remove('passage-target');
  }
  function closePassages(focus = false) {
    if (!passage) return;
    clearTimeout(passageTimer); passageTimer = null;
    clearPassageMark(); passage = null;
    $('passage-navigation').hidden = true; $('reading-default').hidden = false;
    $('passage-query').textContent = $('passage-position').textContent = '';
    $('passage-edit').removeAttribute('title');
    if (focus) $('reading-default').querySelector('[data-action=find-message]').focus({preventScroll:true});
  }
  function passageRange(field, at, length) {
    let offset = 0, range;
    for (const text of field.nodes) {
      const end = offset + text.length;
      if (!range && at < end) { range = document.createRange(); range.setStart(text, at - offset); }
      if (range && at + length <= end) { range.setEnd(text, at + length - offset); return range; }
      offset = end;
    }
    return null;
  }
  function showPassage(index, scroll, focus = false) {
    const {rows, total, pattern} = passage;
    clearPassageMark();
    const position = $('passage-position');
    if (!total) {
      position.textContent = 'No matches'; passage.index = 0;
      $('passage-next').disabled = $('passage-previous').disabled = true;
      return;
    }
    $('passage-next').disabled = $('passage-previous').disabled = total < 2;
    index = (index + total) % total;
    const row = rows.find(item => item.start + item.count > index);
    pattern.lastIndex = 0;
    let hit;
    for (let i = row.start; i <= index; i++) hit = pattern.exec(row.field.text);
    const {element} = row.field, range = passageRange(row.field, hit.index, hit[0].length);
    if (range && window.CSS.highlights && window.Highlight) CSS.highlights.set('chat-passage', new Highlight(range));
    else { element.classList.add('passage-target'); passage.marked = element; }
    passage.index = index; passage.field = row.field; passage.local = index - row.start;
    const label = (index + 1) + ' of ' + total;
    if (position.textContent !== label) position.textContent = label;
    if (focus) { closeDrawers(false); row.article.tabIndex = -1; row.article.focus({preventScroll:true}); }
    if (!scroll) return;
    const rect = range?.getBoundingClientRect() || element.getBoundingClientRect();
    const horizontal = element.closest('pre,.table-scroll');
    if (horizontal) {
      const bounds = horizontal.getBoundingClientRect();
      if (rect.left < bounds.left || rect.right > bounds.right) horizontal.scrollLeft += rect.left - bounds.left - 12;
    }
    const region = $('scroll-region'), bounds = region.getBoundingClientRect();
    region.scrollTop += rect.top - bounds.top - (region.clientHeight - Math.min(rect.height, region.clientHeight)) / 2;
  }
  function indexPassages(entries, target) {
    const {role, pattern, counts, field, index} = passage;
    let total = 0, selected = null;
    const rows = [];
    for (const {message, article, fields} of entries) {
      if (role !== 'all' && role !== message.role) continue;
      for (const part of fields) {
        let count = counts.get(part);
        if (count === undefined) {
          pattern.lastIndex = 0; count = 0;
          while (pattern.test(part.text)) count++;
          counts.set(part, count);
        }
        if (!count) continue;
        if (selected === null && message.id === target) selected = total;
        if (!target && part === field) selected = total + Math.min(passage.local, count - 1);
        rows.push({article, field:part, count, start:total}); total += count;
      }
    }
    passage.total = total; passage.rows = rows;
    return selected ?? Math.min(index, Math.max(0, total - 1));
  }
  function refreshPassages() {
    if (!passage || passageTimer !== null) return;
    passageTimer = setTimeout(() => {
      passageTimer = null;
      const entries = searchEntries(dialogueMessages(), passage.cache);
      showPassage(indexPassages(entries), false);
    }, 180);
  }
  function startPassages(term, role, id, cache) {
    closePassages();
    passage = {term, role, cache, counts:new WeakMap(), pattern:literalPattern(term,'giu'), rows:[], index:0};
    const index = indexPassages(searchEntries(dialogueMessages(), cache), id);
    if (!passage.rows.some(row => row.article.dataset.message === id)) {
      closePassages(); jumpToMessage(id); return;
    }
    $('reading-default').hidden = true; $('passage-navigation').hidden = false;
    $('passage-query').textContent = term; $('passage-edit').title = 'Edit search: ' + term;
    showPassage(index, true, true);
  }
  function stepPassage(direction) {
    if (!passage) return;
    if (passageTimer !== null) {
      clearTimeout(passageTimer); passageTimer = null;
      passage.index = indexPassages(searchEntries(dialogueMessages(), passage.cache));
    }
    showPassage(passage.index + direction, true);
  }
  $('passage-previous').onclick = () => stepPassage(-1);
  $('passage-next').onclick = () => stepPassage(1);
  $('passage-close').onclick = () => closePassages(true);
  $('passage-edit').onclick = () => { if (passage) findMessage(false, passage); };

  function findMessage(outline = false, previous = null) {
    const dialog = $('find-message-dialog'), query = $('find-message-query'), list = $('find-message-list'), more = $('find-message-more');
    const role = $('find-message-role'), earlier = $('find-message-earlier'), messages = dialogueMessages();
    const cache = new WeakMap();
    const turns = new Map();
    let turn = 0, entries;
    for (const message of messages) {
      if (message.role === 'user') turn++;
      turns.set(message.id, turn);
    }
    let matches = [], shown = 0;
    function appendPage() {
      const start = shown;
      for (const {message, text, filename, raw, at, length, fileIndex} of matches.slice(start, start + 25)) {
        const item = node('button', 'find-message-result');
        item.type = 'button';
        item.dataset.target = message.id;
        if (fileIndex !== undefined) item.dataset.file = fileIndex;
        if (raw && length) item.dataset.source = 'true';
        const from = Math.max(0, at - 40), end = Math.min(text.length, at + length + 100);
        const excerpt = node('span');
        if (filename) excerpt.append('File: ');
        else if (raw && length) excerpt.append('Markdown: ');
        excerpt.append((from ? '...' : '') + text.slice(from, at));
        if (length) excerpt.append(node('mark', '', text.slice(at, at + length)));
        excerpt.append(text.slice(at + length, end) + (end < text.length ? '...' : ''));
        const who = message.role === 'user' ? 'You' : 'Assistant';
        const number = fileIndex === undefined ? 0 : turns.get(message.id);
        item.append(node('strong', '', number ? 'Turn ' + number : who), excerpt);
        list.append(item);
        shown++;
      }
      more.hidden = shown >= matches.length;
      if (start) list.children[start]?.focus();
    }
    function filter() {
      const term = query.value.trim();
      const pattern = term ? literalPattern(term, 'iu') : null;
      matches = [];
      const filesOnly = role.value === 'files';
      if (filesOnly) {
        for (const message of messages.slice().reverse()) {
          (message.attachments || []).forEach((file, fileIndex) => {
            const text = file.name || 'Attachment';
            const hit = pattern?.exec(text);
            if ((!pattern || hit) && (typeof file.text === 'string' || typeof file.data === 'string'))
              matches.push({message, text, fileIndex, at: hit?.index || 0, length: hit?.[0].length || 0});
          });
        }
      } else {
        entries ??= searchEntries(messages, cache).reverse();
        for (const {message, fields} of entries) {
          if (role.value !== 'all' && message.role !== role.value) continue;
          const source = {text:message.content, source:true};
          const searchable = pattern ? [...fields, source] : [source, ...fields];
          for (const {text, filename, source: raw} of searchable) {
            if (!text) continue;
            const hit = pattern?.exec(text);
            if (pattern && !hit) continue;
            matches.push({message, text, filename, raw, at: hit?.index || 0, length: hit?.[0].length || 0});
            break;
          }
        }
      }
      const count = matches.length;
      const noun = filesOnly ? 'attachment' : 'message';
      $('find-message-count').textContent = count ? count + (term ? ' matching ' : ' ') + noun + (count === 1 ? '' : 's') + ', newest first' : 'No matching ' + noun + 's';
      shown = 0;
      list.replaceChildren();
      appendPage();
      list.scrollTop = 0;
    }
    dialog.classList.toggle('prompt-outline', outline);
    $('find-message-title').textContent = outline ? 'Conversation outline' : 'Find in conversation';
    dialog.querySelector('.find-message-controls').hidden = outline;
    $('outline-modes').hidden = !outline;
    list.removeAttribute('aria-description');
    earlier.hidden = true; earlier.onclick = list.onkeydown = null;
    more.textContent = outline ? 'Later turns' : 'Show more';
    query.value = previous?.term || '';
    role.value = previous?.role || 'all';
    role.onchange = filter;
    query.oninput = filter;
    query.onkeydown = event => {
      if (event.key === 'Enter' && !event.isComposing) {
        event.preventDefault();
        list.firstElementChild?.click();
      }
    };
    more.onclick = appendPage;
    list.onclick = event => {
      const item = event.target.closest('button[data-target]');
      if (!item) return;
      const term = query.value.trim();
      dialog.close();
      if (item.dataset.file !== undefined) {
        const article = $('messages').querySelector('[data-message="' + CSS.escape(item.dataset.target) + '"]');
        const control = article?.querySelector('.message-attachments')?.children[Number(item.dataset.file)];
        if (!control?.matches('button')) return toast('This attachment is unavailable.');
        jumpToMessage(item.dataset.target);
        control.focus({preventScroll: true});
        control.click();
        return;
      }
      if (term && !item.dataset.source) startPassages(term, role.value, item.dataset.target, cache);
      else { closePassages(); jumpToMessage(item.dataset.target); }
    };
    dialog.onclose = () => {
      if (dialog.open) return;
      query.value = '';
      $('outline-turns').onclick = $('outline-headings').onclick = role.onchange = query.oninput = query.onkeydown = earlier.onclick = more.onclick = list.onclick = list.onkeydown = dialog.onclose = null;
      list.replaceChildren();
    };
    dialog.showModal();
    if (outline) { closePassages(); showOutline(messages, list, earlier, more); }
    else { filter(); query.focus(); }
  }
  function replyGroups() {
    const groups = []; let prompt = '', previous = null;
    for (const message of path()) {
      if (message.role === 'tool') continue;
      if (message.role === 'user') { prompt = message.content || 'Attachment'; previous = null; continue; }
      if (message.role !== 'assistant') { previous = null; continue; }
      if (!previous || !message.run_id || previous.run_id !== message.run_id)
        groups.push({id: message.id, prompt, calls: []});
      groups.at(-1).calls.push(message); previous = message;
    }
    return groups;
  }
  function tokenTotals(calls) {
    return Object.fromEntries(['input', 'output', 'cached', 'total'].map(key => {
      const values = calls.map(m => usage(m)[key]);
      return [key, values.length && values.every(v => v !== undefined) ? values.reduce((a, b) => a + b, 0) : undefined];
    }));
  }
  function button(text, action, id, cls = 'text-button') { const el = node('button', cls, text); el.type = 'button'; el.dataset.action = action; if (id) el.dataset.id = id; return el; }
  let actionTooltip = null, tooltipTarget = null, tooltipSequence = 0;
  function hideActionTooltip(target, force = false) {
    if (!actionTooltip || target && tooltipTarget !== target) return;
    if (!force && target && (target.matches(':hover') || target === document.activeElement)) return;
    tooltipTarget?.removeAttribute('aria-describedby'); actionTooltip.remove(); actionTooltip = null; tooltipTarget = null;
  }
  function showActionTooltip(target) {
    hideActionTooltip();
    const tip = node('div', 'message-action-tooltip', target.dataset.tooltip || target.getAttribute('aria-label'));
    tip.id = 'message-action-tooltip-' + (++tooltipSequence); tip.setAttribute('role', 'tooltip'); document.body.append(tip);
    target.setAttribute('aria-describedby', tip.id); tooltipTarget = target; actionTooltip = tip;
    const box = target.getBoundingClientRect(), size = tip.getBoundingClientRect(), pad = 8;
    const left = Math.max(pad, Math.min(box.left + (box.width - size.width) / 2, innerWidth - size.width - pad));
    const below = box.bottom + size.height + pad <= innerHeight;
    const top = below ? box.bottom + pad : Math.max(pad, box.top - size.height - pad);
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
  }
  function bindActionTooltip(el) {
    el.addEventListener('pointerenter', () => showActionTooltip(el)); el.addEventListener('pointerleave', () => hideActionTooltip(el));
    el.addEventListener('blur', () => hideActionTooltip(el)); el.addEventListener('click', () => hideActionTooltip(el, true));
  }
  function actionIcon(label, action, id, path, cls = 'message-icon-action') {
    const el = button('', action, id, cls); el.setAttribute('aria-label', label);
    el.append(icon(path));
    if (action !== 'reply-stats') bindActionTooltip(el);
    return el;
  }
  bindActionTooltip(document.querySelector('[data-action=attach]'));
  document.addEventListener('focusin', event => { if (event.target.matches('.message-icon-action:not([data-action=reply-stats]),.branch-action,[data-tooltip]')) showActionTooltip(event.target); }, true);
  document.addEventListener('keydown', event => { if (event.key === 'Escape') hideActionTooltip(); }, true);
  document.addEventListener('scroll', () => { if (tooltipTarget && tooltipTarget === document.activeElement) showActionTooltip(tooltipTarget); else hideActionTooltip(); }, true);
  window.addEventListener('resize', () => hideActionTooltip());
  function renderAttachment(attachment, draft = false) {
    const image = /^data:image\/(png|jpeg);base64,[A-Za-z0-9+/=\r\n]+$/.test(attachment.data || '');
    const name = attachment.name || (image ? 'Attached image' : 'Text attachment');
    if (!image && typeof attachment.text !== 'string') return node('span', draft ? '' : 'attachment-pill', name);
    const open = node('button', draft ? 'draft-attachment-open' : image ? 'attachment-picture' : 'attachment-pill', name);
    open.type = 'button';
    open.setAttribute('aria-haspopup', 'dialog');
    open.setAttribute('aria-label', (image ? 'View image ' : 'Read attachment ') + name);
    if (image && !draft) {
      const thumbnail = node('img', 'attachment-image'); thumbnail.src = attachment.data; thumbnail.alt = name; thumbnail.loading = 'lazy';
      open.replaceChildren(thumbnail);
    }
    open.addEventListener('click', () => {
      const generation = S.generation;
      const addToDraft = draft ? null : () => {
        if (generation !== S.generation || !open.isConnected) throw new Error('Reopen the attachment in the current conversation.');
        appendAttachments([attachment]);
        return $('draft');
      };
      if (image) window.ChatArtifacts.openImage(name, attachment.data, addToDraft);
      else window.ChatArtifacts.open({name, source: attachment.text}, {readOnlyAttachment: true, allowWorkspaceEdit: !S.chat?.temporary, addToDraft});
    });
    return open;
  }
  function toolJson(value) {
    if (typeof value !== 'string') return value;
    try { return JSON.parse(value); } catch (_) { return value; }
  }
  function validWorkspacePath(path) {
    return typeof path === 'string' && path && [...path].length <= 240 && !/[:\\\x00-\x1f]/.test(path)
      && path.split('/').every(part => part && part !== '.' && part !== '..');
  }
  function workspaceToolTarget(call, result, parts) {
    if (call.name === 'workspace_write') {
      const path = toolJson(call.arguments)?.path;
      if (!validWorkspacePath(path)) return null;
      const error = call.error || (result?.isError && parts.length === 1 && parts[0]?.type === 'text' && parts[0].text);
      const conflict = typeof error === 'string' && (error === 'Workspace changed. Reload before saving'
        || error === 'Wait for the active command to finish or cancel it' || /^Workspace revision is \d+\./.test(error));
      if (call.status === 'failed' && conflict) return {path, savedCopy: true};
      if (call.status !== 'completed' || result?.isError || parts.length !== 1 || parts[0]?.type !== 'text') return null;
      const value = toolJson(parts[0].text);
      return Array.isArray(value?.files) && value.files.some(file => file?.path === path) ? {path} : null;
    }
    if (parts.length !== 1 || parts[0]?.type !== 'text') return null;
    const value = toolJson(parts[0].text);
    if (call.name === 'workspace_run' && typeof value?.id === 'string' && /^[0-9a-f]{32}$/.test(value.id)) return {jobId: value.id};
    if (result?.isError) return null;
    if (call.name === 'workspace_read' && call.status === 'completed' && validWorkspacePath(value?.path)) return {path: value.path};
    return null;
  }
  let workspaceFileRead;
  function workspaceFileActions(target, fromRead) {
    const row = node('div', 'workspace-tool-file');
    const openLabel = target.savedCopy ? 'Open saved copy' : 'Open file';
    const open = node('button', fromRead ? 'workspace-tool-open' : 'workspace-tool-view-file');
    const attach = node('button', 'workspace-tool-attach');
    const error = node('p', 'error-text workspace-tool-file-error');
    error.setAttribute('role', 'alert'); error.hidden = true;
    for (const [control, label, glyph] of [
      [open, openLabel, 'M9 5h10v10m-1-9L7 17M6 5H4v15h15v-3'],
      [attach, 'Attach to draft', 'm8 12 6-6a3 3 0 0 1 4 4l-8 8a5 5 0 0 1-7-7l9-9']
    ]) {
      control.type = 'button';
      control.title = control === attach ? 'Attach the current saved copy of ' + target.path + ' to the draft' : label + ' ' + target.path;
      control.setAttribute('aria-label', control.title);
      const caption = node('span', '', control === attach ? 'Attach' : label);
      if (control === attach) caption.append(node('span', 'file-action-suffix', ' to draft'));
      control.append(icon(glyph), caption); row.append(control);
    }
    row.append(error);
    row.addEventListener('click', async event => {
      const control = event.target.closest('button');
      if (!control || control.getAttribute('aria-busy') === 'true') return;
      if (control === open && fromRead) { window.ChatWorkspace.openToolResult(target); return; }
      const draft = control === attach ? projectDraft() : null;
      workspaceFileRead?.abort();
      const request = new AbortController(), generation = S.generation;
      workspaceFileRead = request; error.hidden = true;
      control.setAttribute('aria-busy', 'true'); control.setAttribute('aria-disabled', 'true');
      const leave = event => { if (event.detail !== 'chat') request.abort(); };
      const expire = () => request.abort();
      document.addEventListener('companion:view', leave);
      document.addEventListener('companion:unauthorized', expire, {once: true});
      const isCurrent = () => !request.signal.aborted && S.generation === generation && row.isConnected
        && $('app').dataset.view === 'chat' && !document.querySelector('dialog[open]');
      try {
        const signal = AbortSignal.any([request.signal, AbortSignal.timeout(10000)]);
        const file = await api('/workspace/file?path=' + encodeURIComponent(target.path), 'GET', undefined, signal);
        if (!isCurrent()) return;
        if (file.path !== target.path || typeof file.content !== 'string') throw new Error('Invalid saved file response.');
        if (draft) {
          draft.apply({files: [{path: file.path, text: file.content}]}, false);
          $('draft').focus(); return;
        }
        const basename = target.path.split('/').at(-1), html = /\.html?$/i.test(basename);
        const language = html ? 'html' : basename.match(/\.([^.]+)$/)?.[1];
        window.ChatArtifacts.open({name: basename, source: file.content, language}, {
          initialPreview: html, workspacePath: target.path
        });
      } catch (problem) {
        if (isCurrent()) {
          error.textContent = problem.status === 404 ? 'This saved file is no longer available.'
            : problem.name === 'TimeoutError' ? 'Reading the saved file timed out. Try again.'
            : draft ? problem.message : 'Could not open the saved file.';
          error.hidden = false;
        }
      } finally {
        document.removeEventListener('companion:view', leave);
        document.removeEventListener('companion:unauthorized', expire);
        if (workspaceFileRead === request) workspaceFileRead = null;
        control.removeAttribute('aria-busy'); control.removeAttribute('aria-disabled');
      }
    });
    return row;
  }
  function imageResultLinks(result, parts) {
    let payload = result?.structuredContent ?? result;
    if (!Array.isArray(payload?.images) && parts.length === 1 && parts[0]?.type === 'text'
        && typeof parts[0].text === 'string' && parts[0].text.length <= 65536) {
      payload = toolJson(parts[0].text);
    }
    if (!Array.isArray(payload?.images) || !payload.images.length) return null;
    const list = node('ul', 'tool-image-results');
    list.setAttribute('aria-label', 'Image search results');
    for (const item of payload.images.slice(0, 20)) {
      if (!item || typeof item !== 'object') continue;
      const title = typeof item.title === 'string' && item.title.trim();
      const image = window.ChatRender.link(title || 'Open image', item.image_url);
      const thumbnail = window.ChatRender.link(image ? 'Thumbnail' : title || 'Open thumbnail', item.thumbnail_url);
      if (!image && !thumbnail) continue;
      const row = node('li'), links = node('span');
      row.append(image || thumbnail);
      const source = window.ChatRender.link('Source', item.source_url);
      if (source) {
        if (title) source.setAttribute('aria-label', 'Source for ' + title);
        links.append(source);
      }
      if (image && thumbnail && image.href !== thumbnail.href) {
        if (title) thumbnail.setAttribute('aria-label', 'Thumbnail for ' + title);
        links.append(thumbnail);
      }
      if (links.childElementCount) row.append(links);
      list.append(row);
    }
    if (!list.childElementCount) return null;
    if (payload.images.length <= 20) return list;
    const group = node('div');
    group.append(list, node('small', 'muted', 'The full result includes more entries.'));
    return group;
  }
  function toolRow(call, message) {
    const wrap = node('div', 'tool-call'); wrap.dataset.toolCall = call.id || ''; const detail = node('details'); const summary = node('summary'); summary.tabIndex = 0;
    const status = node('span', 'badge' + (call.error ? ' error' : ''), call.status || 'pending');
    status.dataset.status = call.status || 'pending';
    const result = toolJson(call.result), parts = Array.isArray(result?.content) ? result.content : [];
    const target = workspaceToolTarget(call, result, parts);
    const fromRead = call.name === 'workspace_read';
    const name = target?.path ? (fromRead ? 'Read' : target.savedCopy ? 'Write failed' : 'Wrote') : call.name || 'Call';
    const body = node('div', 'tool-body');
    if (target?.path) {
      summary.append(node('span', 'tool-name tool-file-verb', name), node('code', 'tool-file-path', target.path), status);
      body.append(node('code', 'tool-operation', call.name));
      wrap.append(workspaceFileActions(target, fromRead));
    } else {
      summary.append(icon('m8 8-4 4 4 4m8-8 4 4-4 4m-3-10-2 12', 'tool-symbol'), node('span', 'tool-name', name), status);
    }
    body.append(node('div', 'section-label', 'ARGUMENTS'), node('pre', '', json(call.arguments ?? {})));
    if (call.result !== undefined) {
      body.append(node('div', 'section-label', 'RESULT'));
      const full = json(result); const readable = parts.filter(part => part?.type === 'text' && typeof part.text === 'string').map(part => part.text).join('\n\n');
      const preview = readable || full;
      const images = call.status === 'completed' && !call.error && !result?.isError && imageResultLinks(result, parts);
      body.append(images || node('pre', '', preview.slice(0, 16384) + (preview.length > 16384 ? '\n[Preview truncated]' : '')));
      const structured = node('details'); structured.append(node('summary', '', 'Structured result'), node('pre', '', full.slice(0, 16384) + (full.length > 16384 ? '\n[Preview truncated]' : ''))); body.append(structured);
      if (full.length > 16384 || preview.length > 16384) {
        const download = node('button', 'secondary-button', 'Download full result');
        download.type = 'button';
        download.addEventListener('click', () => window.ChatArtifacts.downloadBlob(new Blob([full], {type: 'application/json'}), 'tool-result.json'));
        body.append(download);
      }
      if (!target?.path && ['workspace_list', 'workspace_read', 'workspace_run'].includes(call.name)) {
        const label = target?.jobId ? 'Open job output' : 'Open saved files';
        const open = node('button', 'secondary-button workspace-tool-open', label);
        open.type = 'button'; open.addEventListener('click', () => window.ChatWorkspace.openToolResult(target));
        const link = node('div', 'workspace-tool-link');
        link.append(open); wrap.append(link);
      }
    }
    if (call.error) body.append(node('p', 'error-text', json(call.error)));
    if (call.duration_ms !== undefined) body.append(node('small', 'muted', number(call.duration_ms, ' ms')));
    if (call.name === 'workspace_write' && target?.path) {
      const edit = node('button', 'secondary-button workspace-tool-open', 'Open in Workspace');
      edit.type = 'button'; edit.addEventListener('click', () => window.ChatWorkspace.openToolResult(target)); body.append(edit);
    }
    detail.append(summary, body); wrap.prepend(detail);
    if (call.status === 'approval') {
      detail.open = true;
      const approval = node('div', 'approval', 'Review the requested arguments above.'); const actions = node('div', 'approval-actions');
      const allow = button('Allow once', 'approve', call.id, 'primary-button'); const deny = button('Deny this call', 'deny', call.id, 'secondary-button');
      allow.dataset.run = message.run_id; deny.dataset.run = message.run_id;
      const label = node('label'); const remember = node('input'); remember.type = 'checkbox'; remember.className = 'remember-approval';
      label.append(remember, document.createTextNode(' Also allow later calls to this tool in this reply, even with different arguments.'));
      remember.addEventListener('change', () => { allow.textContent = remember.checked ? 'Allow for this reply' : 'Allow once'; });
      actions.append(allow, deny, label); approval.append(actions); wrap.append(approval);
    }
    return wrap;
  }
  function messageMarkdown(text, mounted) {
    if (!mounted) return markdown(text);
    updateMarkdown(mounted, text);
    // The replacement marks its slot while the mounted body keeps focus and selection.
    return node('div', 'markdown');
  }
  function messageView(message, siblings, content, reasoningBody) {
    const el = node('article', 'message ' + message.role); el.dataset.message = message.id;
    const header = node('div', 'message-header');
    header.append(node('span', '', message.role === 'assistant' ? 'Assistant' : message.role === 'tool' ? 'Tool result' : 'You'));
    if (message.created) { const time = timestamp(message.created); if (!isNaN(time)) header.append(node('span', 'message-time', time.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}))); }
    el.append(header);
    if (message.attachments?.length) { const attachments = node('div', 'message-attachments'); message.attachments.forEach(a => attachments.append(renderAttachment(a))); el.append(attachments); }
    if (message.reasoning) {
      const reasoning = node('details', 'reasoning'); reasoning.dataset.disclosure = 'reasoning';
      const summary = node('summary');
      summary.append(icon('M4 8h10m-6 4h12M4 16h9'), document.createTextNode('Reasoning'));
      reasoning.append(summary, messageMarkdown(message.reasoning, reasoningBody)); el.append(reasoning);
    }
    if (message.content) el.append(messageMarkdown(message.content, content));
    (message.tool_calls || []).forEach(call => {
      const tool = toolRow(call, message);
      tool.querySelectorAll('details').forEach((detail, index) => { detail.dataset.disclosure = 'tool:' + call.id + ':' + index; });
      el.append(tool);
    });
    if (!message.content && !message.reasoning && !message.tool_calls?.length && message.role === 'assistant' && !terminal(message.status)) el.append(node('span', 'muted fine-print', 'Preparing a response...'));
    if (message.role === 'assistant' && message.status === 'completed' && message.reasoning?.trim()
        && !message.content?.trim() && !message.tool_calls?.length) {
      const notice = message.finish_reason === 'stop' ? 'The model stopped before sending an answer.'
        : message.finish_reason === 'length' ? 'The model reached its reply limit before sending an answer.'
        : 'The model did not send an answer.';
      el.append(node('p', 'muted fine-print reply-no-answer', notice));
    }
    const actions = node('div', 'message-actions');
    if (message.content) actions.append(actionIcon('Copy message', 'copy', message.id, 'M8 5H5v16h12v-3 M9 3h11v14H9z'));
    if (message.role === 'user') actions.append(actionIcon('Edit message', 'edit', message.id, 'M4 16.5V20h3.5L19 8.5 15.5 5 4 16.5z M13.8 6.7l3.5 3.5'));
    if (message.role === 'assistant') actions.append(actionIcon('Retry response', 'retry', message.id, 'M20 7v5h-5 M19 12a7 7 0 1 1-2-5'));
    if (message.role === 'assistant' && message.content) {
      const quote = actionIcon('Quote in reply', 'quote', message.id, 'M4 6h6v6l-4 6H4l3-6H4z M14 6h6v6l-4 6h-2l3-6h-3z');
      quote.addEventListener('mousedown', event => { if (event.button === 0) event.preventDefault(); });
      actions.append(quote);
    }
    if (siblings.length > 1) {
      const index = siblings.findIndex(m => m.id === message.id); const branch = node('span', 'branch-switch');
      const prev = actionIcon('Previous branch', 'branch', siblings[(index + siblings.length - 1) % siblings.length].id, 'M15 18l-6-6 6-6', 'branch-action');
      const next = actionIcon('Next branch', 'branch', siblings[(index + 1) % siblings.length].id, 'M9 18l6-6-6-6', 'branch-action');
      branch.append(prev, node('span', '', (index + 1) + ' / ' + siblings.length), next); actions.append(branch);
    }
    if (message.status && ['failed', 'cancelled', 'interrupted'].includes(message.status)) actions.append(node('span', 'error-text', message.status));
    if (message.role === 'assistant') {
      const stats = actionIcon('Reply statistics', 'reply-stats', message.id, 'M5 20V10 M12 20V4 M19 20v-7');
      stats.setAttribute('aria-haspopup', 'dialog'); stats.setAttribute('aria-controls', 'reply-statistics'); stats.setAttribute('aria-expanded', 'false');
      actions.append(stats);
    }
    el.append(actions); return el;
  }
  function reconcileChildren(current, replacement, preserved) {
    const keep = new Set(preserved.map(([, stable]) => stable));
    const stableFor = new Map(preserved);
    Array.from(current.children).forEach(child => { if (!keep.has(child)) child.remove(); });
    let cursor = current.firstChild;
    Array.from(replacement.children).forEach(child => {
      const stable = stableFor.get(child);
      if (stable) {
        if (stable !== cursor) current.insertBefore(stable, cursor);
        cursor = stable.nextSibling;
      } else current.insertBefore(child, cursor);
    });
  }
  function reconcileFocusedAction(current, replacement) {
    const focused = document.activeElement;
    const actions = focused?.closest('.message-actions');
    if (!actions || !current.contains(actions)) return null;
    const nextActions = replacement.querySelector('.message-actions');
    const direction = focused.dataset.action === 'branch' ? focused.getAttribute('aria-label') : null;
    const next = Array.from(nextActions.querySelectorAll('button')).find(button =>
      button.dataset.action === focused.dataset.action && button.dataset.id === focused.dataset.id &&
      (!direction || button.getAttribute('aria-label') === direction));
    if (!next) return null;
    if (focused.parentElement === actions) {
      reconcileChildren(actions, nextActions, [[next, focused]]);
    } else {
      const group = focused.parentElement, nextGroup = next.parentElement;
      reconcileChildren(group, nextGroup, [[next, focused]]);
      reconcileChildren(actions, nextActions, [[nextGroup, group]]);
    }
    return [nextActions, actions];
  }
  function render() {
    const nearBottom = followsLatest();
    const selected = path().filter(m => m.role !== 'tool'); const container = $('messages'); const wanted = new Set(selected.map(m => m.id));
    const childrenByParent = new Map();
    for (const message of S.messages) {
      const children = childrenByParent.get(message.parent_id);
      if (children) children.push(message);
      else childrenByParent.set(message.parent_id, [message]);
    }
    const articles = new Map();
    for (const el of container.querySelectorAll(':scope > .message')) {
      const id = el.dataset.message;
      if (wanted.has(id)) articles.set(id, el);
      else { el.remove(); S.fingerprint.delete(id); }
    }
    turnArticles = [];
    selected.forEach((message, index) => {
      const siblings = childrenByParent.get(message.parent_id).filter(m => m.role === message.role);
      const fingerprint = JSON.stringify(messageKey(message, true)) + siblings.map(m => m.id).join();
      let old = articles.get(message.id);
      if (S.fingerprint.get(message.id) !== fingerprint || !old) {
        const disclosures = new Map(old ? Array.from(old.querySelectorAll('details[data-disclosure]'), detail => [detail.dataset.disclosure, detail]) : []);
        const content = old?.querySelector(':scope > .markdown'), reasoning = old?.querySelector('.reasoning');
        const replacement = messageView(message, siblings, content, reasoning?.querySelector('.markdown'));
        replacement.querySelectorAll('details[data-disclosure]').forEach(detail => {
          const previous = disclosures.get(detail.dataset.disclosure);
          const enteringApproval = detail.open && !previous?.parentElement.querySelector('.approval');
          if (previous && !enteringApproval) detail.open = previous.open;
        });
        if (old) {
          const preserved = [];
          const action = reconcileFocusedAction(old, replacement);
          if (action) preserved.push(action);
          const nextContent = replacement.querySelector(':scope > .markdown'), nextReasoning = replacement.querySelector('.reasoning');
          if (content && nextContent) preserved.push([nextContent, content]);
          if (reasoning && nextReasoning) preserved.push([nextReasoning, reasoning]);
          const focused = document.activeElement?.matches('.tool-call details[data-disclosure^="tool:"][data-disclosure$=":0"] > summary') ? document.activeElement.parentElement : null;
          const keepDetail = focused && old.contains(focused) ? focused : null;
          if (keepDetail) {
            const next = Array.from(replacement.querySelectorAll('details[data-disclosure^="tool:"]')).find(item => item.dataset.disclosure === keepDetail.dataset.disclosure);
            const body = next?.querySelector(':scope > .tool-body');
            const nextWrap = next?.closest('.tool-call');
            const nextSummary = next?.querySelector(':scope > summary'); const summary = keepDetail.querySelector(':scope > summary');
            if (nextWrap && body && nextSummary && summary) {
              keepDetail.querySelector(':scope > .tool-body')?.replaceWith(body);
              keepDetail.open = next.open;
              summary.replaceChildren(...nextSummary.childNodes);
              const stableWrap = keepDetail.closest('.tool-call');
              reconcileChildren(stableWrap, nextWrap, [[next, keepDetail]]);
              preserved.push([nextWrap, stableWrap]);
            }
          }
          if (preserved.length) reconcileChildren(old, replacement, preserved);
          else old.replaceChildren(...replacement.childNodes);
        } else { container.append(replacement); old = replacement; articles.set(message.id, old); }
        S.fingerprint.set(message.id, fingerprint);
      }
      if (message.role === 'user') turnArticles.push(old);
      const previous = selected[index - 1];
      const continuation = message.role === 'assistant' && previous?.role === 'assistant' && Boolean(message.run_id) && message.run_id === previous.run_id;
      old.classList.toggle('continuation', continuation);
      if (continuation) old.setAttribute('aria-label', 'Assistant continuation'); else old.removeAttribute('aria-label');
      if (container.children[index] !== old) container.insertBefore(old, container.children[index] || null);
    });
    if (tooltipTarget && !tooltipTarget.isConnected) hideActionTooltip();
    container.querySelector('.live-state')?.remove();
    const run = currentRun();
    if (run) { const live = node('div', 'live-state'); live.append(node('span', 'status-dot'), node('span', 'live-phase', run.status === 'approval' ? 'Waiting for your approval' : 'Working'), node('span', 'live-metrics'), node('span', 'run-elapsed')); container.append(live); }
    $('welcome').hidden = selected.length > 0; $('messages').hidden = selected.length === 0 && !run;
    $('reading-navigation').hidden = !selected.length;
    $('chat-title').textContent = S.chat ? S.chat.title || 'Untitled conversation' : 'Chat';
    $('chat-title').disabled = !S.chat;
    const temporaryMode = Boolean(S.chat?.temporary);
    $('temporary-notice').hidden = !temporaryMode;
    $('temporary-button').querySelector('.chat-create-label').textContent = temporaryMode ? 'Normal chat' : 'Temporary chat';
    $('temporary-button').querySelector('.temporary-short-label').textContent = temporaryMode ? 'Normal' : 'Temporary';
    $('temporary-button').setAttribute('aria-label', temporaryMode ? 'Switch to a normal chat' : 'Start a temporary chat');
    $('app').dataset.temporary = String(temporaryMode);
    window.ChatProjectContext?.setLanding(!selected.length && !temporaryMode);
    const lifetime = S.config.temporary?.ttl_seconds;
    $('temporary-description').textContent = 'Not saved to history or backups. Tools are off.' + (Number.isFinite(lifetime) ? ' Expires after ' + number(lifetime / 60) + ' minutes of inactivity.' : '') + ' The model server may retain logs or caches.';
    const modelUnavailable = !(S.config.models || []).some(model => model.id === $('model').value);
    $('send').hidden = Boolean(run); $('stop').hidden = !run; $('send').disabled = S.busy || modelUnavailable || settingsWrites.has(S.chat?.id);
    if (!run && modelUnavailable && S.config.upstream_available) banner('This model is not available on the connected server. Choose another model before sending.');
    if ($('app').dataset.view === 'chat') document.title = (S.chat?.title ? S.chat.title + ' / ' : '') + 'llmx Chat';
    const groups = replyGroups();
    groups.forEach(group => group.calls.forEach((m, index) => {
      const control = articles.get(m.id)?.querySelector('[data-action=reply-stats]');
      if (control) { control.dataset.reply = group.id; control.hidden = index !== group.calls.length - 1; control.setAttribute('aria-expanded', String(replyStats?.id === group.id)); }
    }));
    positionReplyStats(); syncLiveMetrics(); renderInspector(); renderContextLimits(); tick();
    syncJump(); refreshPassages(); if (nearBottom && !passage) requestAnimationFrame(bottom);
  }
  function syncLiveMetrics() {
    const run = currentRun();
    const message = run?.status === 'running' ? path().findLast(m => m.role === 'assistant' && m.status === 'running' && m.run_id === run.id) : null;
    const visible = $('app').dataset.view === 'chat' && !document.hidden;
    const id = visible ? message?.request?.id || null : null;
    const needed = Boolean(id || visible && S.tab === 'settings' && !$('app').classList.contains('inspector-closed'));
    if (id === liveMetrics.id && Boolean(liveMetrics.stream) === needed) return;
    liveMetrics.stream?.close();
    Object.assign(liveMetrics, {id, stream: null, history: [], row: null, received: 0, collector: null});
    runtimeLimits = null; renderContextLimits();
    if (!needed) return;
    const stream = new EventSource('/api/monitor/stream'); liveMetrics.stream = stream;
    stream.addEventListener('history', event => {
      if (liveMetrics.stream !== stream) return;
      try {
        const data = JSON.parse(event.data), live = data.latest?.requests?.live;
        const interval = Number.isFinite(data.interval_s) && data.interval_s > 0 ? data.interval_s : 2;
        runtimeLimits = data.ok && Number.isFinite(data.last_success) ? {server: data.latest?.server, observed: data.last_success, maxAge: Math.max(5, interval * 3)} : null;
        renderContextLimits();
        if (data.collector_id !== liveMetrics.collector || data.samples?.some(sample => sample.reset)) liveMetrics.history = [];
        liveMetrics.collector = data.collector_id;
        if (!data.ok || !Number.isFinite(live?.sample_elapsed_ms) || !Array.isArray(live.items)) {
          liveMetrics.history = []; liveMetrics.row = null; renderLiveMetrics(); return;
        }
        const item = live.items.find(row => row.id === id);
        if (!item) { liveMetrics.history = []; liveMetrics.row = null; renderLiveMetrics(); return; }
        const previous = liveMetrics.history.at(-1);
        if (previous && live.sample_elapsed_ms < previous.timestamp) liveMetrics.history = [];
        const sample = {timestamp: live.sample_elapsed_ms, observedAt: performance.now() / 1000, items: new Map([[id, item]])};
        liveMetrics.history.push(sample); liveMetrics.history = liveMetrics.history.slice(-64);
        const rate = key => previous?.timestamp === sample.timestamp ? null : window.LLMXRates.requestWindowRate(liveMetrics.history, id, key, interval);
        liveMetrics.row = {...item, pp: rate('prefilled_tokens'), tg: rate('published_tokens')};
        liveMetrics.received = performance.now();
        renderLiveMetrics();
      } catch (_) { runtimeLimits = null; renderContextLimits(); liveMetrics.history = []; liveMetrics.row = null; renderLiveMetrics(); }
    });
    stream.onerror = () => { if (liveMetrics.stream === stream) { runtimeLimits = null; renderContextLimits(); liveMetrics.history = []; liveMetrics.row = null; renderLiveMetrics(); } };
  }
  function renderLiveMetrics() {
    const phase = document.querySelector('.live-phase'), metrics = document.querySelector('.live-metrics');
    if (!phase || !metrics) return;
    const run = currentRun();
    const row = performance.now() - liveMetrics.received <= 15000 ? liveMetrics.row : null;
    const labels = {queued: 'Queued', paused: 'Paused', encode: 'Processing image', prefill: 'Processing prompt', generate: 'Generating'};
    phase.textContent = run?.status === 'approval' ? 'Waiting for your approval' : row ? labels[row.phase] || 'Working' : 'Working';
    metrics.textContent = '';
    if (!liveMetrics.id) return;
    if (!row) { metrics.textContent = 'Live rates unavailable'; return; }
    const parts = [];
    if (row.phase === 'prefill' && Number.isSafeInteger(row.prefilled_tokens) && row.prefilled_tokens >= 0)
      parts.push(number(row.prefilled_tokens) + ' prompt tokens processed');
    if (row.phase === 'prefill' && Number.isFinite(row.pp)) parts.push(number(row.pp) + ' prompt tok/s');
    if (row.phase === 'generate' && Number.isFinite(row.tg)) parts.push(number(row.tg) + ' generated tok/s');
    metrics.textContent = parts.join(' / ') || 'Waiting for a measured interval';
  }
  function scheduleRender() { if (S.scheduled) return; S.scheduled = true; requestAnimationFrame(() => { S.scheduled = false; render(); }); }
  function valueAt(object, keys) { for (const key of keys) { const value = key.split('.').reduce((o, k) => o?.[k], object); if (typeof value === 'number' && Number.isFinite(value)) return value; } return undefined; }
  function usage(message) { const u = message.usage || {}; return {input: valueAt(u, ['prompt_tokens', 'input_tokens']), output: valueAt(u, ['completion_tokens', 'output_tokens']), cached: valueAt(u, ['prompt_tokens_details.cached_tokens', 'input_tokens_details.cached_tokens', 'cached_tokens']), total: valueAt(u, ['total_tokens'])}; }
  function statRow(target, label, value) { const row = node('div', 'stat-row'); row.append(node('span', '', label), node('strong', '', value)); target.append(row); }
  let replyStats = null, statsTimer = null, closingStats = false;
  function statsAnchor() {
    return replyStats && $('messages').querySelector('[data-action=reply-stats][data-reply="' + CSS.escape(replyStats.id) + '"]:not([hidden])');
  }
  function hideReplyStats(restoreFocus = false) {
    clearTimeout(statsTimer);
    if (!replyStats) return;
    const anchor = statsAnchor();
    replyStats.panel.remove(); replyStats = null;
    anchor?.setAttribute('aria-expanded', 'false');
    if (restoreFocus && anchor) { closingStats = true; anchor.focus({preventScroll: true}); closingStats = false; }
  }
  function positionReplyStats() {
    if (!replyStats) return;
    const anchor = statsAnchor();
    if (!anchor || $('app').dataset.view !== 'chat') { hideReplyStats(); return; }
    const box = anchor.getBoundingClientRect(), panel = replyStats.panel, size = panel.getBoundingClientRect(), pad = 12;
    panel.style.left = Math.max(pad, Math.min(box.left, innerWidth - size.width - pad)) + 'px';
    panel.style.top = Math.max(pad, Math.min(box.bottom + 8 + size.height <= innerHeight ? box.bottom + 8 : box.top - size.height - 8, innerHeight - size.height - pad)) + 'px';
  }
  function scheduleStatsClose() {
    clearTimeout(statsTimer);
    statsTimer = setTimeout(() => {
      if (!replyStats || replyStats.pinned) return;
      const anchor = statsAnchor(), panel = replyStats.panel;
      if (anchor?.matches(':hover') || panel.matches(':hover') || anchor === document.activeElement || panel.contains(document.activeElement)) return;
      hideReplyStats();
    }, 160);
  }
  function showReplyStats(anchor, pin = false) {
    if (closingStats) return;
    clearTimeout(statsTimer);
    const id = anchor.dataset.reply;
    if (replyStats?.pinned && !pin) return;
    if (replyStats?.id === id) {
      if (pin && replyStats.pinned) { hideReplyStats(true); return; }
    } else {
      hideReplyStats();
      const group = replyGroups().find(g => g.id === id); if (!group) return;
      const panel = node('section', 'reply-statistics'); panel.id = 'reply-statistics'; panel.setAttribute('popover', 'manual');
      panel.setAttribute('role', 'dialog'); panel.setAttribute('aria-label', 'Reply statistics'); panel.tabIndex = -1;
      const head = node('div', 'reply-statistics-heading'); head.append(node('h3', '', 'Reply statistics'), button('Close', 'close-stats', '', 'text-button')); panel.append(head);
      const totals = tokenTotals(group.calls);
      [['Input tokens', 'input'], ['Output tokens', 'output'], ['Cached input', 'cached'], ['Total tokens', 'total']].forEach(([label, key]) => statRow(panel, label, number(totals[key])));
      panel.append(node('p', 'fine-print muted', 'Cached input is part of input. Totals include ' + group.calls.length + (group.calls.length === 1 ? ' model call.' : ' model calls.')));
      group.calls.forEach((m, index) => {
        const detail = node('details', 'reply-call-metrics'); detail.open = group.calls.length === 1;
        detail.append(node('summary', '', 'Call ' + (index + 1) + ' / ' + (m.status || 'recorded')));
        const t = m.timings || {}, u = usage(m), body = node('div');
        if (group.calls.length > 1) {
          statRow(body, 'Input tokens', number(u.input)); statRow(body, 'Output tokens', number(u.output));
        }
        statRow(body, 'Prompt processing', number(valueAt(t, ['prompt_per_second', 'prompt_tps', 'prefill_tps']), ' tok/s'));
        statRow(body, 'Generation', number(valueAt(t, ['predicted_per_second', 'decode_tps', 'generation_tps']), ' tok/s'));
        const parts = [t.prepared_ms, t.queued_ms, t.prompt_ms];
        const ttft = parts.every(v => typeof v === 'number' && Number.isFinite(v)) ? parts.reduce((a, b) => a + b, 0) : valueAt(t, ['ttft_ms', 'time_to_first_token_ms']);
        statRow(body, 'Server time to first token', number(ttft, ' ms'));
        const extra = node('details', 'reply-extra-metrics'); extra.append(node('summary', '', 'More measurements'));
        [['Prompt time', 'prompt_ms'], ['Generation time', 'predicted_ms'], ['Queued time', 'queued_ms']].forEach(([label, key]) => statRow(extra, label, number(valueAt(t, [key]), ' ms')));
        statRow(extra, 'Context tokens', number(valueAt(m.context || {}, ['tokens']))); statRow(extra, 'Context limit', number(valueAt(m.context || {}, ['limit'])));
        const raw = node('details'); raw.append(node('summary', '', 'Reported data'), node('pre', '', json({usage: m.usage, timings: m.timings, context: m.context}))); extra.append(raw); body.append(extra);
        detail.append(body); panel.append(detail);
      });
      panel.append(node('p', 'fine-print muted', 'Snapshot when opened. Reopen to refresh. Unavailable values were not reported by the server.'));
      const footer = node('div', 'reply-statistics-footer'); footer.append(node('span', 'fine-print stats-hint', 'Click the statistics button to keep this open.'), button('Copy values', 'copy-stats', '', 'secondary-button')); panel.append(footer);
      panel.dataset.copy = json({totals, calls: group.calls.map(m => ({usage: m.usage, timings: m.timings, context: m.context}))});
      panel.addEventListener('pointerenter', () => clearTimeout(statsTimer)); panel.addEventListener('pointerleave', scheduleStatsClose);
      panel.addEventListener('focusout', scheduleStatsClose);
      document.body.append(panel); replyStats = {id, panel, pinned: false}; panel.showPopover();
    }
    if (pin) { replyStats.pinned = true; replyStats.panel.querySelector('.stats-hint').textContent = 'Pinned. Select text or copy the values.'; replyStats.panel.focus({preventScroll: true}); }
    anchor.setAttribute('aria-expanded', 'true'); positionReplyStats();
  }
  document.addEventListener('pointerover', event => {
    const anchor = event.target.closest('[data-action=reply-stats]');
    if (anchor && event.pointerType !== 'touch' && !anchor.contains(event.relatedTarget)) showReplyStats(anchor);
  });
  document.addEventListener('pointerout', event => { if (event.target.closest('[data-action=reply-stats]')) scheduleStatsClose(); });
  document.addEventListener('focusin', event => { if (event.target.matches('[data-action=reply-stats]')) showReplyStats(event.target); });
  document.addEventListener('focusout', event => { if (event.target.matches('[data-action=reply-stats]')) scheduleStatsClose(); });
  document.addEventListener('pointerdown', event => { if (replyStats && !replyStats.panel.contains(event.target) && !event.target.closest('[data-action=reply-stats]')) hideReplyStats(); });
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && replyStats) { event.preventDefault(); event.stopPropagation(); hideReplyStats(true); } }, true);
  document.addEventListener('scroll', event => { if (replyStats && !replyStats.panel.contains(event.target)) positionReplyStats(); }, true);
  window.addEventListener('resize', positionReplyStats);
  document.addEventListener('companion:view', event => {
    hideReplyStats();
    if (event.detail !== 'chat' && $('attach-source').getAttribute('aria-expanded') === 'true') window.ChatShell.closeMenu(false);
  });
  function renderInspector() {
    const selected = path(), calls = selected.filter(m => m.role === 'assistant'), groups = replyGroups();
    const inspectorKey = JSON.stringify([S.chat?.id, S.messages.map(m => messageKey(m)), selected.map(m => m.id), S.runs.map(r => [r.id, r.status, r.error])]);
    if (inspectorKey === S.inspectorKey) return; S.inspectorKey = inspectorKey;
    const activity = $('panel-activity'), stats = $('panel-stats');
    const openActivity = new Set(Array.from(activity.querySelectorAll('details[open]'), el => el.dataset.activity));
    const focused = activity.contains(document.activeElement) ? document.activeElement.dataset.activityKey : null;
    const allOpen = Boolean(stats.querySelector('details')?.open);
    activity.replaceChildren();
    const run = currentRun();
    const state = node('div', 'stat-row'); state.append(node('span', '', 'Now'), node('strong', '', run ? run.status === 'approval' ? 'Needs approval' : 'Generating' : 'Ready')); activity.append(state);
    function activityError(recordedRun) {
      const issue = node('div', 'activity-error'), message = runError(recordedRun);
      issue.append(node('p', 'error-text', message));
      if (recordedRun.error && recordedRun.error !== message) {
        const detail = node('details'); detail.append(node('summary', '', 'Reported error'), node('pre', '', recordedRun.error)); issue.append(detail);
      }
      return issue;
    }
    function activityCard(group, older = false) {
      const last = group.calls.at(-1), recordedRun = S.runs.find(r => r.id === last.run_id);
      const card = node(older ? 'details' : 'section', 'activity-card');
      const title = node(older ? 'summary' : 'div', 'activity-title');
      title.append(node('span', '', group.prompt ? group.prompt.slice(0, 90) : 'Assistant reply'), node('span', 'badge', recordedRun?.status || last.status || 'recorded'));
      if (older) { card.dataset.activity = group.id; card.open = openActivity.has(group.id); title.dataset.activityKey = group.id; }
      card.append(title);
      const jump = button('View reply', 'reply-detail', last.id, 'text-button'); card.append(jump);
      group.calls.forEach(m => (m.tool_calls || []).forEach(call => {
        const row = node('button', 'stat-row activity-tool-link'); row.type = 'button'; row.dataset.action = 'tool-detail'; row.dataset.message = m.id; row.dataset.call = call.id || '';
        row.dataset.activityKey = m.id + ':' + (call.id || '');
        row.setAttribute('aria-label', 'Show ' + (call.name || 'tool call') + ' arguments and result in conversation');
        row.append(node('span', '', call.name || 'Tool call'), node('strong', '', call.status || 'pending')); card.append(row);
      }));
      if (recordedRun?.error) card.append(activityError(recordedRun));
      return card;
    }
    if (groups.length) {
      activity.append(node('h3', 'section-label', 'LATEST REPLY'), activityCard(groups.at(-1)));
      if (groups.length > 1) {
        const history = node('details', 'activity-history'); history.dataset.activity = 'history'; history.open = openActivity.has('history');
        const summary = node('summary', '', 'Earlier replies (' + (groups.length - 1) + ')'); summary.dataset.activityKey = 'history'; history.append(summary);
        groups.slice(0, -1).reverse().forEach(group => history.append(activityCard(group, true))); activity.append(history);
      }
    } else activity.append(node('p', 'activity-empty', run ? 'Waiting for the first response.' : 'No reply yet. Progress and tool activity will appear here.'));
    const lastRun = S.runs.filter(r => selected.some(m => m.id === r.message_id)).at(-1);
    if (lastRun?.error && !groups.some(g => g.calls.some(m => m.run_id === lastRun.id))) activity.append(activityError(lastRun));
    stats.replaceChildren(node('h3', 'section-label', 'THIS BRANCH'), node('p', 'panel-intro', 'Totals across model calls on the selected conversation branch. Cached input is included in input.'));
    const sums = tokenTotals(calls), grid = node('div', 'metric-grid');
    [['Input tokens', 'input'], ['Output tokens', 'output'], ['Cached input', 'cached'], ['Total tokens', 'total']].forEach(([label, key]) => { const card = node('div', 'metric-card'); card.append(node('span', '', label), node('strong', sums[key] === undefined ? 'unknown' : '', number(sums[key]))); grid.append(card); }); stats.append(grid);
    statRow(stats, 'Replies', String(groups.length)); statRow(stats, 'Model calls', String(calls.length));
    const allCalls = S.messages.filter(m => m.role === 'assistant');
    if (allCalls.length !== calls.length) {
      const all = node('details', 'conversation-totals'); all.open = allOpen; all.append(node('summary', '', 'Including other branches'));
      const totals = tokenTotals(allCalls);
      for (const [label, key] of [['Input tokens', 'input'], ['Output tokens', 'output'], ['Cached input (subset)', 'cached'], ['Total tokens', 'total']]) statRow(all, label, number(totals[key]));
      statRow(all, 'Model calls', String(allCalls.length)); stats.append(all);
    }
    stats.append(node('p', 'muted fine-print', 'Open the statistics icon beside a reply for its timings and model-call details. Unavailable means a call did not report that value.'));
    if (focused) activity.querySelector('[data-activity-key="' + CSS.escape(focused) + '"]')?.focus({preventScroll: true});
  }
  function subscribe(run) {
    closeStream(); S.cursor = Number(run.last_seq || 0); const chatId = S.chat?.id;
    const stream = new EventSource('/api/runs/' + encodeURIComponent(run.id) + '/events?after=' + S.cursor); S.stream = stream;
    stream.onopen = () => connection('Connected');
    stream.onerror = () => { if (S.stream === stream) connection('Reconnecting', true); };
    stream.addEventListener('update', event => {
      if (S.chat?.id !== chatId || S.stream !== stream) return;
      try {
        const update = JSON.parse(event.data); const seq = Number(update.seq || event.lastEventId);
        if (!Number.isFinite(seq) || seq <= S.cursor) return; S.cursor = seq;
        const data = update.data;
        if (update.kind === 'message') {
          const index = S.messages.findIndex(m => m.id === data.id); if (index < 0) S.messages.push(data); else S.messages[index] = data;
          S.chat.active_leaf = data.id;
        } else if (update.kind === 'delta') {
          const message = S.messages.find(m => m.id === data.message_id);
          if (message) {
            if (data.content) message.content = (message.content || '') + data.content;
            if (data.reasoning) message.reasoning = (message.reasoning || '') + data.reasoning;
            for (const fragment of data.tool_calls || []) {
              message.tool_calls ||= [];
              const index = message.tool_calls.findIndex(call => call.id === fragment.id);
              const old = index < 0 ? {id: fragment.id, index: fragment.index, name: '', arguments: '', status: 'preparing'} : message.tool_calls[index];
              const call = {...old, name: old.name + fragment.name, arguments: old.arguments + fragment.arguments};
              if (index < 0) message.tool_calls.push(call); else message.tool_calls[index] = call;
            }
            message.tool_calls?.sort((a, b) => (a.index || 0) - (b.index || 0));
          }
        } else if (update.kind === 'metrics') {
          const message = S.messages.find(m => m.id === data.message_id); if (message) for (const key of ['usage', 'timings', 'context', 'request']) if (data[key]) message[key] = {...message[key], ...data[key]};
        } else if (update.kind === 'tool') {
          const message = S.messages.find(m => m.id === data.message_id); if (message) { message.tool_calls ||= []; const index = message.tool_calls.findIndex(c => c.id === data.call.id); if (index < 0) message.tool_calls.push(data.call); else message.tool_calls[index] = data.call; }
        } else if (update.kind === 'state') {
          const index = S.runs.findIndex(r => r.id === data.id); if (index < 0) S.runs.push(data); else S.runs[index] = data;
          if (terminal(data.status)) { closeStream(); connection('Connected'); if (data.error) banner(runError(data)); syncSnapshot(chatId).catch(error => banner(error.message)); refreshChats().catch(error => toast(error.message)); }
        }
        scheduleRender();
      } catch (error) { banner('Could not apply a stream update: ' + error.message); }
    });
  }
  async function syncSnapshot(chatId) { const generation = S.generation; const data = await chatSnapshot(chatId); if (S.chat?.id !== chatId || S.generation !== generation) return; S.chat = data.chat; S.messages = data.messages || []; S.runs = data.runs || []; snapshotBanner(); scheduleRender(); }
  function tick() {
    renderLiveMetrics(); renderContextLimits();
    const run = currentRun(); if (!run) { $('run-caption').textContent = 'Enter to send / Shift + Enter for a new line'; return; }
    const seconds = Math.max(0, Math.floor((Date.now() - timestamp(run.created).getTime()) / 1000));
    const elapsed = Number.isFinite(seconds) ? (seconds >= 60 ? Math.floor(seconds / 60) + 'm ' : '') + seconds % 60 + 's elapsed' : 'In progress';
    document.querySelectorAll('.run-elapsed').forEach(el => { el.textContent = elapsed; }); $('run-caption').textContent = elapsed + ' / ' + (run.status === 'approval' ? 'Permission required' : 'Generating');
  }
  async function send() {
    if (currentRun() || S.busy || creatingChat || settingsWrites.has(S.chat?.id)) return;
    const originalChat = S.chat, generation = S.generation, epoch = routeEpoch, edit = S.edit;
    rememberDraft(); const submittedKey = draftKey(), submittedDraft = S.chat?.temporary ? composeValue() : drafts.get(submittedKey);
    const content = $('draft').value.trim(), revision = composerRevision, attachments = S.attachments.slice();
    if (!content && !attachments.length) { $('draft').focus(); return; }
    const requestSettings = settingsForRun();
    let chatId = originalChat?.id, parent = edit ? edit.parent_id : originalChat?.active_leaf;
    let chatTitle = originalChat?.title || 'New conversation';
    let phase = 'create a conversation for this send';
    const isCurrentView = () => S.generation === generation && (chatId ? S.chat?.id === chatId : S.chat === originalChat);
    S.busy = true; $('send').disabled = true;
    banner(configurationWarning());
    try {
      if (!chatId) {
        const chat = await api('/chats', 'POST', {settings: requestSettings});
        chatId = chat.id; parent = null; chatTitle = chat.title || chatTitle;
        S.chats.unshift(chat);
        if (S.generation === generation && S.chat === originalChat && epoch === routeEpoch) {
          rememberDraft(); const value = drafts.get('new'); forgetDraft('new');
          S.chat = chat; localStorage.setItem('llmx-chat', chat.id); saveRoute();
          if (value) { drafts.set(chat.id, value); changedDrafts.add(chat.id); }
        }
        renderSidebar();
      }
      phase = 'send to "' + chatTitle + '"';
      await admitRun(chatId, {content, attachments, parent_id: parent || null, settings: requestSettings});
      const acceptedDraft = originalChat?.temporary ? composerRevision === revision && S.edit === edit : drafts.get(chatId) === submittedDraft;
      for (const key of new Set([submittedKey, chatId])) if (drafts.get(key) === submittedDraft) forgetDraft(key);
      if (S.chat?.id !== chatId) return;
      const acceptedGeneration = S.generation;
      if (acceptedDraft) {
        $('draft').value = ''; S.attachments = []; S.edit = null; composerRevision++; $('edit-note').hidden = true; resizeDraft(); renderAttachments();
      }
      phase = 'refresh "' + chatTitle + '"';
      await syncSnapshot(chatId);
      if (S.chat?.id !== chatId || S.generation !== acceptedGeneration) return;
      const run = currentRun(); if (run) subscribe(run);
      phase = 'refresh conversations after sending to "' + chatTitle + '"';
      await refreshChats();
      if (S.chat?.id === chatId && S.generation === acceptedGeneration) bottom();
    } catch (error) {
      if (!isCurrentView()) { toast('Could not ' + phase + ': ' + error.message); return; }
      throw error;
    } finally { S.busy = false; scheduleRender(); }
  }
  async function retry(id) {
    if (currentRun()) throw new Error('Stop the current response before retrying.');
    let message = S.messages.find(m => m.id === id); const seen = new Set();
    while (message && message.role !== 'user' && !seen.has(message.id)) { seen.add(message.id); message = S.messages.find(m => m.id === message.parent_id); }
    if (!message) throw new Error('The original user message could not be found.');
    const chatId = S.chat.id;
    await admitRun(chatId, {regenerate: true, parent_id: message.id, settings: settingsForRun()});
    if (S.chat?.id !== chatId) return;
    const generation = S.generation;
    await syncSnapshot(chatId);
    if (S.chat?.id !== chatId || S.generation !== generation) return;
    const run = currentRun(); if (run) subscribe(run); bottom();
  }
  async function patchChat(changes, chatId = S.chat?.id) {
    if (!chatId) return;
    await api('/chats/' + encodeURIComponent(chatId), 'PATCH', changes);
    if (S.chat?.id === chatId) await syncSnapshot(chatId);
    await refreshChats();
  }
  async function branch(id) {
    if (currentRun()) throw new Error('Stop the current response before changing branches.');
    let leaf = id; const seen = new Set();
    while (!seen.has(leaf)) { seen.add(leaf); const children = S.messages.filter(m => m.parent_id === leaf); if (!children.length) break; leaf = children.at(-1).id; }
    await patchChat({active_leaf: leaf}); closePassages(); bottom();
  }
  function renderAttachments() {
    const root = $('attachments'); root.replaceChildren(); let total = 0;
    S.attachments.forEach((a, i) => {
      const size = attachmentBytes(a); total += size;
      const pill = node('span', 'attachment-pill'); pill.append(renderAttachment(a, true));
      const label = node('small', 'attachment-file-size', formatAttachmentSize(size));
      label.title = size.toLocaleString() + ' bytes'; label.setAttribute('aria-label', a.name + ', ' + size.toLocaleString() + ' bytes'); pill.append(label);
      const remove = button('x', 'remove-attachment', String(i)); remove.setAttribute('aria-label', 'Remove ' + a.name); pill.append(remove); root.append(pill);
    });
    if (S.attachments.length) {
      const summary = node('span', 'attachment-total', 'Total: ' + S.attachments.length + (S.attachments.length === 1 ? ' file' : ' files') + ' - ' + formatAttachmentSize(total) + ' / 10 MiB');
      summary.setAttribute('role', 'status'); summary.setAttribute('aria-label', 'Total attachment size: ' + total.toLocaleString() + ' bytes of 10 MiB'); root.append(summary);
    }
    rememberDraft();
  }
  function attachmentBytes(attachment) {
    if (typeof attachment.text === 'string') return new TextEncoder().encode(attachment.text).byteLength;
    const encoded = (attachment.data || '').split(',', 2)[1] || '';
    const compact = encoded.replace(/\s/g, '');
    return Math.floor(compact.length * 3 / 4) - (compact.endsWith('==') ? 2 : compact.endsWith('=') ? 1 : 0);
  }
  function formatAttachmentSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(bytes % 1024 ? 1 : 0) + ' KiB';
    return (bytes / (1024 * 1024)).toFixed(bytes % (1024 * 1024) ? 1 : 0) + ' MiB';
  }
  async function attach(files, {fromWorkspace = false} = {}) {
    const chatId = S.chat?.id, generation = S.generation, attachments = S.attachments;
    if (!fromWorkspace && attachments.length + files.length > 8) throw new Error('Attach up to 8 files per message.');
    const pending = [];
    for (let file of files) {
      if (!fromWorkspace && (file.type === 'application/pdf' || /\.pdf$/i.test(file.name))) {
        if (!S.config.pdf_reader) throw new Error('PDF import is unavailable. Open Settings > Optional features to install PDF text import.');
        if (pdfSelection) throw new Error('Finish the open PDF selection first.');
        if (chatId !== S.chat?.id || generation !== S.generation || attachments !== S.attachments) return;
        const selection = new AbortController();
        pdfSelection = selection;
        try {
          const {selectPdfText} = await import('/pdf_attachment.js');
          if (selection.signal.aborted) return;
          file = await selectPdfText(file, {signal: selection.signal});
          if (!file || selection.signal.aborted) return;
        } finally { pdfSelection = null; }
      }
      if (['image/png', 'image/jpeg'].includes(file.type)) {
        if (file.size > 8 * 1024 * 1024) throw new Error(file.name + ' is larger than 8 MiB.');
        const data = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = () => reject(new Error('Could not read ' + file.name)); reader.readAsDataURL(file); }); pending.push({name: file.name, type: file.type, data});
      } else if (file.type.startsWith('text/') || textFileExtensions.has(file.name.toLowerCase().match(/\.[^.]+$/)?.[0])) {
        if (file.size > 2 * 1024 * 1024 + 3) throw new Error(file.name + ' is larger than 2 MiB.');
        const bytes = await file.arrayBuffer(); let text; try { text = new TextDecoder('utf-8', {fatal: true, ignoreBOM: fromWorkspace}).decode(bytes); } catch (_) { throw new Error(file.name + ' must be UTF-8 text.'); }
        if (text.includes('\u0000')) throw new Error(file.name + ' looks like a binary file.');
        const size = new TextEncoder().encode(text).byteLength;
        if (size > 2 * 1024 * 1024) throw new Error(file.name + ' is larger than 2 MiB.');
        pending.push({name: file.name, type: file.type || 'text/plain', text});
      } else throw new Error('Unsupported file: ' + file.name + '. Attach UTF-8 text, PNG or JPEG.');
    }
    if (chatId !== S.chat?.id || generation !== S.generation || attachments !== S.attachments)
      throw new Error(chatId !== S.chat?.id || generation !== S.generation
        ? 'Attachment read cancelled because the selected conversation changed.'
        : 'The draft changed while the attachment was being read. Attach the file again.');
    if (fromWorkspace) appendAttachments(pending, true);
    else setDraftAttachments([...S.attachments, ...pending]);
    if ($('app').dataset.view === 'chat') $('draft').focus();
  }
  function setDraftAttachments(files) {
    if (files.length > 8) throw new Error('Attach up to 8 files per message.');
    if (files.reduce((sum, file) => sum + attachmentBytes(file), 0) > 10 * 1024 * 1024)
      throw new Error('Attachments exceed 10 MiB. Remove a file before adding more.');
    S.attachments.splice(0, S.attachments.length, ...files);
    composerRevision++; renderAttachments();
  }
  function appendAttachments(files, fromEditor = false) {
    const pending = files.filter(file => {
      const existing = S.attachments.filter(item => item.name === file.name);
      if (existing.some(item => item.text !== file.text || item.data !== file.data)) {
        if (fromEditor && files.length === 1 && existing.length === 1 && typeof existing[0].text === 'string') {
          const problem = new Error('The draft has another version of ' + file.name + '.');
          const revision = composerRevision, generation = S.generation;
          problem.replaceInDraft = () => {
            if (composerRevision !== revision || generation !== S.generation || !S.attachments.includes(existing[0]))
              throw new Error('The draft changed. Choose Attach to Chat again.');
            setDraftAttachments(S.attachments.map(item => item === existing[0] ? file : item));
          };
          throw problem;
        }
        throw new Error('Remove the existing ' + file.name + ' attachment to use this version.');
      }
      return !existing.length;
    });
    if (pending.length) setDraftAttachments([...S.attachments, ...pending]);
  }
  function workspaceName(path) { return path.replaceAll('/', ':'); }
  function projectDraft() {
    const chat = S.chat?.id, generation = S.generation, revision = composerRevision, system = $('setting-system').value;
    return {system, apply(context, copyInstructions) {
      if (chat !== S.chat?.id || generation !== S.generation || revision !== composerRevision || system !== $('setting-system').value || $('app').dataset.view !== 'chat')
        throw new Error('The conversation or draft changed. Choose the file or project again.');
      appendAttachments(context.files.map(({path, text}) => ({name: workspaceName(path), type: 'text/plain', text})));
      if (copyInstructions) $('setting-system').value = context.project.instructions;
      closeDrawers(false);
    }};
  }
  window.ChatComposer = {attach, workspaceName, projectDraft, formatAttachmentSize};
  function followsLatest() {
    const region = $('scroll-region');
    return region.scrollHeight - region.scrollTop - region.clientHeight <= Math.min(120, region.clientHeight / 3);
  }
  function syncJump() {
    syncReadingPosition();
    $('jump-bottom').hidden = !$('welcome').hidden || followsLatest();
  }
  function bottom() { const region = $('scroll-region'); region.scrollTop = region.scrollHeight; syncJump(); }
  function resizeDraft() {
    const draft = $('draft');
    draft.style.overflowY = 'hidden';
    draft.style.height = 'auto';
    const height = draft.scrollHeight;
    draft.style.height = Math.min(height, 220) + 'px';
    draft.style.overflowY = height > 220 ? 'auto' : 'hidden';
  }
  function reflectDetailsState() {
    const app = $('app');
    const open = app.dataset.view === 'chat' && !app.classList.contains('inspector-closed');
    $('details-toggle').setAttribute('aria-expanded', String(open));
    $('chat-settings-toggle').setAttribute('aria-expanded', String(open && S.tab === 'settings'));
    return open;
  }
  function showTab(name) { S.tab = name; document.querySelectorAll('[data-tab]').forEach(el => { el.setAttribute('aria-selected', String(el.dataset.tab === name)); el.tabIndex = el.dataset.tab === name ? 0 : -1; }); for (const tab of ['activity', 'stats', 'settings']) $('panel-' + tab).hidden = tab !== name; reflectDetailsState(); syncLiveMetrics(); renderContextLimits(); }
  function usesDetailsDrawer() { return document.documentElement.dataset.compact === 'true' || window.innerWidth <= 1200; }
  function canFocusDrawerControl(el) {
    return el && !el.matches(':disabled') && !el.closest('[inert]') && el.checkVisibility({visibilityProperty: true});
  }
  function updateShade(opener, destination) {
    const app = $('app'), sidebar = $('sidebar'), details = $('inspector');
    const detailsOpen = reflectDetailsState();
    const sidebarOpen = document.documentElement.dataset.compact === 'true' && app.classList.contains('sidebar-open');
    const next = sidebarOpen ? sidebar : detailsOpen && usesDetailsDrawer() ? details : null;
    $('sidebar-toggle').setAttribute('aria-expanded', String(sidebarOpen));
    $('drawer-shade').hidden = !next;
    if (next !== modalDrawer) {
      const previous = modalDrawer, returnTo = drawerOpener;
      if (previous) {
        window.ChatShell.closeMenu(false);
        previous.removeAttribute('role'); previous.removeAttribute('aria-modal');
      }
      modalDrawer = next; drawerOpener = next ? opener || document.activeElement : null;
      document.querySelector('main').inert = Boolean(next);
      document.querySelector('.skip-link').inert = Boolean(next);
      sidebar.inert = next === details; details.inert = next === sidebar;
      if (next) {
        hideReplyStats();
        next.setAttribute('role', 'dialog'); next.setAttribute('aria-modal', 'true');
        (next === sidebar ? $('sidebar-close') : $('tab-' + S.tab)).focus({preventScroll: true});
      } else if (destination !== false && !document.querySelector('dialog[open]')) {
        (destination || (canFocusDrawerControl(returnTo) ? returnTo : window.ChatShell.viewFocusTarget())).focus({preventScroll: true});
      }
    }
    syncLiveMetrics(); renderContextLimits();
  }
  function closeDrawers(destination) {
    $('app').classList.remove('sidebar-open');
    if (usesDetailsDrawer()) $('app').classList.add('inspector-closed');
    updateShade(null, destination);
  }
  function showToolCall(messageId, callId) {
    const message = path().find(item => item.id === messageId && item.role === 'assistant');
    if (!message || !(message.tool_calls || []).some(call => call.id === callId)) throw new Error('This tool call is no longer on the selected branch.');
    if (usesDetailsDrawer()) closeDrawers(false);
    const article = $('messages').querySelector('[data-message="' + CSS.escape(messageId) + '"]');
    const wrap = Array.from(article?.querySelectorAll('.tool-call') || []).find(item => item.dataset.toolCall === callId);
    const detail = wrap?.querySelector(':scope > details'); const summary = detail?.querySelector(':scope > summary');
    if (!summary) throw new Error('Tool details are not available in this conversation view.');
    detail.open = true;
    summary.focus({preventScroll: true});
    summary.scrollIntoView({block: 'center', behavior: 'instant'});
  }
  async function action(name, el) {
    if (name === 'attach') {
      if (el.getAttribute('aria-expanded') === 'true') return window.ChatShell.closeMenu();
      const sources = [
        {action: 'attach-device', label: 'From this device'},
        {action: 'attach-files', label: 'Saved files'},
        {action: 'attach-project', label: 'Saved project'}
      ];
      if (dialogueMessages().some(message => message.attachments?.length))
        sources.push({action: 'attach-conversation', label: 'From this conversation'});
      return window.ChatShell.menu(el, 'Attach', sources, choice => action(choice, el).catch(error => toast(error.message)), {above: true});
    }
    if (name !== 'chat-menu') window.ChatShell.closeMenu();
    if (name === 'reply-stats') return showReplyStats(el, true);
    if (name === 'close-stats') return hideReplyStats(true);
    if (name === 'copy-stats') { if (replyStats) { await navigator.clipboard.writeText(replyStats.panel.dataset.copy); toast('Statistics copied.'); } return; }
    if (name === 'new') return newChat();
    if (name === 'home') return newChat({landing: true});
    if (name === 'temporary') return temporaryChat();
    if (name === 'end-temporary') return endTemporaryChat();
    if (name === 'reply-detail') return jumpToMessage(el.dataset.id);
    if (name === 'find-message' || name === 'outline') return findMessage(name === 'outline');
    if (name === 'transcript') {
      const title = S.chat.title || 'Conversation';
      const turns = dialogueMessages().filter(m => m.content);
      const sections = turns.map(m => '## ' + (m.role === 'user' ? 'You' : 'Assistant') + '\n\n' + m.content);
      window.ChatArtifacts.download(['# ' + title, ...sections].join('\n\n') + '\n', title.slice(0, 175) + '.md', 'markdown');
      return;
    }
    if (name === 'tool-detail') return showToolCall(el.dataset.message, el.dataset.call);
    if (name === 'sidebar') {
      if ($('app').classList.contains('sidebar-open')) return closeDrawers();
      $('app').classList.add('sidebar-open');
      if (usesDetailsDrawer()) $('app').classList.add('inspector-closed');
      updateShade(el); return;
    }
    if (name === 'close-drawers') return closeDrawers();
    if (name === 'inspector' || name === 'settings' || name === 'open-settings') {
      if (name === 'open-settings') {
        window.ChatShell.show('chat');
        el = $('chat-settings-toggle');
        if (!usesDetailsDrawer()) el.focus({preventScroll: true});
      }
      $('app').classList.remove('sidebar-open');
      if (name === 'open-settings' || (name === 'settings' && el.getAttribute('aria-expanded') !== 'true')) {
        $('app').classList.remove('inspector-closed'); showTab('settings');
      } else {
        const closed = $('app').classList.toggle('inspector-closed');
        if (!usesDetailsDrawer()) {
          dockedDetailsOpen = !closed;
          localStorage.setItem('llmx-conversation-details', closed ? 'closed' : 'open');
          if (closed && el.id === 'details-close') $('details-toggle').focus({preventScroll: true});
        }
      }
      updateShade(el); return;
    }
    if (name === 'theme') { setTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light', true); return; }
    if (name === 'archive-view') { S.archive = !S.archive; queueChatSearch(); return; }
    if (name === 'attach-conversation') return findMessage(false, {role: 'files'});
    if (name === 'attach-device') return $('file-input').click();
    if (name === 'attach-files') return window.ChatProjectContext.openFiles();
    if (name === 'attach-project') return window.ChatProjectContext.openProjects();
    if (name === 'remove-attachment') { S.attachments.splice(Number(el.dataset.id), 1); composerRevision++; renderAttachments(); return; }
    if (name === 'bottom') return bottom();
    if (name === 'import') return $('import-input').click();
    if (name === 'logout') {
      sidebarRuns.ready = false; stopSidebarRuns();
      try { await api('/logout', 'POST', {}); }
      catch (error) { if (error.status !== 401) { sidebarRuns.ready = true; syncSidebarRuns(); } throw error; }
      clearDrafts(); closeStream(); location.reload(); return;
    }
    if (name === 'cancel-edit') { S.edit = null; $('edit-note').hidden = true; $('draft').value = ''; S.attachments = []; composerRevision++; renderAttachments(); resizeDraft(); return; }
    if (name === 'save-settings') return saveConversationSettings();
    if (name === 'chat-menu') return showChatMenu(el.dataset.id || S.chat?.id, el);
    if (['rename', 'pin', 'archive', 'folder', 'delete', 'export', 'duplicate'].includes(name)) {
      const id = el.dataset.id || (name === 'rename' ? S.chat?.id : menuId);
      const chat = sidebarChat(id);
      if (!chat) throw new Error('This conversation is no longer available.');
      if (name === 'rename') return window.ChatShell.conversationForm({title: 'Rename conversation', label: 'Conversation name', value: chat.title}, title => patchChat({title}, chat.id));
      if (name === 'pin') await patchChat({pinned: !chat.pinned}, chat.id);
      if (name === 'archive') await patchChat({archived: !chat.archived}, chat.id);
      if (name === 'folder') return window.ChatShell.conversationForm({title: 'Move to folder', description: 'Choose an existing folder or enter a new name. Leave empty to remove the folder.', label: 'Folder', value: chat.folder || '', folders: [...new Set(S.chats.map(item => item.folder).filter(Boolean))]}, folder => patchChat({folder}, chat.id));
      if (name === 'delete') {
        if (chat.temporary) return endTemporaryChat();
        const epoch = routeEpoch;
        return window.ChatShell.conversationForm({title: 'Delete conversation?', description: 'Delete "' + chat.title + '" and all its branches permanently. This cannot be undone.', destructive: true, submit: 'Delete'}, async () => {
          await api('/chats/' + encodeURIComponent(chat.id), 'DELETE');
          forgetDraft(chat.id);
          if (S.chat?.id === chat.id && epoch === routeEpoch) { clearChat(); saveRoute(true); }
          await refreshChats();
        });
      }
      if (name === 'duplicate') {
        const data = await api('/chats/' + encodeURIComponent(chat.id) + '/export');
        data.chat.title = (chat.title || 'Conversation').slice(0, 193) + ' (copy)';
        await api('/import', 'POST', data); await refreshChats(); toast('Conversation duplicated.');
      }
      if (name === 'export') {
        const response = await fetch('/api/chats/' + encodeURIComponent(chat.id) + '/export');
        if (!response.ok) throw new Error('Export failed.');
        window.ChatArtifacts.downloadBlob(await response.blob(), (chat.title || 'conversation').replace(/[^a-z0-9_-]/gi, '_') + '.json');
      }
      return;
    }
    if (!S.chat) return;
    if (name === 'stop') { const run = currentRun(); if (run) { await api('/runs/' + encodeURIComponent(run.id) + '/cancel', 'POST', {}); toast('Stop requested.'); } }
    if (name === 'copy') { const message = S.messages.find(m => m.id === el.dataset.id); await navigator.clipboard.writeText(message.content || ''); toast('Message copied.'); }
    if (name === 'quote') {
      const message = S.messages.find(item => item.id === el.dataset.id);
      if (!message?.content) return;
      const body = el.closest('.message').querySelector(':scope > .markdown');
      const selection = window.getSelection();
      let passage = message.content;
      if (selection?.rangeCount && !selection.isCollapsed) {
        const range = selection.getRangeAt(0);
        if (!body?.contains(range.commonAncestorContainer)) throw new Error('Select text within this answer to quote it.');
        passage = selection.toString();
        const fragment = range.cloneContents();
        const controls = fragment.querySelectorAll('.code-heading');
        if (controls.length) {
          controls.forEach(heading => heading.remove());
          const excerpt = node('div', 'markdown');
          excerpt.style.position = 'fixed';
          excerpt.style.left = '-200vw';
          excerpt.append(fragment);
          document.body.append(excerpt);
          passage = excerpt.innerText.trimEnd();
          excerpt.remove();
        }
      }
      const draft = $('draft');
      const quoted = passage.replace(/^/gm, '> ') + '\n\n';
      const separator = draft.value && !draft.value.endsWith('\n\n') ? (draft.value.endsWith('\n') ? '\n' : '\n\n') : '';
      const addition = separator + quoted;
      if (draft.value.length + addition.length > draft.maxLength) throw new Error('Select a shorter passage to fit the message limit.');
      draft.setRangeText(addition, draft.value.length, draft.value.length, 'end');
      draft.dispatchEvent(new Event('input', {bubbles: true}));
      draft.focus();
      draft.scrollTop = draft.scrollHeight;
    }
    if (name === 'edit') { if (currentRun()) throw new Error('Stop the current response before editing.'); const message = S.messages.find(m => m.id === el.dataset.id); S.edit = message; $('draft').value = message.content || ''; S.attachments = [...(message.attachments || [])]; composerRevision++; renderAttachments(); $('edit-note').hidden = false; resizeDraft(); $('draft').focus(); }
    if (name === 'retry') await retry(el.dataset.id);
    if (name === 'branch') await branch(el.dataset.id);
    if (name === 'approve' || name === 'deny') {
      const approval = el.closest('.approval'); const allow = name === 'approve';
      const remember = allow && approval.querySelector('.remember-approval').checked;
      await api('/runs/' + encodeURIComponent(el.dataset.run || currentRun()?.id) + '/approve', 'POST', {call_id: el.dataset.id, allow, remember});
      approval.querySelectorAll('button').forEach(b => { b.disabled = true; });
    }
  }
  function configurePdfPicker(enabled) {
    S.config.pdf_reader = enabled;
    $('file-input').accept = baseFileTypes + (enabled ? ',application/pdf,.pdf' : '');
    $('pdf-help').hidden = !enabled;
    $('attach-source').dataset.tooltip = enabled
      ? 'Choose a source. Device: UTF-8 text or code, PNG, JPEG or selected PDF pages. See Settings for limits.'
      : 'Choose a source. Device: UTF-8 text or code, PNG or JPEG. See Settings for limits.';
  }
  document.addEventListener('companion:features', event => {
    const pdf = event.detail.features.find(feature => feature.id === 'pdf-reader');
    if (pdf) configurePdfPicker(pdf.state === 'installed');
  });
  async function initialize() {
    chatCreating(true);
    sidebarRuns.ready = false; syncSidebarRuns();
    try {
      const session = await api('/session'); $('logout-button').hidden = !session.password_required;
      if (!session.authenticated && session.password_required) { pendingView = null; $('login-dialog').showModal(); $('password').focus(); return; }
      sessionReady = true;
      if (pendingView) { const view = pendingView; pendingView = null; selectView(view); }
      S.config = await api('/config'); document.dispatchEvent(new CustomEvent('companion:config', {detail: S.config})); connection(S.config.upstream_available ? 'Connected' : 'Unavailable', !S.config.upstream_available);
      configurePdfPicker(Boolean(S.config.pdf_reader));
      if (!S.config.upstream_available) banner(S.config.upstream_error || 'The model server is unavailable. Check the companion connection and try again.');
      if (S.config.tools_error) banner([S.config.upstream_error, 'Tool connection issue: ' + S.config.tools_error].filter(Boolean).join(' '));
      await refreshChats(); loadSettings();
      sidebarRuns.ready = true; syncSidebarRuns();
      try { for (const key of Object.keys(sessionStorage)) if (key.startsWith(draftPrefix) && key !== draftPrefix + 'new' && !S.chats.some(chat => draftPrefix + chat.id === key)) forgetDraft(key.slice(draftPrefix.length)); } catch (_) { /* Browser storage may be disabled. */ }
      $('temporary-button').hidden = !S.config.temporary?.enabled;
      navigationReady = true; await restoreRoute();
    } catch (error) { connection('Offline', true); banner(error.message + ' Reload the page to reconnect.'); }
    finally { chatCreating(false); }
  }
  document.addEventListener('click', event => {
    const el = event.target.closest('[data-action],[data-chat],[data-tab]'); if (!el) return;
    const execute = async () => {
      if (el.dataset.action) await action(el.dataset.action, el);
      else if (el.dataset.chat) {
        await openChat(el.dataset.chat);
        if (S.chat?.id === el.dataset.chat && $('app').dataset.view === 'chat' && (document.activeElement === el || document.activeElement === document.body)) {
          if (el.classList.contains('draft-row')) $('draft').focus({preventScroll: true});
          else if (el.classList.contains('recent-chat')) $('chat-title').focus({preventScroll: true});
        }
      }
      else if (el.dataset.tab) showTab(el.dataset.tab);

    };
    execute().catch(error => toast(error.message));
  });
  $('composer').addEventListener('submit', event => { event.preventDefault(); send().catch(error => banner(error.message)); });
  $('draft').addEventListener('input', () => { composerRevision++; resizeDraft(); rememberDraft(); });
  let composerWidth = 0;
  const composerSize = new ResizeObserver(([entry]) => {
    const width = entry.contentRect.width;
    if (width === composerWidth) return;
    composerWidth = width;
    if (width > 0) requestAnimationFrame(resizeDraft);
  });
  composerSize.observe($('composer'));
  $('model').addEventListener('change', () => { snapshotBanner(); render(); });
  $('setting-reply-limit').addEventListener('change', () => { renderReplyLimit(); if (!$('setting-max_tokens').disabled) $('setting-max_tokens').focus(); });
  $('draft').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); send().catch(error => banner(error.message)); } });
  $('draft').addEventListener('paste', event => { const files = Array.from(event.clipboardData?.files || []); if (files.length) { event.preventDefault(); attach(files).catch(error => toast(error.message)); } });
  const chatStage = document.querySelector('.chat-stage'), dropHint = $('file-drop-hint');
  chatStage.addEventListener('dragover', event => {
    if (!event.dataTransfer.types.includes('Files')) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
    dropHint.hidden = false;
  });
  chatStage.addEventListener('dragleave', event => {
    if (!chatStage.contains(event.relatedTarget)) dropHint.hidden = true;
  });
  chatStage.addEventListener('drop', event => {
    dropHint.hidden = true;
    if (!event.dataTransfer.types.includes('Files')) return;
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files);
    if (files.length) attach(files).catch(error => toast(error.message));
  });
  $('file-input').addEventListener('change', event => { attach(Array.from(event.target.files)).catch(error => toast(error.message)); event.target.value = ''; });
  $('import-input').addEventListener('change', async event => {
    const file = event.target.files[0]; event.target.value = ''; if (!file) return;
    const epoch = ++routeEpoch;
    try {
      if (file.size > 32 * 1024 * 1024) throw new Error('Import must be smaller than 32 MB.');
      const chat = await api('/import', 'POST', JSON.parse(await file.text()));
      await refreshChats(); await openChat(chat.id || chat.chat?.id, null, {epoch}); toast('Conversation imported.');
    } catch (error) { toast('Import failed: ' + error.message); }
  });
  $('search').addEventListener('input', queueChatSearch);
  $('scroll-region').addEventListener('scroll', syncJump, {passive: true});
  const transcriptSize = new ResizeObserver(syncJump);
  transcriptSize.observe($('scroll-region')); transcriptSize.observe($('messages'));
  $('login-dialog').addEventListener('cancel', event => event.preventDefault());
  $('login-form').addEventListener('submit', async event => { event.preventDefault(); const submit = event.target.querySelector('button'); submit.disabled = true; try { await api('/login', 'POST', {password: $('password').value}); $('password').value = ''; $('login-error').textContent = ''; $('login-dialog').close(); await initialize(); } catch (error) { $('login-error').textContent = error.message; } finally { submit.disabled = false; } });
  document.addEventListener('keydown', event => {
    if (event.defaultPrevented || document.querySelector('dialog[open]')) return;
    if ((event.ctrlKey || event.metaKey) && event.shiftKey && !event.altKey && event.key.toLowerCase() === 'f' &&
        !modalDrawer && $('app').dataset.view === 'chat' && S.messages.length) {
      event.preventDefault();
      findMessage();
      return;
    }
    if (passage && !modalDrawer && !event.ctrlKey && !event.metaKey && !event.altKey) {
      if (event.key === 'F3') { event.preventDefault(); stepPassage(event.shiftKey ? -1 : 1); return; }
      if (event.key === 'Escape') {
        event.preventDefault(); closePassages(!event.target.closest('input,textarea,[contenteditable=true]')); return;
      }
    }
    if (modalDrawer && event.key === 'Escape') { event.preventDefault(); closeDrawers(); return; }
    if (modalDrawer && event.key === 'Tab') {
      const controls = Array.from(modalDrawer.querySelectorAll('a[href],button,input,textarea,select,details > summary:first-child,[tabindex]'))
        .filter(el => (!el.hasAttribute('tabindex') || el.tabIndex >= 0) && canFocusDrawerControl(el));
      const active = document.activeElement, first = controls[0], last = controls.at(-1);
      if (first && (!modalDrawer.contains(active) || active === (event.shiftKey ? first : last))) {
        event.preventDefault(); (event.shiftKey ? last : first).focus();
      }
    }
    if (modalDrawer !== $('inspector') && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      document.querySelector('[data-action="new"]').click();
    }
    if (event.target.matches('[role=tab]') && ['ArrowLeft', 'ArrowRight'].includes(event.key)) { event.preventDefault(); const tabs = ['activity', 'stats', 'settings']; const index = (tabs.indexOf(S.tab) + (event.key === 'ArrowRight' ? 1 : 2)) % 3; showTab(tabs[index]); $('tab-' + tabs[index]).focus(); }
  });
  window.addEventListener('resize', () => {
    const focused = document.activeElement, previousModal = modalDrawer;
    const drawer = usesDetailsDrawer();
    if (drawer !== detailsInDrawer) {
      detailsInDrawer = drawer;
      $('app').classList.toggle('inspector-closed', drawer || !dockedDetailsOpen);
    }
    if (document.documentElement.dataset.compact !== 'true') $('app').classList.remove('sidebar-open');
    updateShade(null, previousModal?.contains(focused) && canFocusDrawerControl(focused) ? false : undefined);
    if (!previousModal && !modalDrawer && !canFocusDrawerControl(focused) && !document.querySelector('dialog[open]')) {
      if ($('inspector').contains(focused)) $('details-toggle').focus({preventScroll: true});
      if ($('sidebar').contains(focused)) $('sidebar-toggle').focus({preventScroll: true});
    }
  });
  document.addEventListener('companion:view', event => {
    if (event.detail !== 'chat') { pdfSelection?.abort(); closePassages(); }
    if (modalDrawer && (modalDrawer === $('sidebar') || event.detail !== 'chat')) closeDrawers(window.ChatShell.viewFocusTarget());
    else updateShade();
    renderLiveMetrics();
  });
  document.addEventListener('visibilitychange', () => { syncLiveMetrics(); renderLiveMetrics(); });
  document.addEventListener('companion:view', syncSidebarRuns);
  document.addEventListener('visibilitychange', syncSidebarRuns);
  document.addEventListener('companion:unauthorized', () => { sidebarRuns.ready = false; stopSidebarRuns(); });
  window.addEventListener('pagehide', stopSidebarRuns);
  window.addEventListener('pageshow', syncSidebarRuns);
  window.addEventListener('pagehide', () => { rememberDraft(); flushDrafts(); liveMetrics.stream?.close(); liveMetrics.stream = null; liveMetrics.id = null; });
  window.addEventListener('pageshow', syncLiveMetrics);
  document.addEventListener('visibilitychange', () => { if (!document.hidden && currentRun() && !S.stream) syncSnapshot(S.chat.id).then(() => { const run = currentRun(); if (run) subscribe(run); }).catch(error => toast(error.message)); });
  setTheme(themeChosen ? storedTheme : systemTheme.matches ? 'light' : 'dark');
  $('app').classList.toggle('inspector-closed', detailsInDrawer || !dockedDetailsOpen);
  updateShade();
  showTab('activity'); setInterval(tick, 1000); initialize();
})();
