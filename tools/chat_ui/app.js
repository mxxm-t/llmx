/* The service owns conversation state; this file owns browser presentation. */
(() => {
  'use strict';
  const {node, markdown, number, json} = window.ChatRender;
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
  const systemTheme = window.matchMedia('(prefers-color-scheme: light)');
  const storedTheme = localStorage.getItem('llmx-theme');
  let themeChosen = storedTheme === 'light' || storedTheme === 'dark';
  function setTheme(theme, save = false) {
    document.documentElement.dataset.theme = theme;
    $('theme-button').textContent = theme === 'light' ? 'Dark theme' : 'Light theme';
    if (save) { localStorage.setItem('llmx-theme', theme); themeChosen = true; }
  }
  systemTheme.addEventListener('change', event => { if (!themeChosen) setTheme(event.matches ? 'light' : 'dark'); });
  const terminal = status => ['completed', 'failed', 'cancelled', 'interrupted'].includes(status);
  const S = {chats: [], chat: null, messages: [], runs: [], config: {models: [], tools: []}, attachments: [], edit: null, stream: null, cursor: 0, archive: false, tab: 'activity', scheduled: false, fingerprint: new Map(), searchText: new Map(), inspectorKey: '', generation: 0, busy: false};
  let toastTimer, creatingChat = false, menuId = null, composerRevision = 0;
  const liveMetrics = {id: null, stream: null, history: [], row: null, received: 0, collector: null};
  let runtimeLimits = null;
  const objectIds = new WeakMap(); let nextObjectId = 0;
  function identity(value) { if (!value || typeof value !== 'object') return value; if (!objectIds.has(value)) objectIds.set(value, ++nextObjectId); return objectIds.get(value); }
  function messageKey(m, text = false) { return [m.id, m.role, m.status, text ? m.content : undefined, text ? m.reasoning : undefined, identity(m.usage), identity(m.timings), identity(m.context), (m.tool_calls || []).map(identity), (m.attachments || []).map(identity)]; }
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
  document.addEventListener('companion:unauthorized', closeStream);
  function connection(text, offline = false) { const el = $('connection'); el.replaceChildren(node('span', 'status-dot'), document.createTextNode(text)); el.classList.toggle('offline', offline); document.querySelector('.inspector-bottom').replaceChildren(node('span', 'status-dot'), document.createTextNode(offline ? text + ' / local runtime' : 'Connected to your local runtime')); }
  function settings() {
    const value = {model: $('model').value, system: $('setting-system').value};
    if (!(S.config.models || []).some(model => model.id === value.model)) throw new Error('This model is not available on the connected server. Choose another model before sending.');
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
    renderReplyLimit(); renderContextLimits(); renderToolSelections(value.tools || []);
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
      section.append(node('strong', 'tool-selection-title', group.name));
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
  async function refreshChats() { const data = await api('/chats'); S.chats = data.chats || []; renderSidebar(); }
  function renderSidebar() {
    const search = $('search').value.trim().toLowerCase();
    const chats = S.chats.filter(c => Boolean(c.archived) === S.archive && (!search || (c.title + ' ' + (c.folder || '') + ' ' + (S.searchText.get(c.id) || '')).toLowerCase().includes(search)));
    chats.sort((a, b) => Number(Boolean(b.pinned)) - Number(Boolean(a.pinned)) || (String(b.updated).localeCompare(String(a.updated))));
    const groups = new Map();
    chats.forEach(chat => { const group = chat.pinned ? 'Pinned' : chat.folder || 'Recent conversations'; if (!groups.has(group)) groups.set(group, []); groups.get(group).push(chat); });
    const out = $('chat-list'); out.replaceChildren();
    groups.forEach((items, name) => {
      out.append(node('div', 'folder-heading', name));
      items.forEach(chat => {
        const button = node('button', 'chat-item' + (S.chat?.id === chat.id ? ' active' : '')); button.type = 'button'; button.dataset.chat = chat.id; button.title = chat.title;
        if (S.chat?.id === chat.id) button.setAttribute('aria-current', 'page');
        button.append(node('span', 'chat-glyph', '#'), node('span', 'chat-name', chat.title || 'Untitled conversation'));
        if (chat.pinned) button.append(node('span', 'pin-mark', '*'));
        const row = node('div', 'chat-row' + (S.chat?.id === chat.id ? ' active' : ''));
        const menu = node('button', 'chat-row-menu', '...'); menu.type = 'button'; menu.dataset.action = 'chat-menu'; menu.dataset.id = chat.id;
        menu.setAttribute('aria-label', 'Actions for ' + (chat.title || 'Untitled conversation')); menu.setAttribute('aria-haspopup', 'menu'); menu.title = 'Conversation actions';
        row.append(button, menu); out.append(row);
      });
    });
    if (!chats.length) out.append(node('p', 'empty-history', search ? 'No matching conversations.' : S.archive ? 'Your archive is empty.' : 'Your next good idea starts here. Create a conversation to get going.'));
    $('archive-toggle').textContent = S.archive ? 'Back to chats' : 'Show archive';
  }
  async function openChat(id, snapshot = null) {
    if (S.chat?.temporary && S.chat.id !== id && !await leaveTemporary()) return;
    hideReplyStats();
    const generation = ++S.generation; closeStream(); banner(configurationWarning());
    const data = snapshot || await api('/chats/' + encodeURIComponent(id)); if (generation !== S.generation) return;
    S.chat = data.chat; S.messages = data.messages || []; S.runs = data.runs || []; S.edit = null; S.fingerprint.clear(); S.inspectorKey = '';
    snapshotBanner();
    S.searchText.set(id, S.messages.map(m => m.content || '').join(' '));
    S.attachments = []; $('draft').value = ''; composerRevision++; resizeDraft(); renderAttachments(); $('edit-note').hidden = true;
    if (S.chat.temporary) sessionStorage.setItem('llmx-temporary-chat', id); else localStorage.setItem('llmx-chat', id);
    loadSettings(); renderSidebar(); render(); closeDrawers();
    requestAnimationFrame(() => bottom());
    const run = currentRun(); if (run) subscribe(run);
  }
  function chatCreating(value) {
    creatingChat = value;
    document.querySelectorAll('[data-action="new"],[data-action="temporary"]').forEach(control => { control.disabled = value; control.setAttribute('aria-busy', String(value)); });
  }
  async function newChat() {
    if (creatingChat || S.busy) return;
    if (S.chat && !S.chat.temporary && !S.chat.archived && !S.messages.length && !S.runs.length) {
      closeDrawers(); $('draft').focus(); return;
    }
    chatCreating(true);
    try {
      if (S.chat?.temporary && !await leaveTemporary()) return;
      const generation = S.generation, selectedChat = S.chat?.id;
      const chat = await api('/chats', 'POST', {settings: settings()}); await refreshChats();
      if (generation === S.generation && selectedChat === S.chat?.id) { await openChat(chat.id, {chat, messages: [], runs: []}); $('draft').focus(); }
    } finally { chatCreating(false); }
  }
  function clearChat() {
    hideReplyStats();
    const old = S.chat; closeStream(); ++S.generation;
    if (old?.temporary) sessionStorage.removeItem('llmx-temporary-chat'); else localStorage.removeItem('llmx-chat');
    if (old) S.searchText.delete(old.id);
    S.chat = null; S.messages = []; S.runs = []; S.edit = null; S.attachments = []; S.fingerprint.clear(); S.inspectorKey = '';
    $('draft').value = ''; composerRevision++; $('edit-note').hidden = true; resizeDraft(); renderAttachments(); loadSettings(); render();
  }
  async function leaveTemporary() {
    if (!S.chat?.temporary) return true;
    if (currentRun() || S.busy) throw new Error('Stop the response before ending this temporary chat.');
    const end = async () => {
      try { await api('/chats/' + encodeURIComponent(S.chat.id), 'DELETE'); }
      catch (error) { if (error.status !== 404) throw error; }
      clearChat(); await refreshChats();
    };
    if (S.messages.length || $('draft').value.trim() || S.attachments.length)
      return Boolean(await window.ChatShell.conversationForm({title: 'End temporary chat?', description: 'Its messages will be discarded.', destructive: true, submit: 'End chat'}, end));
    await end(); return true;
  }
  async function temporaryChat() {
    if (creatingChat || S.busy) return;
    if (S.chat?.temporary) {
      chatCreating(true);
      try { await leaveTemporary(); }
      finally { chatCreating(false); }
      return;
    }
    if (!S.config.temporary?.enabled) throw new Error('Temporary chats are unavailable on this server.');
    chatCreating(true);
    try {
      const values = settings(); values.tools = [];
      const chat = await api('/chats', 'POST', {temporary: true, settings: values});
      await openChat(chat.id, {chat, messages: [], runs: []}); $('draft').focus();
    } finally { chatCreating(false); }
  }
  function showChatMenu(id, anchor) {
    const chat = S.chats.find(item => item.id === id) || (S.chat?.id === id ? S.chat : null);
    if (!chat) throw new Error('Create a conversation to manage it.');
    menuId = chat.id;
    const items = chat.temporary ? [] : [
      {action: 'rename', label: 'Rename'},
      {action: 'pin', label: chat.pinned ? 'Unpin' : 'Pin'},
      {action: 'folder', label: 'Move to folder'},
      {action: 'duplicate', label: 'Duplicate', separator: true},
      {action: 'archive', label: chat.archived ? 'Restore from archive' : 'Archive'}
    ];
    items.push({action: 'export', label: 'Export JSON', separator: true}, {action: 'delete', label: chat.temporary ? 'End temporary chat' : 'Delete conversation', danger: true, separator: true, disabled: Boolean(chat.temporary && (currentRun() || S.busy))});
    window.ChatShell.menu(anchor, chat.temporary ? 'Temporary chat' : chat.title || 'Conversation', items, name => action(name, {dataset: {id: chat.id}}).catch(error => toast(error.message)));
  }
  function path() {
    const map = new Map(S.messages.map(m => [m.id, m])); const selected = []; const seen = new Set();
    let id = S.chat?.active_leaf;
    while (id && map.has(id) && !seen.has(id)) { seen.add(id); const message = map.get(id); selected.unshift(message); id = message.parent_id; }
    return selected;
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
    const tip = node('div', 'message-action-tooltip', target.getAttribute('aria-label'));
    tip.id = 'message-action-tooltip-' + (++tooltipSequence); tip.setAttribute('role', 'tooltip'); document.body.append(tip);
    target.setAttribute('aria-describedby', tip.id); tooltipTarget = target; actionTooltip = tip;
    const box = target.getBoundingClientRect(), size = tip.getBoundingClientRect(), pad = 8;
    const left = Math.max(pad, Math.min(box.left + (box.width - size.width) / 2, innerWidth - size.width - pad));
    const below = box.bottom + size.height + pad <= innerHeight;
    const top = below ? box.bottom + pad : Math.max(pad, box.top - size.height - pad);
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
  }
  function actionIcon(label, action, id, path, cls = 'message-icon-action') {
    const el = button('', action, id, cls); el.setAttribute('aria-label', label);
    const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); icon.setAttribute('viewBox', '0 0 24 24'); icon.setAttribute('fill', 'none'); icon.setAttribute('stroke', 'currentColor'); icon.setAttribute('stroke-width', '1.8'); icon.setAttribute('stroke-linecap', 'round'); icon.setAttribute('stroke-linejoin', 'round'); icon.setAttribute('aria-hidden', 'true');
    const shape = document.createElementNS('http://www.w3.org/2000/svg', 'path'); shape.setAttribute('d', path); icon.append(shape); el.append(icon);
    if (action === 'reply-stats') return el;
    el.addEventListener('pointerenter', () => showActionTooltip(el)); el.addEventListener('pointerleave', () => hideActionTooltip(el));
    el.addEventListener('blur', () => hideActionTooltip(el)); el.addEventListener('click', () => hideActionTooltip(el, true));
    return el;
  }
  document.addEventListener('focusin', event => { if (event.target.matches('.message-icon-action:not([data-action=reply-stats]),.branch-action')) showActionTooltip(event.target); }, true);
  document.addEventListener('keydown', event => { if (event.key === 'Escape') hideActionTooltip(); }, true);
  document.addEventListener('scroll', () => { if (tooltipTarget && tooltipTarget === document.activeElement) showActionTooltip(tooltipTarget); else hideActionTooltip(); }, true);
  window.addEventListener('resize', () => hideActionTooltip());
  function renderAttachment(attachment) {
    if (/^data:image\/(png|jpeg);base64,[A-Za-z0-9+/=\r\n]+$/.test(attachment.data || '')) {
      const img = node('img', 'attachment-image'); img.src = attachment.data; img.alt = attachment.name || 'Attached image'; img.loading = 'lazy'; return img;
    }
    return node('span', 'attachment-pill', attachment.name || 'Text attachment');
  }
  function toolRow(call, message) {
    const wrap = node('div', 'tool-call'); wrap.dataset.toolCall = call.id || ''; const detail = node('details'); const summary = node('summary'); summary.tabIndex = 0;
    summary.append(node('span', '', '>_'), node('span', '', call.name || 'Tool call'), node('span', 'badge' + (call.error ? ' error' : ''), call.status || 'pending'));
    const body = node('div', 'tool-body'); body.append(node('div', 'section-label', 'ARGUMENTS'), node('pre', '', json(call.arguments ?? {})));
    if (call.result !== undefined) {
      body.append(node('div', 'section-label', 'RESULT'));
      let result = call.result; if (typeof result === 'string') { try { result = JSON.parse(result); } catch (_) { /* Plain text results need no decoding. */ } }
      const full = json(result); const readable = result?.content?.filter(part => part.type === 'text' && typeof part.text === 'string').map(part => part.text).join('\n\n');
      const preview = readable || full; body.append(node('pre', '', preview.slice(0, 16384) + (preview.length > 16384 ? '\n[Preview truncated]' : '')));
      const structured = node('details'); structured.append(node('summary', '', 'Structured result'), node('pre', '', full.slice(0, 16384) + (full.length > 16384 ? '\n[Preview truncated]' : ''))); body.append(structured);
      if (full.length > 16384 || preview.length > 16384) { const download = node('button', 'secondary-button', 'Download full result'); download.type = 'button'; download.addEventListener('click', () => { const url = URL.createObjectURL(new Blob([full], {type: 'application/json'})); const link = node('a'); link.href = url; link.download = 'tool-result.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }); body.append(download); }
    }
    if (call.result !== undefined && ['workspace_list', 'workspace_read', 'workspace_write', 'workspace_run'].includes(call.name)) {
      const open = node('button', 'secondary-button workspace-tool-open', 'Open Workspace'); open.type = 'button'; open.addEventListener('click', () => window.ChatShell.show('workbench')); wrap.append(open);
    }
    if (call.error) body.append(node('p', 'error-text', json(call.error)));
    if (call.duration_ms !== undefined) body.append(node('small', 'muted', number(call.duration_ms, ' ms')));
    detail.append(summary, body); wrap.append(detail);
    if (call.status === 'approval') {
      const approval = node('div', 'approval', 'This tool is requesting permission to run.'); const actions = node('div', 'approval-actions');
      const allow = button('Allow', 'approve', call.id, 'primary-button'); const deny = button('Deny', 'deny', call.id, 'secondary-button');
      allow.dataset.run = message.run_id; deny.dataset.run = message.run_id;
      const label = node('label'); const remember = node('input'); remember.type = 'checkbox'; remember.className = 'remember-approval'; label.append(remember, document.createTextNode(' For this run'));
      actions.append(allow, deny, label); approval.append(actions); wrap.append(approval);
    }
    return wrap;
  }
  function messageView(message) {
    const el = node('article', 'message ' + message.role); el.dataset.message = message.id;
    const header = node('div', 'message-header'); header.append(node('span', 'message-avatar', message.role === 'assistant' ? 'll' : message.role === 'tool' ? '>_' : 'Y'), node('span', '', message.role === 'assistant' ? 'Assistant' : message.role === 'tool' ? 'Tool result' : 'You'));
    if (message.created) { const time = timestamp(message.created); if (!isNaN(time)) header.append(node('span', 'message-time', time.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}))); }
    el.append(header);
    if (message.attachments?.length) { const attachments = node('div', 'message-attachments'); message.attachments.forEach(a => attachments.append(renderAttachment(a))); el.append(attachments); }
    if (message.reasoning) { const reasoning = node('details', 'reasoning'); reasoning.dataset.disclosure = 'reasoning'; reasoning.append(node('summary', '', 'Reasoning'), markdown(message.reasoning)); el.append(reasoning); }
    if (message.content) el.append(markdown(message.content));
    (message.tool_calls || []).forEach(call => {
      const tool = toolRow(call, message);
      tool.querySelectorAll('details').forEach((detail, index) => { detail.dataset.disclosure = 'tool:' + call.id + ':' + index; });
      el.append(tool);
    });
    if (!message.content && !message.reasoning && !message.tool_calls?.length && message.role === 'assistant' && !terminal(message.status)) el.append(node('span', 'muted fine-print', 'Preparing a response...'));
    const actions = node('div', 'message-actions');
    if (message.content) actions.append(actionIcon('Copy message', 'copy', message.id, 'M8 5H5v16h12v-3 M9 3h11v14H9z'));
    if (message.role === 'user') actions.append(actionIcon('Edit message', 'edit', message.id, 'M4 16.5V20h3.5L19 8.5 15.5 5 4 16.5z M13.8 6.7l3.5 3.5'));
    if (message.role === 'assistant') actions.append(actionIcon('Retry response', 'retry', message.id, 'M20 7v5h-5 M19 12a7 7 0 1 1-2-5'));
    const siblings = S.messages.filter(m => m.role === message.role && m.parent_id === message.parent_id);
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
  function render() {
    const region = $('scroll-region'); const nearBottom = region.scrollHeight - region.scrollTop - region.clientHeight < 120;
    const selected = path().filter(m => m.role !== 'tool'); const container = $('messages'); const wanted = new Set(selected.map(m => m.id));
    container.querySelectorAll('[data-message]').forEach(el => { if (!wanted.has(el.dataset.message)) { el.remove(); S.fingerprint.delete(el.dataset.message); } });
    selected.forEach((message, index) => {
      const fingerprint = JSON.stringify(messageKey(message, true)) + S.messages.filter(m => m.parent_id === message.parent_id && m.role === message.role).map(m => m.id).join();
      let old = container.querySelector('[data-message="' + CSS.escape(message.id) + '"]');
      if (S.fingerprint.get(message.id) !== fingerprint || !old) {
        const open = new Map(old ? Array.from(old.querySelectorAll('details[data-disclosure]'), detail => [detail.dataset.disclosure, detail.open]) : []);
        const replacement = messageView(message);
        replacement.querySelectorAll('details[data-disclosure]').forEach(detail => { detail.open = open.get(detail.dataset.disclosure) || false; });
        if (old) {
          const preserved = [];
          const reasoning = old.querySelector('.reasoning'); const nextReasoning = replacement.querySelector('.reasoning');
          if (reasoning && nextReasoning) { reasoning.querySelector('.markdown').replaceWith(nextReasoning.querySelector('.markdown')); preserved.push([nextReasoning, reasoning]); }
          const focused = document.activeElement?.matches('.tool-call details[data-disclosure^="tool:"][data-disclosure$=":0"] > summary') ? document.activeElement.parentElement : null;
          const keepDetail = focused && old.contains(focused) ? focused : null;
          if (keepDetail) {
            const next = Array.from(replacement.querySelectorAll('details[data-disclosure^="tool:"]')).find(item => item.dataset.disclosure === keepDetail.dataset.disclosure);
            const body = next?.querySelector(':scope > .tool-body');
            const nextWrap = next?.closest('.tool-call');
            const nextSummary = next?.querySelector(':scope > summary'); const summary = keepDetail.querySelector(':scope > summary');
            if (nextWrap && body && nextSummary && summary) {
              keepDetail.querySelector(':scope > .tool-body')?.replaceWith(body);
              summary.children[1].textContent = nextSummary.children[1].textContent;
              summary.children[2].textContent = nextSummary.children[2].textContent;
              summary.children[2].classList.toggle('error', nextSummary.children[2].classList.contains('error'));
              const stableWrap = keepDetail.closest('.tool-call');
              reconcileChildren(stableWrap, nextWrap, [[next, keepDetail]]);
              preserved.push([nextWrap, stableWrap]);
            }
          }
          if (preserved.length) reconcileChildren(old, replacement, preserved);
          else old.replaceChildren(...replacement.childNodes);
        } else { container.append(replacement); old = replacement; }
        S.fingerprint.set(message.id, fingerprint);
      }
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
    $('chat-title').textContent = S.chat?.title || 'New conversation';
    const temporaryMode = Boolean(S.chat?.temporary);
    $('temporary-notice').hidden = !temporaryMode;
    $('temporary-button').querySelector('.chat-create-label').textContent = temporaryMode ? 'Normal chat' : 'Temporary chat';
    $('temporary-button').setAttribute('aria-label', temporaryMode ? 'Switch to a normal chat' : 'Start a temporary chat');
    $('app').dataset.temporary = String(temporaryMode);
    const lifetime = S.config.temporary?.ttl_seconds;
    $('temporary-description').textContent = 'Not saved to history or backups. Tools are off.' + (Number.isFinite(lifetime) ? ' Expires after ' + number(lifetime / 60) + ' minutes of inactivity.' : '') + ' The model server may retain logs or caches.';
    const modelUnavailable = !(S.config.models || []).some(model => model.id === $('model').value);
    $('send').hidden = Boolean(run); $('stop').hidden = !run; $('send').disabled = S.busy || modelUnavailable;
    if (!run && modelUnavailable && S.config.upstream_available) banner('This model is not available on the connected server. Choose another model before sending.');
    if ($('app').dataset.view === 'chat') document.title = (S.chat?.title ? S.chat.title + ' / ' : '') + 'llmx Chat';
    const groups = replyGroups();
    groups.forEach(group => group.calls.forEach((m, index) => {
      const control = container.querySelector('[data-message="' + CSS.escape(m.id) + '"] [data-action=reply-stats]');
      if (control) { control.dataset.reply = group.id; control.hidden = index !== group.calls.length - 1; control.setAttribute('aria-expanded', String(replyStats?.id === group.id)); }
    }));
    positionReplyStats(); syncLiveMetrics(); renderInspector(); renderContextLimits(); tick(); if (nearBottom) requestAnimationFrame(bottom);
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
  document.addEventListener('companion:view', () => hideReplyStats());
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
  async function syncSnapshot(chatId) { const generation = S.generation, data = await api('/chats/' + encodeURIComponent(chatId)); if (S.chat?.id !== chatId || S.generation !== generation) return; S.chat = data.chat; S.messages = data.messages || []; S.runs = data.runs || []; snapshotBanner(); scheduleRender(); }
  function tick() {
    renderLiveMetrics(); renderContextLimits();
    const run = currentRun(); if (!run) { $('run-caption').textContent = 'Enter to send / Shift + Enter for a new line'; return; }
    const seconds = Math.max(0, Math.floor((Date.now() - timestamp(run.created).getTime()) / 1000));
    const elapsed = Number.isFinite(seconds) ? (seconds >= 60 ? Math.floor(seconds / 60) + 'm ' : '') + seconds % 60 + 's elapsed' : 'In progress';
    document.querySelectorAll('.run-elapsed').forEach(el => { el.textContent = elapsed; }); $('run-caption').textContent = elapsed + ' / ' + (run.status === 'approval' ? 'Permission required' : 'Generating');
  }
  async function send() {
    if (currentRun() || S.busy || creatingChat) return;
    const originalChat = S.chat, generation = S.generation, edit = S.edit;
    const draft = $('draft').value, content = draft.trim(), revision = composerRevision, attachments = S.attachments.slice(), requestSettings = settings();
    if (!content && !attachments.length) { $('draft').focus(); return; }
    if (!requestSettings.model) throw new Error('Select a model before sending a message.');
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
        if (S.generation === generation && S.chat === originalChat) { S.chat = chat; localStorage.setItem('llmx-chat', chat.id); }
        renderSidebar();
      }
      phase = 'send to "' + chatTitle + '"';
      await api('/chats/' + encodeURIComponent(chatId) + '/runs', 'POST', {content, attachments, parent_id: parent || null, settings: requestSettings});
      if (!isCurrentView()) return;
      if (composerRevision === revision && S.edit === edit) {
        $('draft').value = ''; S.attachments = []; S.edit = null; composerRevision++; $('edit-note').hidden = true; resizeDraft(); renderAttachments();
      }
      phase = 'refresh "' + chatTitle + '"';
      await syncSnapshot(chatId);
      if (!isCurrentView()) return;
      const run = currentRun(); if (run) subscribe(run);
      phase = 'refresh conversations after sending to "' + chatTitle + '"';
      await refreshChats();
      if (isCurrentView()) bottom();
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
    const chatId = S.chat.id, generation = S.generation;
    const isCurrentView = () => S.chat?.id === chatId && S.generation === generation;
    await api('/chats/' + encodeURIComponent(chatId) + '/runs', 'POST', {regenerate: true, parent_id: message.id, settings: settings()});
    if (!isCurrentView()) return;
    await syncSnapshot(chatId);
    if (!isCurrentView()) return;
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
    await patchChat({active_leaf: leaf}); bottom();
  }
  function renderAttachments() {
    const root = $('attachments'); root.replaceChildren(); let total = 0;
    S.attachments.forEach((a, i) => {
      const size = attachmentBytes(a); total += size;
      const pill = node('span', 'attachment-pill', a.name);
      const label = node('small', 'attachment-file-size', formatAttachmentSize(size));
      label.title = size.toLocaleString() + ' bytes'; label.setAttribute('aria-label', a.name + ', ' + size.toLocaleString() + ' bytes'); pill.append(label);
      const remove = button('x', 'remove-attachment', String(i)); remove.setAttribute('aria-label', 'Remove ' + a.name); pill.append(remove); root.append(pill);
    });
    if (S.attachments.length) {
      const summary = node('span', 'attachment-total', 'Total: ' + S.attachments.length + (S.attachments.length === 1 ? ' file' : ' files') + ' - ' + formatAttachmentSize(total) + ' / 10 MiB');
      summary.setAttribute('role', 'status'); summary.setAttribute('aria-label', 'Total attachment size: ' + total.toLocaleString() + ' bytes of 10 MiB'); root.append(summary);
    }
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
  async function attach(files) {
    const chatId = S.chat?.id, generation = S.generation, attachments = S.attachments;
    if (attachments.length + files.length > 8) throw new Error('Attach up to 8 files per message.');
    const pending = [];
    for (const file of files) {
      if (['image/png', 'image/jpeg'].includes(file.type)) {
        if (file.size > 8 * 1024 * 1024) throw new Error(file.name + ' is larger than 8 MiB.');
        const data = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = () => reject(new Error('Could not read ' + file.name)); reader.readAsDataURL(file); }); pending.push({name: file.name, type: file.type, data});
      } else if (file.type.startsWith('text/') || /\.(txt|md|csv|json|py|js|ts|jsx|tsx|cpp|hpp|h|c|rs|go|java|sh|ps1|html|css|xml|log|yaml|yml|toml|ini|sql)$/i.test(file.name)) {
        if (file.size > 2 * 1024 * 1024 + 3) throw new Error(file.name + ' is larger than 2 MiB.');
        const bytes = await file.arrayBuffer(); let text; try { text = new TextDecoder('utf-8', {fatal: true}).decode(bytes); } catch (_) { throw new Error(file.name + ' must be UTF-8 text.'); }
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
    if (S.attachments.length + pending.length > 8) throw new Error('Attach up to 8 files per message.');
    const total = S.attachments.reduce((sum, attachment) => sum + attachmentBytes(attachment), 0) + pending.reduce((sum, attachment) => sum + attachmentBytes(attachment), 0);
    if (total > 10 * 1024 * 1024) throw new Error('Attachments exceed 10 MiB. Remove a file before adding more.');
    S.attachments.push(...pending); composerRevision++; renderAttachments(); $('draft').focus();
  }
  function bottom() { const region = $('scroll-region'); region.scrollTop = region.scrollHeight; $('jump-bottom').hidden = true; }
  function resizeDraft() { const el = $('draft'); el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 220) + 'px'; }
  function showTab(name) { S.tab = name; document.querySelectorAll('[data-tab]').forEach(el => { el.setAttribute('aria-selected', String(el.dataset.tab === name)); el.tabIndex = el.dataset.tab === name ? 0 : -1; }); for (const tab of ['activity', 'stats', 'settings']) $('panel-' + tab).hidden = tab !== name; syncLiveMetrics(); renderContextLimits(); }
  function updateShade() { const compact = document.documentElement.dataset.compact === 'true'; $('drawer-shade').hidden = !(compact && $('app').classList.contains('sidebar-open') || (compact || window.innerWidth <= 1200) && $('app').dataset.view === 'chat' && !$('app').classList.contains('inspector-closed')); syncLiveMetrics(); renderContextLimits(); }
  function closeDrawers() { $('app').classList.remove('sidebar-open'); if (document.documentElement.dataset.compact === 'true' || window.innerWidth <= 1200) $('app').classList.add('inspector-closed'); updateShade(); }
  function showToolCall(messageId, callId) {
    const message = path().find(item => item.id === messageId && item.role === 'assistant');
    if (!message || !(message.tool_calls || []).some(call => call.id === callId)) throw new Error('This tool call is no longer on the selected branch.');
    if (window.innerWidth <= 1200 || document.documentElement.dataset.compact === 'true') closeDrawers();
    const article = $('messages').querySelector('[data-message="' + CSS.escape(messageId) + '"]');
    const wrap = Array.from(article?.querySelectorAll('.tool-call') || []).find(item => item.dataset.toolCall === callId);
    const detail = wrap?.querySelector(':scope > details'); const summary = detail?.querySelector(':scope > summary');
    if (!summary) throw new Error('Tool details are not available in this conversation view.');
    detail.open = true;
    summary.focus({preventScroll: true});
    summary.scrollIntoView({block: 'center', behavior: 'instant'});
  }
  async function action(name, el) {
    if (name !== 'chat-menu') window.ChatShell.closeMenu();
    if (name === 'reply-stats') return showReplyStats(el, true);
    if (name === 'close-stats') return hideReplyStats(true);
    if (name === 'copy-stats') { if (replyStats) { await navigator.clipboard.writeText(replyStats.panel.dataset.copy); toast('Statistics copied.'); } return; }
    if (name === 'new') return newChat();
    if (name === 'temporary') return temporaryChat();
    if (name === 'end-temporary') return leaveTemporary();
    if (name === 'reply-detail') { const target = $('messages').querySelector('[data-message="' + CSS.escape(el.dataset.id) + '"]'); if (target) { closeDrawers(); target.tabIndex = -1; target.focus({preventScroll: true}); target.scrollIntoView({block: 'center', behavior: 'smooth'}); } return; }
    if (name === 'tool-detail') return showToolCall(el.dataset.message, el.dataset.call);
    if (name === 'sidebar') { $('app').classList.toggle('sidebar-open'); updateShade(); return; }
    if (name === 'close-drawers') return closeDrawers();
    if (name === 'inspector' || name === 'settings') { if (name === 'settings') { $('app').classList.remove('inspector-closed'); showTab('settings'); } else $('app').classList.toggle('inspector-closed'); updateShade(); return; }
    if (name === 'theme') { setTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light', true); return; }
    if (name === 'archive-view') { S.archive = !S.archive; renderSidebar(); return; }
    if (name === 'attach') return $('file-input').click();
    if (name === 'remove-attachment') { S.attachments.splice(Number(el.dataset.id), 1); composerRevision++; renderAttachments(); return; }
    if (name === 'bottom') return bottom();
    if (name === 'import') return $('import-input').click();
    if (name === 'logout') { await api('/logout', 'POST', {}); closeStream(); location.reload(); return; }
    if (name === 'cancel-edit') { S.edit = null; $('edit-note').hidden = true; $('draft').value = ''; S.attachments = []; composerRevision++; renderAttachments(); resizeDraft(); return; }
    if (name === 'save-settings') { if (S.chat) await patchChat({settings: settings()}); toast('Settings saved for the next response.'); return; }
    if (name === 'chat-menu') return showChatMenu(el.dataset.id || S.chat?.id, el);
    if (['rename', 'pin', 'archive', 'folder', 'delete', 'export', 'duplicate'].includes(name)) {
      const id = el.dataset.id || (name === 'rename' ? S.chat?.id : menuId);
      const chat = S.chats.find(item => item.id === id) || (S.chat?.id === id ? S.chat : null);
      if (!chat) throw new Error('This conversation is no longer available.');
      if (name === 'rename') return window.ChatShell.conversationForm({title: 'Rename conversation', label: 'Conversation name', value: chat.title}, title => patchChat({title}, chat.id));
      if (name === 'pin') await patchChat({pinned: !chat.pinned}, chat.id);
      if (name === 'archive') await patchChat({archived: !chat.archived}, chat.id);
      if (name === 'folder') return window.ChatShell.conversationForm({title: 'Move to folder', description: 'Choose an existing folder or enter a new name. Leave empty to remove the folder.', label: 'Folder', value: chat.folder || '', folders: [...new Set(S.chats.map(item => item.folder).filter(Boolean))]}, folder => patchChat({folder}, chat.id));
      if (name === 'delete') {
        if (chat.temporary) return leaveTemporary();
        return window.ChatShell.conversationForm({title: 'Delete conversation?', description: 'Delete "' + chat.title + '" and all its branches permanently. This cannot be undone.', destructive: true, submit: 'Delete'}, async () => {
          await api('/chats/' + encodeURIComponent(chat.id), 'DELETE');
          S.searchText.delete(chat.id); if (S.chat?.id === chat.id) clearChat(); await refreshChats();
        });
      }
      if (name === 'duplicate') {
        const data = await api('/chats/' + encodeURIComponent(chat.id) + '/export');
        data.chat.title = (chat.title || 'Conversation').slice(0, 193) + ' (copy)';
        await api('/import', 'POST', data); await refreshChats(); toast('Conversation duplicated.');
      }
      if (name === 'export') { const response = await fetch('/api/chats/' + encodeURIComponent(chat.id) + '/export'); if (!response.ok) throw new Error('Export failed.'); const url = URL.createObjectURL(await response.blob()); const a = node('a'); a.href = url; a.download = (chat.title || 'conversation').replace(/[^a-z0-9_-]/gi, '_') + '.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
      return;
    }
    if (!S.chat) return;
    if (name === 'stop') { const run = currentRun(); if (run) { await api('/runs/' + encodeURIComponent(run.id) + '/cancel', 'POST', {}); toast('Stop requested.'); } }
    if (name === 'copy') { const message = S.messages.find(m => m.id === el.dataset.id); await navigator.clipboard.writeText(message.content || ''); toast('Message copied.'); }
    if (name === 'edit') { if (currentRun()) throw new Error('Stop the current response before editing.'); const message = S.messages.find(m => m.id === el.dataset.id); S.edit = message; $('draft').value = message.content || ''; S.attachments = [...(message.attachments || [])]; composerRevision++; renderAttachments(); $('edit-note').hidden = false; resizeDraft(); $('draft').focus(); }
    if (name === 'retry') await retry(el.dataset.id);
    if (name === 'branch') await branch(el.dataset.id);
    if (name === 'approve' || name === 'deny') { await api('/runs/' + encodeURIComponent(el.dataset.run || currentRun()?.id) + '/approve', 'POST', {call_id: el.dataset.id, allow: name === 'approve', remember: el.closest('.approval').querySelector('.remember-approval').checked}); el.closest('.approval').querySelectorAll('button').forEach(b => { b.disabled = true; }); }
  }
  async function initialize() {
    chatCreating(true);
    try {
      const session = await api('/session'); $('logout-button').hidden = !session.password_required;
      if (!session.authenticated && session.password_required) { $('login-dialog').showModal(); $('password').focus(); return; }
      S.config = await api('/config'); document.dispatchEvent(new CustomEvent('companion:config', {detail: S.config})); connection(S.config.upstream_available ? 'Connected' : 'Unavailable', !S.config.upstream_available);
      if (!S.config.upstream_available) banner(S.config.upstream_error || 'The model server is unavailable. Check the companion connection and try again.');
      if (S.config.tools_error) banner([S.config.upstream_error, 'Tool connection issue: ' + S.config.tools_error].filter(Boolean).join(' '));
      await refreshChats(); loadSettings();
      $('temporary-button').hidden = !S.config.temporary?.enabled;
      const temporary = sessionStorage.getItem('llmx-temporary-chat');
      if (temporary) {
        try { await openChat(temporary); return; }
        catch (error) { if (error.status !== 404) throw error; sessionStorage.removeItem('llmx-temporary-chat'); toast('Your temporary chat has ended or expired.'); }
      }
      const previous = localStorage.getItem('llmx-chat');
      if (previous && S.chats.some(c => c.id === previous)) await openChat(previous); else render();
    } catch (error) { connection('Offline', true); banner(error.message + ' Reload the page to reconnect.'); }
    finally { chatCreating(false); }
  }
  document.addEventListener('click', event => {
    const el = event.target.closest('[data-action],[data-chat],[data-tab],[data-prompt]'); if (!el) return;
    const execute = async () => {
      if (el.dataset.action) await action(el.dataset.action, el);
      else if (el.dataset.chat) await openChat(el.dataset.chat);
      else if (el.dataset.tab) showTab(el.dataset.tab);
      else if (el.dataset.prompt) { $('draft').value = el.dataset.prompt; composerRevision++; resizeDraft(); $('draft').focus(); }
    };
    execute().catch(error => toast(error.message));
  });
  $('composer').addEventListener('submit', event => { event.preventDefault(); send().catch(error => banner(error.message)); });
  $('draft').addEventListener('input', () => { composerRevision++; resizeDraft(); });
  $('model').addEventListener('change', () => { snapshotBanner(); render(); });
  $('setting-reply-limit').addEventListener('change', () => { renderReplyLimit(); if (!$('setting-max_tokens').disabled) $('setting-max_tokens').focus(); });
  $('draft').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); send().catch(error => banner(error.message)); } });
  $('draft').addEventListener('paste', event => { const files = Array.from(event.clipboardData?.files || []); if (files.length) { event.preventDefault(); attach(files).catch(error => toast(error.message)); } });
  $('file-input').addEventListener('change', event => { attach(Array.from(event.target.files)).catch(error => toast(error.message)); event.target.value = ''; });
  $('import-input').addEventListener('change', async event => { const file = event.target.files[0]; event.target.value = ''; if (!file) return; try { if (file.size > 32 * 1024 * 1024) throw new Error('Import must be smaller than 32 MB.'); const chat = await api('/import', 'POST', JSON.parse(await file.text())); await refreshChats(); await openChat(chat.id || chat.chat?.id); toast('Conversation imported.'); } catch (error) { toast('Import failed: ' + error.message); } });
  $('search').addEventListener('input', renderSidebar);
  $('scroll-region').addEventListener('scroll', () => { const el = $('scroll-region'); $('jump-bottom').hidden = el.scrollHeight - el.scrollTop - el.clientHeight < 180; }, {passive: true});
  $('login-dialog').addEventListener('cancel', event => event.preventDefault());
  $('login-form').addEventListener('submit', async event => { event.preventDefault(); const submit = event.target.querySelector('button'); submit.disabled = true; try { await api('/login', 'POST', {password: $('password').value}); $('password').value = ''; $('login-error').textContent = ''; $('login-dialog').close(); await initialize(); } catch (error) { $('login-error').textContent = error.message; } finally { submit.disabled = false; } });
  document.addEventListener('keydown', event => {
    if (document.querySelector('dialog[open]')) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); newChat().catch(error => toast(error.message)); }
    if (event.key === 'Escape') closeDrawers();
    if (event.key === 'Tab' && !$('drawer-shade').hidden) {
      const drawer = $('app').dataset.view === 'chat' && !$('app').classList.contains('inspector-closed') ? $('inspector') : $('sidebar');
      const controls = Array.from(drawer.querySelectorAll('button,input,textarea,select,[href]')).filter(el => !el.disabled && el.getClientRects().length);
      if (controls.length && (!drawer.contains(document.activeElement) || event.shiftKey && document.activeElement === controls[0] || !event.shiftKey && document.activeElement === controls.at(-1))) { event.preventDefault(); (event.shiftKey ? controls.at(-1) : controls[0]).focus(); }
    }
    if (event.target.matches('[role=tab]') && ['ArrowLeft', 'ArrowRight'].includes(event.key)) { event.preventDefault(); const tabs = ['activity', 'stats', 'settings']; const index = (tabs.indexOf(S.tab) + (event.key === 'ArrowRight' ? 1 : 2)) % 3; showTab(tabs[index]); $('tab-' + tabs[index]).focus(); }
  });
  window.addEventListener('resize', updateShade);
  document.addEventListener('companion:view', () => { syncLiveMetrics(); renderLiveMetrics(); });
  document.addEventListener('visibilitychange', () => { syncLiveMetrics(); renderLiveMetrics(); });
  window.addEventListener('pagehide', () => { liveMetrics.stream?.close(); liveMetrics.stream = null; liveMetrics.id = null; });
  window.addEventListener('pageshow', syncLiveMetrics);
  document.addEventListener('visibilitychange', () => { if (!document.hidden && currentRun() && !S.stream) syncSnapshot(S.chat.id).then(() => { const run = currentRun(); if (run) subscribe(run); }).catch(error => toast(error.message)); });
  setTheme(themeChosen ? storedTheme : systemTheme.matches ? 'light' : 'dark');
  if (window.innerWidth <= 1200) $('app').classList.add('inspector-closed');
  showTab('activity'); setInterval(tick, 1000); initialize();
})();
