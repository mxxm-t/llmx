/* Shared navigation keeps chat, monitoring and workspace in one session. */
window.ChatShell = (() => {
  'use strict';
  const {node} = window.ChatRender;
  const $ = id => document.getElementById(id);
  const labels = {chat: 'Chat', monitor: 'Monitor', workbench: 'Workspace', preferences: 'Settings'};
  let config = null;
  const toolTestErrors = new Map();
  function fitViewport() {
    const zoom = Number.parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
    const width = window.innerWidth / zoom, height = window.innerHeight / zoom;
    document.documentElement.style.setProperty('--studio-width', width + 'px'); document.documentElement.style.setProperty('--studio-height', height + 'px');
    document.documentElement.dataset.compact = String(width <= 720);
  }
  async function api(path, method = 'GET', body) {
    const response = await fetch('/api' + path, {method, credentials: 'same-origin', headers: method === 'GET' ? {} : {'Content-Type': 'application/json', 'X-Chat-Request': '1'}, body: body === undefined ? undefined : JSON.stringify(body)});
    let data;
    try { data = await response.json(); } catch (_) { data = {}; }
    if (!response.ok) {
      if (response.status === 401) {
        document.dispatchEvent(new CustomEvent('companion:unauthorized'));
      }
      const error = new Error(data?.error || 'Request failed (' + response.status + ')'); error.status = response.status; throw error;
    }
    return data;
  }
  function themeMonitor() {
    try { const root = $('monitor-frame').contentDocument?.documentElement; if (root) root.dataset.theme = document.documentElement.dataset.theme; } catch (_) { /* The monitor is expected to remain same-origin. */ }
  }
  function loadMonitor(force = false) {
    const frame = $('monitor-frame'); const notice = $('monitor-notice');
    if (config?.monitor?.enabled === false || !config?.monitor?.url) { notice.textContent = config?.monitor?.error || 'Runtime monitoring is not available. The monitor uses the configured inference server health endpoint.'; notice.hidden = false; frame.hidden = true; return; }
    notice.hidden = true; frame.hidden = false;
    if (!frame.getAttribute('src') || force) frame.src = '/monitor/view';
  }
  function show(view) {
    if (!labels[view]) return;
    $('app').dataset.view = view;
    document.querySelectorAll('.global-nav [data-view]').forEach(button => { if (button.dataset.view === view) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current'); });
    for (const name of ['monitor', 'workbench', 'preferences']) $('view-' + name).hidden = view !== name;
    $('shell-title').textContent = labels[view]; $('shell-eyebrow').textContent = view === 'monitor' ? 'RUNTIME' : view === 'workbench' ? 'WORKSPACE' : 'LLMX';
    document.querySelector('.shell-heading').hidden = view === 'chat';
    $('app').classList.remove('sidebar-open'); $('drawer-shade').hidden = true;
    document.title = 'llmx / ' + labels[view];
    if (view === 'monitor') loadMonitor();
    document.dispatchEvent(new CustomEvent('companion:view', {detail: view}));
  }
  function configure(value) {
    config = value;
    $('preferences-upstream').textContent = (config.upstream_available ? 'Connected' : 'Unavailable') + (config.upstream ? ' / ' + config.upstream : '') + (config.upstream_error ? '\n' + config.upstream_error : '');
    $('preferences-auth').textContent = config.authentication === 'owner-password' ? 'Owner password is enabled for this companion.' : config.authentication === 'local-only' ? 'Local access. No owner password is configured.' : 'Authentication details were not reported.';
    $('preferences-models').replaceChildren(...(config.models || []).map(model => node('span', 'model-chip', model.id)));
    const monitor = config.monitor || {};
    $('preferences-monitor').textContent = monitor.url && monitor.enabled !== false ? 'Monitoring the configured inference server.' + (typeof monitor.interval_s === 'number' ? ' Samples refresh every ' + monitor.interval_s + ' seconds.' : '') : monitor.error || 'Monitoring is not available for the configured inference server.';
    renderToolConnections();
    if ($('app').dataset.view === 'monitor') loadMonitor();
    document.dispatchEvent(new CustomEvent('companion:ready', {detail: config}));
  }
  function renderToolConnections() {
    const out = $('preferences-tools'); out.replaceChildren();
    const servers = config?.tool_servers || [];
    if (!servers.length) {
      if (config?.tools?.length) {
        const card = node('section', 'tool-connection');
        card.append(node('strong', '', 'Available tools'), node('p', 'tool-connection-meta', 'Connection status unavailable'));
        (config.tools || []).forEach(tool => {
          const row = node('div', 'connected-tool'); row.append(node('strong', '', tool.name));
          row.append(node('span', '', tool.approval === true ? 'Approval required' : tool.approval === false ? 'Approval not required' : tool.connection === 'builtin:workspace' ? 'Approval varies by tool' : 'Approval policy is managed by the operator'));
          if (tool.description) row.append(node('span', '', tool.description));
          card.append(row);
        });
        out.append(card);
      } else out.append(node('p', 'muted', config?.tools_error || 'No external tool connections are configured. Start the companion with --tools PATH to add MCP servers.'));
      return;
    }
    if (!servers.some(server => server.kind === 'mcp')) out.append(node('p', 'muted', 'No external tool connections are configured. Start the companion with --tools PATH to add MCP servers.'));
    servers.forEach(server => {
      const card = node('section', 'tool-connection');
      const heading = node('div', 'tool-connection-heading');
      heading.append(node('strong', '', server.name || (server.kind === 'builtin' ? 'Workspace' : 'MCP server')));
      if (server.kind === 'builtin') heading.append(node('span', 'tool-kind', 'Built in'));
      const state = server.status === 'ready' ? 'Ready' : server.status === 'checking' ? 'Testing connection' : server.status === 'error' ? 'Connection error' : 'Not tested';
      const meta = node('p', 'tool-connection-meta', state + (Number.isFinite(server.tool_count) ? ' | ' + server.tool_count + ' tools' : ''));
      if (Number.isFinite(server.checked_at)) meta.append(document.createTextNode(' | Last checked ' + new Date(server.checked_at * 1000).toLocaleString()));
      if (Number.isFinite(server.duration_ms)) meta.append(document.createTextNode(' | ' + server.duration_ms + ' ms'));
      const groupTools = (config.tools || []).filter(tool => tool.connection === server.id);
      const list = node('div', 'connected-tool-list');
      groupTools.forEach(tool => {
        const row = node('div', 'connected-tool'); row.append(node('strong', '', tool.name));
        row.append(node('span', '', tool.approval === true ? 'Approval required' : tool.approval === false ? 'Approval not required' : 'Approval policy is managed by the operator'));
        if (tool.description) row.append(node('span', '', tool.description));
        list.append(row);
      });
      const error = server.status === 'error' && server.error ? server.error : toolTestErrors.get(server.id);
      const actions = node('div', 'tool-connection-actions');
      const policy = server.approval === false ? 'Operator policy: approval not required' : server.approval === true ? 'Operator policy: approval required' : 'Approval policy is managed by the operator';
      actions.append(node('span', 'muted fine-print', server.kind === 'builtin' && server.id === 'builtin:workspace' ? 'Approval varies by tool' : policy));
      if (server.kind === 'mcp') {
        const button = node('button', 'secondary-button', server.status === 'checking' ? 'Testing' : 'Test connection'); button.type = 'button'; button.disabled = server.status === 'checking'; button.dataset.testServer = server.id;
        button.addEventListener('click', async () => {
          const restoreFocus = document.activeElement === button;
          const priorStatus = server.status;
          toolTestErrors.delete(server.id);
          button.disabled = true; button.textContent = 'Testing';
          config = {...config, tool_servers: (config.tool_servers || []).map(item => item.id === server.id ? {...item, status: 'checking'} : item)};
          renderToolConnections();
          try {
            const result = await api('/tools/test', 'POST', {server: server.id});
            config = {...config, tools: result.tools || [], tool_servers: result.tool_servers || [], tools_error: result.tools_error || null};
            document.dispatchEvent(new CustomEvent('companion:tools-updated', {detail: config}));
          } catch (error) {
            toolTestErrors.set(server.id, error.message);
            config = {...config, tool_servers: (config.tool_servers || []).map(item => item.id === server.id ? {...item, status: priorStatus} : item)};
          }
          renderToolConnections();
          if (restoreFocus) $('preferences-tools').querySelector('[data-test-server="' + CSS.escape(server.id) + '"]')?.focus();
        });
        actions.append(button);
      }
      card.append(heading);
      if (error) card.append(node('p', 'tool-connection-error', error));
      card.append(meta, list, actions); out.append(card);
    });
  }
  document.addEventListener('companion:config', event => configure(event.detail));
  document.addEventListener('companion:unauthorized', () => { $('monitor-frame').removeAttribute('src'); if (!$('login-dialog').open) $('login-dialog').showModal(); });
  document.addEventListener('click', event => {
    const control = event.target.closest('button[data-view]'); if (control) show(control.dataset.view);
    if (event.target.closest('[data-chat],[data-action="new"],[data-action="temporary"],[data-prompt]')) show('chat');
    if (event.target.closest('[data-action="logout"]')) $('monitor-frame').removeAttribute('src');
  }, true);
  $('preferences-chat-settings').addEventListener('click', () => { show('chat'); document.querySelector('[data-action="settings"]').click(); });
  $('monitor-reload').addEventListener('click', () => loadMonitor(true)); $('monitor-frame').addEventListener('load', themeMonitor);
  new MutationObserver(themeMonitor).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
  window.addEventListener('resize', fitViewport); fitViewport();
  let menuAnchor = null;
  let actionFormOpen = false;
  function restoreMenuFocus(origin) {
    const target = origin?.isConnected ? origin : origin?.dataset.id ? document.querySelector('.chat-row-menu[data-id="' + CSS.escape(origin.dataset.id) + '"]') : null;
    (target || document.querySelector('.topbar [data-action="chat-menu"]'))?.focus();
  }
  function closeMenu(restore = true) {
    const menu = $('action-menu');
    if (!menu.matches(':popover-open')) return;
    menu.hidePopover(); menuAnchor?.setAttribute('aria-expanded', 'false');
    if (restore) restoreMenuFocus(menuAnchor);
  }
  function menu(anchor, title, items, select) {
    closeMenu(false); menuAnchor = anchor;
    const panel = $('action-menu'); panel.replaceChildren(node('div', 'conversation-menu-title', title)); panel.firstChild.id = 'action-menu-title';
    items.forEach(item => {
      if (item.separator) { const line = node('div', 'conversation-menu-divider'); line.setAttribute('role', 'separator'); panel.append(line); }
      const control = node('button', item.danger ? 'danger' : ''); control.type = 'button'; control.setAttribute('role', 'menuitem');
      control.dataset.action = item.action; control.disabled = Boolean(item.disabled); control.tabIndex = -1;
      const paths = {rename: 'm16 3 5 5-12 12-6 1 1-6Z M14 5l5 5', pin: 'm9 3 6 0-1 6 4 4H6l4-4Z M12 13v8', folder: 'M3 6h7l2 2h9v12H3Z', duplicate: 'M8 8h13v13H8Z M16 8V3H3v13h5', archive: 'M3 3h18v5H3Z M5 8v13h14V8 M9 12h6', export: 'M12 3v12 M7 10l5 5 5-5 M4 16v5h16v-5', delete: 'M3 6h18 M9 6V3h6v3 M5 6l1 15h12l1-15 M10 10v7 M14 10v7'};
      const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); icon.classList.add('conversation-menu-icon'); icon.setAttribute('viewBox', '0 0 24 24'); icon.setAttribute('fill', 'none'); icon.setAttribute('stroke', 'currentColor'); icon.setAttribute('stroke-width', '1.6'); icon.setAttribute('stroke-linecap', 'round'); icon.setAttribute('stroke-linejoin', 'round'); icon.setAttribute('aria-hidden', 'true');
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path'); path.setAttribute('d', paths[item.action]); icon.append(path); control.append(icon, node('span', '', item.label));
      control.addEventListener('click', event => { event.stopPropagation(); closeMenu(); select(item.action); }); panel.append(control);
    });
    anchor?.setAttribute('aria-expanded', 'true'); panel.showPopover();
    const zoom = Number.parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
    const box = anchor.getBoundingClientRect(), bounds = panel.getBoundingClientRect();
    const width = innerWidth / zoom, height = innerHeight / zoom;
    panel.style.left = Math.max(8, Math.min(box.right / zoom - bounds.width / zoom, width - bounds.width / zoom - 8)) + 'px';
    panel.style.top = Math.max(8, Math.min(box.bottom / zoom + 6, height - bounds.height / zoom - 8)) + 'px';
    panel.querySelector('button:not(:disabled)')?.focus();
  }
  $('action-menu').addEventListener('toggle', event => { if (event.newState === 'closed' && !$('action-menu').matches(':popover-open')) menuAnchor?.setAttribute('aria-expanded', 'false'); });
  $('action-menu').addEventListener('keydown', event => {
    const controls = Array.from($('action-menu').querySelectorAll('button:not(:disabled)'));
    const index = controls.indexOf(document.activeElement);
    if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) { event.preventDefault(); controls[event.key === 'Home' ? 0 : event.key === 'End' ? controls.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + controls.length) % controls.length]?.focus(); }
    if (event.key === 'Escape' || event.key === 'Tab') { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); } closeMenu(); }
  });
  document.addEventListener('pointerdown', event => { if (!$('action-menu').contains(event.target) && !menuAnchor?.contains(event.target)) closeMenu(false); }, true);
  window.addEventListener('resize', () => closeMenu());
  function conversationForm({title, description = '', label = '', value = '', maxLength = 200, folders = [], destructive = false, submit = 'Save'}, save) {
    if (actionFormOpen) return Promise.resolve(undefined);
    actionFormOpen = true;
    closeMenu(); const dialog = $('action-dialog'), origin = document.activeElement;
    if (dialog.open) { actionFormOpen = false; return Promise.resolve(undefined); }
    $('action-dialog-title').textContent = title; $('action-dialog-description').textContent = description;
    $('conversation-field').hidden = !label; $('conversation-label').textContent = label;
    const input = $('conversation-value'); input.value = value; input.maxLength = maxLength; input.setCustomValidity(''); input.disabled = !label; input.required = Boolean(label && label !== 'Folder');
    $('conversation-folders').replaceChildren(...folders.map(folder => { const option = node('option'); option.value = folder; return option; }));
    input.setAttribute('list', label === 'Folder' ? 'conversation-folders' : '');
    const confirm = $('conversation-submit'), cancel = $('conversation-cancel'); confirm.textContent = submit; confirm.className = destructive ? 'primary-button danger' : 'primary-button';
    $('conversation-error').textContent = ''; confirm.disabled = false; cancel.disabled = false;
    dialog.oncancel = event => { if (confirm.disabled) event.preventDefault(); };
    let result;
    return new Promise(resolve => {
      dialog.onclose = () => { restoreMenuFocus(origin); actionFormOpen = false; resolve(result); };
      cancel.onclick = () => dialog.close();
      $('conversation-form').onsubmit = async event => {
        event.preventDefault(); if (input.required && !input.value.trim()) { input.setCustomValidity('Enter a value for this field.'); input.reportValidity(); return; }
        confirm.disabled = true; cancel.disabled = true; confirm.setAttribute('aria-busy', 'true');
        try { const next = input.value.trim(); if (save) await save(next); result = label ? next : true; dialog.close(); }
        catch (error) { $('conversation-error').textContent = error.message; }
        finally { confirm.disabled = false; cancel.disabled = false; confirm.removeAttribute('aria-busy'); }
      };
      input.oninput = () => input.setCustomValidity(''); dialog.showModal(); if (label) { input.focus(); input.select(); } else cancel.focus();
    });
  }
  function backupStatus(text, failed = false) { $('backup-status').textContent = text; $('backup-status').hidden = false; $('backup-status').classList.toggle('error-text', failed); }
  $('backup-download').addEventListener('click', async () => {
    $('backup-download').disabled = true;
    try { const response = await fetch('/api/backup', {credentials: 'same-origin'}); if (!response.ok) { if (response.status === 401) document.dispatchEvent(new CustomEvent('companion:unauthorized')); const data = await response.json(); throw new Error(data.error || 'Backup failed.'); } const url = URL.createObjectURL(await response.blob()); const link = node('a'); link.href = url; link.download = 'llmx-chat-' + new Date().toISOString().slice(0, 10) + '.sqlite3'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); backupStatus('Conversation backup downloaded.'); }
    catch (error) { backupStatus(error.message, true); } finally { $('backup-download').disabled = false; }
  });
  $('backup-restore').addEventListener('click', () => $('backup-file').click());
  $('backup-file').addEventListener('change', async event => {
    const file = event.target.files[0]; event.target.value = ''; if (!file) return;
    const maximum = config?.backup_max_bytes;
    if (typeof maximum === 'number' && file.size > maximum) { backupStatus('Choose a SQLite backup of at most ' + window.ChatRender.number(maximum / (1024 * 1024)) + ' MiB.', true); return; }
    await conversationForm({title: 'Restore conversation backup?', description: 'Replace every conversation and branch with this backup. Workspace files and checkpoints are not affected.', destructive: true, submit: 'Restore'}, async () => {
      $('backup-restore').disabled = true;
      try {
        const response = await fetch('/api/restore', {method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/vnd.sqlite3', 'X-Chat-Request': '1'}, body: file});
        if (!response.ok) {
          if (response.status === 401) document.dispatchEvent(new CustomEvent('companion:unauthorized'));
          let data; try { data = await response.json(); } catch (_) { data = {}; }
          throw new Error(data?.error || 'Backup restore failed (' + response.status + ').');
        }
        backupStatus('Backup restored. Reloading conversations...'); location.reload();
      }
      catch (error) { backupStatus(error.message, true); throw error; } finally { $('backup-restore').disabled = false; }
    });
  });
  return {show, api, menu, closeMenu, conversationForm, get config() { return config; }};
})();
