/* Shared navigation keeps chat, monitoring and workspace in one session. */
window.ChatShell = (() => {
  'use strict';
  const {node, icon} = window.ChatRender;
  const $ = id => document.getElementById(id);
  const labels = {chat: 'Chat', monitor: 'Monitor', workbench: 'Saved files', preferences: 'Settings'};
  let config = null;
  const toolTestErrors = new Map();
  function fitViewport() {
    const zoom = Number.parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
    const width = window.innerWidth / zoom, height = window.innerHeight / zoom;
    document.documentElement.style.setProperty('--studio-width', width + 'px'); document.documentElement.style.setProperty('--studio-height', height + 'px');
    document.documentElement.dataset.compact = String(width <= 720);
  }
  async function api(path, method = 'GET', body, signal) {
    const response = await fetch('/api' + path, {method, credentials: 'same-origin', headers: method === 'GET' ? {} : {'Content-Type': 'application/json', 'X-Chat-Request': '1'}, body: body === undefined ? undefined : JSON.stringify(body), signal});
    let data;
    try { data = await response.json(); } catch (_) { data = {}; }
    signal?.throwIfAborted();
    if (!response.ok) {
      if (response.status === 401) {
        document.dispatchEvent(new CustomEvent('companion:unauthorized'));
      }
      const error = new Error(data?.error || 'Request failed (' + response.status + ')'); error.status = response.status; error.details = data; throw error;
    }
    return data;
  }
  const featureRows = new Map();
  const featureStates = {available: 'Available', installing: 'Installing', installed: 'Installed', failed: 'Installation failed'};
  let features = [], publishedFeatures = '', featureRequest, featureInstall, featureTimer, featureDelay = 1000;
  let featuresAuthorized = false;

  function viewingFeatures() {
    return featuresAuthorized && !document.hidden && $('app').dataset.view === 'preferences';
  }
  function featuresStatus(message = '', failed = false) {
    const status = $('preferences-features-status');
    status.textContent = message; status.hidden = !message; status.classList.toggle('error-text', failed);
  }
  function featureRow(id) {
    const element = node('section', 'tool-connection optional-feature'); element.dataset.feature = id;
    const heading = node('strong', 'tool-connection-heading'); heading.id = 'optional-feature-' + id; element.setAttribute('aria-labelledby', heading.id);
    const description = node('p');
    const state = node('span', 'optional-feature-state'); state.tabIndex = -1; state.setAttribute('role', 'status');
    const error = node('p', 'tool-connection-error optional-feature-error'); error.hidden = true; error.setAttribute('role', 'alert');
    const button = node('button', 'secondary-button optional-feature-action'); button.type = 'button';
    button.addEventListener('click', () => {
      if (button.getAttribute('aria-disabled') === 'true' || !viewingFeatures()) return;
      requestFeatures(id);
    });
    const actions = node('div', 'tool-connection-actions'); actions.append(state, button);
    element.append(heading, description, actions, error);
    $('preferences-features').append(element);
    return {element, heading, description, state, button, error};
  }
  function renderFeatures() {
    const remaining = new Set(featureRows.keys());
    for (const feature of features) {
      const row = featureRows.get(feature.id) || featureRow(feature.id);
      featureRows.set(feature.id, row); remaining.delete(feature.id);
      row.element.dataset.state = feature.state;
      row.heading.textContent = feature.name;
      row.description.textContent = feature.description;
      const state = featureStates[feature.state] + (feature.version ? ' / ' + feature.version : '');
      if (row.state.textContent !== state) row.state.textContent = state;
      row.error.hidden = !feature.error;
      if (row.error.textContent !== (feature.error || '')) row.error.textContent = feature.error || '';
      if (feature.state === 'installed' && document.activeElement === row.button && viewingFeatures()) row.state.focus({preventScroll: true});
      row.button.hidden = feature.state === 'installed';
      const label = feature.state === 'installing' ? 'Installing' : feature.state === 'failed' ? 'Retry' : 'Install';
      row.button.textContent = label;
      row.button.setAttribute('aria-label', label + ' ' + feature.name);
      row.button.setAttribute('aria-disabled', String(Boolean(featureRequest || featureInstall && featureInstall.id !== feature.id) || feature.state === 'installing'));
    }
    for (const id of remaining) {
      featureRows.get(id).element.remove(); featureRows.delete(id);
    }
  }
  function applyFeatures(snapshot) {
    features = snapshot;
    if (featureInstall && !features.some(feature => feature.id === featureInstall.id && feature.state === 'installing')) featureInstall = null;
    featuresStatus(features.length ? '' : 'No optional features are available.');
    const key = JSON.stringify(features);
    if (key !== publishedFeatures) {
      publishedFeatures = key;
      document.dispatchEvent(new CustomEvent('companion:features', {detail: {features}}));
    }
  }
  function failFeature(id, message) {
    features = features.map(feature => feature.id === id ? {...feature, state: 'failed', error: message} : feature);
  }
  function stopFeatureTimer() {
    clearTimeout(featureTimer); featureTimer = null;
  }
  function scheduleFeatures() {
    if (featureRequest || featureTimer) return;
    if (featureInstall && Date.now() >= featureInstall.deadline) {
      failFeature(featureInstall.id, 'The install status could not be confirmed. Retry or reopen Settings to check again.');
      featureInstall = null; renderFeatures();
    }
    if (!featuresAuthorized || document.hidden || !featureInstall && !(viewingFeatures() && features.some(feature => feature.state === 'installing'))) {
      featureDelay = 1000; return;
    }
    featureTimer = setTimeout(() => {
      featureTimer = null; requestFeatures();
    }, featureDelay);
    featureDelay = Math.min(5000, featureDelay * 2);
  }
  async function requestFeatures(id) {
    if (featureRequest || !(viewingFeatures() || featureInstall && featuresAuthorized && !document.hidden)) return;
    stopFeatureTimer();
    const controller = new AbortController();
    const current = {id, controller};
    featureRequest = current;
    if (id) {
      featureInstall = {id, deadline: Date.now() + 180000}; featureDelay = 1000;
      features = features.map(feature => feature.id === id ? {...feature, state: 'installing', error: ''} : feature);
      featuresStatus();
    } else if (!features.length) featuresStatus('Loading optional features...');
    renderFeatures();
    const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(id ? 30000 : 8000)]);
    try {
      const result = id ? await api('/features/' + encodeURIComponent(id) + '/install', 'POST', {}, signal) : await api('/features', 'GET', undefined, signal);
      if (!Array.isArray(result.features)) throw new Error('The server did not return optional feature status.');
      applyFeatures(result.features);
    } catch (error) {
      if (controller.signal.aborted) return;
      if (Array.isArray(error.details?.features)) applyFeatures(error.details.features);
      else if (id) failFeature(id, error.name === 'TimeoutError' ? 'The install request timed out. Checking its status...' : error.message);
      else featuresStatus(error.name === 'TimeoutError' ? 'Checking optional features timed out.' : 'Could not check optional features. ' + error.message, true);
    } finally {
      if (featureRequest === current) featureRequest = null;
      renderFeatures();
      if (controller.signal.aborted && viewingFeatures()) requestFeatures();
      else scheduleFeatures();
    }
  }
  function syncFeatures() {
    if (viewingFeatures()) requestFeatures();
    else if (!featureInstall) { stopFeatureTimer(); featureRequest?.controller.abort(); }
    else if (document.hidden) stopFeatureTimer();
    else scheduleFeatures();
  }
  document.addEventListener('companion:view', syncFeatures);
  document.addEventListener('visibilitychange', syncFeatures);
  document.addEventListener('companion:ready', () => {
    if (featuresAuthorized) return;
    featuresAuthorized = true; syncFeatures();
  });
  document.addEventListener('companion:unauthorized', () => {
    featuresAuthorized = false; featureInstall = null; publishedFeatures = ''; features = [];
    stopFeatureTimer(); featureRequest?.controller.abort(); featuresStatus(); renderFeatures();
  });
  function themeMonitor() {
    try { const root = $('monitor-frame').contentDocument?.documentElement; if (root) root.dataset.theme = document.documentElement.dataset.theme; } catch (_) { /* The monitor is expected to remain same-origin. */ }
  }
  function loadMonitor(force = false) {
    const frame = $('monitor-frame'); const notice = $('monitor-notice');
    if (config?.monitor?.enabled === false || !config?.monitor?.url) { notice.textContent = config?.monitor?.error || 'Runtime monitoring is not available. The monitor uses the configured inference server health endpoint.'; notice.hidden = false; frame.hidden = true; return; }
    notice.hidden = true; frame.hidden = false;
    if (!frame.getAttribute('src') || force) frame.src = '/monitor/view';
  }
  function viewFocusTarget() {
    if ($('app').dataset.view !== 'chat') return $('shell-title');
    return $('chat-title').disabled ? $('draft') : $('chat-title');
  }
  function show(view) {
    if (!labels[view]) return;
    $('app').dataset.view = view;
    document.querySelectorAll('.global-nav [data-view]').forEach(button => { if (button.dataset.view === view) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current'); });
    for (const name of ['monitor', 'workbench', 'preferences']) $('view-' + name).hidden = view !== name;
    $('shell-title').textContent = labels[view];
    document.querySelector('.shell-heading').hidden = view === 'chat';
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
    if (event.target.closest('[data-chat],[data-action="new"],[data-action="temporary"]')) show('chat');
    if (event.target.closest('[data-action="logout"]')) $('monitor-frame').removeAttribute('src');
  }, true);
  $('preferences-chat-settings').addEventListener('click', () => { show('chat'); document.querySelector('[data-action="settings"]').click(); });
  $('monitor-reload').addEventListener('click', () => loadMonitor(true)); $('monitor-frame').addEventListener('load', themeMonitor);
  new MutationObserver(themeMonitor).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
  window.addEventListener('resize', fitViewport); fitViewport();
  let menuAnchor = null;
  let actionFormOpen = false;
  function restoreMenuFocus(origin) {
    const rowId = origin?.matches('.chat-row-menu') ? origin.dataset.id : null;
    const candidates = [origin];
    if (rowId) candidates.push(document.querySelector('.chat-row-menu[data-id="' + CSS.escape(rowId) + '"]'), $('search'), $('sidebar-toggle'));
    else candidates.push(document.querySelector('.topbar [data-action="chat-menu"]'));
    for (const target of candidates) {
      if (!target?.isConnected) continue;
      target.focus();
      if (document.activeElement === target) return;
    }
  }
  function closeMenu(restore = true) {
    const menu = $('action-menu');
    if (!menu.matches(':popover-open')) return;
    menu.hidePopover(); menuAnchor?.setAttribute('aria-expanded', 'false');
    if (menu.parentElement !== document.body) document.body.append(menu);
    if (restore) restoreMenuFocus(menuAnchor);
  }
  function menu(anchor, title, items, select, {above = false} = {}) {
    closeMenu(false); menuAnchor = anchor;
    const panel = $('action-menu'), drawer = anchor.closest('aside[aria-modal=true]');
    if (drawer) drawer.append(panel);
    panel.replaceChildren(node('div', 'conversation-menu-title', title)); panel.firstChild.id = 'action-menu-title';
    items.forEach(item => {
      if (item.separator) { const line = node('div', 'conversation-menu-divider'); line.setAttribute('role', 'separator'); panel.append(line); }
      const control = node('button', item.danger ? 'danger' : ''); control.type = 'button'; control.setAttribute('role', 'menuitem');
      control.dataset.action = item.action; control.disabled = Boolean(item.disabled); control.tabIndex = -1;
      const paths = {rename: 'm16 3 5 5-12 12-6 1 1-6Z M14 5l5 5', pin: 'm9 3 6 0-1 6 4 4H6l4-4Z M12 13v8', folder: 'M3 6h7l2 2h9v12H3Z', duplicate: 'M8 8h13v13H8Z M16 8V3H3v13h5', archive: 'M3 3h18v5H3Z M5 8v13h14V8 M9 12h6', export: 'M12 3v12 M7 10l5 5 5-5 M4 16v5h16v-5', delete: 'M3 6h18 M9 6V3h6v3 M5 6l1 15h12l1-15 M10 10v7 M14 10v7',
        outline: 'M4 5h2m3 0h11M4 12h2m3 0h11M4 19h2m3 0h11', transcript: 'M4 4h10M4 9h8M17 5v12m-4-4 4 4 4-4M4 20h16', 'find-message': 'M16 16l5 5M18 10a7 7 0 1 1-14 0 7 7 0 0 1 14 0Z', 'attach-device': 'M3 4h18v12H3Z M8 21h8 M12 16v5', 'attach-files': 'M6 3h8l4 4v14H6Z M14 3v5h4 M9 13h6 M9 16h6', 'attach-project': 'M3 6h7l2 2h9v12H3Z'};
      control.append(icon(paths[item.action], 'conversation-menu-icon'), node('span', '', item.label));
      control.addEventListener('click', event => { event.stopPropagation(); closeMenu(); select(item.action); }); panel.append(control);
    });
    anchor?.setAttribute('aria-expanded', 'true'); panel.showPopover();
    const zoom = Number.parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
    const box = anchor.getBoundingClientRect(), bounds = panel.getBoundingClientRect();
    const width = innerWidth / zoom, height = innerHeight / zoom;
    const left = above ? box.left / zoom : (box.right - bounds.width) / zoom;
    const top = above ? (box.top - bounds.height) / zoom - 6 : box.bottom / zoom + 6;
    panel.style.left = Math.max(8, Math.min(left, width - bounds.width / zoom - 8)) + 'px';
    panel.style.top = Math.max(8, Math.min(top, height - bounds.height / zoom - 8)) + 'px';
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
      dialog.onclose = () => {
        const focused = document.activeElement;
        if (focused === document.body || dialog.contains(focused)) restoreMenuFocus(origin);
        actionFormOpen = false; resolve(result);
      };
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
    try { const response = await fetch('/api/backup', {credentials: 'same-origin'}); if (!response.ok) { if (response.status === 401) document.dispatchEvent(new CustomEvent('companion:unauthorized')); const data = await response.json(); throw new Error(data.error || 'Backup failed.'); } window.ChatArtifacts.downloadBlob(await response.blob(), 'llmx-chat-' + new Date().toISOString().slice(0, 10) + '.sqlite3'); backupStatus('Conversation backup downloaded.'); }
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
  return {show, viewFocusTarget, api, menu, closeMenu, conversationForm, get config() { return config; }};
})();
