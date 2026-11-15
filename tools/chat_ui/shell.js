/* Shared navigation keeps chat, monitoring and workspace in one session. */
window.ChatShell = (() => {
  'use strict';
  const {node} = window.ChatRender;
  const $ = id => document.getElementById(id);
  const labels = {chat: 'Chat', monitor: 'Monitor', workbench: 'Workspace', preferences: 'Settings'};
  let config = null;
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
    $('shell-title').textContent = labels[view]; $('shell-eyebrow').textContent = view === 'monitor' ? 'RUNTIME' : view === 'workbench' ? 'WORKSPACE' : 'STUDIO';
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
    const tools = $('preferences-tools'); tools.replaceChildren();
    (config.tools || []).forEach(tool => { const row = node('div', 'connected-tool'); row.append(node('strong', '', tool.name), node('span', '', tool.description || tool.server || 'MCP tool')); tools.append(row); });
    if (!config.tools?.length) tools.append(node('p', 'muted', config.tools_error || 'No MCP tools connected.'));
    if ($('app').dataset.view === 'monitor') loadMonitor();
    document.dispatchEvent(new CustomEvent('companion:ready', {detail: config}));
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
    if (!confirm('Replace every conversation and branch with this backup? Current conversation data will be replaced. Workspace files and checkpoints are not affected.')) return;
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
    catch (error) { backupStatus(error.message, true); } finally { $('backup-restore').disabled = false; }
  });
  return {show, api, get config() { return config; }};
})();
