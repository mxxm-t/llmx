/* Code stays text; executable HTML is loaded only in an opaque-origin frame. */
window.ChatArtifacts = (() => {
  'use strict';
  const {node} = window.ChatRender;
  const extensions = {javascript: 'js', js: 'js', typescript: 'ts', ts: 'ts', python: 'py', py: 'py', shell: 'sh', bash: 'sh', html: 'html', css: 'css', json: 'json', cpp: 'cpp', c: 'c', diff: 'diff', patch: 'patch', markdown: 'md', md: 'md', sql: 'sql', rust: 'rs', go: 'go', yaml: 'yaml'};
  const keywords = new Set(('as async await break case catch class const continue def default delete do elif else enum except export extends false finally fn for from function if impl import in interface let match new None null of pass private public raise return self static struct super switch this throw true try type typeof undefined use var void while with yield').split(' '));
  let current = null, frame = null, bridgeToken = '', previewVersion = 0;
  function language(value) { return String(value || 'text').toLowerCase().split(/\s/)[0].replace(/[^a-z0-9_+-]/g, '') || 'text'; }
  function highlight(source, name) {
    const fragment = document.createDocumentFragment(); const text = String(source);
    if (text.length > 262144) { fragment.append(document.createTextNode(text)); return fragment; }
    if (['diff', 'patch'].includes(language(name))) {
      text.split(/(?<=\n)/).forEach(line => { const cls = /^@@|^diff |^index |^--- |^\+\+\+ /.test(line) ? 'syntax-meta' : line.startsWith('+') ? 'syntax-add' : line.startsWith('-') ? 'syntax-remove' : ''; fragment.append(node('span', cls, line)); }); return fragment;
    }
    const pattern = /(?:\/\*[\s\S]*?\*\/|\/\/[^\n]*|#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|\b\d+(?:\.\d+)?\b|\b[A-Za-z_$][\w$]*\b|<\/?[A-Za-z][^<>]*>)/g;
    let end = 0;
    for (const match of text.matchAll(pattern)) {
      fragment.append(document.createTextNode(text.slice(end, match.index)));
      const token = match[0]; let cls = '';
      if (/^(\/\/|\/\*|#)/.test(token)) cls = 'syntax-comment';
      else if (/^["'`]/.test(token)) cls = 'syntax-string';
      else if (/^\d/.test(token)) cls = 'syntax-number';
      else if (token.startsWith('<')) cls = 'syntax-tag';
      else if (keywords.has(token)) cls = 'syntax-keyword';
      fragment.append(cls ? node('span', cls, token) : document.createTextNode(token)); end = match.index + token.length;
    }
    fragment.append(document.createTextNode(text.slice(end))); return fragment;
  }
  function filename(name, lang) { return String(name || 'artifact.' + (extensions[language(lang)] || 'txt')).replace(/[\\/:*?"<>|\x00-\x1f]/g, '_').slice(0, 180); }
  function downloadBlob(blob, name) {
    const url = URL.createObjectURL(blob);
    const link = node('a');
    link.href = url;
    link.download = name;
    try { link.click(); } finally { setTimeout(() => URL.revokeObjectURL(url), 1000); }
  }
  function download(source, name, lang) {
    const type = language(lang) === 'html' ? 'text/html;charset=utf-8' : 'text/plain;charset=utf-8';
    downloadBlob(new Blob([source], {type}), filename(name, lang));
  }
  const dialog = node('dialog', 'artifact-dialog'); dialog.id = 'artifact-dialog'; dialog.setAttribute('aria-label', 'Artifact viewer');
  const header = node('header', 'artifact-header'); const titles = node('div'); const title = node('h2', '', 'Artifact'); const subtitle = node('p', '', 'Source file'); titles.append(title, subtitle);
  const close = node('button', 'icon', 'x'); close.type = 'button'; close.setAttribute('aria-label', 'Close artifact'); header.append(titles, close);
  const toolbar = node('div', 'artifact-toolbar'); const modes = node('div', 'artifact-modes'); modes.setAttribute('role', 'tablist'); modes.setAttribute('aria-label', 'Artifact view');
  const sourceButton = node('button', '', 'Source'); sourceButton.type = 'button'; sourceButton.setAttribute('role', 'tab'); sourceButton.setAttribute('aria-selected', 'true'); sourceButton.id = 'artifact-source-tab';
  const previewButton = node('button', '', 'Preview'); previewButton.type = 'button'; previewButton.setAttribute('role', 'tab'); previewButton.setAttribute('aria-selected', 'false'); previewButton.id = 'artifact-preview-tab';
  modes.append(sourceButton, previewButton);
  const actions = node('div', 'artifact-toolbar-actions'); const edit = node('button', 'secondary-button'); edit.type = 'button'; const copy = node('button', 'secondary-button', 'Copy'); const save = node('button', 'secondary-button', 'Download'); const wrap = node('button', 'secondary-button', 'Wrap lines'); wrap.type = 'button';
  actions.append(edit, wrap, copy, save); toolbar.append(modes, actions);
  const status = node('div', 'artifact-status'); status.setAttribute('role', 'status'); status.hidden = true;
  const content = node('div', 'artifact-content'); const sourcePanel = node('div', 'artifact-source'); sourcePanel.setAttribute('role', 'tabpanel'); sourcePanel.setAttribute('aria-labelledby', sourceButton.id); const pre = node('pre'); const code = node('code'); pre.append(code); sourcePanel.append(pre);
  const previewPanel = node('div', 'artifact-preview'); previewPanel.hidden = true; previewPanel.setAttribute('role', 'tabpanel'); previewPanel.setAttribute('aria-labelledby', previewButton.id);
  const previewNote = node('div', 'artifact-preview-note', 'Isolated browser preview. Scripts may run; resource requests and access to the studio are restricted.');
  const projectInfo = node('details'); projectInfo.hidden = true; previewNote.append(projectInfo);
  const projectSummary = node('summary'); const projectAssets = node('p'); projectInfo.append(projectSummary, projectAssets);
  const consolePanel = node('details', 'artifact-console'); const consoleSummary = node('summary', '', 'Console / 0 entries'); const consoleOutput = node('div', 'artifact-console-output'); consoleOutput.setAttribute('aria-live', 'polite'); consolePanel.append(consoleSummary, consoleOutput); previewPanel.append(previewNote, consolePanel);
  content.append(sourcePanel, previewPanel); dialog.append(header, toolbar, status, content); document.body.append(dialog);
  function stopPreview() { previewVersion++; bridgeToken = ''; if (frame) frame.remove(); frame = null; }
  function showSource() {
    stopPreview(); wrap.hidden = false; sourcePanel.hidden = false; previewPanel.hidden = true; status.hidden = true;
    sourceButton.setAttribute('aria-selected', 'true'); previewButton.setAttribute('aria-selected', 'false');
  }
  function bridge(token) {
    return '<script>(function(){"use strict";const token=' + JSON.stringify(token) + ',origin=' + JSON.stringify(location.origin) + ';let sent=0;function emit(kind,args){if(sent++>=256)return;let text;try{text=args.map(v=>typeof v==="string"?v:JSON.stringify(v)).join(" ")}catch(e){text="[Unserializable console value]"}parent.postMessage({channel:"llmx-preview",token,kind,text:String(text).slice(0,16384)},origin)}for(const kind of ["log","info","warn","error"]){const original=console[kind];console[kind]=function(...args){emit(kind,args);original.apply(console,args)}}addEventListener("error",e=>emit("error",[e.message]));addEventListener("unhandledrejection",e=>emit("error",[String(e.reason)]))})()<\/script>';
  }
  function projectMarkup(source, instrumentation) {
    const trivia = /^(?:[ \t\r\n\f\uFEFF]|<!--[\s\S]*?-->)*/;
    let offset = trivia.exec(source)[0].length;
    for (const tag of [/^<!doctype\b[^>]*>/i, /^<html(?=[\s>])(?:[^>"']|"[^"]*"|'[^']*')*>/i, /^<head(?=[\s>])(?:[^>"']|"[^"]*"|'[^']*')*>/i]) {
      const match = tag.exec(source.slice(offset));
      if (!match) continue;
      offset += match[0].length;
      offset += trivia.exec(source.slice(offset))[0].length;
    }
    return source.slice(0, offset) + instrumentation + source.slice(offset);
  }
  async function showPreview() {
    if (!current || previewButton.hidden) return;
    stopPreview(); const version = previewVersion;
    const bytes = crypto.getRandomValues(new Uint8Array(24)); bridgeToken = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
    wrap.hidden = true; sourcePanel.hidden = true; previewPanel.hidden = false; status.hidden = false; status.textContent = 'Preparing isolated preview...'; consoleOutput.replaceChildren(); consoleSummary.textContent = 'Console / 0 entries';
    projectInfo.hidden = true; projectInfo.open = false;
    sourceButton.setAttribute('aria-selected', 'false'); previewButton.setAttribute('aria-selected', 'true');
    try {
      const instrumentation = bridge(bridgeToken);
      const body = current.workspacePath
        ? {html: projectMarkup(current.source, instrumentation), workspace_path: current.workspacePath}
        : {html: instrumentation + current.source};
      const data = await window.ChatShell.api('/previews', 'POST', body);
      const target = new URL(data.url, location.href); if (target.origin !== location.origin || !target.pathname.startsWith('/preview/')) throw new Error('The preview service returned an invalid URL.');
      if (version !== previewVersion || !dialog.open) return;
      if (data.workspace) {
        const {revision, assets, path} = data.workspace;
        projectSummary.textContent = 'Workspace revision ' + revision + ' / ' + assets.length + (assets.length === 1 ? ' saved asset' : ' saved assets');
        projectAssets.textContent = path + (assets.length ? ' uses ' + assets.join(', ') + '.' : ' uses no saved CSS or JavaScript.');
        projectInfo.hidden = false;
      }
      frame = node('iframe', 'artifact-frame'); frame.title = current.name + ' preview'; frame.setAttribute('sandbox', 'allow-scripts'); frame.referrerPolicy = 'no-referrer'; frame.src = target.href;
      frame.addEventListener('load', () => { if (version === previewVersion) { status.hidden = true; } }); previewPanel.insertBefore(frame, consolePanel);
    } catch (error) { if (version === previewVersion) { status.textContent = error.message; status.hidden = false; } }
  }
  function open(artifact, {initialPreview = false, allowWorkspaceEdit = true, workspacePath = '', readOnlyAttachment = false} = {}) {
    current = {name: filename(artifact.name, artifact.language), language: language(artifact.language), source: String(artifact.source ?? ''), workspacePath};
    const displayName = readOnlyAttachment ? artifact.name : current.name;
    if (readOnlyAttachment && /\.pdf$/i.test(displayName)) current.name = current.name.slice(0, 176) + '.txt';
    setWrap(['text', 'txt', 'md', 'markdown'].includes(current.language));
    showSource(); title.textContent = displayName; subtitle.textContent = current.language.toUpperCase() + ' / ' + new Intl.NumberFormat().format(new TextEncoder().encode(current.source).length) + ' bytes';
    if (readOnlyAttachment) code.textContent = current.source;
    else code.replaceChildren(highlight(current.source, current.language));
    previewButton.hidden = readOnlyAttachment || (current.language !== 'html' && !/\.html?$/i.test(current.name));
    copy.textContent = 'Copy';
    edit.hidden = readOnlyAttachment || !allowWorkspaceEdit;
    edit.textContent = workspacePath ? 'Open in editor' : 'Edit in workspace';
    if (!dialog.open) dialog.showModal();
    if (initialPreview && !previewButton.hidden) { previewButton.focus(); showPreview(); }
    else sourceButton.focus();
  }
  close.addEventListener('click', () => dialog.close()); dialog.addEventListener('close', stopPreview); sourceButton.addEventListener('click', showSource); previewButton.addEventListener('click', showPreview);
  modes.addEventListener('keydown', event => { if (['ArrowLeft', 'ArrowRight'].includes(event.key)) { event.preventDefault(); event.stopPropagation(); if (!previewButton.hidden && event.target === sourceButton) { previewButton.focus(); showPreview(); } else { sourceButton.focus(); showSource(); } } });
  function setWrap(enabled) {
    pre.classList.toggle('wrapped', enabled);
    wrap.setAttribute('aria-pressed', String(enabled));
  }
  wrap.addEventListener('click', () => setWrap(!pre.classList.contains('wrapped')));
  copy.addEventListener('click', async () => { try { await navigator.clipboard.writeText(current.source); copy.textContent = 'Copied'; } catch (_) { status.textContent = 'Clipboard unavailable. Use Download to save the file.'; status.hidden = false; } });
  save.addEventListener('click', () => download(current.source, current.name, current.language));
  edit.addEventListener('click', () => { const artifact = {...current}; dialog.close(); document.dispatchEvent(new CustomEvent('companion:artifact-edit', {detail: artifact})); });
  window.addEventListener('message', event => {
    const data = event.data;
    if (!frame || event.source !== frame.contentWindow || event.origin !== 'null' || !data || typeof data !== 'object' || data.channel !== 'llmx-preview' || data.token !== bridgeToken || !['log', 'info', 'warn', 'error'].includes(data.kind) || typeof data.text !== 'string' || data.text.length > 16384 || consoleOutput.children.length >= 256) return;
    consoleOutput.append(node('pre', 'console-' + data.kind, data.text)); consoleSummary.textContent = 'Console / ' + consoleOutput.children.length + ' entries'; if (data.kind === 'error') consolePanel.open = true;
  });
  return {open, download, downloadBlob, highlight, language};
})();
