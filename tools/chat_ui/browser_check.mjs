/* Real Chrome integration without npm packages. Invoked by test_ui.py. */
import {spawn} from 'node:child_process';
import {readFile, writeFile, mkdir, readdir} from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

const [url, executable, profile, artifacts, fixture, imageFixture, originalDashboard, upstreamFixture,
  textMaxFixture, textOverFixture, imageMaxFixture, imageOverFixture, tinyFixture, textRawOverFixture, textBomMaxFixture,
  textRaceAFixture, textRaceBFixture] = process.argv.slice(2);
const pdfMode = process.argv.find(arg => arg.startsWith('--pdf-only='))?.split('=')[1];
const featuresMode = process.argv.includes('--features-only');
const homeContextMode = process.argv.includes('--home-context-only');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const startupTimeoutArg = process.argv.find(arg => arg.startsWith('--startup-timeout-ms='));
const startupTimeoutCandidate = startupTimeoutArg ? Number(startupTimeoutArg.slice('--startup-timeout-ms='.length)) : NaN;
const defaultStartupTimeoutMs = process.platform === 'win32' ? 60000 : 30000;
const startupTimeoutMs = Number.isSafeInteger(startupTimeoutCandidate) && startupTimeoutCandidate > 0
  ? startupTimeoutCandidate : defaultStartupTimeoutMs;
const launchArgs = ['--headless=new', '--no-first-run', '--no-default-browser-check',
  '--disable-background-networking', '--remote-debugging-port=0', '--user-data-dir=' + profile,
  '--window-size=1440,1000', 'about:blank'];
if (process.platform === 'linux') launchArgs.unshift('--no-sandbox');
const startupStarted = Date.now();
const chrome = spawn(executable, launchArgs, {stdio: ['ignore', 'ignore', 'pipe'], windowsHide: true});
let launchError = null, launchStderr = '';
chrome.on('error', error => { launchError = error; });
chrome.stderr.on('data', chunk => { launchStderr = (launchStderr + chunk.toString()).slice(-8192); });
function launchDiagnostic(reason) {
  const processState = chrome.exitCode === null && chrome.signalCode === null
    ? 'still running' : 'exited with code=' + chrome.exitCode + ', signal=' + chrome.signalCode;
  return new Error(reason + '\nExecutable: ' + executable + '\nPlatform: ' + process.platform +
    '\nChrome process: pid=' + chrome.pid + ', ' + processState +
    '\nChrome argv: ' + JSON.stringify(launchArgs) +
    '\nChrome stderr (last 8 KiB):\n' + (launchStderr || '[No stderr received]'));
}
let socket;
let nextId = 0;
const pending = new Map();
const errors = [];
const runStreams = [];
let heldMonitorLoad = null;
let heldRestoreRequest = null;
let heldSendResponse = null;
let heldChatCreateResponse = null;
let attachmentChooser = null;
const nativeDialogs = [];
const apiMutations = [];
const imageLinkRequests = [];
const savedFileReads = [];
const browserRequests = [];
const pdfWorkers = [];
async function cdp(method, params = {}, sessionId) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    const timeoutError = new Error('CDP timed out: ' + method);
    const timeout = setTimeout(() => { pending.delete(id); reject(timeoutError); }, 12000);
    pending.set(id, {resolve, reject, timeout, method});
    socket.send(JSON.stringify({id, method, params, sessionId}));
  });
}
async function evaluate(expression) {
  const result = await cdp('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text + ': ' + JSON.stringify(result.exceptionDetails.exception));
  return result.result.value;
}
async function wait(expression, description, timeout = 12000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (typeof expression === 'function' ? await expression() : await evaluate(expression)) return; await delay(80); }
  let detail = '';
  if (description === 'saved theme after reload') {
    try { detail = '\nTheme state: ' + JSON.stringify(await evaluate("({url:location.href,readyState:document.readyState,theme:document.documentElement.dataset.theme,storedTheme:localStorage.getItem('llmx-theme')})")); } catch {}
  }
  try {
    detail += '\nFocus state: ' + JSON.stringify(await evaluate("(()=>{const active=document.activeElement;return {focused:document.hasFocus(),visibility:document.visibilityState,view:document.querySelector('#app')?.dataset.view,compact:document.documentElement.dataset.compact,active:active?{tag:active.tagName,id:active.id,className:active.className,action:active.dataset.action,describedBy:active.getAttribute('aria-describedby'),inert:Boolean(active.closest('[inert]'))}:null,dialog:document.querySelector('dialog[open]')?.id,drawer:document.querySelector('aside[aria-modal=true]')?.id};})()"));
  } catch { /* Retain the original failure if the browser no longer responds. */ }
  throw new Error('Timed out waiting for ' + description + '\n' + await evaluate('document.body.innerText') + detail);
}
async function documentIdentity() {
  const {frameTree} = await cdp('Page.getFrameTree');
  return {loader: frameTree.frame.loaderId, origin: await evaluate('performance.timeOrigin')};
}
async function waitForDocument(previous, description) {
  const end = Date.now() + 12000;
  while (Date.now() < end) {
    try {
      const {frameTree} = await cdp('Page.getFrameTree');
      if (frameTree.frame.loaderId !== previous.loader && await evaluate(`performance.timeOrigin!==${JSON.stringify(previous.origin)} && document.readyState==='complete'`)) return;
    } catch (error) {
      if (!/Cannot find (?:default execution )?context|Execution context was destroyed|Inspected target navigated/i.test(error.message)) throw error;
    }
    await delay(80);
  }
  throw new Error('Timed out waiting for a new ready document: ' + description);
}
async function navigateDocument(target) {
  const previous = await documentIdentity(); const result = await cdp('Page.navigate', {url: target});
  if (result.errorText) throw new Error('Navigation failed: ' + result.errorText);
  await waitForDocument(previous, 'navigate to ' + target);
}
async function reloadDocument() {
  const previous = await documentIdentity(); await cdp('Page.reload'); await waitForDocument(previous, 'reload');
}
const click = selector => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
const closeAction = selector => evaluate(`new Promise(resolve=>{const dialog=document.querySelector('#action-dialog');dialog.addEventListener('close',()=>resolve(true),{once:true});document.querySelector(${JSON.stringify(selector)}).click();})`);
async function pointerClick(selector) {
  const point = await evaluate(`{const e=document.querySelector(${JSON.stringify(selector)});e.scrollIntoView({block:'nearest'});const r=e.getBoundingClientRect();const x=r.x+r.width/2,y=r.y+r.height/2;const hit=document.elementFromPoint(x,y);({x,y,hit:e.contains(hit),target:hit?.outerHTML.slice(0,200),classes:document.querySelector('#app').className,sidebar:document.querySelector('#sidebar').getBoundingClientRect().toJSON(),transform:getComputedStyle(document.querySelector('#sidebar')).transform,viewport:{x:visualViewport.offsetLeft,y:visualViewport.offsetTop,width:visualViewport.width},rootScroll:document.documentElement.scrollLeft,main:document.querySelector('.main').getBoundingClientRect().toJSON()})}`);
  assert(point.hit, 'Pointer blocked for '+selector+': '+JSON.stringify(point));
  await cdp('Input.dispatchMouseEvent', {type: 'mousePressed', x: point.x, y: point.y, button: 'left', clickCount: 1});
  await cdp('Input.dispatchMouseEvent', {type: 'mouseReleased', x: point.x, y: point.y, button: 'left', clickCount: 1});
}
async function actionHoverPoint(selector) {
  await wait("!document.querySelector('#send').disabled", 'completed send before pointer checks');
  await cdp('Input.dispatchMouseEvent', {type: 'mouseMoved', x: 10, y: 90});
  return evaluate(`new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(() => {
    const button = document.querySelector(${JSON.stringify(selector)});
    button.scrollIntoView({block: 'center', behavior: 'instant'});
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const r = button.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
      resolve({x, y, hit: button.contains(document.elementFromPoint(x, y)), rect: r.toJSON()});
    }));
  })))`);
}
async function reasoningClickDuringDelta(text, expectedOpen) {
  const point = await evaluate(`{const summary=document.querySelector('.reasoning > summary');summary.scrollIntoView({block:'nearest'});window.reasoningSummary=summary;window.reasoningDetached=false;window.reasoningObserver=new MutationObserver(records=>{if(records.some(r=>Array.from(r.removedNodes).some(n=>n===summary||n.contains?.(summary))))window.reasoningDetached=true;});reasoningObserver.observe(document.querySelector('#messages'),{childList:true,subtree:true});const r=summary.getBoundingClientRect();({x:r.x+r.width/2,y:r.y+r.height/2})}`);
  await cdp('Input.dispatchMouseEvent', {type: 'mousePressed', ...point, button: 'left', clickCount: 1});
  assert.equal((await fetch(upstreamFixture + '/browser-step')).status, 204);
  await wait(`document.querySelector('.reasoning').textContent.includes(${JSON.stringify(text)})`, 'reasoning delta during pointer gesture');
  await cdp('Input.dispatchMouseEvent', {type: 'mouseReleased', ...point, button: 'left', clickCount: 1});
  assert.equal(await evaluate("document.querySelector('.reasoning').open"), expectedOpen, 'stream delta cancelled reasoning pointer click');
  assert.equal(await evaluate("reasoningSummary===document.querySelector('.reasoning > summary') && !reasoningDetached"), true, 'stream delta detached reasoning summary');
  await evaluate('reasoningObserver.disconnect()');
}
async function fill(selector, text) { await evaluate(`{const e=document.querySelector(${JSON.stringify(selector)});e.value=${JSON.stringify(text)};e.dispatchEvent(new Event('input',{bubbles:true}));}`); }
async function send(text) { await fill('#draft', text); await evaluate("document.querySelector('#composer').requestSubmit()"); }
async function signIn() {
  await wait("document.querySelector('#login-dialog')?.open", 'password dialog');
  await fill('#password', 'browser-test'); await evaluate("document.querySelector('#login-form').requestSubmit()");
  await wait("!document.querySelector('#login-dialog').open && document.querySelector('#model').value==='qwen3-local' && !document.querySelector('[data-action=\"new\"]').disabled", 'authenticated workspace');
}
async function holdHistoryRefresh() {
  await evaluate(`(() => {
    const original = window.fetch;
    window.releaseChatHistory = null;
    window.fetch = async (...args) => {
      const response = await original(...args);
      if (args[0] !== '/api/chats' || (args[1]?.method || 'GET') !== 'GET') return response;
      window.fetch = original;
      return new Promise(resolve => { window.releaseChatHistory = () => resolve(response); });
    };
  })()`);
}
async function idle() { await wait("document.querySelector('#stop').hidden && !document.querySelector('#send').disabled && document.querySelectorAll('.message.assistant').length>0", 'completed response'); }
async function upload(selector, filename) { const root = await cdp('DOM.getDocument'); const target = await cdp('DOM.querySelector', {nodeId: root.root.nodeId, selector}); await cdp('DOM.setFileInputFiles', {nodeId: target.nodeId, files: [filename]}); }
async function uploadMany(selector, files) { const root = await cdp('DOM.getDocument'); const target = await cdp('DOM.querySelector', {nodeId: root.root.nodeId, selector}); await cdp('DOM.setFileInputFiles', {nodeId: target.nodeId, files}); }
async function screenshot(name) { await delay(240); const result = await cdp('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false}); await writeFile(path.join(artifacts, name), Buffer.from(result.data, 'base64')); }
async function attachmentSource(source) {
  const item = `#action-menu [data-action="attach-${source}"]`;
  await pointerClick('[data-action=attach]');
  await wait(`document.querySelector(${JSON.stringify(item)})?.checkVisibility()`, 'Attach menu source ' + source);
  await pointerClick(item);
}
async function checkSavedFilesFromChat(files) {
  const chat = await evaluate("document.querySelector('.chat-item.active')?.dataset.chat");
  assert(chat, 'saved-file check requires an existing conversation');
  const beforeDraft = await evaluate("document.querySelector('#draft').value");
  const beforeMutations = apiMutations.length;
  const selected = files.map(([path,text])=>({path,text,name:path.replaceAll('/', ':')}));
  const currentDraft = "JSON.parse(sessionStorage.getItem('llmx-draft:'+localStorage.getItem('llmx-chat'))||'{}')";
  const marker = 'Keep this existing draft while choosing a saved file.';
  const remove = selected.map(({name})=>`[aria-label=${JSON.stringify('Remove ' + name)}]`);
  const modal = '#saved-file-dialog';
  const counts = () => evaluate(`${JSON.stringify(remove)}.map(selector=>document.querySelectorAll(selector).length)`);
  const select = async () => {
    await attachmentSource('files');
    await wait(`document.querySelector('${modal}')?.open`, 'saved files modal in Chat');
    for (const {path} of selected) {
      const option = `${modal} input[type=checkbox][value=${JSON.stringify(path)}]`;
      await fill('#saved-file-filter', path);
      await wait(`document.querySelector(${JSON.stringify(option)})?.checkVisibility()`, 'filtered saved file ' + path);
      await pointerClick(option);
      await wait(`document.querySelector(${JSON.stringify(option)}).checked && !document.querySelector('#saved-file-add').disabled`, 'saved file selected');
    }
  };
  const closed = async () => wait(`!document.querySelector('${modal}')?.open`, 'saved-file picker closed');
  const currentChatAndDraft = () => evaluate("({view:document.querySelector('#app').dataset.view,chat:document.querySelector('.chat-item.active')?.dataset.chat,draft:document.querySelector('#draft').value})");
  const expected = {view:'chat',chat,draft:marker};
  const noFiles = selected.map(()=>0), onceEach = selected.map(()=>1);
  const initialReads = savedFileReads.length;
  assert.deepEqual(await counts(), noFiles, 'saved file already exists in the test draft');
  await fill('#draft', marker);

  await select();
  const bounds = await evaluate(`(() => {const dialog=document.querySelector('${modal}');const r=dialog.getBoundingClientRect();return r.left>=0&&r.right<=innerWidth&&dialog.scrollWidth<=dialog.clientWidth;})()`);
  assert.equal(bounds, true, 'saved-file picker overflows the viewport');
  await cdp('Input.dispatchKeyEvent', {type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  await cdp('Input.dispatchKeyEvent', {type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  await closed();
  assert.deepEqual(await counts(), noFiles, 'cancelling the saved-file picker attached a file');
  assert.deepEqual(await currentChatAndDraft(), expected, 'cancelling changed the active draft');
  assert(savedFileReads.length > initialReads, 'saved-file picker did not load a metadata list');
  assert(savedFileReads.slice(initialReads).every(query=>!query.length), 'opening or cancelling fetched file contents');

  for (let attempt = 0; attempt < 2; attempt++) {
    await select();
    await pointerClick('#saved-file-add');
    await closed();
    assert.deepEqual(await counts(), onceEach, 'a selected saved file was missing or attached twice');
    assert.deepEqual(await currentChatAndDraft(), expected, 'adding a saved file changed the current conversation or draft');
  }
  await wait(`${JSON.stringify(selected)}.every(({name,text})=>(${currentDraft}).attachments?.some(file=>file.name===name&&file.text===text))`, 'exact saved text for all selected files in the draft');
  const reads = savedFileReads.slice(initialReads).filter(query=>query.length);
  const expectedPaths = selected.map(file=>file.path).sort();
  assert(reads.length >= 1, 'confirmation did not fetch saved files');
  for (const query of reads) assert.deepEqual(query.sort(), expectedPaths, 'selected files were not fetched together');
  assert.equal(apiMutations.length, beforeMutations, 'choosing a saved file sent or changed server data');
  for (const button of remove) await click(button);
  await fill('#draft', beforeDraft);
}
function pdfAssetRequests() {
  return browserRequests.filter(request => {
    const pathname = new URL(request.url).pathname;
    return pathname === '/pdf_attachment.js' || pathname.startsWith('/pdf-reader/');
  });
}
async function checkSentSource(name, source, exportName) {
  const button = `Array.from(document.querySelectorAll('.message-attachments button')).find(button=>button.textContent.includes(${JSON.stringify(name)}))`;
  await wait(`Boolean(${button})`, 'sent attachment button for ' + name);
  const beforeOpen = browserRequests.length;
  assert.equal(await evaluate(`(() => {const button=${button}; button.focus();return button.tabIndex>=0 && document.activeElement===button;})()`), true, 'sent attachment is not keyboard focusable');
  await cdp('Input.dispatchKeyEvent', {type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r'});
  await cdp('Input.dispatchKeyEvent', {type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
  await wait("document.querySelector('#artifact-dialog').open", 'sent attachment source for ' + name);
  assert.deepEqual(await evaluate(`(() => {
    const dialog=document.querySelector('#artifact-dialog'), code=dialog.querySelector('.artifact-source code');
    return {
      name:dialog.querySelector('h2').textContent,
      source:code.textContent,
      visible:code.checkVisibility(),
      editable:code.isContentEditable || Array.from(dialog.querySelectorAll('textarea,input,[contenteditable]')).some(control=>control.checkVisibility()),
      preview:dialog.querySelector('#artifact-preview-tab').checkVisibility() || Boolean(dialog.querySelector('iframe')),
      editCopy:Array.from(dialog.querySelectorAll('button')).some(button=>button.textContent==='Edit a copy' && button.checkVisibility()),
      markup:code.childElementCount,
    };
  })()`), {name, source, visible:true, editable:false, preview:false, editCopy:true, markup:0});
  assert.equal(await evaluate('typeof window.sentHtmlRan'), 'undefined', 'uploaded HTML executed in the companion');
  if (exportName) {
    const exported = await evaluate(`(async () => {
      const createURL=URL.createObjectURL, anchor=HTMLAnchorElement.prototype;
      const clickDescriptor=Object.getOwnPropertyDescriptor(anchor,'click');
      let blob, filename;
      URL.createObjectURL=file=>{blob=file; return createURL.call(URL,file);};
      anchor.click=function(){filename=this.download;};
      try {
        Array.from(document.querySelectorAll('#artifact-dialog button')).find(button=>button.textContent==='Download').click();
        return {filename, text:await blob.text(), type:blob.type};
      } finally {
        URL.createObjectURL=createURL;
        if (clickDescriptor) Object.defineProperty(anchor,'click',clickDescriptor); else delete anchor.click;
      }
    })()`);
    assert.deepEqual(exported, {filename:exportName,text:source,type:'text/plain;charset=utf-8'});
  }
  await cdp('Input.dispatchKeyEvent', {type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  await cdp('Input.dispatchKeyEvent', {type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  await wait(`!document.querySelector('#artifact-dialog').open && document.activeElement===${button}`, 'source viewer Escape restores attachment focus');
  assert.deepEqual(browserRequests.slice(beforeOpen).filter(request => request.method !== 'GET' ||
    new URL(request.url).pathname !== '/api/chats/run-status'), [], 'opening sent source made a network request');
}
function workspaceWriteCases(path, missing, revision, files) {
  const call = (id, target) => ({id:'linked-'+id, name:'workspace_write', arguments:JSON.stringify({path:target,content:'Unapplied write proposal.',revision:revision-1})});
  const stale = (id, target) => ({...call(id,target), status:'failed', error:'Workspace revision is '+revision+'. Reload before saving.'});
  return [
    {id:'linked-verified-read', name:'workspace_read', arguments:JSON.stringify({path}), status:'completed',
      result:{content:[{type:'text', text:JSON.stringify({path,content:'Earlier tool output'})}]}},
    {...call('write',path), status:'completed', result:{content:[{type:'text',text:JSON.stringify({revision,files})}]}},
    stale('stale-write',path), stale('missing-write',missing), stale('unsafe-write','../outside.txt'),
  ];
}
async function checkChatSavedWriteActions(filename, source) {
  const currentChat = () => evaluate("({view:document.querySelector('#app').dataset.view,chat:document.querySelector('.chat-item.active')?.dataset.chat,draft:document.querySelector('#draft').value})");
  const original = await currentChat();
  assert.equal(original.view, 'chat');
  assert(original.chat);
  const expected = {...original,draft:'Unsent Chat draft while viewing the saved file.'};
  await fill('#draft', expected.draft);
  const mutations = apiMutations.length, streams = runStreams.length;
  const action = id => `[data-tool-call="linked-${id}"] .workspace-tool-view-file`;
  for (const [id,label] of [['write','Open file'], ['stale-write','Open saved copy']]) {
    const selector = action(id);
    await wait(`document.querySelector(${JSON.stringify(selector)})?.checkVisibility()`, label + ' for ' + id);
    assert.equal(await evaluate(`document.querySelector(${JSON.stringify(selector)}).textContent`), label);
    assert.equal(await evaluate(`document.querySelector(${JSON.stringify(selector)}).getAttribute('aria-label')`), label + ' ' + filename);
    await pointerClick(selector);
    await wait("document.querySelector('#artifact-dialog').open", label + ' in Chat');
    assert.deepEqual(await evaluate("(() => {const dialog=document.querySelector('#artifact-dialog');return {name:dialog.querySelector('h2').textContent,text:dialog.querySelector('.artifact-source code').textContent,preview:dialog.querySelector('#artifact-preview-tab').checkVisibility()};})()"), {name:filename.split('/').at(-1),text:source,preview:false});
    assert.deepEqual(await currentChat(), expected, 'opening a tool file changed the active conversation or draft');
    await pointerClick('#artifact-dialog [aria-label="Close artifact"]');
    await wait("!document.querySelector('#artifact-dialog').open", 'close saved source');
  }
  await pointerClick(action('missing-write'));
  await wait("document.querySelector('[data-tool-call=linked-missing-write] .workspace-tool-file-error')?.checkVisibility() && /no longer available/.test(document.querySelector('[data-tool-call=linked-missing-write] .workspace-tool-file-error').textContent)", 'missing saved-file write feedback');
  assert.equal(await evaluate("!document.querySelector('[data-tool-call=linked-unsafe-write] :is(.workspace-tool-open,.workspace-tool-view-file,.workspace-tool-attach)') && !document.querySelector('#artifact-dialog').open"), true, 'an invalid or unavailable path opened a file');

  const attach = id => '[data-tool-call="linked-'+id+'"] .workspace-tool-attach';
  const saved = 'JSON.parse(sessionStorage.getItem('+JSON.stringify('llmx-draft:'+original.chat)+'))';
  await pointerClick(attach('verified-read'));
  await wait(saved + '?.attachments?.length===1', 'current saved file attached from read result');
  assert.deepEqual(await evaluate(saved + '.attachments'), [{name:filename.replaceAll('/',':'),type:'text/plain',text:source}]);
  for (const id of ['write','stale-write']) {
    await pointerClick(attach(id));
    await wait("document.activeElement.id==='draft'", 'attach action returns focus to draft');
    assert.equal(await evaluate("document.querySelectorAll('#attachments .attachment-pill').length"), 1, 'repeated file actions duplicated the attachment');
  }
  assert.deepEqual(await currentChat(), expected, 'attaching changed the conversation or unsent message');
  await pointerClick('#attachments [data-action=remove-attachment]');
  const fileURL = '/api/workspace/file?path=' + encodeURIComponent(filename);
  await evaluate("(() => { const fetchFile=window.fetch; window.releaseToolFile=null; window.fetch=async (...args)=>{ " +
    "const response=await fetchFile(...args); if(args[0]!=="+JSON.stringify(fileURL)+")return response; " +
    "window.fetch=fetchFile; await new Promise(resolve=>window.releaseToolFile=resolve); return response; }; })()");
  await pointerClick(attach('write'));
  await wait("typeof window.releaseToolFile==='function'", 'saved-file response held for draft edit');
  await fill('#draft', 'Newer unsent text');
  await evaluate('window.releaseToolFile()');
  await wait("document.querySelector('[data-tool-call=linked-write] .workspace-tool-file-error')?.checkVisibility()", 'changed draft kept while reading saved file');
  assert.equal(await evaluate("document.querySelectorAll('#attachments .attachment-pill').length"), 0);
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Newer unsent text');
  await fill('#draft', expected.draft);

  const inspectorOpened = await evaluate("document.querySelector('#inspector').checkVisibility()");
  const initialTab = await evaluate("document.querySelector('#inspector [role=tab][aria-selected=true]')?.id");
  if (!inspectorOpened) await pointerClick('#details-toggle');
  await pointerClick('#tab-settings');
  await wait("document.querySelector('#details-saved-files')?.checkVisibility()", 'Browse saved files in Chat details');
  assert.equal(await evaluate("document.querySelector('#details-saved-files').textContent.trim()"), 'Browse saved files');
  await pointerClick('#details-saved-files[data-action=attach-files]');
  const savedOption = `#saved-file-dialog input[type=checkbox][value=${JSON.stringify(filename)}]`;
  await wait(`document.querySelector('#saved-file-dialog').open && document.querySelector(${JSON.stringify(savedOption)})?.checkVisibility()`, 'saved file picker from Chat details');
  await pointerClick('#saved-file-cancel');
  await wait("!document.querySelector('#saved-file-dialog').open", 'cancel details file picker');
  assert.deepEqual(await currentChat(), expected, 'details file picker changed the active conversation or draft');
  assert.equal(apiMutations.length, mutations, 'reading saved files sent or mutated server state');
  assert.equal(runStreams.length, streams, 'reading saved files started a model run');
  if (initialTab && initialTab !== 'tab-settings') await pointerClick('#'+initialTab);
  if (!inspectorOpened) await pointerClick('#details-toggle');
  await fill('#draft', original.draft);
}
async function checkNewChatShortcut() {
  const shortcut = async () => {
    for (const type of ['keyDown', 'keyUp']) await cdp('Input.dispatchKeyEvent', {type, key:'k', code:'KeyK', modifiers:2});
  };
  const creates = () => apiMutations.filter(request => request.method === 'POST' && new URL(request.url).pathname === '/api/chats').length;
  const navigate = async view => {
    if (await evaluate("document.documentElement.dataset.compact==='true'")) {
      await pointerClick('#sidebar-toggle');
      await wait("document.querySelector('#sidebar').getBoundingClientRect().left>=0", 'navigation drawer opened');
    }
    await pointerClick('.global-nav [data-view=' + view + ']');
  };
  const before = creates();
  await navigate('workbench');
  await shortcut();
  await wait("document.querySelector('#app').dataset.view==='chat' && document.activeElement.id==='draft' && document.querySelector('#welcome').checkVisibility()", 'new chat shortcut opens Chat and focuses the draft');
  assert.equal(creates(), before + 1, 'new chat shortcut creates one conversation');
  const id = await evaluate("document.querySelector('.chat-item.active').dataset.chat");
  await fill('#draft', 'Unsent keyboard draft');
  await navigate('preferences');
  await shortcut();
  await wait("document.querySelector('#app').dataset.view==='chat' && document.activeElement.id==='draft'", 'repeat shortcut returns to the empty conversation');
  assert.equal(await evaluate("document.querySelector('.chat-item.active').dataset.chat"), id);
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Unsent keyboard draft');
  assert.equal(creates(), before + 1, 'repeat shortcut duplicated an empty conversation');
  await fill('#draft', '');
}
async function checkSpeakerLayout(width) {
  const layout = await evaluate(`['user','assistant'].map(role=>{
    const article=document.querySelector('.message.'+role),header=article.querySelector('.message-header').getBoundingClientRect();
    const body=article.querySelector(':scope > .markdown').getBoundingClientRect(),actions=article.querySelector('.message-actions').getBoundingClientRect();
    return {headerLeft:header.right<=body.left+1,headerAbove:header.bottom<=body.top+1,
      actionsBelow:actions.top>=body.bottom-1,actionsAligned:Math.abs(actions.left-body.left)<=1};
  })`);
  assert(layout.every(turn=>width===1440 ? turn.headerLeft : turn.headerAbove),'speaker placement at '+width+': '+JSON.stringify(layout));
  assert(layout[0].actionsBelow && (width!==1440 || layout[0].actionsAligned),'prompt actions below the text at '+width+': '+JSON.stringify(layout[0]));
}
async function checkPassageNavigation() {
  const term = 'tide window';
  const chat = await evaluate(`ChatShell.api('/import','POST',{version:1,chat:{title:'Harbor survey plan',active_leaf:'t'},messages:${JSON.stringify([
    {id:'p1',role:'user',content:'How should we use the tide window for our morning survey?'},
    {id:'a1',parent_id:'p1',role:'assistant',content:'## Morning survey plan\n\nUse the **tide** window to inspect the outer markers before the channel begins to drain. Start at the harbor entrance and work back toward shelter.\n\n### Field checklist\n\n- Photograph the marker from the channel side.\n- Record the time and water depth alongside the photograph.\n- Compare the gauge reading with the harbor station.\n\n~~~text\ncheck("tide window")\n~~~\n\nKeep the instrument case closed between readings.'},
    {id:'p2',parent_id:'a1',role:'user',content:'Please use the attached observations to refine the route.',attachments:[{name:'tide window.csv',type:'text/plain',text:'Private attachment tide window'}]},
    {id:'a2',parent_id:'p2',role:'assistant',content:'## Route adjustment\n\nThe tide **window** closes first at the eastern marker. Visit it before checking the sheltered pier.',reasoning:'Private reasoning tide window',tool_calls:[{id:'lookup',name:'echo',arguments:'Private arguments tide window',status:'completed'}]},
    {id:'t',parent_id:'a2',role:'tool',content:'Private result tide window',tool_call_id:'lookup'},
    {id:'other',parent_id:'p2',role:'assistant',content:'Alternate route tide window'},
  ])}})`);
  await reloadDocument();
  await wait(`document.querySelector('[data-chat="${chat.id}"]')`,'passage fixture in history');
  await pointerClick(`[data-chat="${chat.id}"]`);
  await wait("document.querySelectorAll('.message.assistant').length===2",'active survey conversation');
  await fill('#draft','I will check the harbor gauge before we launch.');
  const reply = await evaluate("document.querySelector('.message.assistant').dataset.message");
  const find = '#reading-navigation [data-action=find-message]';
  const open = async () => {
    await pointerClick(find);
    await wait("document.activeElement.id==='find-message-query'", 'passage finder focused');
    await fill('#find-message-query',term);
  };
  const activate = async () => {
    await open();
    assert.equal(await evaluate("document.querySelectorAll('#find-message-list [data-target]').length"),4,'one result per visible matching message');
    await pointerClick(`#find-message-list [data-target="${reply}"]`);
  };
  const press = async (key, modifiers=0) => {
    for (const type of ['keyDown','keyUp']) await cdp('Input.dispatchKeyEvent',{type,key,code:key,modifiers,windowsVirtualKeyCode:key==='Escape'?27:114});
  };
  const highlight = () => evaluate(`(() => {
    const ranges = [...(CSS.highlights.get('chat-passage') || [])];
    if (ranges.length!==1) return null;
    const range=ranges[0], parent=range.startContainer.parentElement,region=document.querySelector('#scroll-region').getBoundingClientRect();
    const bounds=range.getBoundingClientRect();
    return {text:range.toString(),article:parent.closest('.message')?.dataset.message,inline:range.startContainer!==range.endContainer && !!parent.closest('strong'),
      code:!!parent.closest('pre code'),file:!!parent.closest('.attachment-pill'),visible:bounds.top>=region.top-1 && bounds.bottom<=region.bottom+1,
      position:document.querySelector('#passage-position').textContent};
  })()`);
  const writes = apiMutations.length;
  const draftSize = () => evaluate(`new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>{
    const el=document.querySelector('#draft');resolve({height:el.clientHeight,content:el.scrollHeight,start:el.selectionStart,end:el.selectionEnd});
  })))`);
  await evaluate("document.querySelector('#draft').setSelectionRange(3,9)");
  for (const [width,height] of [[1440,1000],[320,800]]) {
    await cdp('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width===320});
    await wait(`innerWidth===${width} && document.documentElement.dataset.compact==='${width===320}'`,'survey reading viewport');
    await checkSpeakerLayout(width);
    const size = await draftSize();
    assert(size.height>=size.content,'draft clipped at '+width+': '+JSON.stringify(size));
    assert.deepEqual([size.start,size.end],[3,9],'resizing changed the draft selection');
    for (const theme of ['light','dark']) {
      await cdp('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:theme}]});
      await wait(`document.documentElement.dataset.theme==='${theme}'`,'survey reading theme');
      await evaluate(`{const region=document.querySelector('#scroll-region'),heading=document.querySelector('.message.assistant > .markdown h3');
        region.scrollTop+=heading.getBoundingClientRect().top-region.getBoundingClientRect().top-Math.min(105,region.clientHeight*.18);}`);
      await screenshot(`passage-reader-${width}-${theme}.png`);
      await activate();
      assert.deepEqual(await highlight(),{text:term,article:reply,inline:true,code:false,file:false,visible:true,position:'2 of 5'});
      assert.equal(await evaluate(`document.activeElement.dataset.message==='${reply}'`),true);
      const fit = await evaluate(`(() => {
        const composer=document.querySelector('#composer').getBoundingClientRect();
        return document.documentElement.scrollWidth<=innerWidth && composer.left>=0 && composer.right<=innerWidth && composer.bottom<=innerHeight &&
          ['#passage-edit','#passage-previous','#passage-next','#passage-close','#send'].every(s=>{
            const button=document.querySelector(s),r=button.getBoundingClientRect();
            return r.left>=0 && r.right<=innerWidth && button.contains(document.elementFromPoint(r.left+r.width/2,r.top+r.height/2));
          });
      })()`);
      assert.equal(fit,true,'passage controls and composer fit at '+width);
      await screenshot(`passage-found-${width}-${theme}.png`);
      await press('Escape');
      await wait(`document.activeElement.matches('${find}') && !CSS.highlights.has('chat-passage')`,'escape clears passage and restores Find');
    }
  }
  await activate();
  await evaluate(`{const range=document.createRange();range.selectNodeContents(document.querySelector('.message.assistant > .markdown h3'));
    const selection=getSelection();selection.removeAllRanges();selection.addRange(range);}`);
  await press('F3');
  assert.equal((await highlight()).code,true,'F3 advances to the fenced code text');
  assert.equal(await evaluate('getSelection().toString()'),'Morning survey plan','passage navigation changed the reader selection');
  await press('F3',8);
  assert.equal((await highlight()).inline,true,'Shift+F3 goes back across inline markup');
  await pointerClick('#passage-next');
  assert.equal(await evaluate('document.activeElement.id'),'passage-next');
  await pointerClick('#passage-next');
  assert.equal((await highlight()).file,true,'filename passage excludes the attachment body');
  await pointerClick('#passage-next');
  const last = await highlight();
  assert.equal(last.position,'5 of 5');
  assert.equal(last.visible,true,'cycling scrolls the selected passage into the transcript viewport');
  await pointerClick('#passage-next');
  assert.equal((await highlight()).position,'1 of 5','forward wrap');
  await pointerClick('#passage-previous');
  assert.equal((await highlight()).position,'5 of 5','backward wrap');
  await pointerClick('#passage-close');
  await wait(`document.activeElement.matches('${find}') && !CSS.highlights.has('chat-passage')`,'Close returns to Find');
  await open();
  await evaluate("{const filter=document.querySelector('#find-message-role');filter.value='assistant';filter.dispatchEvent(new Event('change'))}");
  assert.equal(await evaluate("document.querySelectorAll('#find-message-list [data-target]').length"),2,'role filter selects replies');
  await pointerClick(`#find-message-list [data-target="${reply}"]`);
  assert.equal((await highlight()).position,'1 of 3','role filter bounds the reading strip');
  await pointerClick('#passage-edit');
  assert.deepEqual(await evaluate("[document.activeElement.id,document.querySelector('#find-message-query').value,document.querySelector('#find-message-role').value]"),['find-message-query',term,'assistant']);
  await fill('#find-message-query','**tide**');
  await pointerClick(`#find-message-list [data-target="${reply}"]`);
  assert.equal(await evaluate(`document.activeElement.dataset.message==='${reply}' && !CSS.highlights.has('chat-passage') && !document.querySelector('#passage-navigation').checkVisibility()`),true,'raw Markdown match jumps to its message');
  await open(); await fill('#find-message-query','');
  await pointerClick(`#find-message-list [data-target="${reply}"]`);
  assert.equal(await evaluate(`document.activeElement.dataset.message==='${reply}' && !CSS.highlights.has('chat-passage') && !document.querySelector('#passage-navigation').checkVisibility()`),true,'empty query jumps to the message');
  assert.equal(await evaluate("document.querySelector('#draft').value"),'I will check the harbor gauge before we launch.');
  assert.equal(apiMutations.length,writes,'passage navigation changed server state');
  await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await wait("document.documentElement.dataset.compact==='false'",'desktop draft after phone navigation');
  const restored = await draftSize();
  assert(restored.height>=restored.content,'draft clipped after returning to desktop: '+JSON.stringify(restored));
  assert.deepEqual([restored.start,restored.end],[3,9]);
  await fill('#draft','Chart observation.\n'.repeat(50));
  const long = await draftSize();
  assert(long.height<=220 && long.content>long.height,'long draft should scroll under its height cap: '+JSON.stringify(long));
  await pointerClick('[data-action=new]');
  await wait("document.querySelector('#welcome').checkVisibility() && !document.querySelector('#draft').value",'new conversation for a streamed passage');
  await send('Check the latest station reading.');
  await wait("document.querySelector('.message.assistant > .markdown')?.innerText.includes('opens in the morning') && !document.querySelector('#stop').hidden",'first visible streamed occurrence');
  await fill('#draft','Keep the chart for tomorrow.');
  await open();
  await pointerClick('#find-message-list [data-target]');
  assert.equal((await highlight()).position,'1 of 1');
  const upstream = process.argv.find(arg=>arg.startsWith('--fixture-upstream=')).slice('--fixture-upstream='.length);
  assert.equal((await fetch(upstream+'/browser-step',{signal:AbortSignal.timeout(4000)})).status,204);
  await wait("document.querySelector('#passage-position').textContent==='1 of 2' && document.querySelector('#stop').hidden",'passages update when the streamed paragraph is replaced');
  assert.equal(await evaluate("Array.from(CSS.highlights.get('chat-passage'))[0].startContainer.isConnected"),true);
  assert.equal((await highlight()).visible,true,'stream growth scrolled the selected passage out of view');
  await pointerClick('#passage-next');
  const finalHit=await highlight();
  assert.equal(finalHit.position,'2 of 2');
  assert.equal(finalHit.visible,true,'new streamed passage is visible after advancing');
  await pointerClick('.global-nav [data-view=workbench]');
  await wait("!CSS.highlights.has('chat-passage') && document.querySelector('#passage-navigation').hidden",'leaving Chat clears passage navigation');
  await pointerClick('.global-nav [data-view=chat]');
  assert.equal(await evaluate("document.querySelector('#draft').value"),'Keep the chart for tomorrow.');
}
async function checkPromptOutline() {
  const key = async (key, code) => {
    for (const type of ['keyDown','keyUp']) {
      await cdp('Input.dispatchKeyEvent',{type,key,code:key,windowsVirtualKeyCode:code,text: type === 'keyDown' && key === 'Enter' ? '\r' : ''});
    }
  };
  const menu = '.topbar [data-action=chat-menu]';
  const open = async () => {
    await pointerClick(menu);
    await pointerClick('#action-menu [data-action=outline]');
    await wait("document.querySelector('#find-message-dialog').open && document.activeElement.matches('.find-message-result')", 'outline focused');
  };
  await evaluate("document.querySelectorAll('#messages .message.user')[16].scrollIntoView({block:'center'})");
  await evaluate("document.querySelector("+JSON.stringify(menu)+").focus({preventScroll:true})");
  await key('Enter',13);
  await wait("document.activeElement.dataset.action==='find-message'", 'conversation menu focused its first action');
  await key('ArrowDown',40);
  assert.equal(await evaluate("document.activeElement.dataset.action"),'outline');
  await key('Enter',13);
  await wait("document.querySelector('#find-message-dialog').open", 'open outline using keyboard');
  assert.equal(await evaluate("document.querySelector('#find-message-title').textContent"), 'Conversation outline');
  assert.match(await evaluate("document.activeElement.textContent"), /summary\.md, budget\[1\]\.csv/);
  assert.equal(await evaluate("document.activeElement.getAttribute('aria-current')"), 'location');
  assert.equal(await evaluate("document.querySelectorAll('.find-message-result').length"), 25);
  assert.equal(await evaluate("document.querySelector('#find-message-count').textContent"), 'Turns 1 to 25 of 27');
  assert.doesNotMatch(await evaluate("document.querySelector('#find-message-list').innerText"), /Inactive prompt branch|Private attachment body|Answer \[/);
  await screenshot('conversation-outline-desktop.png');
  await key('End',35);
  assert.match(await evaluate("document.activeElement.textContent"), /Prompt 26$/);
  await key('Home',36);
  assert.match(await evaluate("document.activeElement.textContent"), /Prompt 0$/);
  await pointerClick('#find-message-more');
  assert.match(await evaluate("document.activeElement.textContent"), /Prompt 25$/);
  await key('ArrowUp',38);
  assert.match(await evaluate("document.activeElement.textContent"), /Prompt 24$/);
  await key('ArrowDown',40); await key('Enter',13);
  assert.equal(await evaluate("document.activeElement.matches('article.message.user') && document.activeElement.innerText.includes('Prompt 25')"), true);
  await cdp('Emulation.setDeviceMetricsOverride',{width:320,height:600,deviceScaleFactor:1,mobile:true});
  await wait("document.documentElement.dataset.compact==='true'", 'phone outline');
  await open();
  await key('End',35);
  await pointerClick('#find-message-earlier');
  const theme = await evaluate("document.documentElement.dataset.theme");
  await evaluate("document.documentElement.dataset.theme='light'");
  await screenshot('conversation-outline-phone.png');
  await evaluate("document.documentElement.dataset.theme="+JSON.stringify(theme));
  assert.equal(await evaluate("document.documentElement.scrollWidth<=innerWidth && document.querySelector('#find-message-dialog').getBoundingClientRect().right<=innerWidth && !document.querySelector('#find-message-query').checkVisibility()"),true);
  await key('Escape',27);
  await wait("!document.querySelector('#find-message-dialog').open && document.activeElement.dataset.action==='chat-menu'", 'Escape restores conversation menu trigger');
  assert.equal(await evaluate("document.querySelector('#draft').value==='Unsent draft' && document.querySelectorAll('#attachments .attachment-pill').length===2"), true);
  await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await wait("document.documentElement.dataset.compact==='false'", 'desktop after outline');
}
async function checkLandingDrafts() {
  const homeDraft = 'Prepare a route for the afternoon survey.';
  const textDraft = 'Check the eastern marker before the tide turns.';
  const titles = ['Harbor survey route','Berth measurements','Harbor station notes','Winter archive'];
  const [textChat,fileChat,cleanChat,archivedChat] = await evaluate(`(async()=>{
    const ids=[];
    for (const title of ${JSON.stringify(titles)}) {
      const empty = title==='Berth measurements';
      const chat=await ChatShell.api('/import','POST',{version:1,chat:{title,active_leaf:empty?null:'r'},
        messages:empty?[]:[{id:'r',role:'assistant',content:'Saved station notes for '+title+'.'}]});
      ids.push(chat.id);
    }
    let {revision}=await ChatShell.api('/workspace');
    for (const [path,content] of [['harbor/route.md','Morning route and marker positions.'],['observations/tide.csv','station,depth\\nA2,4.6\\n']]) {
      ({revision}=await ChatShell.api('/workspace/file','POST',{revision,path,content}));
    }
    await ChatShell.api('/workspace/projects','POST',{name:'Harbor fieldwork',paths:['harbor/route.md','observations/tide.csv']});
    return ids;
  })()`);
  assert.equal(await evaluate("document.querySelector('#landing-drafts').checkVisibility()"),false,'new session starts with no draft suggestions');
  const sidebar = async () => {
    if (await evaluate("document.documentElement.dataset.compact==='true'")) {
      await pointerClick('#sidebar-toggle');
      await wait("document.querySelector('#sidebar').getBoundingClientRect().left>=0",'sidebar open');
    }
  };
  const select = async id => {
    await sidebar(); await pointerClick(`.chat-item[data-chat="${id}"]`);
    await wait(`document.querySelector('.chat-item.active')?.dataset.chat==='${id}'`,'selected saved conversation');
  };
  const home = async () => {
    await sidebar(); await pointerClick('#sidebar [data-action=home]');
    await wait("!document.querySelector('.chat-item.active') && document.querySelector('#welcome').checkVisibility() && document.activeElement.id==='draft'",'ordinary new-chat home');
    assert.equal(await evaluate("document.querySelector('#draft').value"),homeDraft);
  };
  await reloadDocument();
  await wait(`document.querySelector('.chat-item[data-chat="${textChat}"]')`,'history for landing drafts');
  await fill('#draft',homeDraft);
  await select(archivedChat); await fill('#draft','Keep these winter measurements private.'); await home();
  await evaluate(`ChatShell.api('/chats/${archivedChat}','PATCH',{archived:true})`);
  await select(textChat); await fill('#draft',textDraft); await home();
  await select(fileChat);
  await upload('#file-input',fixture);
  await wait("document.querySelector('#attachments .attachment-pill')?.innerText.includes('depth-log.csv')",'file-only draft attached');
  assert.equal(await evaluate("document.querySelector('#draft').value"),'');
  await home();
  await reloadDocument();
  await wait(`document.querySelector('#draft').value===${JSON.stringify(homeDraft)} && document.querySelector('.landing-context-row')`,'restored home draft and workspace files');
  for (const [width,height,theme] of [[1440,1000,'light'],[320,800,'dark']]) {
    await cdp('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width===320});
    await cdp('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:theme}]});
    await wait(`innerWidth===${width} && document.documentElement.dataset.theme==='${theme}' && document.documentElement.dataset.compact==='${width===320}'`,'matched home viewport and theme');
    await screenshot(`landing-${width}-${theme}.png`);
    assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true);
  }
  const draftRows = '#landing-draft-list button[data-chat]';
  const ids = () => evaluate(`Array.from(document.querySelectorAll(${JSON.stringify(draftRows)}),row=>row.dataset.chat).sort()`);
  const row = id => `${draftRows}[data-chat="${id}"]`;
  assert.deepEqual(await ids(),[textChat,fileChat].sort(),'real unsent saved chats only, excluding the archived, untouched, and home drafts');
  assert.equal(await evaluate("document.querySelector('#landing-drafts').checkVisibility() && document.querySelector('#recent-conversations').checkVisibility() && document.querySelector('#recent-list button').tabIndex===0"),true,'mobile drafts and recent conversations remain available');
  const textLabel=await evaluate(`document.querySelector(${JSON.stringify(row(textChat))}).innerText`);
  const fileLabel=await evaluate(`document.querySelector(${JSON.stringify(row(fileChat))}).innerText`);
  assert(textLabel.includes(titles[0]) && textLabel.includes(textDraft),'text draft shows its title and preview');
  assert(fileLabel.includes(titles[1]) && fileLabel.includes('depth-log.csv') && fileLabel.includes('1 file'),'attachment-only draft identifies its file');
  const writes = apiMutations.length;
  await evaluate(`document.querySelector(${JSON.stringify(row(textChat))}).focus()`);
  for (const type of ['keyDown','keyUp']) await cdp('Input.dispatchKeyEvent',{type,key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:type==='keyDown'?'\r':''});
  await wait(`document.activeElement.id==='draft' && document.querySelector('#draft').value===${JSON.stringify(textDraft)} && document.querySelector('.chat-item.active')?.dataset.chat==='${textChat}'`,'keyboard resumed saved text draft after reload');
  await home();
  await pointerClick(row(fileChat));
  await wait(`document.activeElement.id==='draft' && document.querySelector('.chat-item.active')?.dataset.chat==='${fileChat}' && document.querySelector('#attachments .attachment-pill')?.innerText.includes('depth-log.csv')`,'attachment-only draft resumed after reload');
  assert.equal(await evaluate("document.querySelector('#draft').value"),'');
  assert.equal(await evaluate("document.querySelector('#welcome').checkVisibility() && document.querySelectorAll('#messages .message').length===0"),true,'resumed empty saved conversation');
  assert.equal(await evaluate(`document.querySelector(${JSON.stringify(row(fileChat))})?.checkVisibility() || false`),false,'current saved chat offered as a resumable draft');
  await home();
  assert.equal(apiMutations.length,writes,'resuming saved drafts wrote to the server');
  await pointerClick(`#recent-list [data-chat="${cleanChat}"]`);
  await wait(`document.activeElement.id==='chat-title' && document.querySelector('.chat-item.active')?.dataset.chat==='${cleanChat}'`,'mobile recent conversation accessible');
  await home();
  await pointerClick('#temporary-button');
  await wait("document.querySelector('#app').dataset.temporary==='true' && !document.querySelector('#temporary-button').disabled",'temporary landing');
  await fill('#draft','Private temporary survey note.');
  assert.equal(await evaluate("document.querySelector('#landing-drafts').checkVisibility()"),false,'temporary mode must hide saved drafts');
  await pointerClick('#temporary-button');
  await wait("document.querySelector('#action-dialog').open",'confirm ending the temporary draft');
  await pointerClick('#conversation-submit');
  await wait("document.querySelector('#app').dataset.temporary!=='true' && !document.querySelector('#temporary-button').disabled",'ordinary home after temporary chat');
  assert.deepEqual(await ids(),[textChat,fileChat].sort(),'temporary draft did not leak into home');
  assert.equal(await evaluate("document.querySelector('#draft').value"),homeDraft);
  await cdp('Network.clearBrowserCookies');
  assert.equal(await evaluate("ChatShell.api('/chats').then(()=>200,error=>error.status)"),401);
  await wait("document.querySelector('#login-dialog').open && !document.querySelector('#landing-drafts').checkVisibility() && !document.querySelector('#landing-draft-list').childElementCount",'clear draft previews when the session locks');
}
async function checkLandingContext() {
  const source = key => '#landing-context [data-source='+key+']';
  const row = label => '.landing-context-row[title='+JSON.stringify(label)+']';
  const shown = selector => wait('document.querySelector('+JSON.stringify(selector)+')?.checkVisibility()', selector);
  const empty = text => wait('document.querySelector("#landing-context-status").innerText==='+JSON.stringify(text), text);
  const actionMenu = '.topbar [data-action=chat-menu]';
  const menuState = () => evaluate(`(() => {
    const button = document.querySelector(${JSON.stringify(actionMenu)});
    const visible = button.checkVisibility({visibilityProperty:true});
    button.focus();
    return {visible, focused:document.activeElement===button};
  })()`);
  await empty('No saved files yet.');
  assert.equal(await evaluate("document.querySelector('#landing-context-heading').textContent"),'Attach from workspace');
  await evaluate("ChatShell.api('/import','POST',{version:1,chat:{title:'Resume this draft',active_leaf:'reply'},messages:[{id:'reply',role:'assistant',content:'Saved reply'}]})");
  await reloadDocument();
  await shown('.chat-item[data-chat]');
  assert.equal(await evaluate("document.querySelector('#recent-list button').textContent.includes('Resume this draft') && !document.querySelector('#recent-conversations').checkVisibility()"), true, 'desktop history is in the sidebar');
  assert.equal(await evaluate("document.querySelector('#chat-title').disabled"), true, 'an unsaved chat has no rename action');
  assert.equal(await evaluate("document.querySelector('#landing-context-browse').checkVisibility() && !document.querySelector('.temporary-saved-files').checkVisibility()"),true,'ordinary landing shows one browse shortcut');
  const blankMenu = await menuState();
  assert.equal(blankMenu.visible, false, 'blank draft offers a conversation menu without a target');
  assert.equal(blankMenu.focused, false, 'keyboard focus entered the unavailable conversation menu');
  await fill('#draft','Review the release notes.');
  await pointerClick(source('projects'));
  await empty('No saved projects yet.');
  assert.equal(await evaluate("document.querySelector('#landing-context-heading').textContent"),'Use a saved project');
  await pointerClick('#landing-context-browse');
  await wait("document.activeElement.id==='workspace-project-name' && document.querySelector('#workspace-project-form').checkVisibility()", 'create a project from home');
  const ids = await evaluate(`(async()=>{
    let {revision}=await ChatShell.api('/workspace');
    for(const [path,content] of [['brief/outline.md','Current saved outline'],['notes/releases.md','Release notes'],['src/theme.css','body {}'],['z.txt','Other file']]) {
      ({revision}=await ChatShell.api('/workspace/file','POST',{path,content,revision}));
    }
    const ids=[];
    for (const [name,paths] of [['Documentation',['notes/releases.md']],['Release review',['brief/outline.md','notes/releases.md']]]) {
      const {project}=await ChatShell.api('/workspace/projects','POST',{name,paths,instructions:'Read the release notes.'});ids.push(project.id);
    }
    return ids;
  })()`);
  await pointerClick('.workspace-back');
  await shown(row('Release review'));
  assert.equal(await evaluate("document.activeElement.id"), 'draft', 'Back to Chat should resume the unsent draft');
  await cdp('Input.insertText', {text:' Next step.'});
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Review the release notes. Next step.');
  await fill('#draft', 'Review the release notes.');
  await screenshot('home-context-1440.png');
  const writes = apiMutations.length, streams = runStreams.length;
  const contextReads = () => browserRequests.filter(r=>new URL(r.url).pathname.endsWith('/context')).length;
  assert.equal(contextReads(),0,'home fetched project file bodies before selection');
  await pointerClick(row('Release review'));
  await wait("document.querySelector('#project-context-files').children.length===2",'review named project');
  assert.equal(await evaluate("document.querySelector('#project-context-select').value"),ids[1]);
  assert.equal(await evaluate("document.querySelector('#project-context-copy').checked"),false);
  await pointerClick('#project-context-cancel');
  assert.equal(await evaluate("document.querySelectorAll('#attachments .attachment-pill').length"),0);
  await pointerClick(source('files'));
  await shown(row('brief/outline.md'));
  assert.equal(await evaluate("document.querySelectorAll('.landing-context-row').length"),3,'home file list is bounded');
  const reads = savedFileReads.filter(paths=>paths.length).length;
  await pointerClick(row('brief/outline.md'));
  await wait("document.querySelector('#saved-file-dialog input:checked')?.value==='brief/outline.md'", 'named file selected');
  assert.equal(savedFileReads.filter(paths=>paths.length).length,reads,'preselection read a file body');
  await pointerClick('#saved-file-add');
  await wait("document.querySelectorAll('#attachments .attachment-pill').length===1 && document.activeElement.id==='draft'", 'confirmed selected file in draft');
  await pointerClick(source('projects')); await shown(row('Release review')); await pointerClick(row('Release review'));
  await wait("!document.querySelector('#project-context-apply').disabled", 'project review ready');
  await pointerClick('#project-context-apply');
  await wait("document.querySelectorAll('#attachments .attachment-pill').length===2", 'deduplicate file and reviewed project');
  assert.equal(await evaluate("document.querySelector('#draft').value"),'Review the release notes.');
  assert.equal(apiMutations.length,writes,'home selections changed saved state');
  assert.equal(runStreams.length,streams,'home selections submitted the draft');
  await cdp('Emulation.setDeviceMetricsOverride',{width:320,height:700,deviceScaleFactor:1,mobile:true});
  const filesTab = source('files');
  await wait(`(async () => {
    if (innerWidth !== 320 || document.documentElement.dataset.compact !== 'true') return false;
    const tab = document.querySelector(${JSON.stringify(filesTab)});
    const before = tab.getBoundingClientRect();
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const after = tab.getBoundingClientRect();
    return before.x === after.x && before.y === after.y && before.width === after.width && before.height === after.height;
  })()`, 'Files tab settled after the phone layout and draft resize');
  await pointerClick(filesTab);
  assert.equal(await evaluate(`document.querySelector(${JSON.stringify(filesTab)}).getAttribute('aria-pressed')`),'true','phone pointer selected Files');
  await shown(row('brief/outline.md'));
  assert.equal(await evaluate("document.querySelector('#landing-context-heading').textContent"),'Attach from workspace');
  await shown('.recent-chat');
  assert.match(await evaluate("document.querySelector('.recent-chat').textContent"), /Resume this draft/);
  await screenshot('home-context-320.png');
  assert.equal(await evaluate("document.documentElement.scrollWidth<=innerWidth && document.querySelector('.landing-context-row').getBoundingClientRect().right<=innerWidth"),true);
  await evaluate("ChatShell.api('/workspace').then(s=>ChatShell.api('/workspace/delete','POST',{path:'brief/outline.md',revision:s.revision}))");
  await pointerClick(row('brief/outline.md'));
  await wait("document.querySelector('#saved-file-error').checkVisibility() && !document.querySelector('#saved-file-dialog input:checked') && document.querySelector('#saved-file-add').disabled", 'deleted file cannot select another file');
  await pointerClick('#saved-file-cancel');
  await evaluate("(()=>{const original=window.fetch;window.releaseProjectList=null;window.fetch=async(...args)=>{const response=await original(...args);if(args[0]!=='/api/workspace/projects')return response;window.fetch=original;await new Promise(resolve=>window.releaseProjectList=resolve);return response;};})()");
  await pointerClick(source('projects'));
  await wait("typeof releaseProjectList==='function'", 'pending project list');
  await pointerClick(source('files')); await shown(row('notes/releases.md'));
  await evaluate("releaseProjectList()");
  await wait("document.querySelector('#landing-context-browse').dataset.action==='attach-files' && !Array.from(document.querySelectorAll('.landing-context-row')).some(row=>row.title==='Release review')", 'stale project list kept out of files');
  await pointerClick('#temporary-button');
  await wait("document.querySelector('#app').dataset.temporary==='true' && !document.querySelector('.landing-context-row') && !document.querySelector('#landing-context').checkVisibility()", 'temporary chat isolates landing workspace metadata');
  const temporaryMenu = await menuState();
  assert.equal(temporaryMenu.visible && temporaryMenu.focused, true, 'temporary conversation actions unavailable');
  await pointerClick(actionMenu);
  await shown('#action-menu [data-action=export]');
  await cdp('Input.dispatchKeyEvent', {type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  const requests = () => browserRequests.filter(r=>/^\/api\/workspace\/(?:files|projects)$/.test(new URL(r.url).pathname)).length;
  const beforeReload = requests();
  await reloadDocument();
  await wait("document.querySelector('#app').dataset.temporary==='true' && !document.querySelector('#temporary-button').disabled", 'temporary chat restored');
  assert.equal(requests(), beforeReload, 'restoring a temporary chat fetched landing metadata');
  await shown('.temporary-saved-files');
  await pointerClick('.temporary-saved-files');
  await wait("document.querySelector('#saved-file-dialog').open && document.querySelector('#saved-file-list input')?.checkVisibility()", 'direct saved file browser in temporary chat');
  assert.equal(requests(),beforeReload+1,'temporary picker reads metadata only on request');
  await pointerClick('#saved-file-cancel');
  assert.equal(await evaluate("document.querySelector('#app').dataset.temporary==='true' && !document.querySelector('#attachments .attachment-pill')"),true);
}
async function checkWorkspaceImport() {
  const dialog = '#workspace-import-dialog', list = '#workspace-import-list', submit = '#workspace-import-submit';
  const fileInput = '#workspace-local-files', folderInput = '#workspace-local-folder';
  const q = selector => 'document.querySelector(' + JSON.stringify(selector) + ')';
  const open = async () => {
    await pointerClick('#workspace-import');
    await wait(q(dialog) + '.open', 'open workspace import review');
  };
  const close = () => wait('!' + q(dialog) + '.open', 'workspace import closed');
  const choose = (...names) => uploadMany(fileInput, names.map(name => path.join(fixture, name)));
  const batchPosts = () => apiMutations.filter(request => request.method === 'POST' && new URL(request.url).pathname === '/api/workspace/files').length;
  const write = (path, content) => evaluate("ChatShell.api('/workspace').then(({revision}) => ChatShell.api('/workspace/file','POST',"
    + "Object.assign({revision}," + JSON.stringify({path, content}) + ")))");
  const read = path => evaluate("ChatShell.api(" + JSON.stringify('/workspace/file?path=' + encodeURIComponent(path)) + ").then(f=>f.content)");
  const rows = () => evaluate("Array.from(document.querySelectorAll('#workspace-import-list input'), box => [box.value,box.checked])");
  const input = name => list + ' input[value=' + JSON.stringify(name) + ']';
  const reads = () => evaluate('window.workspaceImportReads');
  await fill('#draft', 'Keep this chat draft');
  await write('active.md', 'original');
  await pointerClick('.global-nav [data-view=workbench]');
  await wait(q('.workspace-file[title="active.md"]'), 'initial saved file');
  await pointerClick('.workspace-actions .primary-button');
  await wait(q('#action-dialog') + '.open', 'new file name');
  await fill('#conversation-value', 'pending.txt');
  await closeAction('#conversation-submit');
  await wait(q('.workspace-file-path') + ".textContent === 'pending.txt'", 'unsaved new path');
  await fill('#workspace-editor', 'Editor-only text');
  await evaluate("window.workspaceImportReads = []; const readBuffer = File.prototype.arrayBuffer; File.prototype.arrayBuffer = function() { workspaceImportReads.push(this.name); return readBuffer.call(this); }");
  await open();
  assert.equal(await evaluate(q(submit) + '.disabled'), true);
  await upload(folderInput, path.join(fixture, 'Excluded'));
  await wait(q('#workspace-import-note') + ".textContent.includes('No files available')", 'excluded-only folder');
  assert.deepEqual(await rows(), []);
  assert.equal(await evaluate(q(submit) + '.disabled'), true);
  await choose('pending.txt');
  await wait(q('#workspace-import-summary') + ".textContent.includes('unsaved editor')", 'reserved unsaved editor path');
  assert.equal(await evaluate(q(submit) + '.disabled'), true);
  await choose('large.txt');
  await wait(q('#workspace-import-summary') + ".textContent.includes('file limit')", 'workspace byte limit');
  assert.deepEqual(await reads(), [], 'blocked selections were read');
  await choose('a.txt', 'z-invalid.txt');
  await wait(q(submit) + ".textContent === 'Import 2 files'", 'two reviewed files');
  assert.deepEqual(await rows(), [['a.txt',true], ['z-invalid.txt',true]]);
  assert.deepEqual(await reads(), [], 'validity was checked before confirmation');
  await pointerClick(submit);
  await wait(q('#workspace-import-error') + ".textContent.includes('z-invalid.txt is not valid UTF-8')", 'invalid UTF-8 refused');
  assert.deepEqual(await reads(), ['a.txt','z-invalid.txt']);
  assert.equal(batchPosts(), 0, 'partial batch submitted before an invalid file');
  await choose('binary.txt');
  await wait(q('#workspace-import-error') + '.hidden', 'reselection cleared previous error');
  await pointerClick(submit);
  await wait(q('#workspace-import-error') + ".textContent.includes('NUL')", 'binary content refused');
  assert.equal(batchPosts(), 0);

  await choose('empty.txt', 'a.txt');
  await wait(q(submit) + ".textContent === 'Import 2 files'", 'empty and nonempty UTF-8 files');
  await pointerClick(submit);
  await close();
  assert.equal(batchPosts(), 1);
  assert.equal(await read('empty.txt'), '');
  assert.equal(await read('a.txt'), 'ordinary\r\n');
  assert.deepEqual(await evaluate('[' + q('.workspace-file-path') + '.textContent,' + q('#workspace-editor') + '.value]'), ['pending.txt','Editor-only text']);
  await pointerClick('#workspace-save');
  await wait(q('.workspace-file[title="pending.txt"]') + ' && ' + q('#workspace-save') + '.disabled', 'unsaved editor still saves at the new revision');
  assert.equal(await read('pending.txt'), 'Editor-only text');
  await pointerClick('.workspace-file[title="active.md"]');
  await wait(q('#workspace-editor') + ".value === 'original'", 'open saved editor');
  await fill('#workspace-editor', 'local amended text');
  const changed = await write('active.md', 'external edit');
  await pointerClick('.workspace-actions button:first-of-type');
  await wait(q('.workspace-revision') + '.textContent === ' + JSON.stringify('Revision ' + changed.revision), 'fresh workspace snapshot with stale editor');
  await open();
  await choose('a.txt');
  await wait(q('#workspace-import-summary') + ".textContent.includes('already exists')", 'saved path collision');
  assert.equal(await evaluate(q(submit) + '.disabled'), true);
  await upload(folderInput, path.join(fixture, 'Field project'));
  await wait(q(input('docs/notes.md')), 'native folder selection');
  assert.deepEqual(await rows(), [['docs/notes.md',false], ['omit.txt',false], ['src/helper.txt',false]], 'hidden and dependency files in review');
  assert.equal(await evaluate(q(submit) + '.disabled'), true, 'an entire folder selected by default');
  const previousReads = await reads();
  await fill('#workspace-import-filter','docs');
  await pointerClick(input('docs/notes.md'));
  await fill('#workspace-import-filter','');
  await pointerClick(input('src/helper.txt'));
  assert.deepEqual(await rows(), [['docs/notes.md',true], ['omit.txt',false], ['src/helper.txt',true]]);
  assert.deepEqual(await reads(), previousReads, 'folder read during review');
  await screenshot('review-1440.png');
  await cdp('Emulation.setDeviceMetricsOverride', {width:320,height:700,deviceScaleFactor:1,mobile:false});
  await wait("innerWidth===320 && document.documentElement.dataset.compact==='true'", 'compact workspace layout');
  const bounds = await evaluate("['#workspace-import-dialog','#workspace-import-list','#workspace-import-submit','#workspace-import-cancel','#workspace-import'].map(s=>{const {left,right,top,bottom}=document.querySelector(s).getBoundingClientRect();return [left,right,top,bottom]})");
  assert(bounds.every(([left,right,top,bottom]) => left >= 0 && right <= 320 && top >= 0 && bottom <= 700), 'clipped controls at 320px: ' + JSON.stringify(bounds));
  await screenshot('review-320.png');
  await write('concurrent.txt', 'new revision');
  await pointerClick(submit);
  await wait('!' + q('#workspace-import-refresh') + '.hidden', 'stale review can be refreshed');
  assert.equal(batchPosts(), 2);
  assert.equal(await evaluate("ChatShell.api('/workspace').then(s=>s.files.some(f=>f.path==='docs/notes.md'))"), false);
  await pointerClick('#workspace-import-refresh');
  await wait('!' + q(submit) + '.disabled', 'review refreshed for explicit retry');
  assert.deepEqual(await rows(), [['docs/notes.md',true], ['omit.txt',false], ['src/helper.txt',true]]);
  await pointerClick(submit);
  await close();
  assert.equal(batchPosts(), 3);
  assert.equal(await read('docs/notes.md'), '\ufeffH\u00e4llo\r\nmiddle\rlast\n');
  assert.equal(await read('src/helper.txt'), 'helper');
  const names = await evaluate("ChatShell.api('/workspace').then(s=>s.files.map(f=>f.path).sort())");
  assert.deepEqual(names, ['a.txt','active.md','concurrent.txt','docs/notes.md','empty.txt','pending.txt','src/helper.txt']);
  assert.deepEqual((await reads()).slice(previousReads.length), ['notes.md','helper.txt','notes.md','helper.txt']);
  assert.equal(await evaluate(q('#workspace-editor') + '.value'), 'local amended text');
  await pointerClick('#workspace-save');
  await wait(q('.workspace-error') + ".textContent.includes('Workspace changed')", 'stale dirty editor refused');
  assert.equal(await read('active.md'), 'external edit');
  assert.equal(await evaluate(q('#workspace-editor') + '.value'), 'local amended text');
  await open();
  assert.deepEqual(await rows(), [], 'previous file objects retained on reopening review');
  await choose('Field project/omit.txt');
  await wait(q(submit) + ".textContent === 'Import 1 file'", 'selected file before cancelling');
  await evaluate("const unpausedRead = File.prototype.arrayBuffer; File.prototype.arrayBuffer = async function() { File.prototype.arrayBuffer=unpausedRead; await new Promise(resolve=>window.resumeImportRead=resolve); const data=await unpausedRead.call(this); setTimeout(()=>{window.importReadFinished=true},0); return data; }");
  await pointerClick(submit);
  await wait("typeof window.resumeImportRead === 'function'", 'file read in progress');
  await pointerClick('#workspace-import-cancel');
  await close();
  await open();
  await evaluate('window.resumeImportRead()');
  await wait('window.importReadFinished', 'cancelled file read completed');
  assert.equal(batchPosts(), 3, 'cancelled read sent another import');
  assert.equal(await evaluate(q('#workspace-import-error') + '.hidden && ' + q(submit) + '.disabled'), true, 'cancelled read changed the next review');
  await cdp('Input.dispatchKeyEvent', {type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
  await close();
  assert.equal(await evaluate('document.activeElement.id'), 'workspace-import');
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await wait("innerWidth===1440 && document.documentElement.dataset.compact==='false'", 'desktop workspace layout');
  await pointerClick('.workspace-back');
  await wait(q('#draft') + ".value === 'Keep this chat draft'", 'preserved unsent conversation draft');
}

async function checkHomeLanding() {
  const selected = await evaluate("document.querySelector('.chat-item.active')?.dataset.chat");
  assert(selected, 'home landing check requires a saved conversation');
  const draft = await evaluate("document.querySelector('#draft').value");
  const history = () => evaluate("Array.from(document.querySelectorAll('#chat-list .chat-item')).map(chat=>chat.dataset.chat).sort()");
  const creates = () => apiMutations.filter(request=>request.method==='POST' && new URL(request.url).pathname==='/api/chats').length;
  const ids = await history();
  const originalCreates = creates();
  for (const view of ['chat', 'workbench', 'preferences', 'monitor']) {
    const marker = 'Unsent home navigation draft from ' + view;
    await fill('#draft', marker);
    if (view !== 'chat') await pointerClick(`.global-nav [data-view="${view}"]`);
    await wait(`document.querySelector('#app').dataset.view===${JSON.stringify(view)}`, view + ' before logo navigation');
    assert.equal(await evaluate("(() => {const home=document.querySelector('#chat-home');home?.focus();return home?.tagName==='BUTTON' && home.tabIndex>=0 && document.activeElement===home;})()"), true, 'logo is not keyboard focusable in ' + view);
    await cdp('Input.dispatchKeyEvent', {type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r'});
    await cdp('Input.dispatchKeyEvent', {type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
    await wait("document.querySelector('#app').dataset.view==='chat' && !document.querySelector('#welcome').hidden && !document.querySelector('.message') && !document.querySelector('.chat-item.active') && document.activeElement.id==='draft'", 'fresh Chat landing from ' + view);
    assert.equal(await evaluate("document.querySelector('#jump-bottom').hidden"), true, 'empty Chat kept the transcript jump control');
    assert.deepEqual(await history(), ids, 'home changed the history from ' + view);
    assert.equal(creates(), originalCreates, 'home created an empty saved conversation from ' + view);
    const recent = await evaluate("fetch('/api/chats').then(r=>r.json()).then(({chats})=>chats.filter(c=>!c.archived && c.active_leaf).sort((a,b)=>b.updated-a.updated).slice(0,3).map(c=>c.id))");
    assert.deepEqual(await evaluate("Array.from(document.querySelectorAll('#recent-list button'), b=>b.dataset.chat)"), recent, 'landing lists recent populated chats by update time');
    if (view === 'chat') {
      await pointerClick('#landing-context [data-action=attach-files]');
      await wait("document.querySelector('#saved-file-dialog').open", 'open saved files from the landing');
      await pointerClick('#saved-file-cancel');
    }
    const fromRecent = recent.includes(selected) && await evaluate("document.querySelector('#recent-conversations').checkVisibility()");
    await pointerClick((fromRecent ? '.recent-chat' : '.chat-item')+'[data-chat="'+selected+'"]');
    await wait(`document.querySelector('.chat-item.active')?.dataset.chat===${JSON.stringify(selected)} && document.querySelector('#draft').value===${JSON.stringify(marker)} && document.querySelector('.message.assistant')`, 'restore saved transcript and draft after leaving ' + view);
    if (fromRecent) assert.equal(await evaluate("document.activeElement.id"), 'chat-title', 'resuming from the landing focuses the conversation');
  }
  await fill('#draft', draft);
}
async function selectWorkspaceSection(section) {
  const selector = `#workspace-sections [data-section="${section}"]`;
  await pointerClick(selector);
  await wait(`document.querySelector(${JSON.stringify(selector)}).getAttribute('aria-pressed')==='true'`, 'Workspace ' + section + ' section');
  assert.equal(await evaluate("document.querySelectorAll('#workspace-sections [aria-pressed=true]').length"), 1, 'Workspace selected more than one section');
}
async function workspaceScreenshots(name) {
  await cdp('Input.dispatchMouseEvent', {type:'mouseMoved', x:1, y:1});
  await screenshot(`workflow-${name}-1440x1000.png`);
  await cdp('Emulation.setDeviceMetricsOverride', {width:320,height:800,deviceScaleFactor:1,mobile:true});
  await wait("document.documentElement.dataset.compact==='true' && innerWidth===320", 'narrow Workspace screenshot');
  await screenshot(`workflow-${name}-320x800.png`);
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await wait("document.documentElement.dataset.compact==='false'", 'return to desktop after Workspace screenshot');
}
async function checkEmptyWorkspace(capture = false) {
  await wait("document.querySelector('#workspace-start')?.checkVisibility() && document.querySelector('.workspace-actions .primary-button:enabled')", 'loaded Workspace empty guidance');
  const guidance = await evaluate("document.querySelector('#workspace-start').textContent");
  assert.match(guidance, /files/i);
  assert.match(guidance, /saved|project/i);
  assert.equal(await evaluate("document.querySelector('#workspace-sections [data-section=files]').getAttribute('aria-pressed')"), 'true');
  if (capture) await workspaceScreenshots('empty');
  await pointerClick('.workspace-actions .primary-button');
  await wait("document.querySelector('#action-dialog').open && document.activeElement.id==='conversation-value'", 'new file from empty Workspace');
}
async function checkWorkspaceFlow() {
  await pointerClick('.global-nav [data-view="workbench"]');
  await checkEmptyWorkspace(true);
  const filename = 'notes/first-workspace-file.html';
  await fill('#conversation-value', filename);
  await closeAction('#conversation-submit');
  await wait("document.querySelector('#workspace-editor').checkVisibility() && !document.querySelector('#workspace-editor').disabled", 'first file editor');
  const source = '<h1>Browser Workspace file</h1>';
  await fill('#workspace-editor', source);
  await pointerClick('#workspace-save');
  await wait("document.querySelector('.workspace-file[title=\"notes/first-workspace-file.html\"]')?.checkVisibility() && document.querySelector('#workspace-save').disabled", 'saved first file in Workspace');
  assert.equal(await evaluate("document.querySelector('#workspace-start').checkVisibility()"), false, 'empty Workspace persists after creating a file');
  await workspaceScreenshots('editor');
  await pointerClick('.workspace-editor-actions button:first-child');
  await wait("document.querySelector('#artifact-dialog').open && document.querySelector('.artifact-frame')", 'preview first Workspace file');
  await click('#artifact-source-tab');
  assert.equal(await evaluate("document.querySelector('.artifact-source code').textContent"), source);
  await pointerClick('#artifact-dialog [aria-label="Close artifact"]');
  const panels = ['#workspace-editor', '#workspace-projects', '.workspace-terminal', '.workspace-checkpoints'];
  for (const [index, section] of ['projects', 'commands', 'history'].entries()) {
    await selectWorkspaceSection(section);
    const visible = panels.map((_, panel) => panel === index + 1);
    assert.deepEqual(await evaluate(`${JSON.stringify(panels)}.map(selector=>document.querySelector(selector).checkVisibility())`), visible, 'Workspace utilities are visible outside their section');
    if (section === 'commands') await workspaceScreenshots('commands');
  }
  await selectWorkspaceSection('files');
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), source);
  const beforeAttach = apiMutations.length;
  await pointerClick('#workspace-attach');
  await wait("document.querySelector('#app').dataset.view==='chat' && document.querySelector('[aria-label=\"Remove notes:first-workspace-file.html\"]')", 'attach first Workspace file to Chat');
  assert.equal(apiMutations.length, beforeAttach, 'attaching from Workspace wrote to the server');
  await click('[aria-label="Remove notes:first-workspace-file.html"]');
  const siblingName = 'examples/first-workspace-file.html';
  const siblingText = '<h1>Second saved copy \u00e9</h1>\n';
  await evaluate(`(async () => {
    const {revision}=await ChatShell.api('/workspace');
    await ChatShell.api('/workspace/file','POST',{revision,path:${JSON.stringify(siblingName)},content:${JSON.stringify(siblingText)}});
  })()`);
  await cdp('Emulation.setDeviceMetricsOverride', {width:320,height:800,deviceScaleFactor:1,mobile:true});
  await wait("document.documentElement.dataset.compact==='true' && innerWidth===320", 'narrow Chat Attach menu');
  await evaluate("document.querySelector('[data-action=attach]').focus()");
  for (const [key, keyCode, focus] of [['Enter',13,'attach-device'], ['ArrowDown',40,'attach-files'], ['Escape',27,'attach']]) {
    const action = {key,code:key,windowsVirtualKeyCode:keyCode};
    await cdp('Input.dispatchKeyEvent', {type:'keyDown', ...action, ...(key==='Enter' ? {text:'\r'} : {})});
    await cdp('Input.dispatchKeyEvent', {type:'keyUp', ...action});
    await wait(`document.activeElement.dataset.action===${JSON.stringify(focus)}`, 'Attach menu focus after ' + key);
  }
  assert.equal(await evaluate("document.querySelector('#action-menu').matches(':popover-open')"), false, 'Escape did not close Attach menu');
  await checkSavedFilesFromChat([[filename,source],[siblingName,siblingText]]);
  await attachmentSource('project');
  await wait("document.querySelector('#project-context-dialog').open", 'saved project entry from the Attach menu');
  await pointerClick('#project-context-cancel');
  await wait("!document.querySelector('#project-context-dialog').open && document.querySelector('#app').dataset.view==='chat'", 'close project picker in the same Chat');
}
async function checkOptionalFeatures() {
  const workerFixture = process.argv.find(arg => arg.startsWith('--fixture-upstream=')).slice('--fixture-upstream='.length);
  const featureCalls = () => browserRequests.filter(request => /^\/api\/features(?:\/|$)/.test(new URL(request.url).pathname));
  const row = '#preferences-features [data-feature="pdf-reader"]';
  const action = row + ' .optional-feature-action';
  const hasPdf = "document.querySelector('#file-input').accept.split(',').includes('.pdf')";
  const state = name => `document.querySelector(${JSON.stringify(row)})?.dataset.state===${JSON.stringify(name)}`;
  async function view(name) {
    await pointerClick('.mobile-nav');
    await evaluate("Promise.all(document.querySelector('.sidebar').getAnimations().map(animation=>animation.finished.catch(()=>{})))");
    await pointerClick(`.global-nav [data-view="${name}"]`);
    await wait(`document.querySelector('#app').dataset.view===${JSON.stringify(name)}`, name + ' view');
  }
  async function releaseInstall() {
    assert.equal((await fetch(workerFixture + '/browser-step', {signal:AbortSignal.timeout(4000)})).status, 204);
  }
  assert.deepEqual(featureCalls(), [], 'feature catalog requested during ordinary boot');
  assert.equal(await evaluate(hasPdf), false, 'PDF is offered before installation');
  await send('Browser plain-text prompt');
  await wait("document.querySelector('#messages').textContent.includes('Plain browser reply.')", 'plain reply');
  await idle();
  const draft = 'Preserve this unsent draft.';
  await fill('#draft', draft);
  await upload('#file-input', fixture);
  await wait("document.querySelector('#attachments').textContent.includes('notes.txt')", 'ordinary text attachment');
  assert.deepEqual(featureCalls(), [], 'feature catalog requested by plain chat');
  assert.deepEqual(pdfAssetRequests(), [], 'PDF resources requested by plain chat');

  const document = await documentIdentity();
  await cdp('Emulation.setDeviceMetricsOverride', {width:320,height:740,deviceScaleFactor:1,mobile:true});
  await wait("document.documentElement.dataset.compact==='true'", 'narrow Optional features viewport');
  await view('preferences');
  await wait(state('available'), 'available PDF feature');
  assert(featureCalls().some(request => request.method === 'GET' && new URL(request.url).pathname === '/api/features'));
  assert.deepEqual(featureCalls().filter(request => request.method !== 'GET'), [], 'opening Settings started installation');
  assert.deepEqual(pdfAssetRequests(), [], 'opening Settings requested PDF resources');
  assert.equal(await evaluate(`document.querySelector(${JSON.stringify(action)}).textContent.trim()`), 'Install');
  assert.equal(await evaluate(`(() => {
    const panel=document.querySelector('#view-preferences'), item=document.querySelector(${JSON.stringify(row)});
    const bounds=item.getBoundingClientRect();
    return document.documentElement.scrollWidth<=innerWidth && panel.scrollWidth<=panel.clientWidth
      && bounds.left>=0 && bounds.right<=innerWidth;
  })()`), true, 'Optional features overflows at 320 pixels');

  await pointerClick(action);
  await wait(state('installing'), 'first PDF install running');
  assert.equal(await evaluate(`document.querySelector(${JSON.stringify(action)}).getAttribute('aria-disabled')`), 'true');
  await releaseInstall();
  await wait(state('failed'), 'retryable PDF install failure');
  const error = await evaluate(`document.querySelector(${JSON.stringify(row + ' .optional-feature-error')}).textContent.trim()`);
  assert.match(error, /Retry/i, 'the failure did not explain how to retry');
  assert.doesNotMatch(error, /fixture download failure/, 'installer exception leaked into Settings');
  assert.equal(await evaluate(`document.querySelector(${JSON.stringify(action)}).textContent.trim()`), 'Retry');

  await pointerClick(action);
  await wait(state('installing'), 'retried PDF install running');
  const duplicate = await evaluate("ChatShell.api('/features/pdf-reader/install','POST',{})");
  assert.equal(duplicate.features.find(feature => feature.id === 'pdf-reader').state, 'installing');
  await view('chat');
  assert.equal(await evaluate(hasPdf), false, 'PDF chooser enabled before installation completed');
  await releaseInstall();
  await wait(hasPdf, 'live PDF chooser upgrade in Chat');
  assert.deepEqual(await documentIdentity(), document, 'PDF chooser upgrade required reloading the page');
  assert.equal(await evaluate("document.querySelector('#draft').value"), draft);
  assert.deepEqual(await evaluate("Array.from(document.querySelectorAll('#attachments .attachment-pill')).map(pill=>pill.firstChild.textContent)"), ['notes.txt']);
  await cdp('Page.setInterceptFileChooserDialog', {enabled:true});
  await attachmentSource('device');
  await wait(() => attachmentChooser !== null, 'live native file chooser after installation');
  await cdp('Page.setInterceptFileChooserDialog', {enabled:false});
  const {node} = await cdp('DOM.describeNode', {backendNodeId:attachmentChooser.backendNodeId});
  assert(node.attributes[node.attributes.indexOf('accept') + 1].split(',').includes('.pdf'));

  const visits = featureCalls().filter(request => request.method === 'GET').length;
  await view('preferences');
  await wait(state('installed'), 'installed feature on return to Settings');
  await wait(() => featureCalls().filter(request => request.method === 'GET').length > visits, 'settings catalog refresh on return');
  assert.equal(await evaluate(`(() => {
    const button=document.querySelector(${JSON.stringify(action)});
    return !button || button.disabled || !button.checkVisibility();
  })()`), true, 'installed feature still has an active install button');
  assert.deepEqual(pdfAssetRequests(), [], 'installing in Settings requested PDF browser assets');
  assert.deepEqual(browserRequests.filter(request => new URL(request.url).origin !== new URL(url).origin), [], 'Optional features made off-origin browser requests');
  assert.deepEqual(errors, [], 'uncaught Optional features browser exceptions');
  console.log('PASS: Optional features lazy catalog, 320 px Settings, install failure and retry, duplicate admission, live PDF chooser and preserved draft');
}
async function checkPdfAttachment() {
  const origin = new URL(url).origin;
  const installed = pdfMode === 'installed';
  assert.equal(await evaluate('ChatShell.config.pdf_reader'), installed);
  assert.equal(await evaluate("document.querySelector('#file-input').accept.split(',').includes('.pdf')"), installed);
  assert.deepEqual(pdfAssetRequests(), [], 'PDF resources were requested during ordinary boot');
  await send('Browser plain-text prompt');
  await wait("document.querySelector('#messages').textContent.includes('Plain browser reply.')", 'plain reply');
  await idle();
  assert.deepEqual(pdfAssetRequests(), [], 'PDF resources were requested for a plain-text chat');

  const draft = 'Keep this PDF draft.';
  await fill('#draft', draft);
  const mutations = apiMutations.length;
  if (!installed) {
    await upload('#file-input', fixture);
    await wait("/PDF import is unavailable/i.test(document.querySelector('#toast').textContent)", 'unavailable-reader guidance');
    assert.deepEqual(pdfAssetRequests(), [], 'an unavailable PDF reader was requested');
  } else {
    const pdfName = 'quarterly.pdf';
    const expected = 'Page 2\nOnly the second page belongs in chat.';
    const htmlFixture = process.argv.find(arg => arg.startsWith('--html-fixture=')).slice('--html-fixture='.length);
    const htmlSource = await readFile(htmlFixture, 'utf8');
    const open = async () => {
      await upload('#file-input', fixture);
      await wait("document.querySelector('#pdf-attachment-dialog')?.open && !document.querySelector('#pdf-attachment-pages').disabled", 'ready PDF selection');
    };
    const closed = async () => wait("!document.querySelector('#pdf-attachment-dialog')", 'closed PDF selection');
    const review = async (pages) => {
      await fill('#pdf-attachment-pages', pages);
      await click('#pdf-attachment-review');
    };
    const selectedPage = async () => {
      await review('2');
      await wait("!document.querySelector('#pdf-attachment-apply').disabled", 'selected page preview');
      assert.equal(await evaluate("document.querySelector('#pdf-attachment-preview').readOnly"), true);
      assert.equal(await evaluate("document.querySelector('#pdf-attachment-preview').value"), expected);
    };
    await cdp('Emulation.setDeviceMetricsOverride', {width:320,height:740,deviceScaleFactor:1,mobile:true});
    await wait("document.documentElement.dataset.compact==='true'", 'narrow PDF viewport');
    await open();
    await selectedPage();
    assert.equal(await evaluate(`(() => {
      const dialog = document.querySelector('#pdf-attachment-dialog');
      const box = dialog.getBoundingClientRect();
      return box.left >= 0 && box.right <= innerWidth && box.top >= 0 && box.bottom <= innerHeight
        && dialog.scrollWidth <= dialog.clientWidth && dialog.contains(document.activeElement);
    })()`), true, 'PDF preview overflowed the narrow viewport or lost modal focus');
    await screenshot('pdf-review-narrow.png');
    await evaluate(`(() => {
      const postMessage = Worker.prototype.postMessage;
      const abortTimeout = AbortSignal.timeout;
      const gate = window.pdfTerminationGate = {message:null, selectedFile:null, heldTimeout:false};
      let fallback;
      gate.release = () => {
        Worker.prototype.postMessage = postMessage;
        AbortSignal.timeout = abortTimeout;
        clearTimeout(fallback);
        gate.message?.();
        gate.message = null;
      };
      document.querySelector('#file-input').addEventListener('change', event => {
        gate.selectedFile = event.target.files[0]?.name;
      }, {capture:true,once:true});
      // Extend the teardown watchdog for this cancellation so the browser can reopen the same file while the first worker is still live.
      AbortSignal.timeout = milliseconds => {
        if (milliseconds !== 250) return abortTimeout.call(AbortSignal, milliseconds);
        AbortSignal.timeout = abortTimeout;
        gate.heldTimeout = true;
        return abortTimeout.call(AbortSignal, 8000);
      };
      Worker.prototype.postMessage = function(message, ...args) {
        if (message?.action !== 'Terminate' || message.data !== null || gate.message) return postMessage.call(this, message, ...args);
        gate.message = () => postMessage.call(this, message, ...args);
      };
      fallback = setTimeout(gate.release, 6000);
    })()`);
    try {
      await cdp('Input.dispatchKeyEvent', {type:'keyDown', key:'Escape', code:'Escape', windowsVirtualKeyCode:27});
      await cdp('Input.dispatchKeyEvent', {type:'keyUp', key:'Escape', code:'Escape', windowsVirtualKeyCode:27});
      await closed();
      await wait('!!pdfTerminationGate.message', 'held first PDF worker termination');
      assert.equal(await evaluate('pdfTerminationGate.heldTimeout'), true, 'first worker cleanup deadline was not held');
      assert.equal(await evaluate("document.querySelector('#draft').value"), draft, 'cancelling PDF preview changed the draft');
      assert.equal(await evaluate("document.querySelectorAll('#attachments .attachment-pill').length"), 0, 'cancelling PDF preview added an attachment');

      await upload('#file-input', fixture);
      assert.equal(await evaluate('pdfTerminationGate.selectedFile'), 'quarterly.pdf', 'same-file selection did not reach the file input');
      await wait("document.querySelector('#pdf-attachment-dialog')?.open || document.querySelector('#toast').textContent.includes('Finish the open PDF selection first')", 'same PDF reopened during teardown', 4000);
      assert.equal(await evaluate("!!document.querySelector('#pdf-attachment-dialog')?.open"), true,
        'same PDF could not reopen during teardown: ' + await evaluate("document.querySelector('#toast').textContent"));
      await wait("!document.querySelector('#pdf-attachment-pages').disabled", 'same PDF ready during first worker teardown', 4000);
      assert.equal(await evaluate('!!pdfTerminationGate.message'), true, 'first worker finished before the same PDF reopened');
    } finally {
      await evaluate('pdfTerminationGate.release(); delete window.pdfTerminationGate');
    }
    await review('3');
    await wait("/No extractable text.*OCR/.test(document.querySelector('#pdf-attachment-error').textContent)", 'graphics-only page guidance');
    assert.equal(await evaluate("document.querySelector('#pdf-attachment-apply').disabled && !document.querySelector('#pdf-attachment-preview').checkVisibility()"), true);
    await click('#pdf-attachment-cancel');
    await closed();

    await open();
    await selectedPage();
    await click('#pdf-attachment-apply');
    await closed();
    await wait(`document.querySelector('#attachments').textContent.includes(${JSON.stringify(pdfName)})`, 'plain-text PDF attachment with original filename');
    await upload('#file-input', htmlFixture);
    await wait("document.querySelector('#attachments').textContent.includes('uploaded.html')", 'ordinary HTML text attachment');
    assert.equal(await evaluate("document.querySelector('#draft').value"), draft);
    assert.equal(apiMutations.length, mutations, 'PDF selection or review mutated the server');
    await evaluate("document.querySelector('#composer').requestSubmit()");
    await wait("document.querySelector('#messages').textContent.includes('PDF browser reply.')", 'reply to approved PDF text');
    await idle();
    const submissions = browserRequests.filter(request => request.method === 'POST' && /\/api\/chats\/[^/]+\/runs$/.test(new URL(request.url).pathname));
    assert.equal(submissions.length, 2, 'unexpected number of chat submissions');
    assert.deepEqual(JSON.parse(submissions[1].postData).attachments, [
      {name:pdfName,type:'text/plain',text:expected},
      {name:'uploaded.html',type:'text/html',text:htmlSource},
    ], 'chat did not send the exact PDF excerpt and HTML source attachments');
    const paths = pdfAssetRequests().map(request => new URL(request.url).pathname);
    for (const asset of ['/pdf_attachment.js', '/pdf-reader/pdf.mjs', '/pdf-reader/pdf.worker.mjs']) {
      assert(paths.includes(asset), 'missing request for ' + asset);
    }
    assert(pdfWorkers.some(worker => worker === origin + '/pdf-reader/pdf.worker.mjs'), 'PDF.js did not use the dedicated same-origin worker');
    const beforeViewing = {assets:pdfAssetRequests().length, mutations:apiMutations.length};
    await checkSentSource(pdfName, expected, pdfName + '.txt');
    await checkSentSource('uploaded.html', htmlSource);
    await reloadDocument();
    await wait("document.querySelector('#messages').textContent.includes('PDF browser reply.')", 'restored PDF conversation');
    await checkSentSource(pdfName, expected);
    assert.deepEqual({assets:pdfAssetRequests().length, mutations:apiMutations.length}, beforeViewing, 'viewing sent attachments fetched the PDF reader or mutated the server');
  }
  assert.deepEqual(browserRequests.filter(request => new URL(request.url).origin !== origin), [], 'PDF check made an off-origin browser request');
  const payload = browserRequests.map(request => request.postData || '').join('\n');
  assert.doesNotMatch(payload, /raw-pdf-fixture-marker|First page private marker|%PDF-|application\/pdf|JVBERi0/);
  assert.deepEqual(errors, [], 'uncaught PDF browser exceptions');
  console.log('PASS: PDF ' + pdfMode + ', zero lazy assets on boot and plain chat' + (installed ? ', reviewed page 2 sent only as text, Escape cancellation, graphics-only refusal, narrow preview, same-origin worker and read-only sent source after reload' : ', unavailable-reader guidance'));
}
async function sidebarFocusSamples(selector) {
  const transitions = await evaluate(`(async () => {
    const animations = document.querySelector('.sidebar').getAnimations();
    await Promise.all(animations.map(animation => animation.finished.catch(() => {})));
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    return animations.length;
  })()`);
  if (transitions) console.log('Sidebar screenshot waited for ' + transitions + ' transition(s)');
  const box = await evaluate(`(()=>{const row=document.querySelector(${JSON.stringify(selector)}).parentElement,r=row.getBoundingClientRect(),probe=document.createElement('div');probe.style.background='var(--hover)';document.body.append(probe);const expected=getComputedStyle(probe).backgroundColor;probe.remove();return {x:r.x,y:r.y,width:r.width,height:r.height,expected}})()`);
  const left = Math.floor(box.x), top = Math.floor(box.y);
  const width = Math.ceil(box.width + box.x - left), height = Math.ceil(box.height + box.y - top);
  const image = await cdp('Page.captureScreenshot', {format: 'png', clip: {x: left, y: top, width, height, scale: 1}});
  return await evaluate(`new Promise(resolve=>{const img=new Image();img.onload=()=>{const canvas=document.createElement('canvas');canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;const context=canvas.getContext('2d',{willReadFrequently:true});context.drawImage(img,0,0);const pixels=[[10,4],[img.naturalWidth-8,4],[10,img.naturalHeight-5],[img.naturalWidth-8,img.naturalHeight-5]].map(([x,y])=>Array.from(context.getImageData(x,y,1,1).data));resolve({expected:${JSON.stringify(box.expected)},pixels})};img.onerror=()=>resolve({expected:${JSON.stringify(box.expected)},pixels:[]});img.src='data:image/png;base64,${image.data}'})`);
}

try {
  let port;
  const startupDeadline = startupStarted + startupTimeoutMs;
  while (Date.now() < startupDeadline) {
    if (launchError) throw launchDiagnostic('Chrome failed to spawn: ' + launchError.message);
    if (chrome.exitCode !== null || chrome.signalCode !== null) throw launchDiagnostic('Chrome exited before opening DevTools: exit=' + chrome.exitCode + ', signal=' + chrome.signalCode);
    try { port = Number((await readFile(path.join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]); if (port) break; } catch {}
    await delay(Math.min(80, Math.max(0, startupDeadline - Date.now())));
  }
  const startupElapsedMs = Date.now() - startupStarted;
  if (!port) throw launchDiagnostic('Chrome did not open DevTools within ' + startupElapsedMs + ' ms (limit ' + startupTimeoutMs + ' ms)');
  const startupUrl = 'http://127.0.0.1:' + port + '/json/list';
  let target, discovery = 'No response received';
  while (!target && Date.now() < startupDeadline) {
    const remaining = Math.max(1, startupDeadline - Date.now());
    try {
      const response = await fetch(startupUrl, {signal: AbortSignal.timeout(remaining)});
      if (response.ok) {
        target = (await response.json()).find(t => t.type === 'page' && t.webSocketDebuggerUrl);
        if (!target) discovery = 'No page with a debugger URL';
      } else {
        discovery = 'HTTP ' + response.status;
      }
    } catch (error) {
      discovery = error.name + ': ' + error.message;
    }
    if (!target) await delay(Math.min(80, Math.max(0, startupDeadline - Date.now())));
  }
  if (!target) throw launchDiagnostic('DevTools page discovery timed out at ' + startupUrl + ' (' + startupTimeoutMs + ' ms limit). Last result: ' + discovery);
  const discoveryElapsedMs = Date.now() - startupStarted;
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    const remaining = startupDeadline - Date.now();
    if (remaining <= 0) { reject(launchDiagnostic('Chrome DevTools WebSocket did not open within ' + (Date.now() - startupStarted) + ' ms (limit ' + startupTimeoutMs + ' ms)')); return; }
    const timeout = setTimeout(() => reject(launchDiagnostic('Chrome DevTools WebSocket did not open within ' + (Date.now() - startupStarted) + ' ms (limit ' + startupTimeoutMs + ' ms)')), remaining);
    socket.onopen = () => { clearTimeout(timeout); resolve(); };
    socket.onerror = () => { clearTimeout(timeout); reject(launchDiagnostic('Chrome DevTools WebSocket connection failed')); };
  });
  console.log('Chrome DevTools ready in ' + (Date.now() - startupStarted) + ' ms (port at ' + startupElapsedMs + ' ms, page at ' + discoveryElapsedMs + ' ms)');
  socket.onmessage = event => {
    const data = JSON.parse(event.data);
    if (data.id && pending.has(data.id)) { const call = pending.get(data.id); pending.delete(data.id); clearTimeout(call.timeout); if (data.error) call.reject(new Error(JSON.stringify(data.error))); else call.resolve(data.result); }
    if (data.method === 'Runtime.exceptionThrown') { const detail = data.params.exceptionDetails; errors.push(detail.exception?.description || detail.text); }
    if (data.method === 'Network.requestWillBeSent' && /\/api\/runs\/[^/]+\/events(?:\?|$)/.test(data.params.request.url)) runStreams.push(data.params.request.url);
    if (data.method === 'Network.requestWillBeSent' && data.params.request.method !== 'GET') apiMutations.push({url: data.params.request.url, method: data.params.request.method});
    if (data.method === 'Network.requestWillBeSent' && data.params.request.url.startsWith('https://example.invalid/')) imageLinkRequests.push(data.params.request.url);
    if (data.method === 'Network.requestWillBeSent' && new URL(data.params.request.url).pathname === '/api/workspace/files') {
      savedFileReads.push(new URL(data.params.request.url).searchParams.getAll('path'));
    }
    if ((pdfMode || featuresMode || homeContextMode) && data.method === 'Network.requestWillBeSent') browserRequests.push(data.params.request);
    if (pdfMode && data.method === 'Target.attachedToTarget') {
      const {sessionId, targetInfo} = data.params;
      pdfWorkers.push(targetInfo.url);
      cdp('Network.enable', {}, sessionId)
        .then(() => cdp('Runtime.runIfWaitingForDebugger', {}, sessionId))
        .catch(error => errors.push(error.message));
    }
    if (data.method === 'Fetch.requestPaused') {
      if (data.params.request.method === 'POST' && /\/api\/chats\/[^/]+\/runs$/.test(new URL(data.params.request.url).pathname)) heldSendResponse = data.params;
      else if (data.params.request.url.endsWith('/api/restore')) heldRestoreRequest = data.params.requestId;
      else if (new URL(data.params.request.url).pathname === '/api/chats') {
        if (data.params.request.method === 'POST' && data.params.responseStatusCode) heldChatCreateResponse = data.params;
        else cdp('Fetch.continueResponse', {requestId: data.params.requestId}).catch(() => {});
      }
      else heldMonitorLoad = data.params.requestId;
    }
    if (data.method === 'Page.javascriptDialogOpening') { nativeDialogs.push(data.params.type); cdp('Page.handleJavaScriptDialog', {accept: true, promptText: 'Research'}).catch(() => {}); }
    if (data.method === 'Page.fileChooserOpened') attachmentChooser = data.params;
  };
  await cdp('Page.enable'); await cdp('Runtime.enable'); await cdp('DOM.enable'); await cdp('Network.enable');
  if (pdfMode) await cdp('Target.setAutoAttach', {autoAttach:true,waitForDebuggerOnStart:true,flatten:true,filter:[{type:'worker'}]});
  await cdp('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
  await cdp('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: 'dark'}]});
  await navigateDocument(url);
  if (process.argv.includes('--image-results-only')) {
    await signIn();
    const origin = 'https://pictures.example.invalid';
    const title = 'Northern lights <b>in winter</b>';
    const sourcePage = process.argv.find(arg=>arg.startsWith('--fixture-upstream=')).slice('--fixture-upstream='.length)+'/v1/models';
    const payload = {query:'aurora', images:[
      {title, image_url:origin+'/aurora.jpg', thumbnail_url:origin+'/thumb.jpg', source_url:sourcePage},
      {title:'Clouds', image_url:'ftp://example.invalid/clouds', thumbnail_url:origin+'/clouds.png'},
      {title:'No image', source_url:origin+'/no-image'}
    ]};
    const mcp = {content:[{type:'text', text:JSON.stringify(payload)}]};
    const makeCall = (id, result, status='completed') => ({id, name:'local_search_images', arguments:'{}', status, result});
    const calls = [makeCall('text-result',mcp),makeCall('structured',{structuredContent:payload,content:[]}),
      makeCall('failed',{...mcp,isError:true},'failed'),makeCall('empty',{images:[]}),makeCall('ordinary',{value:42}),
      makeCall('many',{images:Array.from({length:21},(_,i)=>({image_url:origin+'/'+i}))})];
    const chat = await evaluate(`ChatShell.api('/import','POST',${JSON.stringify({version:1,chat:{title:'Image search history',active_leaf:'image-answer'},messages:[
      {id:'image-answer',parent_id:null,role:'assistant',content:'Search complete.',tool_calls:calls}
    ]})})`);
    await reloadDocument();
    await wait(`document.querySelector('[data-chat="${chat.id}"]')`, 'imported search history');
    await pointerClick(`[data-chat="${chat.id}"]`);
    const tool = '[data-tool-call="text-result"]';
    await wait(`document.querySelector('${tool}')`, 'image search tool');
    await pointerClick(tool+' > details > summary');
    const list = tool+' .tool-image-results';
    const visible = await evaluate(`Array.from(document.querySelectorAll('${list} a'), a=>[a.textContent,a.href,a.target,a.rel])`);
    const expected = [[title,origin+'/aurora.jpg'],['Source',sourcePage],['Thumbnail',origin+'/thumb.jpg'],['Clouds',origin+'/clouds.png']];
    assert.deepEqual(visible,expected.map(([label,href])=>[label,href,'_blank','noopener noreferrer']));
    assert.equal(await evaluate(`document.querySelector('${list} b')`),null);
    assert.equal(await evaluate("document.querySelectorAll('[data-tool-call=structured] .tool-image-results li').length"),2);
    assert.equal(await evaluate("document.querySelectorAll('[data-tool-call=many] .tool-image-results li').length"),20);
    assert(await evaluate("document.querySelector('[data-tool-call=many] .tool-body').textContent.includes('more entries')"));
    for (const id of ['failed','empty','ordinary'])
      assert.equal(await evaluate(`document.querySelector('[data-tool-call="${id}"] .tool-image-results')`),null);
    assert(await evaluate(`document.querySelector('${tool} .tool-body details pre').textContent.includes('No image')`));
    for (const width of [320,1440]) {
      await cdp('Emulation.setDeviceMetricsOverride',{width,height:800,deviceScaleFactor:1,mobile:width===320});
      await wait(`innerWidth===${width}`, 'image result viewport');
      await evaluate(`document.querySelector('${list}').scrollIntoView({block:'center'})`);
      assert(await evaluate(`Array.from(document.querySelectorAll('${list} a')).every(a=>{
        const r=a.getBoundingClientRect(); return r.width>0 && r.left>=0 && r.right<=innerWidth && a.contains(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2));
      })`),'image links reachable at '+width);
      await screenshot('image-results-'+width+'.png');
    }
    assert.equal(browserRequests.some(request=>new URL(request.url).origin===origin),false,'opening results fetched a remote image');
    assert.equal((await cdp('Target.getTargets')).targetInfos.some(tab=>tab.url===sourcePage),false);
    await pointerClick(list+' a[aria-label^="Source for"]');
    const deadline = Date.now()+5000;
    let opened;
    while (Date.now()<deadline && !opened) {
      opened = (await cdp('Target.getTargets')).targetInfos.find(tab=>tab.url===sourcePage);
      if (!opened) await delay(50);
    }
    assert(opened,'source click navigated a new tab to the fixture page');
    await cdp('Target.closeTarget',{targetId:opened.targetId});
    console.log('PASS: image links, source opened in a new tab, MCP envelopes, raw results, fallback and phone layout');
  } else if (process.argv.includes('--write-file-actions-only')) {
    await signIn();
    const filename = 'nested/saved & exact.txt', source = 'Saved bytes \u00e9\n';
    const chat = await evaluate(`(async () => {
      const {revision}=await ChatShell.api('/workspace');
      const state=await ChatShell.api('/workspace/file','POST',{revision,path:${JSON.stringify(filename)},content:${JSON.stringify(source)}});
      const htmlPath='site/index.html';
      const site=await ChatShell.api('/workspace/file','POST',{revision:state.revision,path:htmlPath,content:'<h1>First version</h1>'});
      const tool_calls=(${workspaceWriteCases})(${JSON.stringify(filename)},'missing.txt',site.revision,site.files);
      tool_calls.push({...(${workspaceWriteCases})(htmlPath,'missing.txt',site.revision,site.files)[1],id:'linked-site'});
      return ChatShell.api('/import','POST',{version:1,chat:{title:'Saved write file actions',active_leaf:'saved-write-message'},messages:[{id:'saved-write-message',parent_id:null,role:'assistant',content:'Saved write outcomes.',tool_calls}]});
    })()`);
    await reloadDocument();
    await wait(`document.querySelector('[data-chat="${chat.id}"]')`, 'saved write actions fixture');
    await pointerClick(`[data-chat="${chat.id}"]`);
    await wait("document.querySelector('[data-tool-call=linked-write]')", 'saved write outcomes');
    await checkChatSavedWriteActions(filename, source);
    const openResult = id => pointerClick('[data-tool-call="linked-'+id+'"] .workspace-tool-view-file');
    const editSaved = '.artifact-toolbar-actions button:first-child';
    const editor = '#workspace-editor';
    const chatNav = '.workspace-back';
    const dialogOpen = "document.querySelector('#artifact-dialog').open";
    await fill('#draft', 'Keep the follow-up draft.');
    await cdp('Emulation.setDeviceMetricsOverride',{width:320,height:700,deviceScaleFactor:1,mobile:true});
    await wait("innerWidth===320 && document.documentElement.dataset.compact==='true'", 'phone layout before opening HTML');
    await evaluate(`new Promise(resolve => {
      const button = document.querySelector('[data-tool-call="linked-site"] .workspace-tool-view-file');
      button.scrollIntoView({block:'center',behavior:'instant'});
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    })`);
    await openResult('site');
    await wait(dialogOpen + " && document.querySelector('#artifact-preview-tab').getAttribute('aria-selected')==='true'", 'HTML result preview');
    assert.equal(await evaluate("document.querySelector("+JSON.stringify(editSaved)+").textContent"), 'Open in editor');
    const revision = await evaluate(`(async()=>{const snap=await ChatShell.api('/workspace');return (await ChatShell.api('/workspace/file','POST',{revision:snap.revision,path:'site/index.html',content:'<h1>Later version</h1>'})).revision})()`);
    assert.equal(await evaluate("document.documentElement.scrollWidth<=innerWidth"), true, 'file viewer overflows the phone viewport');
    await screenshot('saved-html-editor-phone.png');
    await pointerClick(editSaved);
    await wait("document.activeElement.id==='workspace-editor' && document.activeElement.value==='<h1>Later version</h1>'", 'editor reads the current saved file');
    await fill(editor, 'Unsaved HTML edit.');
    await pointerClick(chatNav);
    await openResult('site');
    await wait(dialogOpen, 'reopen same saved HTML');
    await pointerClick(editSaved);
    await wait("document.activeElement.id==='workspace-editor' && document.activeElement.value==='Unsaved HTML edit.'", 'retain same-file editor buffer');
    await pointerClick(chatNav);
    await openResult('stale-write');
    await wait(dialogOpen, 'open current copy after a write conflict');
    await pointerClick(editSaved);
    await wait("document.querySelector('#action-dialog').open && document.querySelector('#action-dialog-title').textContent==='Discard unsaved changes?'", 'protect edits in another file');
    await pointerClick('#conversation-cancel');
    assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), 'Unsaved HTML edit.');
    await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await wait("document.documentElement.dataset.compact==='false'", 'desktop file list');
    const fileURL = '/api/workspace/file?path=' + encodeURIComponent(filename);
    await evaluate(`(()=>{const fetchFile=window.fetch;window.releaseEditorRead=null;window.fetch=async (...args)=>{
      const response=await fetchFile(...args); if (args[0]===${JSON.stringify(fileURL)}) {
        window.fetch=fetchFile;await new Promise(resolve=>window.releaseEditorRead=resolve);
      }return response;
    };})()`);
    await pointerClick('.workspace-file[title="'+filename+'"]');
    await wait("document.querySelector('#action-dialog').open", 'explicit discard before another file opens');
    await pointerClick('#conversation-submit');
    await wait("typeof window.releaseEditorRead==='function'", 'hold file read in flight');
    await pointerClick(chatNav);
    await openResult('site');
    await wait(dialogOpen, 'choose saved HTML while another read is in flight');
    await pointerClick(editSaved);
    await wait("document.activeElement.id==='workspace-editor'", 'resume the HTML edit');
    await evaluate("window.releaseEditorRead();new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))");
    assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), 'Unsaved HTML edit.', 'an earlier read replaced the resumed edit');
    await pointerClick(chatNav);
    assert.equal(await evaluate("document.querySelector('#draft').value"), 'Keep the follow-up draft.');
    assert.equal(await evaluate("ChatShell.api('/workspace').then(s=>s.revision)"), revision, 'opening the editor wrote workspace data');
    assert.deepEqual(errors, [], 'saved write action browser exceptions');
    console.log('PASS: tool results open the latest saved file in the editor, preserve unsaved edits and chat drafts, and handle missing or invalid files');
  } else if (process.argv.includes('--draft-viewer-only')) {
    await signIn();
    await fill('#draft','Keep this draft.');
    await evaluate(`(() => {
      const transfer = new DataTransfer();
      transfer.items.add(new File(['export default 42;'], 'answer.mjs', {type:'application/javascript'}));
      transfer.items.add(new File(['module.exports = 42;'], 'answer.CJS'));
      const picker = document.querySelector('#file-input');
      picker.files = transfer.files;
      picker.dispatchEvent(new Event('change', {bubbles:true}));
    })()`);
    await wait("JSON.parse(sessionStorage.getItem('llmx-draft:new') || '{}').attachments?.length===2 || document.querySelector('#toast').textContent.includes('Unsupported file')", 'module files selected');
    const moduleDraft = await evaluate("JSON.parse(sessionStorage.getItem('llmx-draft:new') || '{}')");
    assert.deepEqual(moduleDraft.attachments?.map(file=>[file.name,file.text]), [
      ['answer.mjs','export default 42;'], ['answer.CJS','module.exports = 42;']
    ], 'device picker attaches ES and CommonJS modules as source');
    const accepted = await evaluate("document.querySelector('#file-input').accept.split(',')");
    assert(accepted.includes('.mjs') && accepted.includes('.cjs'), 'module files visible in device picker');
    await pointerClick('#attachments [aria-label="Remove answer.mjs"]');
    await pointerClick('#attachments [aria-label="Remove answer.CJS"]');

    const source = '<b>Source only</b>\r\n';
    await evaluate(`(async () => {
      const canvas=document.createElement('canvas'); canvas.width=640; canvas.height=360;
      const ctx=canvas.getContext('2d'); ctx.fillStyle='#f6f8fb'; ctx.fillRect(0,0,640,360);
      ctx.fillStyle='#193951'; ctx.font='22px sans-serif'; ctx.fillText('Tokens per second',32,48);
      for (const [i,height] of [108,165,125,184,212,198].entries()) ctx.fillRect(40+i*94,316-height,60,height);
      const image=type=>new Promise(resolve=>canvas.toBlob(resolve,type));
      await ChatComposer.attach([new File([${JSON.stringify(source)}],'page.html',{type:'text/html'}),
        new File([await image('image/png')],'plot.png',{type:'image/png'}),
        new File([await image('image/jpeg')],'scan.jpg',{type:'image/jpeg'})]);
    })()`);
    const mutations=apiMutations.length;
    const draftSnapshot="JSON.parse(sessionStorage.getItem('llmx-draft:new')||'{}')";
    await wait(draftSnapshot+".attachments?.length===3",'draft files stored');
    const before=await evaluate(draftSnapshot);
    const textButton='#attachments [aria-label="Read attachment page.html"]';
    const editCopy='#artifact-dialog .artifact-toolbar-actions button:first-child';
    assert(await evaluate(`Boolean(document.querySelector(${JSON.stringify(textButton)}))`),'unsent file has no read control');
    await pointerClick(textButton);
    await wait("document.querySelector('#artifact-dialog').open",'draft source viewer');
    assert.equal(await evaluate(`document.querySelector(${JSON.stringify(editCopy)}).checkVisibility()`),true,'draft source offers an editable workspace copy');
    assert.equal(await evaluate("document.querySelector('.artifact-source code').textContent"),source);
    assert.equal(await evaluate("document.querySelector('.artifact-source code').childElementCount"),0);
    assert.equal(await evaluate("document.querySelector('#artifact-preview-tab').checkVisibility()"),false);
    await click('#artifact-dialog [aria-label="Close artifact"]');
    await wait(`document.activeElement.matches(${JSON.stringify(textButton)})`, 'focus returns to draft filename');
    const downloads=path.join(profile,'attachment-downloads');
    await mkdir(downloads);
    await cdp('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:downloads});
    for (const [width,theme,name] of [[320,'light','plot.png'],[1440,'dark','scan.jpg']]) {
      await cdp('Emulation.setDeviceMetricsOverride',{width,height:800,deviceScaleFactor:1,mobile:width===320});
      await wait(`innerWidth===${width} && document.documentElement.dataset.compact==='${width===320}'`,'attachment viewport ready');
      await evaluate(`document.documentElement.dataset.theme='${theme}'`);
      const selector='#attachments [aria-label="View image '+name+'"]';
      await evaluate(`document.querySelector(${JSON.stringify(selector)}).focus()`);
      await cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r'});
      await cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
      await wait("document.querySelector('#artifact-dialog').open && document.querySelector('#artifact-dialog img')?.naturalWidth===640",'inspect local image bytes');
      assert.equal(await evaluate("document.querySelector('#artifact-dialog h2').textContent"),name);
      assert(await evaluate("document.querySelector('#artifact-dialog img').checkVisibility() && document.documentElement.scrollWidth<=innerWidth"));
      assert.equal(await evaluate("document.querySelector('#artifact-source-tab').checkVisibility()"),false);
      await wait("document.querySelector('.artifact-header p').textContent.includes('640 x 360 px')",'decoded image dimensions');
      assert.equal(await evaluate("document.querySelector('#artifact-dialog img').getBoundingClientRect().right<=innerWidth"),true);
      const actual = '#artifact-actual-size';
      const bounds = () => evaluate("(() => {const img=document.querySelector('.artifact-image'),area=img.parentElement;return {width:img.getBoundingClientRect().width, height:img.getBoundingClientRect().height,left:area.scrollLeft, overflow:area.scrollWidth-area.clientWidth,page:document.documentElement.scrollWidth};})()");
      if (width===320) {
        const fitted=await bounds();
        assert(fitted.width<640);
        assert(await evaluate(`document.querySelector('${actual}')?.checkVisibility()`),'scaled image has no original-size control');
        await pointerClick(actual);
        await wait("document.querySelector('#artifact-actual-size').getAttribute('aria-pressed')==='true'",'image at actual size');
        const zoomed=await bounds();
        assert.equal(zoomed.width,640); assert.equal(zoomed.height,360);
        assert(zoomed.overflow>0); assert.equal(zoomed.page,width);
        await evaluate("document.querySelector('.artifact-image-view').focus()");
        await cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});
        await cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});
        await wait("document.querySelector('.artifact-image-view').scrollLeft>0",'pan image with keyboard');
        await cdp('Emulation.setDeviceMetricsOverride',{width:320,height:320,deviceScaleFactor:1,mobile:true});
        await wait("document.querySelector('.artifact-image-view').scrollHeight>document.querySelector('.artifact-image-view').clientHeight",'short viewport scrolls vertically');
        const farEdge=await evaluate("(() => {const v=document.querySelector('.artifact-image-view');v.scrollTo(10000,10000);return {x:v.scrollLeft,y:v.scrollTop,maxX:v.scrollWidth-v.clientWidth,maxY:v.scrollHeight-v.clientHeight};})()");
        assert.equal(farEdge.x,farEdge.maxX); assert.equal(farEdge.y,farEdge.maxY);
        await cdp('Emulation.setDeviceMetricsOverride',{width,height:800,deviceScaleFactor:1,mobile:true});
        await pointerClick(actual);
        await wait(`document.querySelector('.artifact-image').getBoundingClientRect().width===${fitted.width}`,'return to fitted image width');
        assert.equal((await bounds()).left,0);
        await pointerClick(actual);
        await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:800,deviceScaleFactor:1,mobile:false});
        await wait("!document.querySelector('#artifact-actual-size').checkVisibility() && document.querySelector('#artifact-actual-size').getAttribute('aria-pressed')==='false'",'clear actual size when the image fits after resize');
        await cdp('Emulation.setDeviceMetricsOverride',{width,height:800,deviceScaleFactor:1,mobile:true});
        await wait("document.querySelector('#artifact-actual-size').checkVisibility()", 'size control after narrowing');
      } else {
        assert.equal((await bounds()).width,640);
        assert.equal(await evaluate(`document.querySelector('${actual}').checkVisibility()`),false,'fitting image needs no size switch');
      }
      await pointerClick('#artifact-dialog .artifact-toolbar-actions button:last-child');
      const original=before.attachments.find(a=>a.name===name);
      const bytes=Buffer.from(original.data.split(',')[1],'base64');
      let downloaded;
      for (let i=0;i<100 && !downloaded;i++) {
        try { downloaded=await readFile(path.join(downloads,name)); }
        catch (error) { if (error.code!=='ENOENT') throw error; await delay(50); }
      }
      assert.deepEqual(downloaded,bytes,'downloaded image differs from the draft');
      await screenshot('draft-image-'+width+'.png');
      await cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
      await wait(`!document.querySelector('#artifact-dialog').open && document.activeElement.matches(${JSON.stringify(selector)})`, 'close image with keyboard');
      await wait("!document.querySelector('#artifact-dialog img[src]')",'release closed image');
    }
    await pointerClick(textButton);
    await wait("document.querySelector('.artifact-source code').checkVisibility()", 'switch back from image to text');
    await click('#artifact-dialog [aria-label="Close artifact"]');
    assert.deepEqual(await evaluate(draftSnapshot),before);
    assert.equal(apiMutations.length,mutations,'review sent or changed server data');
    await pointerClick('#attachments [aria-label="Remove plot.png"]');
    await wait(draftSnapshot+".attachments?.length===2",'remove remains separate');
    const messages=[
      {id:'first',role:'user',content:'Original attachment',attachments:[{name:'page.html',type:'text/plain',text:'Previous version'}]},
      {id:'unused',parent_id:'first',role:'user',content:'Alternate branch',attachments:[{name:'private.txt',type:'text/plain',text:'Other branch'}]},
      {id:'img',parent_id:'first',role:'user',content:'Reference image',attachments:[before.attachments[0],before.attachments[2]]}
    ];
    const saved=await evaluate(`ChatShell.api('/import','POST',{version:1,chat:{title:'Image attachment',active_leaf:'img'},messages:${JSON.stringify(messages)}})`);
    await reloadDocument();
    await wait(`document.querySelector('[data-chat="${saved.id}"]')`, 'saved image conversation');
    await pointerClick(`[data-chat="${saved.id}"]`);
    const sentImage='.message-attachments [aria-label="View image scan.jpg"]';
    await wait(`document.querySelector(${JSON.stringify(sentImage)})?.checkVisibility()`, 'sent thumbnail');
    await fill('#draft', 'Follow-up question');
    await pointerClick(sentImage);
    await wait("document.querySelector('#artifact-dialog').open && document.querySelector('.artifact-image').naturalWidth===640", 'inspect sent image');
    const reuse='#artifact-add-to-draft', draftFiles='#attachments .attachment-pill';
    assert.equal(await evaluate(`document.querySelector(${JSON.stringify(editCopy)}).checkVisibility()`),false,'image viewer has no text editor action');
    const writes=apiMutations.length;
    await pointerClick(reuse);
    await wait("document.activeElement.id==='draft' && !document.querySelector('#artifact-dialog').open",'return to follow-up');
    await pointerClick(sentImage);
    await pointerClick(reuse);
    assert.equal(await evaluate(`document.querySelectorAll('${draftFiles}').length`),1,'reuse keeps a single copy');
    const sentText='.message.user:has(.attachment-picture) [aria-label="Read attachment page.html"]';
    await pointerClick('#attach-source');
    await pointerClick('#action-menu [data-action="attach-conversation"]');
    assert.equal(await evaluate("document.querySelector('#find-message-role').value"),'files');
    assert.equal(await evaluate("document.querySelectorAll('#find-message-list [data-file]').length"),3);
    assert(!await evaluate("document.querySelector('#find-message-list').textContent.includes('private.txt')"));
    await fill('#find-message-query','page.htm');
    await wait("document.querySelectorAll('#find-message-list [data-file]').length===2",'filter two versions by name');
    await pointerClick('#find-message-list [data-file]');
    assert.equal(await evaluate("document.querySelector('.artifact-source code').textContent"),source);
    await pointerClick(reuse);
    const stored=`JSON.parse(sessionStorage.getItem('llmx-draft:${saved.id}')||'{}')`;
    await wait(stored+'.attachments?.length===2','follow-up files stored');
    const messageFiles=await evaluate(`ChatShell.api('/chats/'+${JSON.stringify(saved.id)}).then(chat=>chat.messages.find(m=>m.content==='Reference image').attachments)`);
    assert.deepEqual(await evaluate(stored+'.attachments'),[messageFiles[1],messageFiles[0]]);
    assert.equal(await evaluate(stored+'.text'),'Follow-up question');
    await pointerClick('#attachments [aria-label="Read attachment page.html"]');
    assert.equal(await evaluate("document.querySelector('#artifact-add-to-draft').checkVisibility()"),false);
    await click('#artifact-dialog [aria-label="Close artifact"]');
    await pointerClick('#attachments [aria-label="Remove page.html"]');
    await evaluate("ChatComposer.attach([new File(['Changed'], 'page.html',{type:'text/html'})])");
    await pointerClick(sentText);
    await pointerClick(reuse);
    await wait("document.querySelector('.artifact-status').textContent.includes('Remove the existing page.html')",'different version kept');
    await wait(stored+".attachments[1]?.text==='Changed'",'preserve the newer draft file');
    assert.equal(apiMutations.length,writes,'attachment reuse wrote server state');
    assert.equal(await evaluate(`document.querySelector(${JSON.stringify(editCopy)}).checkVisibility()`),true,'sent source offers an editable workspace copy');
    await cdp('Emulation.setDeviceMetricsOverride',{width:320,height:720,deviceScaleFactor:1,mobile:true});
    await wait("innerWidth===320 && document.documentElement.dataset.compact==='true'",'phone attachment viewer');
    assert(await evaluate(`{const r=document.querySelector(${JSON.stringify(editCopy)}).getBoundingClientRect();r.width>0 && r.left>=0 && r.right<=innerWidth}`),'edit copy fits the phone viewer');
    await pointerClick(editCopy);
    await wait("document.querySelector('#action-dialog').open && document.querySelector('#conversation-value').value==='page.html'",'attachment filename in editor');
    await fill('#conversation-value','site/page.html');
    await pointerClick('#conversation-submit');
    await wait("document.activeElement.id==='workspace-editor'",'attachment opened in editor');
    assert.equal(await evaluate("document.querySelector('#workspace-editor').value"),source.replaceAll('\r\n','\n'));
    assert.equal(await evaluate("ChatShell.api('/workspace').then(s=>s.files.length)"),0,'opening a copy saved a file');
    await pointerClick('#workspace-save');
    await wait("document.querySelector('#workspace-save').disabled",'workspace saved');
    const copy=await evaluate("ChatShell.api('/workspace/file?path=site%2Fpage.html')");
    assert.equal(copy.content,source,'saved copy lost attachment bytes');
    await evaluate(`ChatShell.api('/workspace/file','POST',{path:'conflict.html',revision:${copy.revision},content:'Saved original'})`);
    await fill('#workspace-editor','Unsent editor changes.');
    await pointerClick('.workspace-back');
    const writesBeforeCancel = apiMutations.length;
    for (const cancelAt of ['discard','filename','replace']) {
      await pointerClick(sentText);
      await pointerClick(editCopy);
      const titled = title => `document.querySelector('#action-dialog').open && document.querySelector('#action-dialog-title').textContent===${JSON.stringify(title)}`;
      await wait(titled('Discard unsaved changes?'),'protect unsaved editor');
      if (cancelAt !== 'discard') {
        await pointerClick('#conversation-submit');
        await wait(titled('Edit in workspace'),'name the attachment copy');
        if (cancelAt === 'replace') {
          await fill('#conversation-value','conflict.html');
          await pointerClick('#conversation-submit');
          await wait(titled('Replace saved file?'),'confirm conflicting destination');
        }
      }
      await pointerClick('#conversation-cancel');
      await wait("!document.querySelector('#action-dialog').open",'dismiss copy dialog');
      await wait(`document.querySelector('#app').dataset.view==='chat' && document.activeElement.matches(${JSON.stringify(sentText)})`,
        cancelAt+' cancel returns to the chat attachment',3000);
      assert.equal(await evaluate("document.querySelector('#workspace-editor').value"),'Unsent editor changes.');
      assert.equal(await evaluate("document.querySelector('#draft').value"),'Follow-up question');
      assert.equal(apiMutations.length,writesBeforeCancel,'cancel wrote to the server');
    }
    assert.equal((await evaluate("ChatShell.api('/workspace/file?path=conflict.html')")).content,'Saved original');
    assert.equal(await evaluate("document.querySelector('#draft').value"),'Follow-up question');
    const original=await evaluate(`ChatShell.api('/chats/'+${JSON.stringify(saved.id)}).then(chat=>chat.messages.find(m=>m.content==='Reference image').attachments)`);
    assert.deepEqual(original,messageFiles,'opening a workspace copy modified the sent attachment');
    await pointerClick('#temporary-button');
    await wait("document.querySelector('#app').dataset.temporary==='true'",'temporary chat ready');
    await evaluate("ChatComposer.attach([new File(['Ephemeral'], 'temp.txt',{type:'text/plain'})])");
    await pointerClick('#attachments [aria-label="Read attachment temp.txt"]');
    assert.equal(await evaluate(`document.querySelector(${JSON.stringify(editCopy)}).checkVisibility()`),false,'temporary attachment offers persistent editing');
    console.log('PASS: review and reuse attachments, save an independent workspace copy and preserve the draft and original message');
  } else if (process.argv.includes('--editor-attachments-only')) {
    await signIn();
    await evaluate("ChatShell.api('/workspace/file','POST',{path:'notes/plan.txt',content:'Original',revision:0})");
    const goFiles = '.global-nav [data-view="workbench"]';
    const attached = '#attachments .attachment-pill';
    const inChat = "document.querySelector('#app').dataset.view==='chat'";
    await fill('#draft','Review this file.');
    await pointerClick(goFiles);
    await wait("document.querySelector('.workspace-file[title=\"notes/plan.txt\"]')?.checkVisibility()", 'saved path in workspace');
    await pointerClick('.workspace-file[title="notes/plan.txt"]');
    await wait("document.querySelector('#workspace-editor').value==='Original'", 'saved file in editor');
    await pointerClick('#workspace-attach');
    await wait(inChat, 'attach original version');
    await evaluate("ChatComposer.attach(Array.from({length:7},(_,i)=>new File(['spare'], 'spare'+i+'.txt', {type:'text/plain'})))");
    await pointerClick(goFiles);
    await pointerClick('#workspace-attach');
    await wait(inChat, 'reattach identical version');
    assert.equal(await evaluate("document.querySelectorAll("+JSON.stringify(attached)+").length"),8,'identical attachment duplicated at the draft limit');
    for (let i=0;i<7;i++) await click('[aria-label="Remove spare'+i+'.txt"]');
    const enterWithRefreshHeld = async () => {
      await evaluate(`(() => {
        const original = window.fetch;
        window.releaseWorkspaceRefresh = null;
        window.fetch = async (...args) => {
          const response = await original(...args);
          if (args[0] === '/api/workspace') {
            window.fetch = original;
            await new Promise(resolve => { window.releaseWorkspaceRefresh = resolve; });
          }
          return response;
        };
      })()`);
      await pointerClick(goFiles);
      await wait("typeof window.releaseWorkspaceRefresh==='function'", 'hold navigation refresh');
    };
    await enterWithRefreshHeld();
    await fill('#workspace-editor','Revised');
    await pointerClick('#workspace-attach');
    await wait("document.querySelector('.workspace-error').checkVisibility() || "+inChat, 'version conflict resolved');
    assert.equal(await evaluate("document.querySelector('#app').dataset.view"),'workbench','conflicting copy entered the draft');
    await evaluate("window.releaseWorkspaceRefresh();new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))");
    assert.equal(await evaluate("document.querySelector('.workspace-error').checkVisibility()"),true,'older refresh erased the conflict');
    assert.match(await evaluate("document.querySelector('.workspace-error').textContent"),/notes:plan.txt/);
    assert.equal(await evaluate("document.querySelector('#workspace-editor').value"),'Revised');
    await pointerClick('.workspace-back');
    await pointerClick('[aria-label="Remove notes:plan.txt"]');
    await enterWithRefreshHeld();
    const attachPoint = await evaluate("(()=>{const r=document.querySelector('#workspace-attach').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()");
    await cdp('Input.dispatchMouseEvent',{type:'mousePressed',...attachPoint,button:'left',clickCount:1});
    await evaluate("window.releaseWorkspaceRefresh();new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))");
    assert(await evaluate(`Boolean(document.elementFromPoint(${attachPoint.x},${attachPoint.y})?.closest('#workspace-attach'))`),'refresh moved Attach to Chat during the click');
    await cdp('Input.dispatchMouseEvent',{type:'mouseReleased',...attachPoint,button:'left',clickCount:1});
    await wait(inChat, 'attach revised version after removal');
    await wait("JSON.parse(sessionStorage.getItem('llmx-draft:new')||'{}').attachments?.[0]?.text==='Revised'", 'draft has revised editor bytes');
    assert.equal(await evaluate("document.querySelectorAll("+JSON.stringify(attached)+").length"),1);
    assert.equal(await evaluate("document.querySelector('#draft').value"),'Review this file.');
    assert.equal(await evaluate("ChatShell.api('/workspace/file?path=notes%2Fplan.txt').then(file=>file.content)"),'Original');
    await evaluate("ChatComposer.attach(Array.from({length:7},(_,i)=>new File(['spare'], 'neighbor'+i+'.txt', {type:'text/plain'})))");
    await pointerClick(goFiles);
    await cdp('Emulation.setDeviceMetricsOverride',{width:320,height:760,deviceScaleFactor:1,mobile:true});
    await wait("innerWidth===320 && document.documentElement.dataset.compact==='true'", 'phone editor ready for replacement');
    await fill('#workspace-editor','Second revision');
    await pointerClick('#workspace-attach');
    await wait("document.querySelector('.workspace-error').checkVisibility()", 'replacement offered for the editor path');
    assert.equal(await evaluate("document.querySelector('.workspace-error button')?.textContent"),'Replace in draft','no in-place replacement action');
    await fill('#workspace-editor','Last edit');
    assert.equal(await evaluate("document.querySelector('.workspace-error button')?.checkVisibility() || false"),false,'old offer still targets an earlier edit');
    await pointerClick('#workspace-attach');
    await wait("document.querySelector('.workspace-error button')?.checkVisibility()", 'offer for the current edit');
    const draftFiles = "JSON.parse(sessionStorage.getItem('llmx-draft:new')).attachments";
    await wait(draftFiles+".length===8", 'full draft before replacement');
    const before = await evaluate(draftFiles);
    assert.equal(before[0].text,'Revised','offering replacement changed the draft');
    assert(await evaluate("document.documentElement.scrollWidth<=innerWidth"),'replacement notice overflows a phone');
    await screenshot('editor-replace-320.png');
    for (const type of ['keyDown','keyUp']) await cdp('Input.dispatchKeyEvent',{type,key:'Tab',code:'Tab',windowsVirtualKeyCode:9});
    assert(await evaluate("document.activeElement.matches('.workspace-error button')"),'replacement is unreachable from the focused notice');
    await cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r'});
    await cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
    await wait(inChat, 'replace in draft returns to chat');
    await wait(draftFiles+"[0]?.text==='Last edit'", 'edited attachment stored in place');
    const after = await evaluate(draftFiles);
    assert.deepEqual(after,[{...before[0],text:'Last edit'},...before.slice(1)],'replacement disturbed the surrounding draft');
    assert.equal(await evaluate("document.activeElement.id"),'draft','composer focus after replacement');
    assert.equal(await evaluate("document.querySelector('#draft').value"),'Review this file.');
    assert.equal(await evaluate("ChatShell.api('/workspace/file?path=notes%2Fplan.txt').then(file=>file.content)"),'Original');
    await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await wait("document.documentElement.dataset.compact==='false'", 'desktop draft restored');
    for (let i=0;i<7;i++) await click('[aria-label="Remove neighbor'+i+'.txt"]');
    await pointerClick(goFiles);
    await fill('#workspace-editor','An edit that was offered before the draft changed');
    await pointerClick('#workspace-attach');
    await wait("document.querySelector('.workspace-error button')?.checkVisibility()", 'replacement for stale draft check');
    await evaluate("ChatComposer.attach([new File(['new'], 'later.txt', {type:'text/plain'})])");
    await pointerClick('.workspace-error button');
    await wait("document.querySelector('.workspace-error').textContent.includes('draft changed')", 'stale replacement refused');
    assert.equal(await evaluate("document.activeElement.matches('.workspace-error')"),true,'lost focus after stale replacement');
    await pointerClick('.workspace-back');
    await wait(draftFiles+".length===2", 'unrelated change retained after stale action');
    assert.equal(await evaluate(draftFiles+"[0].text"),'Last edit');
    await click('[aria-label="Remove later.txt"]');
    await evaluate("ChatComposer.attach(Array.from({length:5},(_,i)=>new File(['x'.repeat(2*1024*1024-(i===4?12:0))],'big'+i+'.txt',{type:'text/plain'})))");
    await pointerClick(goFiles);
    await pointerClick('#workspace-attach');
    await wait("document.querySelector('.workspace-error button')?.checkVisibility()", 'replacement near total byte limit');
    await pointerClick('.workspace-error button');
    await wait("document.querySelector('.workspace-error').textContent.includes('10 MiB')", 'oversize replacement refused');
    await pointerClick('.workspace-back');
    for (let i=0;i<5;i++) await click('[aria-label="Remove big'+i+'.txt"]');
    await wait(draftFiles+".length===1", 'draft back within browser storage capacity');
    assert.equal(await evaluate(draftFiles+"[0].text"),'Last edit','oversize replacement changed the original');
    const sourceText = '\ufeffFirst line\r\nSecond line';
    const paths = ['source.txt','notes.pdf'];
    const outcomes = [];
    for (const path of paths) {
      await evaluate(`(async () => {
        const {revision} = await ChatShell.api('/workspace');
        await ChatShell.api('/workspace/file','POST',{revision,path:${JSON.stringify(path)},content:${JSON.stringify(sourceText)}});
      })()`);
      await pointerClick(goFiles);
      const saved = '.workspace-file[title="' + path + '"]';
      await wait(`document.querySelector(${JSON.stringify(saved)})?.checkVisibility()`, 'saved editor attachment fixture');
      await pointerClick(saved);
      if (path === paths[0]) {
        await wait("document.querySelector('#action-dialog').open", 'discard the earlier unsaved fixture');
        await pointerClick('#conversation-submit');
      }
      await wait(`document.querySelector('.workspace-file-path').textContent===${JSON.stringify(path)}`, 'open editor attachment fixture');
      await pointerClick('#workspace-attach');
      await wait(inChat + " || document.querySelector('.workspace-error').checkVisibility()", 'attach exact workspace text');
      if (await evaluate(inChat)) {
        const snapshot = `JSON.parse(sessionStorage.getItem('llmx-draft:new')).attachments.find(a=>a.name===${JSON.stringify(path)})`;
        await wait(snapshot, 'workspace attachment stored with draft');
        const text = await evaluate(snapshot + '.text');
        outcomes.push([path,text]);
      } else {
        outcomes.push([path,await evaluate("document.querySelector('.workspace-error').textContent")]);
        await pointerClick('.workspace-back');
      }
    }
    assert.deepEqual(outcomes,paths.map(path=>[path,sourceText]),'editor attachment changed the workspace text or opened the device importer');
    await evaluate(`ChatComposer.projectDraft().apply({files:${JSON.stringify(paths.map(path=>({path,text:sourceText})))},project:{instructions:''}},false)`);
    assert.equal(await evaluate("document.querySelectorAll('#attachments .attachment-pill').length"),3,'picker duplicated the editor snapshot');
    console.log('PASS: reattaching a workspace path preserves one explicit file version');
  } else if (process.argv.includes('--project-download-only')) {
    await signIn();
    const members = JSON.parse(await readFile(fixture,'utf8'));
    const projectName = 'Release draft for the winter field trial and project handover';
    const projectIds = await evaluate(`(async () => {
      const api = ChatShell.api, members = ${JSON.stringify(members)};
      const files = Object.entries(members).map(([path,content])=>({path,content}));
      files.push({path:'unrelated.txt',content:'Outside the selected project'});
      const {revision} = await api('/workspace');
      await api('/workspace/files','POST',{revision,files});
      const project = async (name,paths) => (await api('/workspace/projects','POST',{name,paths,instructions:'Local project instructions'})).project.id;
      return [await project(${JSON.stringify(projectName)},Object.keys(members)),await project('Instructions only',[]),await project('Other files',['unrelated.txt'])];
    })()`);
    await fill('#draft','Unsent chat draft');
    await pointerClick('.global-nav [data-view=workbench]');
    await wait("document.querySelector('.workspace-file[title=\"docs/notes.md\"]')", 'project member available in editor');
    await pointerClick('.workspace-file[title="docs/notes.md"]');
    await wait("document.querySelector('.workspace-file-path').textContent==='docs/notes.md'", 'open saved project member');
    await fill('#workspace-editor','Uncommitted editor text');
    await selectWorkspaceSection('projects');
    const row = id => `.workspace-project-list li:has(.workspace-project-row[data-project="${id}"])`;
    const download = id => row(id)+' .workspace-project-download';
    const target = download(projectIds[0]);
    await wait(`document.querySelector(${JSON.stringify(target)})?.checkVisibility()`, 'download control on the selected project');
    assert.equal(await evaluate(`!!document.querySelector(${JSON.stringify(download(projectIds[1]))})`),false,'instructions-only project has no archive action');
    for (const [id,name] of [[projectIds[0],projectName],[projectIds[2],'Other files']]) {
      assert.deepEqual(await evaluate(`{const b=document.querySelector(${JSON.stringify(download(id))}); [b.textContent.trim(),b.getAttribute('aria-label')]}`),['Download ZIP','Download ZIP for '+name]);
    }
    await screenshot('project-download-1440.png');
    await cdp('Emulation.setDeviceMetricsOverride',{width:320,height:800,deviceScaleFactor:1,mobile:true});
    await wait("innerWidth===320 && document.documentElement.dataset.compact==='true'", 'narrow project actions');
    assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true);
    const projectControls = await evaluate(`Array.from(document.querySelectorAll('${row(projectIds[0])} button'), button => {
      const r=button.getBoundingClientRect();
      return {fits:r.left>=0 && r.right<=innerWidth && r.top>=0 && r.bottom<=innerHeight,
        hit:button.contains(document.elementFromPoint(r.left+r.width/2,r.top+r.height/2))};
    })`);
    assert.deepEqual(projectControls,[{fits:true,hit:true},{fits:true,hit:true}],'edit and download controls on a narrow screen');
    await screenshot('project-download-320.png');
    const [id] = projectIds, route = '/api/workspace/projects/'+id+'/archive';
    await evaluate(`ChatShell.api('/workspace/projects/${id}','PATCH',{version:1,name:'Release review',paths:${JSON.stringify(Object.keys(members))}})`);
    const downloadPath = path.join(path.dirname(fixture),'downloads');
    await mkdir(downloadPath);
    await cdp('Browser.setDownloadBehavior',{behavior:'allow',downloadPath});
    const responses = [];
    socket.addEventListener('message', event => {
      const {method,params} = JSON.parse(event.data);
      if (method==='Network.responseReceived' && new URL(params.response.url).pathname===route) responses.push(params.response);
    });
    const preserved = async writes => {
      assert.deepEqual(await evaluate(`[document.activeElement.matches(${JSON.stringify(target)}),document.querySelector('#workspace-editor').value,
        !document.querySelector('.unsaved-indicator').hidden,document.querySelector('#draft').value]`),[true,'Uncommitted editor text',true,'Unsent chat draft']);
      assert.equal(apiMutations.length,writes,'downloading changed server state');
    };
    const writes = apiMutations.length;
    await pointerClick(target);
    const filename = 'llmx-Release-review.zip';
    const downloaded = path.join(downloadPath,filename);
    await wait(async () => {
      try { return (await readFile(downloaded)).subarray(0,4).equals(Buffer.from('PK\x03\x04')); }
      catch (error) { if (error.code==='ENOENT') return false; throw error; }
    }, 'download under the current server project name');
    assert.equal(responses.length,1);
    assert.equal(responses[0].status,200);
    assert.equal(responses[0].mimeType,'application/zip');
    const disposition = Object.entries(responses[0].headers).find(([key])=>key.toLowerCase()==='content-disposition');
    assert.equal(disposition?.[1],'attachment; filename="'+filename+'"');
    await preserved(writes);
    await evaluate("(async()=>{const {revision}=await ChatShell.api('/workspace'); await ChatShell.api('/workspace/delete','POST',{revision,path:'src/empty.txt'});})()");
    const writesAfterRemoval = apiMutations.length;
    await pointerClick(target);
    await wait("document.querySelector('.workspace-project-status.error-text')?.innerText.includes('src/empty.txt')", 'missing selected file explained');
    assert.deepEqual(responses.map(response=>response.status),[200,409]);
    await preserved(writesAfterRemoval);
    assert.deepEqual(await readdir(downloadPath),[filename],'missing members must not download a partial ZIP');
    await writeFile(path.join(path.dirname(fixture),filename),await readFile(downloaded));
    assert.deepEqual(errors,[],'project download browser exceptions');
    console.log('PASS: a named project download uses the server filename, preserves focus and drafts, and reports a missing file without a partial archive');
  } else if (process.argv.includes('--generated-file-only')) {
    await signIn();
    const read = () => evaluate("ChatShell.api('/workspace/file?path=artifact.py')");
    const openSnippet = async () => {
      await evaluate("ChatShell.show('chat'); ChatArtifacts.open({language:'python',source:'print(42)'})");
      await pointerClick('.artifact-toolbar-actions button:first-child');
      await wait("document.querySelector('#conversation-value').value === 'artifact.py' && document.querySelector('#action-dialog').open", 'generated filename');
    };
    await evaluate(`(async () => {
      const workspace = await ChatShell.api('/workspace');
      await ChatShell.api('/workspace/file','POST',{path:'artifact.py',content:'original',revision:workspace.revision});
    })()`);
    await openSnippet();
    await pointerClick('#conversation-submit');
    await wait("document.querySelector('#action-dialog-title').textContent === 'Replace saved file?' || document.querySelector('#workspace-editor').value === 'print(42)'", 'collision decision or editor');
    if (!await evaluate("document.querySelector('#action-dialog').open")) {
      await pointerClick('#workspace-save');
      await wait("document.querySelector('#workspace-save').disabled && !document.querySelector('#workspace-editor').disabled", 'save without a collision decision');
      assert.equal((await read()).content, 'original', 'the original file was overwritten without a replacement decision');
      assert.fail('missing replacement decision');
    }
    await pointerClick('#conversation-cancel');
    assert.equal((await read()).content, 'original');
    await openSnippet();
    await pointerClick('#conversation-submit');
    await wait("document.querySelector('#action-dialog-title').textContent === 'Replace saved file?' && document.querySelector('#action-dialog').open", 'explicit replacement');
    await pointerClick('#conversation-submit');
    await wait("document.querySelector('#workspace-editor').value === 'print(42)'", 'replacement draft');
    assert.equal((await read()).content, 'original', 'opening the replacement draft wrote to storage');
    await cdp('Emulation.setDeviceMetricsOverride', {width:320,height:800,deviceScaleFactor:1,mobile:true});
    await wait("document.documentElement.dataset.compact==='true' && innerWidth===320", 'narrow file review');
    await pointerClick('#workspace-review');
    await wait("document.querySelector('#artifact-dialog').open && document.querySelector('.artifact-source code').textContent.includes('+print(42)')", 'unsaved replacement comparison');
    assert.match(await evaluate("document.querySelector('.artifact-source code').textContent"), /^-original$/m);
    assert.equal((await read()).content, 'original', 'reviewing a draft wrote to storage');
    await pointerClick('#artifact-dialog [aria-label="Close artifact"]');
    assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), 'print(42)');
    assert.equal(await evaluate("document.activeElement.id"), 'workspace-review');
    await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await wait("document.documentElement.dataset.compact==='false'", 'desktop after file review');
    await pointerClick('#workspace-save');
    await wait("document.querySelector('#workspace-save').disabled && document.querySelector('.unsaved-indicator').hidden", 'saved replacement');
    assert.equal((await read()).content, 'print(42)');
    assert.equal(await evaluate("document.querySelector('#workspace-review').hidden"), true, 'clean edits need no review');
    await openSnippet();
    await pointerClick('#conversation-submit');
    await wait("!document.querySelector('#action-dialog').open && document.activeElement.id === 'workspace-editor' && document.querySelector('#workspace-save').disabled", 'identical content opens cleanly');
    await openSnippet();
    await fill('#conversation-value', 'another.py');
    await pointerClick('#conversation-submit');
    await wait("document.querySelector('.workspace-file-path').textContent === 'another.py' && !document.querySelector('#action-dialog').open", 'unused path opens directly');
    await pointerClick('#workspace-save');
    await wait("document.querySelector('.unsaved-indicator').hidden", 'new file saved');
    assert.equal(await evaluate("(async () => (await ChatShell.api('/workspace/file?path=another.py')).content)()"), 'print(42)');
    await openSnippet();
    await fill('#conversation-value', 'race.py');
    await evaluate(`(async () => {
      const {revision} = await ChatShell.api('/workspace');
      await ChatShell.api('/workspace/file','POST',{path:'race.py',content:'other tab',revision});
    })()`);
    await pointerClick('#conversation-submit');
    await wait("document.querySelector('.workspace-file-path').textContent === 'race.py'", 'draft at raced path');
    await pointerClick('#workspace-save');
    await wait("document.querySelector('.workspace-error').checkVisibility()", 'stale creation refused');
    assert.equal(await evaluate("(async () => (await ChatShell.api('/workspace/file?path=race.py')).content)()"), 'other tab');
    const remove = '.workspace-editor-actions .danger';
    const deletes = () => apiMutations.filter(request => request.url.endsWith('/api/workspace/delete')).length;
    const beforeDiscard = deletes();
    assert.equal(await evaluate("document.querySelector('.workspace-editor-actions .danger').textContent"), 'Discard');
    await pointerClick(remove);
    await closeAction('#conversation-cancel');
    assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), 'print(42)', 'cancel kept the unsaved draft');
    await pointerClick(remove);
    await pointerClick('#conversation-submit');
    await wait("!document.querySelector('#action-dialog').open || document.querySelector('#conversation-error').textContent", 'discard outcome');
    assert.equal(deletes(), beforeDiscard, 'discarding an unsaved file sent a saved-file deletion');
    await wait("document.activeElement.textContent === 'New file'", 'focus returns to New file after discard');
    assert.equal(await evaluate("document.querySelector('#workspace-editor').disabled && document.querySelector('.workspace-error').hidden"), true);
    assert.equal(await evaluate("(async () => (await ChatShell.api('/workspace/file?path=race.py')).content)()"), 'other tab');
    await openSnippet();
    await pointerClick('#conversation-submit');
    await wait("!document.querySelector('#action-dialog').open && document.activeElement.id === 'workspace-editor'", 'existing identical file reopened');
    assert.equal(await evaluate("document.querySelector('.workspace-editor-actions .danger').textContent"), 'Delete');
    await pointerClick(remove);
    await closeAction('#conversation-submit');
    assert.equal(deletes(), beforeDiscard + 1, 'deleting the saved file sent one deletion');
    assert.equal(await evaluate("(async () => (await ChatShell.api('/workspace')).files.some(file => file.path === 'artifact.py'))()"), false);
    assert.deepEqual(errors, [], 'generated file browser errors');
    for (const [label,source,edited,expected] of [
      ['windows','one\r\ntwo\r\n','one\nchanged\nthree\n','one\r\nchanged\r\nthree\r\n'],
      ['mac','one\rtwo','one\nchanged','one\rchanged'],
      ['mixed','one\r\ntwo\nthree\rfour','one\ninserted\ntwo\nthree\nfour','one\r\ninserted\r\ntwo\nthree\rfour'],
    ]) {
      const name = label + '.txt';
      await evaluate(`(async()=>{const s=await ChatShell.api('/workspace');await ChatShell.api('/workspace/file','POST',{path:${JSON.stringify(name)},content:${JSON.stringify(source)},revision:s.revision});ChatWorkspace.openFiles();})()`);
      await wait(`document.querySelector('.workspace-file[title="${name}"]')`, 'saved line-ending fixture');
      await pointerClick('.workspace-file[title="' + name + '"]');
      await wait(`document.querySelector('.workspace-file-path').textContent===${JSON.stringify(name)}`, 'open line-ending fixture');
      assert.equal(await evaluate("document.querySelector('#workspace-save').disabled"), true, 'untouched ' + label + ' file was marked edited');
      await fill('#workspace-editor', edited);
      await pointerClick('#workspace-review');
      await wait("document.querySelector('#artifact-dialog').open", 'review line endings');
      const patch = await evaluate("document.querySelector('#artifact-dialog .artifact-source code').textContent");
      assert.ok(patch.includes(label === 'mixed' ? '+inserted\r\n' : label === 'windows' ? '+changed\r\n' : '+one\rchanged'), 'draft comparison lost ' + label + ' line endings');
      const stored = () => evaluate(`ChatShell.api('/workspace/file?path=${name}').then(f=>f.content)`);
      assert.equal(await stored(),source);
      await pointerClick('#artifact-dialog [aria-label="Close artifact"]');
      await pointerClick('#workspace-save');
      await wait("document.querySelector('#workspace-save').disabled && document.querySelector('.unsaved-indicator').hidden", 'saved line endings');
      assert.equal(await stored(),expected,label+' editor changed newline style');
    }
    console.log('PASS: saved generated file collision, cancellation and deliberate replacement');
  } else if (process.argv.includes('--history-sections-only')) {
    await signIn();
    const ids = await evaluate(`(async () => {
      const ids = [];
      for (const [title, fields] of [
        ['Draft proposal', {folder:'Research'}], ['Field notes', {folder:'Research'}],
        ['Archived notes', {folder:'Research',archived:true}], ['Inbox', {}],
        ['Manual pin folder', {folder:'Pinned'}], ['First priority', {pinned:true}],
      ]) {
        const chat = await ChatShell.api('/chats','POST',{title});
        if (Object.keys(fields).length) await ChatShell.api('/chats/'+chat.id,'PATCH',fields);
        ids.push(chat.id);
      }
      return ids;
    })()`);
    await reloadDocument();
    const section = id => '.history-section:has([data-chat="'+id+'"])';
    const research = section(ids[0]);
    await wait(`document.querySelector('${research}')?.open`, 'expanded folder');
    assert.equal(await evaluate(`document.querySelector('${research} .section-count').textContent`), '2');
    assert.equal(await evaluate(`document.querySelector('${section(ids[4])} .section-name').textContent`), 'Pinned (folder)');
    assert.equal(await evaluate(`document.querySelector('${section(ids[5])} .section-name').textContent`), 'Pinned');
    await evaluate(`document.querySelector('${research} summary').focus()`);
    await cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r'});
    await cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
    await wait(`!document.querySelector('${research}').open`, 'fold folder with keyboard');
    await fill('#search','Draft proposal');
    await wait(`document.querySelector('#chat-list > .chat-row [data-chat="${ids[0]}"]')?.checkVisibility()`, 'search in closed folder');
    await fill('#search','');
    await wait(`document.querySelector('${research}')?.open === false`, 'restore browsing state');
    await click('#archive-toggle');
    await wait(`document.querySelector('${section(ids[2])}')?.open`, 'archive has independent sections');
    await click('#archive-toggle');
    await wait(`document.querySelector('${research}')?.open === false`, 'return to folded library');
    await cdp('Emulation.setDeviceMetricsOverride',{width:320,height:700,deviceScaleFactor:1,mobile:true});
    await pointerClick('#sidebar-toggle');
    await wait("document.querySelector('#app').classList.contains('sidebar-open')", 'mobile drawer');
    await pointerClick(research+' summary');
    await wait(`document.querySelector('${research}').open`, 'open folder in mobile drawer');
    await screenshot('history-sections-mobile.png');
    console.log('PASS: keyboard and mobile folder navigation, archive and search');
  } else if (process.argv.includes('--message-find-only')) {
    await signIn();
    await checkPassageNavigation();
    const chat = await evaluate(`(async () => {
      const messages = [];
      for (let i=0;i<27;i++) {
        messages.push({id:'p'+i,parent_id:i?'a'+(i-1):null,role:'user',content:'Prompt '+i});
        messages.push({id:'a'+i,parent_id:'p'+i,role:'assistant',content:'Answer ['+i+']', reasoning:'Private reasoning marker'});
      }
      const message = id => messages.find(m => m.id === id);
      message('p12').content = '# Costs to review';
      message('a12').tool_calls = [{id:'costs',name:'echo',arguments:'## Tool argument heading',status:'completed'}];
      messages.splice(messages.indexOf(message('p13')),0,{id:'tool',parent_id:'a12',role:'tool',content:'## Tool result heading',tool_call_id:'costs'});
      message('p13').parent_id = 'tool';
      message('a13').content = '## Fixed costs\\n\\nThe monthly baseline covers the production database, backups and the minimum compute allocation.\\n\\n### Open questions\\n\\nConfirm whether the quoted rates include the standby instance.';
      message('a14').reasoning = '## Private reasoning marker';
      message('a14').content = '## Hosting options\\n\\nThe managed service covers scheduled backups. Self-hosting adds the cost of on-call support.\\n\\n### Open questions\\n\\nAsk both providers for their bandwidth and restore fees.\\n\\n### Open questions\\n\\nRecord who owns the recovery drill for each option.';
      message('a15').content = '## Launch checklist\\n\\nVerify the backup retention period and the traffic estimate before signing the contract.';
      message('a25').content += '\\n\\n~~~md\\n## Fenced heading\\n~~~';
      messages.find(m=>m.id==='a24').content = 'First paragraph.\\n\\n~~~python\\nprint(24)\\n#' + 'x'.repeat(240) + '\\n~~~\\n\\nFinal paragraph.';
      const questions = ['Which items affect the release budget?', 'Compare the hosting costs for the two deployment options.',
        'Summarize the remaining assumptions to verify before launch and list the evidence we would need to revise the estimates.'];
      questions.forEach((content,i) => { messages.find(message => message.id==='p'+(i+13)).content = content; });
      const upload = name => ({name,type:'text/plain',text:'Private attachment body'});
      Object.assign(messages.find(m=>m.id==='p16'), {content:'', attachments:[upload('summary.md'), upload('budget[1].csv')]});
      messages.find(m=>m.id==='p22').attachments = [upload('notes.txt'), upload('notes.md')];
      messages.push({id:'other',parent_id:'p26',role:'assistant',content:'## Alternate branch marker'});
      messages.push({id:'hidden-prompt',parent_id:'other',role:'user',content:'Inactive prompt branch'});
      return ChatShell.api('/import','POST',{version:1,chat:{title:'Navigation fixture',active_leaf:'a26'},messages});
    })()`);
    const plainChat = await evaluate(`ChatShell.api('/import','POST',{version:1,chat:{title:'Plain reply',active_leaf:'a'},
      messages:[{id:'p',role:'user',content:'Plain prompt',attachments:[{name:'estuary photo.png',type:'image/png',
        data:'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aBy0AAAAASUVORK5CYII='}]},
        {id:'a',parent_id:'p',role:'assistant',content:'Plain reply'}]})`);
    await reloadDocument();
    await wait(`document.querySelector('[data-chat="${chat.id}"]')`, 'imported conversation');
    await pointerClick(`[data-chat="${plainChat.id}"]`);
    await wait("document.querySelector('.message.assistant')?.textContent.includes('Plain reply')", 'plain conversation');
    const contents = '#conversation-contents[data-action=outline]';
    const key = async (key, code) => {
      for (const type of ['keyDown','keyUp']) await cdp('Input.dispatchKeyEvent',
        {type,key,code:key,windowsVirtualKeyCode:code,text:type==='keyDown' && key==='Enter' ? '\r' : ''});
    };
    const accessible = async selector => {
      const {root} = await cdp('DOM.getDocument');
      const {nodeId} = await cdp('DOM.querySelector',{nodeId:root.nodeId,selector});
      const {nodes} = await cdp('Accessibility.getPartialAXTree',{nodeId,fetchRelatives:false});
      return nodes[0];
    };
    const closeContents = async () => {
      await key('Escape',27);
      await wait("!document.querySelector('#find-message-dialog').open && document.activeElement.id==='conversation-contents'", 'Escape returns to Contents');
    };
    const mode = async (selector,pressed,count) => {
      const button = await accessible(selector);
      assert.equal(button.role.value,'button');
      assert.equal(await evaluate(`document.querySelector(${JSON.stringify(selector)}).getAttribute('aria-pressed')`),String(pressed));
      assert([button.name?.value,button.description?.value].join(' ').split(/\D+/).includes(String(count)), 'accessible mode count: '+button.name?.value);
    };
    const plainWrites = apiMutations.length;
    await pointerClick(contents);
    await pointerClick('#outline-headings');
    assert.equal(await evaluate("document.querySelector('#find-message-list [data-target]')"), null);
    assert.match(await evaluate("document.querySelector('#find-message-dialog').innerText"), /no (?:reply )?headings/i);
    assert.equal(await evaluate("document.querySelector('#outline-headings').getAttribute('aria-pressed')"),'true');
    await closeContents();
    await pointerClick('#reading-navigation [data-action=find-message]');
    await fill('#find-message-query','estuary photo');
    await pointerClick('#find-message-list [data-target]');
    assert.equal(await evaluate(`{const image=document.querySelector('img[alt="estuary photo.png"]'),style=getComputedStyle(image);
      document.querySelector('#passage-position').textContent==='1 of 1' && image.naturalWidth===1 && parseFloat(style.outlineWidth)>0 && style.outlineStyle!=='none'}`),true,'image filename match visibly outlines the attachment');
    await pointerClick('#passage-close');
    assert.equal(apiMutations.length, plainWrites, 'navigating headings and image filenames changed the chat');
    await pointerClick(`[data-chat="${chat.id}"]`);
    await wait("document.querySelectorAll('.message.user').length===27", 'selected branch');
    await fill('#draft', 'Unsent draft');
    await cdp('Emulation.setDeviceMetricsOverride', {width:320,height:360,deviceScaleFactor:1,mobile:true});
    await wait("document.documentElement.dataset.compact==='true' && innerWidth===320", 'short phone transcript');
    const setBottomDistance = distance => evaluate(`new Promise(resolve=>{const r=document.querySelector('#scroll-region');r.scrollTop=r.scrollHeight-r.clientHeight-${distance};requestAnimationFrame(()=>requestAnimationFrame(resolve));})`);
    const jumpState = () => evaluate("({available:document.querySelector('#jump-bottom').checkVisibility(),composerTop:document.querySelector('#composer').getBoundingClientRect().top})");
    await setBottomDistance(0);
    const atLatest = await jumpState();
    assert.equal(atLatest.available, false);
    await setBottomDistance(150);
    assert.equal((await jumpState()).available, true, 'reader outside the auto-follow range has no way to jump');
    await setBottomDistance(240);
    assert.equal((await jumpState()).composerTop, atLatest.composerTop, 'Jump to latest moved the composer');
    assert.equal(await evaluate("Math.round(document.querySelector('#jump-bottom').getBoundingClientRect().right-document.querySelector('#composer').getBoundingClientRect().right)"), 0, 'jump button and composer should align');
    await pointerClick('#jump-bottom');
    await wait("document.querySelector('#jump-bottom').hidden", 'jumped back to latest');
    assert.equal(await evaluate("(()=>{const r=document.querySelector('#scroll-region');return r.scrollHeight-r.scrollTop-r.clientHeight<2;})()"), true);
    assert.equal(await evaluate("document.querySelector('#draft').value"), 'Unsent draft');
    await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await wait("document.documentElement.dataset.compact==='false'", 'return to desktop transcript');
    const writes = apiMutations.length;
    await evaluate(`{
      const pane = document.querySelector('.chat-stage');
      const transfer = new DataTransfer();
      transfer.items.add(new File(['first'], 'one.txt', {type:'text/plain'}));
      transfer.items.add(new File(['second'], 'two.md', {type:'text/markdown'}));
      pane.dispatchEvent(new DragEvent('dragover', {dataTransfer:transfer, cancelable:true}));
      if (!document.querySelector('#file-drop-hint').checkVisibility()) throw Error('No file drop feedback');
      pane.dispatchEvent(new DragEvent('drop', {dataTransfer:transfer, cancelable:true}));
      const text = new DataTransfer();
      text.setData('text/plain','ordinary dragged text');
      if (!pane.dispatchEvent(new DragEvent('drop',{dataTransfer:text,cancelable:true}))) throw Error('Text drop intercepted');
    }`);
    await wait("document.querySelectorAll('#attachments .attachment-pill').length === 2", 'both dropped files attached');
    assert.equal(await evaluate("document.querySelector('#file-drop-hint').hidden && document.querySelector('#draft').value === 'Unsent draft'"), true);
    await checkPromptOutline();
    const ids = await evaluate("Array.from(document.querySelectorAll('#messages .message.assistant'), m => m.dataset.message)");
    const expected = [
      [ids[13],0,'Fixed costs'], [ids[13],1,'Open questions'], [ids[14],0,'Hosting options'],
      [ids[14],1,'Open questions'], [ids[14],2,'Open questions'], [ids[15],0,'Launch checklist'],
    ];
    const duplicateRow = index => `#find-message-list [data-target="${ids[14]}"][data-heading="${index}"]`;
    for (const [width,height] of [[1440,1000],[320,800]]) {
      await cdp('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width===320});
      await wait(`innerWidth===${width} && document.documentElement.dataset.compact==='${width===320}'`, 'contents viewport');
      const title = await accessible(contents);
      assert.equal(title.role.value,'button');
      assert.match(title.name.value,/contents|outline/i);
      const header = await evaluate(`(() => {
        const visible = Array.from(document.querySelectorAll('.topbar button,.topbar #connection,#reading-navigation button,#reading-position')).filter(el => el.checkVisibility());
        return visible.map(el => {
          const {left,right,top,bottom,width,height} = el.getBoundingClientRect();
          return {name:el.id || el.dataset.action,left,right,top,bottom,hit:el.contains(document.elementFromPoint(left+width/2,top+height/2))};
        });
      })()`);
      assert(header.some(control=>control.name==='conversation-contents'), 'missing visible Contents control');
      for (const [index,control] of header.entries()) {
        assert(control.hit && control.left>=0 && control.right<=width, 'clipped header control: '+JSON.stringify(control));
        for (const other of header.slice(index+1)) {
          assert(control.right<=other.left || other.right<=control.left || control.bottom<=other.top || other.bottom<=control.top,
            'overlapping header controls: '+JSON.stringify([control,other]));
        }
      }
      for (const theme of ['light','dark']) {
        await cdp('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:theme}]});
        await wait(`document.documentElement.dataset.theme==='${theme}'`, 'contents theme');
        const current = await evaluate(`{const p=document.querySelectorAll('#messages .message.user')[14];
          p.scrollIntoView({block:'center',behavior:'instant'}); p.dataset.message}`);
        await wait("document.querySelector('#reading-position').textContent==='Turn 15 of 27'", 'reading position after scrolling');
        await screenshot(`contents-chat-${width}-${theme}.png`);
        if (theme==='dark') {
          await evaluate("document.querySelector('#conversation-contents').focus()");
          await key('Enter',13);
        } else await pointerClick(contents);
        await wait(`document.activeElement.dataset.target==='${current}' && document.activeElement.getAttribute('aria-current')==='location'`, 'current prompt from Contents');
        await mode('#outline-turns',true,27);
        await mode('#outline-headings',false,6);
        await screenshot(`contents-turns-${width}-${theme}.png`);
        await pointerClick('#outline-headings');
        await mode('#outline-turns',false,27);
        await mode('#outline-headings',true,6);
        const rows = await evaluate("Array.from(document.querySelectorAll('#find-message-list [data-target]'), row=>[row.dataset.target,Number(row.dataset.heading),row.innerText])");
        assert.deepEqual(rows.map(([id,index])=>[id,index]),expected.map(([id,index])=>[id,index]), 'only active-branch reply headings, in reading order');
        rows.forEach((row,index)=>assert(row[2].includes(expected[index][2]), 'heading label: '+row[2]));
        for (const [index,label] of [[1,/1 of 2\s*\u00b7\s*Ask both providers/],[2,/2 of 2\s*\u00b7\s*Record who owns/]]) {
          assert.match(rows[index+2][2],label,'visible duplicate heading label');
          assert.match((await accessible(duplicateRow(index))).name.value,label,'accessible duplicate heading label');
        }
        assert.equal(await evaluate("document.querySelector('#find-message-list [aria-current=location]')"),null,'a prompt cannot mark an earlier reply heading as current');
        assert.equal(await evaluate("document.documentElement.scrollWidth<=innerWidth && document.querySelector('#find-message-dialog').getBoundingClientRect().right<=innerWidth"),true);
        await screenshot(`contents-headings-${width}-${theme}.png`);
        await closeContents();
      }
    }
    for (const index of [1,2]) {
      await evaluate("document.querySelector('#scroll-region').scrollTop=0");
      await wait("document.querySelector('#reading-position').textContent==='Turn 1 of 27'", 'start before jumping to a distant heading');
      await pointerClick(contents);
      await pointerClick('#outline-headings');
      await pointerClick(duplicateRow(index));
      assert.equal(await evaluate(`(() => {
        const heading = document.activeElement, reply = document.querySelector('[data-message="${ids[14]}"]');
        const bounds = heading.getBoundingClientRect(), region = document.querySelector('#scroll-region').getBoundingClientRect();
        return !document.querySelector('#find-message-dialog').open && heading.textContent==='Open questions' &&
          heading===reply.querySelectorAll(':scope > .markdown :is(h1,h2,h3,h4,h5,h6)')[${index}] && bounds.top>=region.top && bounds.bottom<=region.bottom;
      })()`),true,'jump to the correct identically named heading '+index);
    }
    await evaluate("document.activeElement.scrollIntoView({block:'center',behavior:'instant'})");
    await pointerClick(contents);
    await pointerClick('#outline-headings');
    assert.equal(await evaluate(`document.querySelector('#find-message-list [aria-current=location]')?.matches('[data-target="${ids[14]}"][data-heading="2"]')`),true,'exact current heading within a reply');
    assert.equal(await evaluate('document.activeElement.id'),'outline-headings');
    await key('Tab',9);
    assert.equal(await evaluate("document.activeElement.matches('#find-message-list [data-heading]')"),true);
    await key('End',35);
    await key('Enter',13);
    assert.equal(await evaluate(`document.activeElement.closest('article')?.dataset.message==='${ids[15]}' && document.activeElement.textContent==='Launch checklist'`),true);
    await pointerClick(contents);
    await pointerClick('#outline-headings');
    await evaluate(`{
      const original=document.querySelector('[data-message="${ids[15]}"] > .markdown :is(h2,h3,h4,h5,h6)');
      const updated=original.cloneNode(); updated.textContent='Revised checklist'; original.replaceWith(updated);
    }`);
    await pointerClick(`#find-message-list [data-target="${ids[15]}"][data-heading="0"]`);
    assert.equal(await evaluate(`document.activeElement.matches('article[data-message="${ids[15]}"]') && !document.querySelector('#find-message-dialog').open`),true,'stale heading falls back to its reply');
    assert.match(await evaluate("document.querySelector('#toast').innerText"),/heading changed/i);
    assert.equal(await evaluate("document.querySelector('#draft').value==='Unsent draft' && document.querySelectorAll('#attachments .attachment-pill').length===2"),true);
    assert.equal(apiMutations.length,writes,'navigating reply headings changed server state');
    await pointerClick('#reading-navigation [data-action=find-message]');
    await wait("document.activeElement.id==='find-message-query'", 'direct Find focuses search');
    await key('Escape',27);
    await wait("document.activeElement.matches('#reading-navigation [data-action=find-message]')", 'Escape returns to Find');
    await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await wait("document.documentElement.dataset.compact==='false'", 'desktop after heading navigation');
    await pointerClick('.topbar [data-action=chat-menu]');
    await pointerClick('#action-menu [data-action=find-message]');
    assert.equal(await evaluate("document.activeElement.id"), 'find-message-query');
    assert.equal(await evaluate("document.querySelectorAll('.find-message-result').length"), 25);
    assert.equal(await evaluate("document.querySelector('#find-message-count').textContent"), '54 messages, newest first');
    const speaker = value => evaluate(`{ const field=document.querySelector('#find-message-role'); field.value=${JSON.stringify(value)}; field.dispatchEvent(new Event('change')); }`);
    await speaker('user');
    assert.match(await evaluate("document.querySelector('#find-message-list').innerText"), /summary\.md/);
    await fill('#find-message-query','notes.');
    assert.equal(await evaluate("document.querySelectorAll('.find-message-result').length"), 1, 'two matching filenames on one prompt');
    await speaker('assistant');
    assert.equal(await evaluate("document.querySelectorAll('.find-message-result').length"), 0);
    await speaker('user');
    await fill('#find-message-query','BUDGET[1]');
    assert.equal(await evaluate("document.querySelector('.find-message-result mark')?.textContent"), 'budget[1]');
    await pointerClick('.find-message-result');
    assert.equal(await evaluate("document.activeElement.matches('article.message.user') && !!document.activeElement.querySelector('[aria-label=\"Read attachment budget[1].csv\"]')"), true, 'focused the file-only prompt');
    await pointerClick('.topbar [data-action=chat-menu]');
    await pointerClick('#action-menu [data-action=find-message]');
    await speaker('user');
    await pointerClick('#find-message-more');
    assert.equal(await evaluate("document.querySelectorAll('.find-message-result').length"), 27);
    assert.match(await evaluate("document.activeElement.textContent"), /Prompt 1$/);
    for (const query of ['Private reasoning marker','Private attachment body','Alternate branch marker','.*']) {
      await fill('#find-message-query', query);
      assert.equal(await evaluate("document.querySelector('#find-message-count').textContent"), 'No matching messages');
    }
    await fill('#find-message-query','ANSWER [3]');
    assert.equal(await evaluate("document.querySelectorAll('.find-message-result').length"), 0);
    await speaker('assistant');
    assert.equal(await evaluate("document.querySelector('.find-message-result mark').textContent"), 'Answer [3]');
    await pointerClick('.find-message-result');
    assert.equal(await evaluate("document.activeElement.matches('article.message.assistant') && document.activeElement.innerText.includes('Answer [3]')"), true);
    assert.equal(await evaluate("document.querySelector('#draft').value"), 'Unsent draft');
    await cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'F',code:'KeyF',modifiers:10,windowsVirtualKeyCode:70});
    await wait("document.querySelector('#find-message-dialog').open", 'conversation find shortcut');
    assert.equal(await evaluate("document.querySelector('#find-message-role').value"), 'all');
    await pointerClick('#find-message-dialog button[type=submit]');
    await evaluate(`(async () => {
      const open = () => {
        document.querySelector('.topbar [data-action=chat-menu]').click();
        document.querySelector('#action-menu [data-action=find-message]').click();
      };
      open();
      const dialog = document.querySelector('#find-message-dialog');
      const closed = new Promise(resolve => dialog.addEventListener('close', resolve, {once:true}));
      dialog.close();
      open();
      await closed;
    })()`);
    await fill('#find-message-query', 'answer [0]');
    assert.equal(await evaluate("document.querySelector('.find-message-result mark')?.textContent"), 'Answer [0]', 'a previous close event cleared the reopened search');
    await cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
    await cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
    assert.equal(await evaluate("document.activeElement.innerText.includes('Answer [0]') && !document.querySelector('#find-message-dialog').open"), true);
    const downloadPath = path.join(profile, 'downloads');
    await mkdir(downloadPath, {recursive:true});
    await cdp('Browser.setDownloadBehavior', {behavior:'allow', downloadPath});
    await pointerClick('.topbar [data-action=chat-menu]');
    await pointerClick('#action-menu [data-action=transcript]');
    let transcript;
    for (let i = 0; i < 50 && !transcript; i++) {
      try { transcript = await readFile(path.join(downloadPath, 'Navigation fixture.md'), 'utf8'); }
      catch (error) { if (error.code !== 'ENOENT') throw error; await delay(100); }
    }
    assert.match(transcript, /^# Navigation fixture/);
    assert.match(transcript, /## You\n\nPrompt 0\n\n## Assistant\n\nAnswer \[0\]/);
    assert.doesNotMatch(transcript, /Private reasoning marker|Private attachment body|budget\[1\]|summary\.md|Alternate branch marker|Unsent draft/);
    assert.equal(apiMutations.length, writes, 'search changed server state');
    const quoteTarget = '.message.assistant:last-of-type [data-action=quote]';
    await evaluate(`(() => {
      const answer = document.querySelector('.message.assistant:last-of-type > .markdown p').firstChild;
      const range = document.createRange();
      range.setStart(answer, 0); range.setEnd(answer, 6);
      window.getSelection().removeAllRanges(); window.getSelection().addRange(range);
    })()`);
    await pointerClick(quoteTarget);
    assert.equal(await evaluate("document.querySelector('#draft').value"), 'Unsent draft\n\n> Answer\n\n');
    assert.equal(await evaluate('document.activeElement.id'), 'draft');
    await evaluate('window.getSelection().removeAllRanges()');
    await pointerClick(quoteTarget);
    assert.equal(await evaluate("document.querySelector('#draft').value.endsWith('> Answer [26]\\n\\n')"), true);
    assert.equal(await evaluate("document.querySelectorAll('#attachments .attachment-pill').length"), 2);
    await fill('#draft', '');
    await evaluate(`(() => {
      const body = document.querySelector('.message.assistant .code-block').parentElement;
      const range = document.createRange(); range.selectNodeContents(body);
      window.getSelection().removeAllRanges(); window.getSelection().addRange(range);
    })()`);
    await pointerClick('.message.assistant:has(.code-block) [data-action=quote]');
    const excerpt = await evaluate("document.querySelector('#draft').value");
    assert.match(excerpt, /First paragraph\.[\s\S]*print\(24\)[\s\S]*Final paragraph\./);
    assert.doesNotMatch(excerpt, /Download|Copy code|Open|Private reasoning/);
    assert.equal(apiMutations.length, writes, 'quoting must only change the draft');
    await pointerClick('.code-heading button:last-child');
    await cdp('Emulation.setDeviceMetricsOverride', {width:320,height:800,deviceScaleFactor:1,mobile:true});
    const wrap = '#artifact-dialog button[aria-pressed]';
    const expectedCode = 'print(24)\n#' + 'x'.repeat(240);
    const overflows = "(() => {const panel=document.querySelector('.artifact-source');return panel.scrollWidth > panel.clientWidth + 1})()";
    assert.equal(await evaluate(overflows), true, 'long source line overflows before wrapping');
    assert.equal(await evaluate("document.querySelector('#artifact-dialog pre code').textContent"), expectedCode);
    assert.equal(await evaluate("getComputedStyle(document.querySelector('.artifact-source pre')).whiteSpace"), 'pre');
    await pointerClick(wrap);
    assert.equal(await evaluate("getComputedStyle(document.querySelector('.artifact-source pre')).whiteSpace"), 'pre-wrap');
    assert.equal(await evaluate(overflows), false, 'wrapped code fits the phone screen');
    assert.equal(await evaluate("document.querySelector('#artifact-dialog pre code').textContent"), expectedCode);
    await pointerClick(wrap);
    assert.equal(await evaluate("getComputedStyle(document.querySelector('.artifact-source pre')).whiteSpace"), 'pre');
    assert.deepEqual(errors, [], 'message navigation browser errors');
    console.log('PASS: streamed and rendered passages, scoped navigation, draft reflow, transcript layout, turn and heading outlines, find and export');
  } else if (process.argv.includes('--workspace-import-only')) {
    await signIn();
    await checkWorkspaceImport();
    assert.deepEqual(errors, [], 'workspace import browser exceptions');
    console.log('PASS: reviewed native multi-file and folder import, atomic retry, byte preservation, editor conflicts and responsive keyboard controls');
  } else if (process.argv.includes('--landing-drafts-only')) {
    await signIn(); await checkLandingDrafts();
    assert.deepEqual(errors,[],'ordinary landing browser exceptions');
    console.log('PASS: resume saved text and attachment-only drafts after reload, preserve home draft, archive and temporary isolation, mobile recent navigation');
  } else if (homeContextMode) {
    await signIn(); await checkLandingContext();
    assert.deepEqual(errors,[], 'home context browser exceptions');
    console.log('PASS: saved context from home, project creation, file selection, missing file, review, deduplication and preserved draft');
  } else if (process.argv.includes('--workflow-only')) {
    await signIn();
    await send('Browser workflow prompt');
    await idle();
    await checkHomeLanding();
    await checkNewChatShortcut();
    await checkWorkspaceFlow();
    assert.deepEqual(errors, [], 'workflow browser exceptions');
    console.log('PASS: keyboard Home, Workspace Sections, narrow Attach menu, saved file confirmation and deduplication, and saved project entry from Chat');
  } else if (featuresMode) {
    await signIn();
    await checkOptionalFeatures();
  } else if (pdfMode) {
    await signIn();
    await checkPdfAttachment();
  } else if (process.argv.includes('--nav-only')) {
    await wait("window.ChatShell && window.ChatShell.config!==null && document.querySelector('#connection').textContent!=='Connecting'", 'live studio ready');
    for (const [width,height] of [[1440,1000],[1024,768],[768,1024],[390,844],[360,800],[844,390],[720,500]]) {
      await cdp('Emulation.setDeviceMetricsOverride', {width,height,deviceScaleFactor:width===720?2:1,mobile:width<=390});
      await evaluate("window.dispatchEvent(new Event('resize'));document.querySelector('[data-action=\"close-drawers\"]').click()");
      await delay(240);
      for(const section of ['monitor','workbench','preferences','chat']) {
        if(width<=720){await pointerClick('.mobile-nav');await delay(230);}
        await pointerClick('.global-nav [data-view="'+section+'"]');
        await wait(`document.querySelector('#app').dataset.view===${JSON.stringify(section)}`, 'real pointer navigation '+section+' at '+width);
        assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true,'live navigation overflow at '+width);
      }
      if(width<=720){await pointerClick('.mobile-nav');await delay(230);}
      await screenshot('live-sidebar-'+width+'x'+height+'.png');
      await evaluate("document.querySelector('[data-action=\"close-drawers\"]').click();window.ChatShell.show('monitor')");
      await screenshot('live-monitor-'+width+'x'+height+'.png');
      await evaluate("window.ChatShell.show('workbench')");
      await wait("/^Revision \\d+$/.test(document.querySelector('.workspace-revision').textContent)",'live workspace');
      await screenshot('live-workspace-'+width+'x'+height+'.png');
      console.log('PASS actual sidebar pointer clicks: '+width+'x'+height);
    }
    assert.deepEqual(errors,[],'live sidebar browser exceptions');
    await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await evaluate("document.documentElement.style.zoom='2';window.dispatchEvent(new Event('resize'));document.querySelector('[data-action=\"close-drawers\"]').click()");
    for(const section of ['chat','monitor','workbench','preferences']){await evaluate(`window.ChatShell.show(${JSON.stringify(section)})`);assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true,'live CSS200% zoom overflow '+section);}
    await pointerClick('.mobile-nav');await delay(230);await pointerClick('.global-nav [data-view="workbench"]');
    await screenshot('live-workspace-200percent.png');
  } else {
  await wait("document.querySelector('#login-dialog')?.open", 'password dialog');
  await fill('#password', 'wrong'); await evaluate("document.querySelector('#login-form').requestSubmit()");
  await wait("document.querySelector('#login-error').textContent.includes('Incorrect')", 'invalid-password error');
  await signIn();
  await cdp('Page.setInterceptFileChooserDialog', {enabled: true});
  await attachmentSource('device');
  await wait(() => attachmentChooser !== null, 'native attachment chooser');
  await cdp('Page.setInterceptFileChooserDialog', {enabled: false});
  const {node: chooserInput} = await cdp('DOM.describeNode', {backendNodeId: attachmentChooser.backendNodeId});
  assert.equal(attachmentChooser.mode, 'selectMultiple', 'attachment chooser disabled multiple files');
  assert.equal(chooserInput.attributes[chooserInput.attributes.indexOf('id') + 1], 'file-input');
  const chooserHints = chooserInput.attributes[chooserInput.attributes.indexOf('accept') + 1].split(',');
  assert.deepEqual(['.go', '.rs', '.sh'].filter(extension => !chooserHints.includes(extension)), [], 'chooser omitted accepted UTF-8 source files');
  const attachmentHint = `{
    const attach=document.querySelector('[data-action=attach]'), tip=document.getElementById(attach.getAttribute('aria-describedby'));
    tip?.role==='tooltip' && tip.textContent.length>attach.getAttribute('aria-label').length && tip.checkVisibility();
  }`;
  await evaluate("document.querySelector('#draft').focus()");
  const attachPoint = await actionHoverPoint('[data-action=attach]');
  assert(attachPoint.hit, 'attachment control was obscured');
  await cdp('Input.dispatchMouseEvent', {type:'mouseMoved', x:attachPoint.x, y:attachPoint.y});
  await wait(attachmentHint, 'descriptive attachment hover tooltip');
  await screenshot('attachment-guidance-desktop-dark.png');
  await cdp('Input.dispatchMouseEvent', {type:'mouseMoved', x:10, y:90});
  await cdp('Input.dispatchKeyEvent', {type:'keyDown', key:'Tab', code:'Tab', windowsVirtualKeyCode:9});
  await cdp('Input.dispatchKeyEvent', {type:'keyUp', key:'Tab', code:'Tab', windowsVirtualKeyCode:9});
  await wait("document.activeElement.matches('[data-action=attach]')", 'keyboard focus reaches attachment control');
  await wait(attachmentHint, 'descriptive attachment keyboard tooltip');
  await cdp('Input.dispatchKeyEvent', {type:'keyDown', key:'Escape', code:'Escape', windowsVirtualKeyCode:27});
  await cdp('Input.dispatchKeyEvent', {type:'keyUp', key:'Escape', code:'Escape', windowsVirtualKeyCode:27});
  assert.equal(await evaluate("document.activeElement.matches('[data-action=attach]') && !document.activeElement.hasAttribute('aria-describedby')"), true, 'Escape did not dismiss the hint while preserving attachment focus');
  await cdp('Emulation.setDeviceMetricsOverride', {width:320,height:844,deviceScaleFactor:1,mobile:true});
  await wait("document.documentElement.dataset.compact==='true'", 'narrow attachment layout');
  await cdp('Emulation.setTouchEmulationEnabled', {enabled:true,maxTouchPoints:1});
  assert.equal(await evaluate("{const r=document.querySelector('[data-action=attach]').getBoundingClientRect();r.width>=40 && r.height>=40 && document.documentElement.scrollWidth<=innerWidth}"), true, 'attachment target or composer does not fit a narrow touch viewport');
  const settingsPoint = await evaluate("{const b=document.querySelector('[data-action=settings]').getBoundingClientRect();({x:b.x+b.width/2,y:b.y+b.height/2})}");
  await cdp('Input.dispatchTouchEvent', {type:'touchStart',touchPoints:[settingsPoint]});
  await cdp('Input.dispatchTouchEvent', {type:'touchEnd',touchPoints:[]});
  await wait("!document.querySelector('#panel-settings').hidden && document.querySelector('#attachment-help').checkVisibility()", 'attachment information available through touch Settings');
  await evaluate("document.documentElement.dataset.theme='light';document.querySelector('#attachment-help').scrollIntoView({block:'nearest'})");
  assert.equal(await evaluate("{const r=document.querySelector('#attachment-help').getBoundingClientRect(),p=document.querySelector('#panel-settings').getBoundingClientRect();r.left>=p.left && r.right<=p.right && r.top>=p.top && r.bottom<=p.bottom}"), true, 'attachment help clips in narrow Settings');
  await screenshot('attachment-guidance-mobile-light.png');
  await click('#inspector [data-action=inspector]');
  await cdp('Emulation.setTouchEmulationEnabled', {enabled:false});
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await wait("document.documentElement.dataset.compact==='false' && document.querySelector('#drawer-shade').hidden", 'restored desktop attachment layout');
  await evaluate("document.documentElement.dataset.theme='dark'");
  const sidebarTargets = ['#archive-toggle','.footer-actions [data-action=import]','#theme-button','#logout-button'];
  const composerTargets = ['.composer-left [data-action=settings]'];
  const assertTargets = async (selectors, size, label) => {
    const measurements = await evaluate(`(() => {
      const boxes=${JSON.stringify(selectors)}.map(selector=>{
        const element=document.querySelector(selector), rect=element.getBoundingClientRect();
        const x=rect.x+rect.width/2, y=rect.y+rect.height/2;
        const hits=[[x,rect.top+1],[x,rect.bottom-1],[rect.left+1,y],[rect.right-1,y]]
          .every(([left,top])=>element.contains(document.elementFromPoint(left,top)));
        return {selector,w:rect.width,h:rect.height,hits};
      });
      return {boxes,overflow:document.documentElement.scrollWidth>innerWidth};
    })()`);
    assert(measurements.boxes.every(box=>box.w>=size && box.h>=size && box.hits) && !measurements.overflow,
      label+' hit areas require '+size+' CSS px and must remain unoccluded: '+JSON.stringify(measurements));
  };
  await assertTargets([...sidebarTargets,...composerTargets],24,'desktop');
  await cdp('Emulation.setTouchEmulationEnabled', {enabled:true,maxTouchPoints:1});
  await wait("matchMedia('(pointer:coarse)').matches", 'desktop coarse pointer');
  await assertTargets([...sidebarTargets,...composerTargets],40,'desktop coarse pointer');
  await cdp('Emulation.setTouchEmulationEnabled', {enabled:false});
  for (const width of [320,390]) {
    await cdp('Emulation.setDeviceMetricsOverride', {width,height:844,deviceScaleFactor:1,mobile:true});
    await wait("document.documentElement.dataset.compact==='true' && innerWidth==="+width, 'compact target test viewport');
    await assertTargets(composerTargets,40,'compact composer at '+width);
    await pointerClick('#sidebar-toggle');
    await wait("document.querySelector('#sidebar').getBoundingClientRect().left>=0 && document.querySelector('#sidebar').getAttribute('aria-modal')==='true'", 'open sidebar for target measurement');
    await assertTargets(sidebarTargets,40,'compact sidebar at '+width);
    await pointerClick('#sidebar-close');
  }
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await wait("document.documentElement.dataset.compact==='false' && getComputedStyle(document.querySelector('#sidebar')).visibility==='visible'", 'restore desktop after target measurement');
  for (const theme of ['dark', 'light']) {
    await evaluate(`document.documentElement.dataset.theme=${JSON.stringify(theme)}`);
    const point = await evaluate("(()=>{const r=document.querySelector('.global-nav [data-view=monitor]').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()");
    await cdp('Input.dispatchMouseEvent', {type:'mouseMoved', ...point});
    await delay(180);
    await screenshot('navigation-hover-' + theme + '.png');
    const navigation = await evaluate("(()=>{const selected=document.querySelector('.global-nav [aria-current=page]'), hovered=document.querySelector('.global-nav button:hover');return {selected:selected.dataset.view,hovered:hovered?.dataset.view,selectedFill:getComputedStyle(selected).backgroundColor,hoveredFill:getComputedStyle(hovered).backgroundColor};})()");
    assert.equal(navigation.selected, 'chat'); assert.equal(navigation.hovered, 'monitor');
    assert.notEqual(navigation.selectedFill, navigation.hoveredFill, theme + ' navigation hover looks selected');
  }
  await evaluate("document.documentElement.dataset.theme='dark'");
  assert.equal(await evaluate("document.querySelector('#temporary-button')?.closest('.topbar-actions')!==null"), true, 'Temporary chat is not in the conversation header');
  const detailsToggle = '.topbar [data-action="inspector"]';
  const detailsOpen = () => evaluate("getComputedStyle(document.querySelector('#inspector')).display!=='none'");
  const inspectorPreference = {defaultOpen: await detailsOpen()};
  await screenshot('reading-default-1440-dark.png');
  if (!await detailsOpen()) await click(detailsToggle);
  await reloadDocument();
  inspectorPreference.rememberedOpen = await detailsOpen();
  await cdp('Emulation.setDeviceMetricsOverride', {width:390,height:844,deviceScaleFactor:1,mobile:true});
  await reloadDocument();
  inspectorPreference.mobileOpen = await detailsOpen();
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await reloadDocument();
  inspectorPreference.restoredOpen = await detailsOpen();
  if (await detailsOpen()) await click(detailsToggle);
  await reloadDocument();
  inspectorPreference.rememberedClosed = !await detailsOpen();
  await click('[data-action="settings"]');
  inspectorPreference.settingsOpened = await evaluate("!document.querySelector('#panel-settings').hidden && getComputedStyle(document.querySelector('#inspector')).display!=='none'");
  await reloadDocument();
  inspectorPreference.closedAfterSettings = !await detailsOpen();
  await evaluate("localStorage.removeItem('llmx-theme')");
  await cdp('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: 'light'}]});
  await reloadDocument();
  await wait("!document.querySelector('#login-dialog').open && document.querySelector('#model').value==='qwen3-local' && !document.querySelector('[data-action=\"new\"]').disabled", 'authenticated workspace after theme reload');
  assert.equal(await evaluate('document.documentElement.dataset.theme'), 'light', 'unset theme did not use light system preference');
  assert.equal(await evaluate("document.querySelector('#theme-button').textContent"), 'Dark theme', 'theme button label did not follow system preference');
  await cdp('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: 'dark'}]});
  await wait("document.documentElement.dataset.theme==='dark'", 'unset theme follows system preference change');
  assert.equal(await evaluate("document.querySelector('#theme-button').textContent"), 'Light theme', 'theme button label did not follow system change');
  await click('#theme-button');
  assert.equal(await evaluate('document.documentElement.dataset.theme'), 'light', 'explicit theme toggle did not win');
  await cdp('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: 'dark'}]});
  assert.equal(await evaluate('document.documentElement.dataset.theme'), 'light', 'system change overrode explicit theme');
  await reloadDocument();
  await wait("!document.querySelector('#login-dialog').open && document.querySelector('#model').value==='qwen3-local' && !document.querySelector('[data-action=\"new\"]').disabled", 'saved theme after reload');
  assert.equal(await evaluate('document.documentElement.dataset.theme'), 'light', 'saved explicit theme did not persist');
  assert.equal(await evaluate("document.querySelector('#theme-button').textContent"), 'Dark theme', 'saved theme button label is wrong');
  await click('#theme-button');
  assert.equal(await evaluate('document.documentElement.dataset.theme'), 'dark', 'explicit dark theme toggle did not win');
  await cdp('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: 'light'}]});
  await reloadDocument();
  await wait("!document.querySelector('#login-dialog').open && document.querySelector('#model').value==='qwen3-local' && !document.querySelector('[data-action=\"new\"]').disabled", 'saved dark theme after reload');
  assert.equal(await evaluate('document.documentElement.dataset.theme'), 'dark', 'saved dark theme did not persist');
  await evaluate("localStorage.removeItem('llmx-theme')");
  await cdp('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: 'dark'}]});
  await reloadDocument();
  await wait("!document.querySelector('#login-dialog').open && document.querySelector('#model').value==='qwen3-local' && !document.querySelector('[data-action=\"new\"]').disabled", 'restored legacy dark browser setup');
  await wait("getComputedStyle(document.querySelector('.brand-wordmark')).maskImage.includes('/logo.svg')", 'shared wordmark');
  assert.equal(await evaluate("document.querySelector('link[rel=icon]').getAttribute('href')"), '/favicon.svg', 'shared favicon');
  assert.equal(await evaluate("document.querySelector('.brand-wordmark').getBoundingClientRect().width"), 76, 'sidebar wordmark width');
  assert.equal(await evaluate("document.querySelector('.brand-wordmark').getBoundingClientRect().height"), 30, 'sidebar wordmark height');
  assert.equal(await evaluate("(async()=>{const img=new Image();img.src='/logo.svg';await img.decode();return img.naturalWidth>0})()"), true, 'wordmark asset decoded');
  await screenshot('desktop-empty.png');
  const comparison = await evaluate(String.raw`(() => {
    const md = ChatRender.markdown('| Operation | Unit | Before | After |\n| :--- | :---: | ---: | ---: |\n| Read \\| write | **ms** | 19 | 8 |\n| Pending | ms |\n| Extra | ms | 2 | 1 | discard |');
    document.body.append(md);
    const rows = Array.from(md.querySelectorAll('tr'), row => Array.from(row.cells, cell => Array(cell.colSpan).fill(cell.textContent)).flat());
    const align = Array.from(md.querySelector('tbody tr').cells, cell => getComputedStyle(cell).textAlign);
    const bold = md.querySelector('td strong')?.textContent;
    const headers = Array.from(md.querySelectorAll('th'), cell => cell.scope);
    md.remove();
    const invalid = ['| --- | words |', '| --- |'].map(line => !!ChatRender.markdown('| A | B |\n' + line).querySelector('table'));
    return {rows, align, bold, headers, invalid};
  })()`);
  assert.deepEqual(comparison, {rows:[['Operation','Unit','Before','After'],['Read | write','ms','19','8'],['Pending','ms','',''],['Extra','ms','2','1']],
    align:['left','center','right','right'], bold:'ms', headers:['col','col','col','col'], invalid:[false,false]}, 'comparison table layout');
  const stableMarkdown = await evaluate(String.raw`(() => {
    const samples=['# Title\n\nParagraph with **bold**.\n\n- Item\n  - Nested\n\n\x60\x60\x60python\nprint(1)\n\x60\x60\x60\n\nTail', '| First | Second |\n| --- | --- |\n| A | B |\n\n> Quote\n\n<script>literal</script>'];
    for(const source of samples){
      const current=ChatRender.markdown('');
      for(let i=0;i<=source.length;i++){
        const expected=ChatRender.markdown(source.slice(0,i));
        ChatRender.updateMarkdown(current,source.slice(0,i));
        if(current.innerHTML!==expected.innerHTML)return {pass:false,stage:'prefix',offset:i};
      }
      for(const suffix of ['\n\n'+source,source,'short','',source]){
        ChatRender.updateMarkdown(current,suffix);
        if(current.innerHTML!==ChatRender.markdown(suffix).innerHTML)return {pass:false,stage:'rewrite'};
      }
    }
    const boundaries=['First\n\nName | Rate\n--- | ---:\nCPU | 3', 'First\n\nName | Rate\n\nCPU | 3',
      'First\n\n- One\n\n  More\n\n- Two', 'First\n\n- One\n\nMore\n\n- Two', 'First\n\n> Quote\n\nTail'];
    const rewritten=ChatRender.markdown('');
    for(const source of boundaries){
      ChatRender.updateMarkdown(rewritten,source);
      if(rewritten.innerHTML!==ChatRender.markdown(source).innerHTML)return {pass:false,stage:'changed boundary',source};
    }
    const text='Completed paragraph\n\n\x60\x60\x60python\nprint(1)\n\x60\x60\x60\n\nChanging';
    const current=ChatRender.markdown(text);document.body.append(current);
    const paragraph=current.firstElementChild, button=current.querySelector('.code-heading button');button.textContent='Copied';button.focus();
    const selection=getSelection(),range=document.createRange();range.selectNodeContents(paragraph);selection.removeAllRanges();selection.addRange(range);
    const highlight=ChatArtifacts.highlight;
    let highlights=0;
    ChatArtifacts.highlight=(...args)=>{highlights++;return highlight(...args);};
    try {ChatRender.updateMarkdown(current,text+' final block');}
    finally {ChatArtifacts.highlight=highlight;}
    if(highlights!==0)return {pass:false,stage:'highlighted unchanged code'};
    const result={pass:true,paragraph:current.firstElementChild===paragraph,button:current.querySelector('.code-heading button')===button,feedback:button.textContent,focus:document.activeElement===button,selection:selection.toString()};
    selection.removeAllRanges();current.remove();return result;
  })()`);
  assert.deepEqual(stableMarkdown,{pass:true,paragraph:true,button:true,feedback:'Copied',focus:true,selection:'Completed paragraph'}, 'streamed Markdown changed completed blocks, focus or selection');
  await click('[data-action="settings"]');
  assert.equal(await evaluate("document.querySelector('[data-action=save-settings]').hidden && document.querySelector('#conversation-settings-help')?.textContent.includes('first reply')"), true, 'blank settings must explain persistence without offering a phantom save');
  await fill('#setting-system', 'Keep first-chat settings.');
  await fill('#draft', 'Draft before explicit creation.');
  await click('[data-action="new"]');
  await wait("document.querySelector('.chat-item.active')", 'explicit first chat');
  assert.equal(await evaluate("fetch('/api/chats/'+localStorage.getItem('llmx-chat')).then(r=>r.json()).then(d=>d.chat.settings.system)"), 'Keep first-chat settings.', 'first chat creation lost blank settings');
  const firstSettingsChat = await evaluate("localStorage.getItem('llmx-chat')");
  await click('[data-action="new"]');
  assert.equal(await evaluate("localStorage.getItem('llmx-chat')"), firstSettingsChat, 'New replaced the current empty chat');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Draft before explicit creation.', 'creation lost the blank draft');
  await evaluate("ChatShell.api('/chats/'+localStorage.getItem('llmx-chat'),'DELETE')");
  await evaluate("localStorage.removeItem('llmx-chat')");
  await reloadDocument();
  await wait("window.ChatShell.config && !document.querySelector('[data-action=new]').disabled", 'blank fixture cleaned');
  assert.equal(await evaluate("document.querySelector('#draft').value"), '', 'blank draft resurrected after moving into a chat');
  const composeChats = await evaluate(`(async()=>{
    const a=await ChatShell.api('/chats','POST',{title:'Compose A'});
    const b=await ChatShell.api('/chats','POST',{title:'Compose B'});
    localStorage.setItem('llmx-chat',a.id); return [a.id,b.id];
  })()`);
  await reloadDocument();
  await wait("document.querySelector('#chat-title').textContent==='Compose A'", 'compose fixture');
  await click('[data-action="settings"]');
  await fill('#setting-system', 'Save these settings in A.');
  const settingsPath = JSON.stringify('/api/chats/'+composeChats[0]);
  await evaluate("(()=>{const original=window.fetch;window.releaseSettingsSave=null;window.fetch=async (...args)=>{if(args[0]==="+settingsPath+" && args[1]?.method==='PATCH'){window.fetch=original;await new Promise(resolve=>window.releaseSettingsSave=resolve);}return original(...args);};})()");
  await click('[data-action="save-settings"]');
  await wait("typeof window.releaseSettingsSave==='function'", 'hold settings save');
  await click('[data-chat="'+composeChats[1]+'"]');
  await wait("document.querySelector('#chat-title').textContent==='Compose B'", 'switch during settings save');
  await fill('#setting-system', 'Keep newer settings in B.');
  await evaluate('window.releaseSettingsSave()');
  await wait("fetch("+settingsPath+").then(r=>r.json()).then(d=>d.chat.settings.system==='Save these settings in A.')", 'save applied to original conversation');
  assert.equal(await evaluate("document.querySelector('#setting-system').value"), 'Keep newer settings in B.', 'old settings save replaced destination controls');
  await click('[data-chat="'+composeChats[0]+'"]');
  await wait("document.querySelector('#chat-title').textContent==='Compose A' && document.querySelector('#setting-system').value==='Save these settings in A.'", 'reopen saved settings');
  await fill('#draft', 'Keep this unsent message.');
  await upload('#file-input', fixture);
  await wait("document.querySelector('#attachments').textContent.includes('notes.txt')", 'draft attachment');
  await click('[data-chat="'+composeChats[0]+'"]');
  await delay(200);
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Keep this unsent message.', 'reselect discarded compose text');
  await evaluate(`(()=>{const original=window.fetch;window.releaseComposeOpen=null;window.fetch=async(...args)=>{const response=await original(...args);if(args[0]!=='/api/chats/'+${JSON.stringify(composeChats[1])})return response;window.fetch=original;return new Promise(resolve=>window.releaseComposeOpen=()=>resolve(response));};})()`);
  await click('[data-chat="'+composeChats[1]+'"]');
  await wait("typeof window.releaseComposeOpen==='function'", 'pending compose navigation');
  await click('[data-chat="'+composeChats[0]+'"]');
  await delay(150);
  await evaluate('releaseComposeOpen()');
  await delay(150);
  assert.equal(await evaluate("document.querySelector('#chat-title').textContent"), 'Compose A', 'delayed navigation overrode the latest selection');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Keep this unsent message.', 'delayed navigation discarded the draft');
  await click('[data-chat="'+composeChats[1]+'"]');
  await wait("document.querySelector('#chat-title').textContent==='Compose B'", 'second compose chat');
  await fill('#draft', 'Second independent draft.');
  await click('[data-chat="'+composeChats[0]+'"]');
  await wait("document.querySelector('#chat-title').textContent==='Compose A'", 'first compose chat restored');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Keep this unsent message.', 'navigation discarded compose text');
  assert.equal(await evaluate("document.querySelector('#attachments').textContent.includes('notes.txt')"), true, 'navigation discarded compose attachment');
  await reloadDocument();
  await wait("document.querySelector('#chat-title').textContent==='Compose A'", 'compose reload');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Keep this unsent message.', 'reload discarded compose text');
  assert.equal(await evaluate("document.querySelector('#attachments').textContent.includes('notes.txt')"), true, 'reload discarded compose attachment');
  await click('[data-chat="'+composeChats[1]+'"]');
  await wait("document.querySelector('#chat-title').textContent==='Compose B'", 'second draft after reload');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Second independent draft.', 'drafts leaked between conversations');
  await evaluate("window.savedStorageWrite=Storage.prototype.setItem;Storage.prototype.setItem=function(k,v){if(k.startsWith('llmx-draft:'))throw new DOMException('Full','QuotaExceededError');return savedStorageWrite.call(this,k,v)}");
  await fill('#draft', 'Keep this when draft storage is full.');
  await wait("!document.querySelector('#draft-notice').hidden", 'draft storage failure notice');
  await click('[data-chat="'+composeChats[0]+'"]');
  await wait("document.querySelector('#chat-title').textContent==='Compose A'", 'quota navigation');
  await click('[data-chat="'+composeChats[1]+'"]');
  await wait("document.querySelector('#chat-title').textContent==='Compose B'", 'quota return');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Keep this when draft storage is full.', 'quota failure discarded memory draft');
  await evaluate("Storage.prototype.setItem=window.savedStorageWrite");
  await fill('#draft', 'Storage recovered.');
  await wait("document.querySelector('#draft-notice').hidden", 'draft storage recovered');
  await evaluate(`Promise.all(${JSON.stringify(composeChats)}.map(id=>ChatShell.api('/chats/'+id,'DELETE')))`);
  await evaluate("localStorage.removeItem('llmx-chat')");
  await reloadDocument();
  await wait("window.ChatShell.config && !document.querySelector('[data-action=new]').disabled", 'compose fixture cleaned');
  const imageLink = '![Lake](https://example.invalid/photo_(lake).jpg)';
  const imageLinkSource = [imageLink + ' ![](https://example.invalid/other.jpg) [Source](https://example.invalid/source)',
    '`' + imageLink + '`', '```markdown\n' + imageLink + '\n```', '![Unsafe](javascript:alert%281%29)'].join('\n\n');
  const imageLinkShape = await evaluate(`(async () => {
    const rendered = ChatRender.markdown(${JSON.stringify(imageLinkSource)});
    document.body.append(rendered);
    try {
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      const first = rendered.firstElementChild;
      return {text:first.textContent, links:Array.from(first.querySelectorAll('a[href]'), anchor =>
        [anchor.textContent,anchor.href,anchor.target,anchor.rel]), images:rendered.querySelectorAll('img').length,
        code:Array.from(rendered.querySelectorAll('code'), block => block.textContent),
        unsafe:rendered.lastElementChild.querySelectorAll('a[href]').length};
    } finally { rendered.remove(); }
  })()`);
  assert.deepEqual(imageLinkShape, {text:'Lake Open image Source', links:[
    ['Lake','https://example.invalid/photo_(lake).jpg','_blank','noopener noreferrer'],
    ['Open image','https://example.invalid/other.jpg','_blank','noopener noreferrer'],
    ['Source','https://example.invalid/source','_blank','noopener noreferrer']],
    images:0, code:[imageLink,imageLink], unsafe:0}, 'Markdown image links lost labels, changed code, or created unsafe markup');
  assert.deepEqual(imageLinkRequests, [], 'rendering Markdown requested a remote image');
  const fencedMarkdown = ['Before the inner example', '```js', 'const value = 1;', '```', '~~~~', '```` still source', 'After the inner example'].join('\n');
  const fencedTilde = ['~~~', '```', '~~~~ still source', 'Nested code'].join('\n');
  const partialFence = '````markdown\n' + fencedMarkdown;
  const completeFence = partialFence + '\n````` \t\nOrdinary closing text';
  const nestedFence = '- Parent\n  - Child\n' + ['~~~~text', fencedTilde, '~~~~', '', 'After nested fence'].join('\n').replace(/^(.+)/gm, '    $1');
  const fenceShapes = await evaluate(`(() => {
    const partial = ChatRender.markdown(${JSON.stringify(partialFence)});
    const complete = ChatRender.markdown(${JSON.stringify(completeFence)});
    const nested = ChatRender.markdown(${JSON.stringify(nestedFence)}).querySelector('ul > li > ul > li');
    function shape(root) {
      const block = root?.querySelector('.code-block');
      return {blocks: root?.querySelectorAll('.code-block').length, source: block?.querySelector('code').textContent,
        language: block?.querySelector('[data-language]').dataset.language, following: block?.nextElementSibling?.textContent,
        actions: Array.from(block?.querySelectorAll('.code-actions button') || [], button => button.textContent)};
    }
    const streamed = shape(partial), completeShape = shape(complete);
    ChatRender.updateMarkdown(partial, ${JSON.stringify(completeFence)});
    return {streamed, complete: completeShape, reconciled: shape(partial), nested: shape(nested)};
  })()`);
  const fencedShape = (source, language, following) => ({blocks:1,source,language,
    ...(following ? {following} : {}), actions:['Copy code','Download','Open']});
  const completedShape = fencedShape(fencedMarkdown, 'markdown', 'Ordinary closing text');
  assert.deepEqual(fenceShapes, {streamed:fencedShape(fencedMarkdown, 'markdown'), complete:completedShape,
    reconciled:completedShape, nested:fencedShape(fencedTilde, 'text', 'After nested fence')},
    'a fence closed on a mixed, short or annotated marker, lost source or split its actions');
  assert.equal(await evaluate(`(async () => {
    const fixture = ChatRender.markdown(${JSON.stringify(completeFence)});
    const [copy,,open] = fixture.querySelectorAll('.code-actions button');
    const clipboard = navigator.clipboard, writeText = clipboard.writeText;
    let copied;
    clipboard.writeText = async text => { copied = text; };
    try { copy.click(); await Promise.resolve(); } finally { clipboard.writeText = writeText; }
    open.click();
    return copied;
  })()`), fencedMarkdown, 'Copy code changed the fenced source');
  await wait("document.querySelector('#artifact-dialog').open", 'long-fence source viewer');
  assert.equal(await evaluate("document.querySelector('.artifact-source code').textContent"), fencedMarkdown, 'source viewer changed the fenced source');
  await click('#artifact-dialog [aria-label="Close artifact"]');
  const nestedListSource = '3. First\n   - Child\n     7. Leaf\n        Wrapped line\n\n        ```js\n        const value = 1;\n        ```\n4. Second\n\nAfter\n\n- Root\n  - Nested **text**\n- Next';
  const nestedListShape = await evaluate(`(() => {
    const rendered = ChatRender.markdown(${JSON.stringify(nestedListSource)});
    const first = rendered.firstElementChild;
    const nested = first.querySelector(':scope > li > ul > li > ol');
    return {start: first.start, items: first.querySelectorAll(':scope > li').length,
      leafStart: nested?.start, wrapped: nested?.textContent.includes('Wrapped line'),
      code: nested?.querySelector('code')?.textContent,
      following: first.nextElementSibling?.textContent,
      lastItems: rendered.lastElementChild.querySelectorAll(':scope > li').length,
      bold: rendered.lastElementChild.querySelector('li > ul strong')?.textContent};
  })()`);
  assert.deepEqual(nestedListShape, {start: 3, items: 2, leafStart: 7, wrapped: true,
    code: 'const value = 1;', following: 'After', lastItems: 2, bold: 'text'}, 'nested lists lost structure, numbering or item content');
  assert.equal(await evaluate(`ChatRender.markdown(${JSON.stringify('- Root\n  - <img src=x onerror=alert(1)>')}).querySelector('img')===null`), true, 'nested list parsed model HTML');
  assert.deepEqual(await evaluate(`(() => {
    const rendered=ChatRender.markdown(${JSON.stringify('5) Root\n\t- Tabbed\n6) Next')});
    return {start:rendered.firstElementChild.start, items:rendered.firstElementChild.children.length,
      nested:rendered.querySelector('ol > li > ul > li')?.textContent};
  })()`), {start:5,items:2,nested:'Tabbed'}, 'tabs or parenthesis numbering changed list ownership');
  assert.equal(await evaluate(`(() => {
    const source=Array.from({length:24},(_,i)=>'  '.repeat(i)+'- level:'+i+':').join(String.fromCharCode(10));
    const rendered=ChatRender.markdown(source);
    return rendered.querySelectorAll('ul').length<=8 && Array.from({length:24},(_,i)=>'level:'+i+':').every(text=>rendered.textContent.includes(text));
  })()`), true, 'deep list fallback lost content or exceeded nesting bound');
  assert.equal(await evaluate(`(() => {
    const source=${JSON.stringify('- Parent\n  - Child **text**\n- Next')};
    for(let end=1;end<=source.length;end++) {
      const text=ChatRender.markdown(source.slice(0,end)).textContent;
      if(end>=8&&!text.includes('Parent')) return false;
      if(end>=18&&!text.includes('Child')) return false;
    }
    return ChatRender.markdown(source).querySelector('ul > li > ul > li strong')?.textContent==='text';
  })()`), true, 'partial streamed list dropped visible content');
  await evaluate(`(() => {
    const fixture=document.createElement('article'); fixture.id='markdown-list-fixture'; fixture.className='message assistant';
    fixture.style.cssText='position:fixed;inset:20px;z-index:1000;overflow:auto;padding:20px;background:var(--panel);border:1px solid var(--border);margin:0;max-width:700px';
    fixture.append(ChatRender.markdown(${JSON.stringify(nestedListSource)})); document.body.append(fixture);
  })()`);
  for(const width of [1440,320]) for(const theme of ['dark','light']) {
    await cdp('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<500});
    await evaluate('document.documentElement.dataset.theme='+JSON.stringify(theme));
    assert.equal(await evaluate("document.documentElement.scrollWidth<=innerWidth && document.querySelector('#markdown-list-fixture').scrollWidth<=document.querySelector('#markdown-list-fixture').clientWidth"), true, 'nested list horizontal overflow');
    assert.equal(await evaluate(`Array.from(document.querySelectorAll('#markdown-list-fixture .code-actions button')).every(button=>{
      const range=document.createRange(); range.selectNodeContents(button);
      const bounds=button.getBoundingClientRect(), header=button.closest('.code-heading').getBoundingClientRect();
      return range.getClientRects().length===1 && bounds.left>=header.left && bounds.right<=header.right;
    })`), true, 'nested code actions split labels or leave their header');
    await screenshot('nested-list-'+width+'-'+theme+'.png');
  }
  await evaluate("document.querySelector('#markdown-list-fixture').remove();document.documentElement.dataset.theme='dark'");
  await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await evaluate("for (let i=0;i<12;i++) document.querySelector('[data-action=\"new\"]').click()");
  await wait("document.querySelector('.chat-item.active')!==null", 'new empty conversation');
  await evaluate("new Promise(resolve => setTimeout(resolve, 200))");
  assert.equal(await evaluate("fetch('/api/chats').then(r=>r.json()).then(d=>d.chats.length)"), 1, 'rapid New clicks created duplicate drafts');
  await click('[data-action="settings"]');
  assert.equal(await evaluate("document.querySelector('#setting-reply-limit').value==='none' && document.querySelector('#reply-token-limit').hidden"), true, 'new chat forces a numeric reply cap');
  await wait("document.querySelector('#request-context-limit').textContent==='8,192 tokens'", 'existing collector exposes request capacity');
  assert.equal(await evaluate("document.querySelector('#model-context-limit').textContent"), '40,960 tokens', 'model and request context were conflated');
  await evaluate("document.querySelector('#setting-reply-limit').value='custom';document.querySelector('#setting-reply-limit').dispatchEvent(new Event('change'))");
  await click('[data-action="save-settings"]');
  await wait("document.querySelector('#toast').textContent.includes('positive whole number')", 'empty custom reply cap refused');
  await fill('#setting-max_tokens', '123');
  await click('[data-action="save-settings"]');
  await wait("document.querySelector('#toast').textContent==='Settings saved for this chat.'", 'explicit reply cap saved');
  await reloadDocument();
  await wait("document.querySelector('#setting-max_tokens').value==='123'", 'saved explicit cap restored');
  assert.equal(await evaluate("document.querySelector('#setting-reply-limit').value"), 'custom');
  await click('[data-action="settings"]');
  for (const width of [1440, 390]) for (const theme of ['dark', 'light']) {
    await cdp('Emulation.setDeviceMetricsOverride', {width, height:1000,deviceScaleFactor:1,mobile:width<500});
    await wait(`document.documentElement.dataset.compact==='${width<500}'`, 'settings viewport resize');
    await click('[data-action="settings"]');
    assert.equal(await evaluate("!document.querySelector('#panel-settings').hidden && document.querySelector('#inspector').getBoundingClientRect().width>0"), true, 'settings screenshot did not open its panel');
    await evaluate("document.documentElement.dataset.theme="+JSON.stringify(theme));
    assert.equal(await evaluate("document.documentElement.scrollWidth<=innerWidth"), true, 'settings overflow');
    assert.equal(await evaluate("getComputedStyle(document.querySelector('#setting-reply-limit')).backgroundColor===getComputedStyle(document.querySelector('#setting-temperature')).backgroundColor && getComputedStyle(document.querySelector('#setting-reply-limit')).paddingTop==='10px'"), true, 'reply select does not use shared field styling');
    await screenshot('reply-settings-'+width+'-'+theme+'.png');
  }
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await evaluate("document.documentElement.dataset.theme='dark';document.querySelector('#setting-reply-limit').value='none';document.querySelector('#setting-reply-limit').dispatchEvent(new Event('change'))");
  await click('[data-action="save-settings"]');
  await wait("fetch('/api/chats/'+localStorage.getItem('llmx-chat')).then(r=>r.json()).then(d=>d.chat.settings.max_tokens===null)", 'no additional cap saved');
  await reloadDocument();
  await wait("document.querySelector('.chat-item.active')!==null && !document.querySelector('[data-action=new]').disabled", 'restore uncapped empty chat');
  assert.equal(await evaluate("document.querySelector('#setting-reply-limit').value==='none' && document.querySelector('#setting-max_tokens').disabled"), true, 'null cap became a number after reload');
  await click('[data-action="close-drawers"]');
  await fill('#draft', 'Keep my unsent draft');
  await click('[data-action="new"]');
  await evaluate("new Promise(resolve => setTimeout(resolve, 200))");
  assert.equal(await evaluate("fetch('/api/chats').then(r=>r.json()).then(d=>d.chats.length)"), 1, 'New on an empty conversation created another draft');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Keep my unsent draft');
  await upload('#file-input', fixture);
  await wait("document.querySelector('#attachments').innerText.includes('notes.txt')", 'valid attachment before an oversized batch');
  await upload('#file-input', textOverFixture);
  await delay(400);
  assert.equal(await evaluate("document.querySelector('#attachments').innerText.includes('text-over.txt')"), false, 'oversized UTF-8 text attachment was added');
  assert.match(await evaluate("document.querySelector('#toast').textContent"), /2 MiB/, 'oversized text refusal did not explain the limit');
  assert.deepEqual(await evaluate("Array.from(document.querySelectorAll('#attachments .attachment-pill')).map(el=>el.firstChild.textContent)"), ['notes.txt'], 'text refusal discarded the valid attachment');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Keep my unsent draft', 'text refusal changed the draft');
  await uploadMany('#file-input', [textMaxFixture, textOverFixture]);
  await delay(400);
  assert.deepEqual(await evaluate("Array.from(document.querySelectorAll('#attachments .attachment-pill')).map(el=>el.firstChild.textContent)"), ['notes.txt'], 'rejected attachment batch partially committed');
  await evaluate("window.__largeTextReads=0;window.__blobArrayBuffer=Blob.prototype.arrayBuffer;Blob.prototype.arrayBuffer=function(){if(this.size>2*1024*1024+3)window.__largeTextReads++;return window.__blobArrayBuffer.call(this)}");
  await upload('#file-input', textRawOverFixture);
  await delay(400);
  assert.equal(await evaluate('window.__largeTextReads'), 0, 'oversized raw text was read before refusal');
  assert.equal(await evaluate("document.querySelector('#attachments').innerText.includes('text-raw-over.txt')"), false, 'oversized raw text was added');
  await evaluate("Blob.prototype.arrayBuffer=window.__blobArrayBuffer");
  await upload('#file-input', textBomMaxFixture);
  await wait("document.querySelector('#attachments').innerText.includes('text-bom-max.txt')", 'UTF-8 BOM text whose decoded bytes equal the limit');
  assert.equal(await evaluate("document.querySelector('#attachments .attachment-pill:nth-child(2) .attachment-file-size').getAttribute('aria-label')"), 'text-bom-max.txt, 2,097,152 bytes', 'text size did not use decoded UTF-8 bytes');
  await evaluate("while(document.querySelector('#attachments [data-action=remove-attachment]'))document.querySelector('#attachments [data-action=remove-attachment]').click()");
  assert.equal(await evaluate("document.querySelector('#attachments').children.length"), 0, 'pre-boundary fixtures were not removed');
  await upload('#file-input', textMaxFixture);
  await wait("document.querySelector('#attachments').innerText.includes('text-max.txt')", 'text attachment at its byte limit');
  await upload('#file-input', imageMaxFixture);
  await wait("document.querySelector('#attachments').innerText.includes('image-max.png')", 'image attachment at its byte limit');
  assert.match(await evaluate("document.querySelector('#attachments .attachment-total').textContent"), /10 MiB/, 'total did not show the exact combined attachment size');
  assert.equal(await evaluate("document.querySelector('#attachments .attachment-pill:nth-child(1) .attachment-file-size').getAttribute('aria-label')"), 'text-max.txt, 2,097,152 bytes', 'text attachment size was not reported exactly');
  assert.equal(await evaluate("document.querySelector('#attachments .attachment-pill:nth-child(2) .attachment-file-size').getAttribute('aria-label')"), 'image-max.png, 8,388,608 bytes', 'image attachment size was not reported exactly');
  await screenshot('attachment-feedback-desktop.png');
  await upload('#file-input', tinyFixture);
  await delay(400);
  assert.equal(await evaluate("document.querySelector('#attachments').innerText.includes('tiny.txt')"), false, 'attachment total above 10 MiB was accepted');
  assert.match(await evaluate("document.querySelector('#toast').textContent"), /10 MiB/, 'aggregate refusal did not explain the limit');
  assert.equal(await evaluate("document.querySelectorAll('#attachments .attachment-pill').length"), 2, 'aggregate refusal discarded the existing attachments');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Keep my unsent draft', 'aggregate refusal changed the draft');
  await upload('#file-input', imageOverFixture);
  await delay(400);
  assert.equal(await evaluate("document.querySelector('#attachments').innerText.includes('image-over.png')"), false, 'oversized image attachment was added');
  assert.match(await evaluate("document.querySelector('#toast').textContent"), /8 MiB/, 'oversized image refusal did not explain the limit');
  await evaluate("while(document.querySelector('#attachments [data-action=remove-attachment]'))document.querySelector('#attachments [data-action=remove-attachment]').click()");
  await upload('#file-input', imageMaxFixture);
  await wait("document.querySelector('#attachments').innerText.includes('image-max.png')", 'image attachment for concurrent total check');
  await evaluate("window.__raceReads=0;window.__racePending=[];window.__originalBlobArrayBuffer=Blob.prototype.arrayBuffer;Blob.prototype.arrayBuffer=function(){if(this.size===1572864){const blob=this;window.__raceReads++;return new Promise(resolve=>window.__racePending.push(()=>window.__originalBlobArrayBuffer.call(blob).then(resolve)))}return window.__originalBlobArrayBuffer.call(this)}");
  await upload('#file-input', textRaceAFixture);
  await wait('window.__raceReads===1', 'first concurrent text read');
  await upload('#file-input', textRaceBFixture);
  await wait('window.__raceReads===2', 'second concurrent text read');
  await evaluate("window.__racePending.splice(0).forEach(release=>release())");
  await delay(400);
  assert.equal(await evaluate("document.querySelectorAll('#attachments .attachment-pill').length"), 2, 'concurrent attachments both exceeded the current aggregate limit');
  assert.equal(await evaluate("Array.from(document.querySelectorAll('#attachments .attachment-pill')).some(el=>['race-a.txt','race-b.txt'].includes(el.firstChild.textContent))"), true, 'concurrent read limit check did not commit either admissible file');
  assert.equal(await evaluate("document.querySelector('#attachments .attachment-total').getAttribute('aria-label')"), 'Total attachment size: 9,961,472 bytes of 10 MiB', 'concurrent attachment commit did not recheck the current total');
  assert.match(await evaluate("document.querySelector('#toast').textContent"), /10 MiB/, 'concurrent aggregate refusal was not reported');
  await evaluate("Blob.prototype.arrayBuffer=window.__originalBlobArrayBuffer");
  await cdp('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
  await evaluate("window.dispatchEvent(new Event('resize'))");
  await delay(120);
  await click('[data-action="close-drawers"]');
  await wait("document.querySelector('#app').classList.contains('inspector-closed')", 'attachment composer visible on mobile');
  assert.equal(await evaluate("document.documentElement.scrollWidth<=innerWidth && document.querySelector('#attachments').getBoundingClientRect().right<=innerWidth"), true, 'attachment sizes overflow the mobile layout');
  await screenshot('attachment-feedback-mobile.png');
  await cdp('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
  await evaluate("window.dispatchEvent(new Event('resize'));while(document.querySelector('#attachments [data-action=remove-attachment]'))document.querySelector('#attachments [data-action=remove-attachment]').click()");
  assert.equal(await evaluate("document.querySelector('#attachments').children.length"), 0, 'attachment cleanup left a size summary behind');
  await evaluate(`{
    const Original = window.EventSource;
    window.healthStreams = [];
    window.EventSource = function(url, options) {
      if (url !== '/api/monitor/stream') return new Original(url, options);
      const stream = new EventTarget(); stream.closed = false; stream.close = () => { stream.closed = true; };
      window.healthStreams.push(stream); return stream;
    };
    window.healthSample = (clock, phase, prefilled, published, present = true, ok = true) => {
      const item = {id:'r-browser-1',phase,prompt_tokens:200,reused_tokens:0,prefilled_tokens:prefilled,published_tokens:published};
      const data = {collector_id:'browser-collector',ok,interval_s:2,samples:[],latest:{requests:{live:{sample_elapsed_ms:clock,items:present?[item]:[]}}}};
      window.healthStreams.at(-1).dispatchEvent(new MessageEvent('history',{data:JSON.stringify(data)}));
    };
  }`);
  await send('Make a practical plan.');
  await wait("document.querySelector('.reasoning')!==null", 'streamed reasoning');
  await wait('healthStreams.length===1', 'one shared collector subscription');
  await click('[data-action="settings"]');
  assert.equal(await evaluate('healthStreams.length'), 1, 'Settings added a second monitor subscription during generation');
  await evaluate(`window.capacitySample=(model,limit,age=0,ok=true)=>healthStreams.at(-1).dispatchEvent(new MessageEvent('history',{data:JSON.stringify({collector_id:'browser-collector',ok,interval_s:2,last_success:Date.now()/1000-age,samples:[],latest:{server:{model,request_tokens:limit}}})}))`);
  await evaluate("capacitySample('qwen3-local',8192)");
  assert.equal(await evaluate("document.querySelector('#request-context-limit').textContent"), '8,192 tokens');
  for (const value of ["capacitySample('another-model',8192)", "capacitySample('qwen3-local',true)", "capacitySample('qwen3-local',8192,60)", "capacitySample('qwen3-local',8192,0,false)"]) {
    await evaluate(value);
    assert.equal(await evaluate("document.querySelector('#request-context-limit').textContent"), 'Unavailable', 'stale, invalid or mismatched context shown as current');
  }
  await click('[data-tab="activity"]');
  await evaluate("healthSample(1000,'prefill',0,0)");
  assert.equal(await evaluate("document.querySelector('.live-phase').textContent"), 'Processing prompt');
  assert(!(await evaluate("document.querySelector('.live-metrics').textContent")).includes('tok/s'), 'one sample invented a rate');
  await evaluate("healthSample(3000,'prefill',200,0)");
  assert((await evaluate("document.querySelector('.live-metrics').textContent")).includes('100 prompt tok/s'));
  await evaluate("healthSample(3000,'prefill',200,0)");
  assert(!(await evaluate("document.querySelector('.live-metrics').textContent")).includes('tok/s'), 'repeated snapshot held an old rate');
  await evaluate("healthSample(5000,'generate',200,40)");
  assert((await evaluate("document.querySelector('.live-metrics').textContent")).includes('20 generated tok/s'));
  await evaluate("healthSample(6000,'generate',200,40,false)");
  assert.equal(await evaluate("document.querySelector('.live-metrics').textContent"), 'Live rates unavailable');
  await evaluate("healthSample(7000,'generate',200,60)");
  assert(!(await evaluate("document.querySelector('.live-metrics').textContent")).includes('tok/s'), 'missing row was bridged');
  await evaluate("healthSample(9000,'generate',200,60)");
  assert((await evaluate("document.querySelector('.live-metrics').textContent")).includes('0 generated tok/s'), 'measured zero was lost');
  await evaluate("healthStreams.at(-1).onerror()");
  assert.equal(await evaluate("document.querySelector('.live-metrics').textContent"), 'Live rates unavailable');
  await evaluate("window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:true}));window.dispatchEvent(new PageTransitionEvent('pageshow',{persisted:true}))");
  await wait('healthStreams.length===2 && healthStreams[0].closed', 'back-forward cache restoration resubscribed');
  await click('.global-nav [data-view="preferences"]');
  assert.equal(await evaluate('healthStreams.every(stream=>stream.closed)'), true, 'leaving Chat kept its health subscription');
  await click('.global-nav [data-view="chat"]');
  await wait('healthStreams.length===3', 'chat return resubscribed');
  await evaluate("healthSample(10000,'generate',200,60);healthSample(12000,'generate',200,100)");
  await cdp('Emulation.setDeviceMetricsOverride', {width:390,height:844,deviceScaleFactor:1,mobile:true});
  await evaluate("window.dispatchEvent(new Event('resize'));document.querySelector('[data-action=close-drawers]').click()");
  assert.equal(await evaluate("document.documentElement.scrollWidth<=innerWidth"), true, 'live metrics overflow mobile');
  await screenshot('live-metrics-mobile-dark.png');
  await evaluate("document.documentElement.dataset.theme='light'");
  await screenshot('live-metrics-mobile-light.png');
  await evaluate("document.documentElement.dataset.theme='dark'");
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await evaluate("window.dispatchEvent(new Event('resize'))");
  await pointerClick('[data-action="reply-stats"]');
  await wait("document.querySelector('#reply-statistics .stats-hint')?.textContent.startsWith('Pinned')", 'pin statistics during generation');
  assert.equal(await evaluate("document.querySelector('#reply-statistics .stat-row strong').textContent"), 'Unavailable', 'unreported live usage invented a zero');
  await evaluate("window.__pinnedStats=document.querySelector('#reply-statistics');const range=document.createRange();range.selectNodeContents(__pinnedStats.querySelector('.stat-row'));getSelection().removeAllRanges();getSelection().addRange(range)");
  assert.equal((await fetch(upstreamFixture + '/browser-step')).status, 204);
  await wait("document.querySelector('.reasoning .markdown').textContent.includes('Compare the options.')", 'statistics survives streamed reasoning');
  assert.equal(await evaluate("document.querySelector('#reply-statistics')===__pinnedStats && getSelection().toString().includes('Input tokens')"), true, 'streaming replaced pinned statistics or selection');
  await pointerClick('[data-action="close-stats"]');
  await pointerClick('.reasoning > summary');
  assert.equal(await evaluate("document.querySelector('.reasoning').open"), true, 'reasoning did not open after statistics');
  await reasoningClickDuringDelta('Choose a clear plan.', false);
  await pointerClick('.reasoning > summary');
  assert.equal(await evaluate("document.querySelector('.reasoning').open"), true, 'reasoning did not reopen while streaming');
  assert.equal((await fetch(upstreamFixture + '/browser-step')).status, 204);
  await idle();
  assert.equal(await evaluate('healthStreams.every(stream=>stream.closed)'), true, 'completion kept the health subscription');
  assert.equal(await evaluate("document.querySelector('.live-metrics')"), null, 'completion kept a live rate');
  assert.equal(await evaluate("document.querySelector('.reasoning').open"), true, 'completion discarded open reasoning choice');
  const selectedBeforeDelayedCreate = await evaluate("localStorage.getItem('llmx-chat')");
  const navigationTarget = await evaluate("fetch('/api/chats',{method:'POST',headers:{'Content-Type':'application/json','X-Chat-Request':'1'},body:JSON.stringify({title:'Delayed create navigation target'})}).then(r=>r.json()).then(c=>c.id)");
  await reloadDocument();
  await wait(`document.querySelector('[data-chat="${navigationTarget}"]')!==null && document.querySelector('.chat-item.active')?.dataset.chat===${JSON.stringify(selectedBeforeDelayedCreate)}`, 'existing navigation target available');
  await cdp('Fetch.enable', {patterns: [{urlPattern: url + '/api/chats', requestStage: 'Response'}]});
  await click('[data-action="new"]');
  await wait(() => heldChatCreateResponse !== null, 'successful chat-create response paused');
  const createdBody = await cdp('Fetch.getResponseBody', {requestId: heldChatCreateResponse.requestId});
  const createdDuringDelay = JSON.parse(Buffer.from(createdBody.body, createdBody.base64Encoded ? 'base64' : 'utf8').toString('utf8')).id;
  await click(`[data-chat="${navigationTarget}"]`);
  await wait(`localStorage.getItem('llmx-chat')===${JSON.stringify(navigationTarget)} && document.querySelector('.chat-item.active')?.dataset.chat===${JSON.stringify(navigationTarget)}`, 'switch existing conversation during delayed create');
  await fill('#draft', 'Newer draft typed during chat creation');
  await cdp('Fetch.continueResponse', {requestId: heldChatCreateResponse.requestId});
  heldChatCreateResponse = null;
  await cdp('Fetch.disable');
  await wait(`document.querySelector('[data-chat="${createdDuringDelay}"]')!==null`, 'created conversation discoverable after response');
  await wait("!document.querySelector('[data-action=\"new\"]').disabled", 'chat creation completion');
  assert.equal(await evaluate("document.querySelector('.chat-item.active')?.dataset.chat"), navigationTarget, 'late create response navigated away from selected conversation');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Newer draft typed during chat creation', 'late create response cleared newer draft');
  await click(`[data-chat="${selectedBeforeDelayedCreate}"]`);
  await wait(`localStorage.getItem('llmx-chat')===${JSON.stringify(selectedBeforeDelayedCreate)}`, 'restore original conversation for following checks');
  await reloadDocument();
  await wait("document.querySelectorAll('.message.assistant').length===1 && document.querySelector('#stop').hidden", 'refresh and persisted stream completion');
  assert.equal(await evaluate("document.querySelector('.message.assistant').textContent.split('A practical plan.').length-1"), 1, 'snapshot replay duplicated output');
  const scrollChat = await evaluate("document.querySelector('.chat-item.active').dataset.chat");
  await cdp('Emulation.setDeviceMetricsOverride', {width:850,height:360,deviceScaleFactor:1,mobile:false});
  await wait("document.querySelector('#scroll-region').scrollHeight - document.querySelector('#scroll-region').clientHeight >= 180", 'long transcript in a short viewport');
  await evaluate("document.querySelector('#scroll-region').scrollTop=0");
  await wait("!document.querySelector('#jump-bottom').hidden", 'jump control available above latest reply');
  assert.equal(await evaluate("document.querySelector('#jump-bottom').getBoundingClientRect().top >= document.querySelector('#scroll-region').getBoundingClientRect().bottom"), true, 'jump control obscures transcript actions');
  await pointerClick('#chat-home');
  await wait("!document.querySelector('#welcome').hidden && document.querySelector('#jump-bottom').hidden", 'home clears the previous transcript jump');
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await wait("document.querySelector('#app').getBoundingClientRect().height===1000", 'desktop frame height restored before selecting history');
  await pointerClick('.chat-item[data-chat="'+scrollChat+'"]');
  await wait("document.querySelector('.message.assistant table') && document.querySelector('.chat-item.active')", 'return to saved conversation');
  await checkHomeLanding();
  assert.equal(await evaluate('window.injected===true'), false, 'model HTML executed');
  assert.equal(await evaluate("document.querySelectorAll('.message.assistant table').length"), 1);
  await click('[data-tab="stats"]');
  assert.equal(await evaluate("document.querySelector('.topbar').getBoundingClientRect().top>=0 && document.querySelector('.composer').getBoundingClientRect().bottom<=innerHeight"), true, 'header or composer clipped');
  assert(!(await evaluate("document.querySelector('#panel-stats').innerText")).includes('tok/s'), 'conversation totals repeat per-call rates');
  assert((await evaluate("document.querySelector('#panel-stats').innerText")).includes('32'));
  assert.equal(await evaluate("document.querySelector('.message-metrics')"), null, 'inline statistics still clutter replies');
  for (const width of [1440, 390]) for (const theme of ['dark', 'light']) {
    await cdp('Emulation.setDeviceMetricsOverride', {width, height: 1000, deviceScaleFactor: 1, mobile: width < 500});
    await evaluate("document.documentElement.dataset.theme="+JSON.stringify(theme)+";document.querySelector('[data-action=close-drawers]').click()");
    const point = await actionHoverPoint('[data-action=reply-stats]');
    await cdp('Input.dispatchMouseEvent', {type: 'mouseMoved', x: point.x, y: point.y});
    await wait("document.querySelector('#reply-statistics')?.matches(':popover-open')", 'hover preview statistics');
    const text = await evaluate("document.querySelector('#reply-statistics').innerText");
    assert(text.includes('240 tok/s') && text.includes('510 ms') && text.includes('25 tok/s'), 'reply measurements or units missing');
    assert.equal(await evaluate("{const r=document.querySelector('#reply-statistics').getBoundingClientRect();r.left>=0&&r.right<=innerWidth&&r.top>=0&&r.bottom<=innerHeight}"), true, 'statistics panel clipped');
    const panelPoint = await evaluate("{const r=document.querySelector('#reply-statistics').getBoundingClientRect();({x:r.x+20,y:r.y+20})}");
    await cdp('Input.dispatchMouseEvent', {type: 'mouseMoved', ...panelPoint}); await delay(200);
    assert.equal(await evaluate("document.querySelector('#reply-statistics')?.matches(':popover-open')"), true, 'hover panel vanished while entering it');
    if (width < 500) {
      const point = await evaluate("{const r=document.querySelector('[data-action=reply-stats]').getBoundingClientRect();({x:r.x+r.width/2,y:r.y+r.height/2})}");
      await cdp('Emulation.setTouchEmulationEnabled', {enabled:true,maxTouchPoints:1});
      await cdp('Input.dispatchTouchEvent', {type:'touchStart',touchPoints:[point]});
      await cdp('Input.dispatchTouchEvent', {type:'touchEnd',touchPoints:[]});
      await cdp('Emulation.setTouchEmulationEnabled', {enabled:false});
    } else await pointerClick('[data-action=reply-stats]');
    await wait("document.querySelector('.stats-hint').textContent.startsWith('Pinned')", 'click or touch pins statistics');
    await checkSpeakerLayout(width);
    await screenshot('reply-statistics-'+width+'-'+theme+'.png');
    await click('[data-action=copy-stats]');
    await wait("document.querySelector('#toast').textContent==='Statistics copied.'", 'statistics copy confirmation');
    await cdp('Input.dispatchKeyEvent', {type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
    await cdp('Input.dispatchKeyEvent', {type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
    assert.equal(await evaluate("!document.querySelector('#reply-statistics') && document.activeElement?.dataset.action==='reply-stats'"), true, 'Escape did not close statistics and restore focus');
    await evaluate("document.activeElement.blur();document.querySelector('[data-action=reply-stats]').focus()");
    await cdp('Input.dispatchKeyEvent', {type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,text:'\r'});
    await cdp('Input.dispatchKeyEvent', {type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});
    await wait("document.querySelector('.stats-hint').textContent.startsWith('Pinned')", 'keyboard pins statistics');
    await pointerClick('#draft');
    assert.equal(await evaluate("document.querySelector('#reply-statistics')"), null, 'outside click did not dismiss statistics');
  }
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await evaluate("document.documentElement.dataset.theme='dark'");
  await click('.code-actions button:last-child');
  await wait("document.querySelector('#artifact-dialog').open", 'code artifact viewer');
  assert((await evaluate("document.querySelector('.artifact-source').textContent")).includes("print('hello')"));
  assert.equal(await evaluate("document.querySelector('.artifact-toolbar-actions button:first-child').hidden"), false, 'Chat artifact should still offer Edit in workspace');
  assert.equal(await evaluate("document.querySelectorAll('.code-block .syntax-string').length>0"), true, 'code syntax colors missing');
  await click('#artifact-dialog [aria-label="Close artifact"]');
  await click('.message.assistant [data-action="retry"]');
  await wait("document.querySelector('.reasoning')?.textContent.includes('Consider another approach.') && !document.querySelector('#stop').hidden", 'retry reasoning stream');
  const activeRetry = await evaluate("fetch('/api/chats/'+localStorage.getItem('llmx-chat')).then(r=>r.json()).then(d=>({message:d.chat.active_leaf,run:d.messages.find(m=>m.id===d.chat.active_leaf).run_id}))");
  const streamsBeforeActiveReload = runStreams.length;
  await reloadDocument();
  await wait(`document.querySelector('.message.assistant')?.dataset.message===${JSON.stringify(activeRetry.message)} && document.querySelector('.reasoning')?.textContent.includes('Consider another approach.') && !document.querySelector('#stop').hidden`, 'restored active reasoning after refresh');
  await wait(() => runStreams.slice(streamsBeforeActiveReload).some(stream => new URL(stream).pathname === '/api/runs/' + activeRetry.run + '/events'), 'refresh reconnecting to the same active run');
  assert.equal(await evaluate("document.querySelectorAll('.message.assistant').length"), 1, 'active refresh duplicated assistant message');
  await pointerClick('.reasoning > summary');
  await pointerClick('.reasoning > summary');
  assert.equal(await evaluate("document.querySelector('.reasoning').open"), false, 'reasoning did not close while streaming');
  const branchTargets = await evaluate(`(() => {
    const buttons=Array.from(document.querySelectorAll('.message.assistant .branch-switch button'));
    window.focusedNextBranch=buttons[1]; focusedNextBranch.focus({preventScroll:true});
    return buttons.map(button=>({label:button.getAttribute('aria-label'), id:button.dataset.id, glyph:button.querySelector('path').getAttribute('d')}));
  })()`);
  assert.deepEqual(branchTargets.map(button=>button.label), ['Previous branch','Next branch']);
  assert.equal(branchTargets[0].id, branchTargets[1].id, 'two-branch retry does not exercise the directional target collision');
  assert.equal((await fetch(upstreamFixture + '/browser-step')).status, 204);
  await wait("document.querySelector('.message.assistant')?.innerText.includes('An alternative approach.') && document.querySelector('#stop').hidden", 'retry branch');
  assert.deepEqual(await evaluate(`(() => {
    const buttons=Array.from(document.querySelectorAll('.message.assistant .branch-switch button'));
    return {actions:buttons.map(button=>({label:button.getAttribute('aria-label'), id:button.dataset.id, glyph:button.querySelector('path').getAttribute('d')})), focused:document.activeElement===focusedNextBranch && buttons[1]===focusedNextBranch};
  })()`), {actions:branchTargets, focused:true}, 'retry completion changed the direction, position, or target of Next branch');
  assert.equal(await evaluate("document.querySelector('.reasoning').open"), false, 'completion discarded closed reasoning choice');
  await click('.message.assistant .branch-switch button');
  await wait("document.querySelector('.message.assistant')?.innerText.includes('A practical plan.')", 'previous response branch');
  await click('.message.user [data-action="edit"]');
  await fill('#draft', 'Edited draft before refreshing.');
  await reloadDocument();
  await wait("document.querySelector('#draft').value==='Edited draft before refreshing.' && !document.querySelector('#edit-note').hidden", 'edit draft restored after refresh');
  await send('Make an edited plan.');
  await wait("document.querySelector('.message.assistant')?.innerText.includes('An edited branch.') && document.querySelector('#stop').hidden", 'edited user branch');
  await click('[data-action="settings"]');
  await click('#tool-settings input'); await click('[data-action="save-settings"]');
  await wait("document.querySelector('#toast').textContent.includes('Settings saved')", 'saved tool selection');
  const toolFixture = {upstream_available:true, models:[{id:'qwen3-local'}], tools:[{name:'echo', description:'Echo text', connection:'mcp:fixture',approval:true}], tools_error:null,
    tool_servers:[{id:'mcp:fixture',name:'Fixture MCP',kind:'mcp',status:'unchecked',tool_count:1,approval:true,checked_at:null,duration_ms:null,error:null}]};
  await evaluate(`document.dispatchEvent(new CustomEvent('companion:config',{detail:${JSON.stringify({...toolFixture, tools:[], tool_servers:[{id:'builtin:workspace',name:'Workspace',kind:'builtin',status:'ready',tool_count:0,approval:null}]})}}));`);
  await click('[data-view="preferences"]');
  assert.equal(await evaluate("document.querySelector('#preferences-tools').innerText.includes('Start the companion with --tools PATH')"), true, 'built-in-only configuration does not explain external MCP setup');
  await evaluate(`document.dispatchEvent(new CustomEvent('companion:config',{detail:${JSON.stringify(toolFixture)}}));document.dispatchEvent(new CustomEvent('companion:tools-updated',{detail:${JSON.stringify(toolFixture)}}));`);
  await evaluate("document.querySelector('#setting-system').value='Keep this unsaved system prompt';document.querySelector('#draft').value='Keep this unsent draft';");
  await click('[data-view="preferences"]');
  assert.equal(await evaluate("document.querySelector('#preferences-tools').innerText.includes('Approval required')"), true, 'operator approval policy label missing');
  const toolMetadataSeparated = await evaluate(`{
    const [policy, description] = document.querySelectorAll('#preferences-tools .connected-tool span');
    policy.textContent === 'Approval required' && description.textContent === 'Echo text' &&
      description.getBoundingClientRect().top >= policy.getBoundingClientRect().bottom;
  }`);
  await evaluate(`window.__toolTestCalls=0;window.__realFetch=window.fetch;window.fetch=async (url,options)=>{if(String(url)==='/api/tools/test'){window.__toolTestCalls++;return new Response(JSON.stringify({tools:[{name:'fresh_tool',description:'Freshly listed',connection:'mcp:fixture',approval:false}],tool_servers:[{id:'mcp:fixture',name:'Fixture MCP',kind:'mcp',status:'ready',tool_count:1,approval:false,checked_at:1794760000,duration_ms:14,error:null}],tools_error:null}),{status:200,headers:{'Content-Type':'application/json'}})}return window.__realFetch(url,options)};`);
  await click('[data-test-server="mcp:fixture"]');
  await wait("document.querySelector('#preferences-tools').innerText.includes('Freshly listed')", 'tested tool list refresh');
  await click('[data-view="chat"]'); await click('[data-action="settings"]');
  assert.equal(await evaluate("document.querySelector('#setting-system').value==='Keep this unsaved system prompt' && document.querySelector('#draft').value==='Keep this unsent draft'"), true, 'connection test discarded unsaved settings or draft');
  assert.equal(await evaluate("document.querySelector('#tool-settings input:checked')?.value==='echo' && document.querySelector('#tool-settings .unavailable-tool')?.innerText.includes('Unavailable. Selection preserved')"), true, 'unavailable selected tool is not visibly preserved');
  await click('#tool-settings .unavailable-tool input');
  assert.equal(await evaluate("document.querySelector('#tool-settings .unavailable-tool input')?.checked===false"), true, 'unavailable selection cannot be removed');
  await click('#tool-settings .unavailable-tool input');
  await click('[data-view="preferences"]');
  await evaluate(`window.fetch=async (url,options)=>{if(String(url)==='/api/tools/test')return new Response(JSON.stringify({error:'Fixture connection refused'}),{status:502,headers:{'Content-Type':'application/json'}});return window.__realFetch(url,options)};`);
  await click('[data-test-server="mcp:fixture"]');
  await wait("document.querySelector('#preferences-tools').innerText.includes('Fixture connection refused')", 'connection test failure');
  assert.equal(await evaluate("document.querySelector('.tool-connection-meta').textContent.startsWith('Ready')"), true, 'request failure changed remote status');
  await screenshot('tool-connections-desktop.png');
  await evaluate(`window.fetch=async (url,options)=>{if(String(url)==='/api/tools/test')return new Response(JSON.stringify({tools:[],tool_servers:[{id:'mcp:fixture',name:'Fixture MCP',kind:'mcp',status:'error',tool_count:0,approval:true,checked_at:1794760001,duration_ms:15,error:'Remote server refused discovery'}],tools_error:'Remote server refused discovery'}),{status:200,headers:{'Content-Type':'application/json'}});return window.__realFetch(url,options)};`);
  await click('[data-test-server="mcp:fixture"]');
  await wait("document.querySelector('#preferences-tools').innerText.includes('Remote server refused discovery')", 'remote discovery error');
  assert.equal(await evaluate("document.querySelector('.tool-connection-meta').textContent.startsWith('Connection error')"), true, 'remote failure was not identified');

  await evaluate('window.fetch=window.__realFetch');
  await cdp('Emulation.setDeviceMetricsOverride', {width:360,height:800,deviceScaleFactor:1,mobile:true});
  assert.equal(await evaluate("document.querySelector('#preferences-tools').getBoundingClientRect().right<=innerWidth && document.documentElement.scrollWidth<=innerWidth"), true, 'tool settings overflow at 360px');
  await screenshot('tool-connections-mobile.png');
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await click('[data-view="chat"]');
  await send('Use the echo tool.');
  await wait("document.querySelector('.tool-call summary .badge')?.textContent==='preparing'", 'tool arguments before model completion');
  assert.equal(await evaluate("document.querySelector('.approval')===null"), true, 'partial tool offered approval');
  const preparingCall = await evaluate("document.querySelector('.tool-call').dataset.toolCall");
  await pointerClick('.tool-call summary');
  assert.equal(await evaluate("document.querySelector('.tool-call details').open && document.querySelector('.tool-body pre').textContent==='{" + '\\"text\\":\\"hel' + "'"), true, 'partial tool arguments not inspectable');
  await screenshot('tool-preparing-desktop.png');
  await reloadDocument();
  await wait("document.querySelector('.tool-call summary .badge')?.textContent==='preparing'", 'partial tool survives refresh');
  assert.equal(await evaluate("document.querySelector('.tool-call').dataset.toolCall"), preparingCall, 'refresh changed streamed call identity');
  await pointerClick('.tool-call summary');
  await evaluate("window.preparingSummary=document.querySelector('.tool-call summary')");
  assert.equal((await fetch(upstreamFixture + '/browser-step')).status, 204);
  await wait("document.querySelector('.tool-call summary').textContent.includes('echo') && document.querySelector('.tool-body pre').textContent.endsWith('hello')", 'tool-only stream updates arguments and name');
  assert.equal(await evaluate("document.querySelector('.tool-call summary')===preparingSummary && document.querySelector('.tool-call details').open && document.activeElement===preparingSummary"), true, 'tool fragment detached the active disclosure');
  assert.equal(await evaluate("document.querySelector('.approval')===null && document.querySelectorAll('.tool-call').length===1"), true, 'partial call duplicated or became executable');
  await pointerClick('.tool-call summary');
  await wait("!document.querySelector('.tool-call details').open", 'collapsed preparing tool arguments');
  assert.equal((await fetch(upstreamFixture + '/browser-step')).status, 204);
  await wait("document.querySelector('.approval')!==null", 'tool approval');
  const approvalArguments = "document.querySelector('.tool-call details').open && JSON.parse(document.querySelector('.tool-body pre').textContent).text==='hello'";
  const transition = await evaluate(`({argumentsVisible: ${approvalArguments}, focusKept: document.activeElement===preparingSummary})`);
  await reloadDocument();
  await wait("document.querySelector('.approval')!==null", 'persisted approval after refresh');
  const approvalStyle = await evaluate(`{
    const panel = document.querySelector('.approval');
    const remember = panel.querySelector('.remember-approval');
    const allow = panel.querySelector('[data-action=approve]');
    const defaults = {allow: allow.textContent, deny: panel.querySelector('[data-action=deny]').textContent,
      checked: remember.checked, scope: remember.labels[0].textContent.trim()};
    remember.click();
    const remembered = allow.textContent;
    remember.click();
    ({defaults, remembered, reset: allow.textContent});
  }`);
  assert.deepEqual({toolMetadataSeparated, transition, reloadArguments: await evaluate(approvalArguments), ...approvalStyle}, {
    toolMetadataSeparated: true,
    transition: {argumentsVisible: true, focusKept: true},
    reloadArguments: true,
    defaults: {allow: 'Allow once', deny: 'Deny this call', checked: false,
      scope: 'Also allow later calls to this tool in this reply, even with different arguments.'},
    remembered: 'Allow for this reply',
    reset: 'Allow once',
  }, 'pending tool review, decision scope and Settings tool metadata');
  await click('.activity-tool-link');
  const approvedCall = await evaluate("document.querySelector('.tool-call').dataset.toolCall");
  assert.equal(await evaluate("document.activeElement?.closest('.tool-call')?.dataset.toolCall==="+JSON.stringify(approvedCall)), true, 'pending Activity entry did not focus the live tool summary');
  await evaluate("document.querySelector('.approval [data-action=approve]').click()");
  await wait("document.querySelector('#stop').hidden && document.querySelector('#messages').innerText.includes('The tool returned hello.')", 'tool continuation');
  assert.equal(await evaluate("document.activeElement?.closest('.tool-call')?.dataset.toolCall==="+JSON.stringify(approvedCall)+" && document.activeElement.closest('.tool-call details').open && document.activeElement.closest('.tool-call').querySelector('summary .badge').textContent==='completed' && document.activeElement.closest('.tool-call').innerText.includes('hello')"), true, 'streamed tool update lost summary focus, open state, or current status/result');
  assert.equal(await evaluate("document.querySelectorAll('#messages > .message:nth-last-child(-n+2) [data-action=reply-stats]:not([hidden])').length"), 1, 'tool-assisted reply has duplicate statistics buttons');
  await pointerClick('#messages > .message:last-of-type [data-action=reply-stats]');
  assert.equal(await evaluate("{const p=document.querySelector('#reply-statistics');p.querySelector('.stat-row strong').textContent==='240' && p.querySelectorAll('.reply-call-metrics').length===2}"), true, 'tool-assisted reply totals or per-call breakdown incorrect');
  await click('[data-action=close-stats]');
  const replyFlow = await evaluate(`(() => {
    const messages = [...document.querySelectorAll('#messages > .message')];
    const [tool, answer] = messages.slice(-2);
    return {tool: Boolean(tool.querySelector('.tool-call')), answer: answer.innerText,
      continuation: answer.classList.contains('continuation'),
      continued: [...document.querySelectorAll('.message.continuation')].length,
      firstHeader: getComputedStyle(tool.querySelector('.message-header')).display,
      nextHeader: getComputedStyle(answer.querySelector('.message-header')).display,
      label: answer.getAttribute('aria-label')};
  })()`);
  assert.equal(replyFlow.tool, true);
  assert(replyFlow.answer.includes('The tool returned hello.'));
  assert.equal(replyFlow.continuation, true, 'tool follow-up must read as the same assistant reply');
  assert.equal(replyFlow.continued, 1, 'separate replies must keep their own header');
  assert.notEqual(replyFlow.firstHeader, 'none');
  assert.equal(replyFlow.nextHeader, 'none');
  assert.equal(replyFlow.label, 'Assistant continuation');
  const replyTheme = await evaluate('document.documentElement.dataset.theme');
  const replyInspectorClosed = await evaluate("document.querySelector('#app').classList.contains('inspector-closed')");
  await evaluate("document.querySelector('#app').classList.add('inspector-closed')");
  const replyLayout = [];
  for (const width of [1440, 390, 320]) {
    await cdp('Emulation.setDeviceMetricsOverride', {width,height:width===1440?1000:844,deviceScaleFactor:1,mobile:width<500});
    await wait(`document.documentElement.dataset.compact==='${width<500}'`, 'reply viewport resize');
    for (const theme of ['dark', 'light']) {
      await evaluate(`document.documentElement.dataset.theme=${JSON.stringify(theme)};document.querySelector('#scroll-region').scrollTop=document.querySelector('#scroll-region').scrollHeight`);
      const geometry = await evaluate(`(() => {
        const answer=document.querySelector('.message.continuation'),tool=answer.previousElementSibling;
        const details=tool.querySelector('.tool-call details'),actions=answer.querySelector('.message-actions');
        return {gap:parseFloat(getComputedStyle(tool).paddingBottom)+parseFloat(getComputedStyle(answer).paddingTop),
          details:details.open,actions:actions.getBoundingClientRect().height,
          actionTarget:Math.min(...Array.from(actions.querySelectorAll('button:not([hidden])'),button=>button.getBoundingClientRect().height)),
          actionGap:parseFloat(getComputedStyle(actions).marginTop),
          bottomGap:parseFloat(getComputedStyle(answer).paddingBottom),
          toolHeight:tool.getBoundingClientRect().height,answerHeight:answer.getBoundingClientRect().height,
          overflow:document.documentElement.scrollWidth>innerWidth};
      })()`);
      replyLayout.push({width, theme, ...geometry});
      assert.equal(geometry.actionGap, geometry.bottomGap, 'completed reply has uneven space around its action row');
      assert(geometry.gap<=16, 'one reply still has full inter-message spacing');
      assert.equal(geometry.details, true, 'grouping collapsed the open tool details');
      assert(geometry.actions>=32, 'grouping hid the reply actions');
      assert.equal(geometry.overflow, false);
      await screenshot('reply-flow-'+width+'-'+theme+'.png');
    }
  }
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await evaluate(`document.documentElement.dataset.theme=${JSON.stringify(replyTheme)};document.querySelector('#app').classList.toggle('inspector-closed',${replyInspectorClosed})`);
  assert.equal(await evaluate("document.querySelector('.tool-call .approval')===null"), true, 'completed tool retained its obsolete approval panel');
  const chatBeforeSlowAttachment = await evaluate("localStorage.getItem('llmx-chat')");
  await evaluate("window.__originalArrayBuffer=Blob.prototype.arrayBuffer; Blob.prototype.arrayBuffer=function(){return new Promise(resolve=>{window.__releaseFileRead=()=>resolve(window.__originalArrayBuffer.call(this));});}");
  await upload('#file-input', fixture);
  await wait('typeof window.__releaseFileRead==="function"', 'paused attachment read');
  await click('[data-action="new"]');
  await wait(() => evaluate("localStorage.getItem('llmx-chat') !== " + JSON.stringify(chatBeforeSlowAttachment)), 'new conversation selection');
  await evaluate('window.__releaseFileRead()');
  await wait("document.querySelector('#toast').textContent.includes('Attachment read cancelled')", 'stale attachment notice');
  assert.equal(await evaluate("document.querySelector('#attachments').innerText"), '', 'late attachment read crossed into the newly selected conversation');
  await evaluate('Blob.prototype.arrayBuffer=window.__originalArrayBuffer');
  await click('[data-chat="' + chatBeforeSlowAttachment + '"]');
  await wait(async () => await evaluate("localStorage.getItem('llmx-chat')") === chatBeforeSlowAttachment, 'return to original conversation');
  await evaluate("window.__releaseFileRead=null; Blob.prototype.arrayBuffer=function(){return new Promise(resolve=>{window.__releaseFileRead=()=>resolve(window.__originalArrayBuffer.call(this));});}");
  await upload('#file-input', fixture);
  await wait('typeof window.__releaseFileRead==="function"', 'paused read before editing draft');
  await click('.message.user [data-action="edit"]');
  const editedDraftDuringRead = await evaluate("document.querySelector('#draft').value");
  await evaluate('window.__releaseFileRead()');
  await wait("document.querySelector('#toast').textContent.includes('The draft changed')", 'replaced draft attachment notice');
  assert.equal(await evaluate("document.querySelector('#attachments').innerText"), '', 'late read changed another draft in the same chat');
  assert.equal(await evaluate("document.querySelector('#draft').value"), editedDraftDuringRead);
  await evaluate('Blob.prototype.arrayBuffer=window.__originalArrayBuffer');
  await click('[data-action="cancel-edit"]');
  await upload('#file-input', fixture);
  await wait("document.querySelector('#attachments').innerText.includes('notes.txt')", 'attachment chip');
  await upload('#file-input', imageFixture);
  await wait("document.querySelector('#attachments').innerText.includes('pixel.png')", 'image attachment chip');
  const beforeImageRefusal = await evaluate("document.querySelectorAll('.message').length");
  await send('Reject image test');
  await wait("!document.querySelector('#send').disabled && document.querySelector('#banner').textContent.includes('images')", 'friendly image rejection');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Reject image test', 'rejected image lost draft');
  assert.equal(await evaluate("document.querySelectorAll('#attachments .attachment-pill').length"), 2, 'rejected image lost attachments');
  assert.equal(await evaluate("document.querySelectorAll('.message').length"), beforeImageRefusal, 'rejected image created a message');
  assert.equal(await evaluate("document.querySelector('#stop').hidden"), true, 'rejected image started generation');
  await send('Read the attachment.');
  await wait("!document.querySelector('#stop').hidden && document.querySelector('#messages').innerText.includes('The attached file is readable.')", 'attachment text before metrics');
  const attachmentStats = ".message.assistant:last-of-type [data-action=reply-stats]:not([hidden])";
  await evaluate("document.querySelector("+JSON.stringify(attachmentStats)+").focus({preventScroll:true})");
  await wait("document.querySelector('#reply-statistics .stat-row strong')?.textContent==='Unavailable'", 'attachment statistics before metrics');
  await cdp('Input.dispatchKeyEvent', {type:'keyDown', key:'Escape', code:'Escape', windowsVirtualKeyCode:27});
  await cdp('Input.dispatchKeyEvent', {type:'keyUp', key:'Escape', code:'Escape', windowsVirtualKeyCode:27});
  await evaluate("document.activeElement.blur()");
  await evaluate(`(() => {
    const target='/api/chats/'+localStorage.getItem('llmx-chat'), original=window.fetch;
    window.releaseTerminalTooltipSnapshot=null;
    window.fetch=async (...args)=>{
      const response=await original(...args);
      if(args[0]!==target || args[1]?.method!=='GET') return response;
      window.fetch=original;
      return new Promise(resolve=>window.releaseTerminalTooltipSnapshot=()=>resolve(response));
    };
  })()`);
  await cdp('Input.dispatchMouseEvent', {type:'mouseMoved', x:10, y:90});
  await evaluate("window.metricsFocusedButton=document.querySelector('.message.assistant:last-of-type [data-action=retry]');window.metricsFocusedButton.focus({preventScroll:true})");
  assert.equal(await evaluate("document.activeElement===window.metricsFocusedButton"), true, 'Retry is focused before metrics arrive');
  assert.equal((await fetch(upstreamFixture + '/browser-step')).status, 204);
  await wait("document.querySelector('#stop').hidden && typeof window.releaseTerminalTooltipSnapshot==='function'", 'terminal snapshot held while actions are visible');
  assert.equal(await evaluate("document.activeElement===window.metricsFocusedButton && window.metricsFocusedButton.isConnected && document.querySelector('.message-action-tooltip')?.textContent==='Retry response'"), true, 'reported metrics removed a focused action or tooltip');
  await click('[data-tab="stats"]');
  await click('#panel-stats .conversation-totals > summary');
  assert((await evaluate("document.querySelector('#panel-stats').innerText")).includes('960'), 'all-branch totals omitted retries');
  assert.equal(await evaluate("document.querySelectorAll('.attachment-image').length"), 1, 'image attachment missing');
  assert.equal(await evaluate(`(() => {
    const retry = document.querySelector('.message.assistant [data-action=retry]');
    const edit = document.querySelector('.message.user [data-action=edit]');
    return retry.getAttribute('aria-label') === 'Retry response' && [retry, edit].every(button => {
      const style = getComputedStyle(button), rect = button.getBoundingClientRect();
      return button.querySelector('svg[aria-hidden=true]') && !button.textContent.trim()
        && style.backgroundColor === 'rgba(0, 0, 0, 0)' && style.borderTopColor === 'rgba(0, 0, 0, 0)'
        && rect.width >= 32 && rect.height >= 32;
    });
  })()`), true, 'idle user and assistant actions use accessible, borderless icons and full hit targets');
  await evaluate("{const b=document.querySelector('.message.assistant [data-action=copy]');b.blur();b.focus({preventScroll:true})}");
  await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9});
  await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9});
  await wait("document.activeElement?.getAttribute('aria-describedby')?.startsWith('message-action-tooltip-')", 'keyboard focus tooltip');
  assert.equal(await evaluate("{const b=document.activeElement,t=document.getElementById(b.getAttribute('aria-describedby'));b.closest('.message-actions')!==null && getComputedStyle(b).outlineStyle==='solid' && getComputedStyle(b).outlineWidth==='2px' && t?.getAttribute('role')==='tooltip'}"), true, 'message action keyboard focus or tooltip is not visible');
  await evaluate(`(() => {
    const button=document.activeElement;
    window.snapshotRetryButton=button; window.tooltipSnapshotRendered=false;
    if(button.dataset.action!=='retry') throw new Error('Tab did not focus Retry before snapshot');
    const article=button.closest('.message');
    const observer=new MutationObserver(()=>{window.tooltipSnapshotRendered=true;observer.disconnect();});
    observer.observe(article,{childList:true});
    window.releaseTerminalTooltipSnapshot();
  })()`);
  await wait("window.tooltipSnapshotRendered", 'held terminal snapshot reconciled the focused message');
  const snapshotFocus = await evaluate("({connected:window.snapshotRetryButton.isConnected,focused:document.activeElement===window.snapshotRetryButton,active:document.activeElement.dataset.action||document.activeElement.tagName,tooltip:document.querySelector('.message-action-tooltip')?.textContent||null,statsOpen:!!document.querySelector('#reply-statistics')})");
  assert.deepEqual(snapshotFocus,{connected:true,focused:true,active:'retry',tooltip:'Retry response',statsOpen:false}, 'terminal snapshot must preserve the keyboard-focused Retry control and tooltip');
  await evaluate("document.querySelector("+JSON.stringify(attachmentStats)+").focus({preventScroll:true})");
  await wait("document.querySelector('#reply-statistics .stat-row strong')?.textContent==='120'", 'attachment statistics refresh when reopened after metrics');
  await cdp('Input.dispatchKeyEvent', {type:'keyDown', key:'Escape', code:'Escape', windowsVirtualKeyCode:27});
  await cdp('Input.dispatchKeyEvent', {type:'keyUp', key:'Escape', code:'Escape', windowsVirtualKeyCode:27});
  await evaluate("document.activeElement.blur()");
  const retryPoint = await actionHoverPoint('.message.assistant [data-action=retry]');
  assert(retryPoint.hit, 'Retry hover target is obscured: ' + JSON.stringify(retryPoint));
  await cdp('Input.dispatchMouseEvent', {type: 'mouseMoved', x: retryPoint.x, y: retryPoint.y});
  const retryHover = await evaluate("{const b=document.querySelector('.message.assistant [data-action=retry]'),t=document.getElementById(b.getAttribute('aria-describedby')),r=b.getBoundingClientRect(),tr=t?.getBoundingClientRect(),hit=document.elementFromPoint("+retryPoint.x+","+retryPoint.y+"),style=getComputedStyle(b);({hover:b.matches(':hover'),aria:b.getAttribute('aria-describedby'),tip:t?.textContent||null,tipRect:tr?{left:tr.left,right:tr.right,top:tr.top,bottom:tr.bottom}:null,buttonRect:{left:r.left,right:r.right,top:r.top,bottom:r.bottom,width:r.width,height:r.height},point:"+JSON.stringify(retryPoint)+",hit:hit?{tag:hit.tagName,className:String(hit.className),inside:b.contains(hit)}:null,active:document.activeElement?.outerHTML?.slice(0,180),display:style.display,visibility:style.visibility,pointerEvents:style.pointerEvents,viewport:{width:innerWidth,height:innerHeight,scale:visualViewport.scale,offsetLeft:visualViewport.offsetLeft,offsetTop:visualViewport.offsetTop}})}");
  assert(retryHover.hover && retryHover.tip==='Retry response' && retryHover.tipRect.left>=0 && retryHover.tipRect.right<=retryHover.viewport.width,
    'message action hover tooltip is missing or clipped: '+JSON.stringify(retryHover));
  await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27});
  await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27});
  assert.equal(await evaluate("!document.querySelector('.message-action-tooltip') && !document.querySelector('.message.assistant [data-action=retry]').hasAttribute('aria-describedby')"), true, 'Escape did not dismiss the action tooltip');
  await evaluate(`{
    const region = document.querySelector('#scroll-region');
    region.style.maxHeight = Math.floor(Math.min(region.scrollHeight, region.clientHeight) / 2) + 'px';
  }`);
  const userEditPoint = await actionHoverPoint('.message.user [data-action=edit]');
  assert(userEditPoint.hit, 'Edit hover target is obscured: ' + JSON.stringify(userEditPoint));
  await cdp('Input.dispatchMouseEvent', {type: 'mouseMoved', x: userEditPoint.x, y: userEditPoint.y});
  assert.equal(await evaluate("{const b=document.querySelector('.message.user [data-action=edit]'),t=document.getElementById(b.getAttribute('aria-describedby')),r=t?.getBoundingClientRect(); b.matches(':hover') && t?.textContent==='Edit message' && r.left>=0 && r.right<=innerWidth}"), true, 'edge message tooltip is clipped');
  const moved = await evaluate(`{
    const region = document.querySelector('#scroll-region'), before = region.scrollTop;
    region.scrollTop += before >= 4 ? -4 : 4;
    region.scrollTop !== before;
  }`);
  assert.equal(moved, true, 'the tooltip scroll test must actually move the transcript');
  await wait("!document.querySelector('.message-action-tooltip')", 'tooltip dismissal while scrolling');
  await evaluate("document.querySelector('#scroll-region').style.removeProperty('max-height')");
  await cdp('Input.dispatchMouseEvent', {type: 'mouseMoved', x: 10, y: 90});
  await click('.message.assistant [data-action=copy]');
  await wait("!document.querySelector('#toast').hidden && document.querySelector('#toast').textContent==='Message copied.'", 'accessible message copy confirmation');
  await evaluate("document.querySelector('#toast').hidden=true");
  await screenshot('desktop-conversation.png');
  const exported = await evaluate("fetch('/api/chats/'+localStorage.getItem('llmx-chat')+'/export').then(r=>r.json())");
  const importFile = path.join(profile, 'import.json'); await writeFile(importFile, JSON.stringify(exported));
  await upload('#import-input', importFile);
  await wait("document.querySelector('#toast').textContent.includes('Conversation imported')", 'conversation import');
  await click('.topbar [data-action="chat-menu"]'); await click('[data-action="pin"]');
  await wait("document.querySelector('.chat-item.active .pin-mark')!==null", 'pin conversation');
  await click('.topbar [data-action="chat-menu"]'); await click('[data-action="folder"]');
  await fill('#conversation-value', 'Research'); await closeAction('#conversation-submit');
  await wait("fetch('/api/chats/'+localStorage.getItem('llmx-chat')).then(r=>r.json()).then(d=>d.chat.folder==='Research')", 'persisted folder mutation');
  await send('Trigger a failed upstream response.');
  await wait("document.querySelector('#stop').hidden && !document.querySelector('#banner').hidden", 'upstream failure banner');
  const streamsBeforeFailedReload = runStreams.length;
  await reloadDocument();
  await wait("document.querySelector('#stop').hidden && Array.from(document.querySelectorAll('.message.assistant')).at(-1)?.textContent.includes('Incomplete response')", 'persisted failed response');
  assert.equal(await evaluate("!document.querySelector('#banner').hidden && document.querySelector('#banner').textContent.includes('disconnected')"), true, 'failed snapshot did not restore its error banner');
  assert.equal(runStreams.length, streamsBeforeFailedReload, 'completed failure snapshot unexpectedly subscribed to active generation');
  await evaluate(`(() => {
    const original = window.fetch;
    const path = '/api/chats/' + localStorage.getItem('llmx-chat');
    window.fetch = async (...args) => {
      const response = await original(...args);
      if (args[0] !== path) return response;
      window.fetch = original;
      const snapshot = await response.json();
      const leaf = snapshot.messages.find(m => m.id === snapshot.chat.active_leaf);
      snapshot.runs.find(r => r.id === leaf.run_id).error = 'llmx returned HTTP 400: the chat template refused the messages: can only concatenate str (not "list") to str';
      return new Response(JSON.stringify(snapshot), {status: 200, headers: {'Content-Type': 'application/json'}});
    };
  })()`);
  await click('.chat-item.active');
  await wait("document.querySelector('#banner').textContent.includes('Edit the message')", 'legacy image failure explanation');
  assert.equal(await evaluate("document.querySelector('#banner').textContent.includes('concatenate')"), false, 'legacy template error leaked into banner');
  assert.equal(await evaluate("document.querySelector('.activity-error details pre')?.textContent.includes('concatenate')"), true, 'Activity lost original error details');
  await reloadDocument();
  await wait("document.querySelector('#banner').textContent.includes('disconnected')", 'unrelated failure stays specific');

  await send('Cancel this response.');
  await wait("document.querySelector('#messages').textContent.includes('Cancellation test') && !document.querySelector('#stop').hidden", 'cancellable live response');
  await click('#stop');
  await wait("document.querySelector('#stop').hidden && document.querySelector('#messages').innerText.includes('cancelled')", 'cancelled response');
  await cdp('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
  await evaluate("window.dispatchEvent(new Event('resize'))");
  await click('[data-action="close-drawers"]');
  await click('[data-action="bottom"]');
  assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'), true, 'mobile page overflows');
  await screenshot('mobile-conversation.png');
  await click('[data-action="sidebar"]');
  assert.equal(await evaluate("!document.querySelector('#drawer-shade').hidden"), true);
  await screenshot('mobile-history.png');
  await click('[data-action="close-drawers"]');
  await click('[data-action="sidebar"]');
  await click('[data-action="theme"]');
  await click('[data-action="close-drawers"]');
  assert.equal(await evaluate('document.documentElement.dataset.theme'), 'light');
  assert.equal(await evaluate("getComputedStyle(document.querySelector('.brand-wordmark')).maskImage.includes('/logo.svg')"), true, 'light theme wordmark');
  await screenshot('mobile-light.png');
  assert.equal(await evaluate("{const b=document.querySelector('.message.assistant [data-action=retry]'),r=b?.getBoundingClientRect(),m=document.querySelector('.message.assistant [data-action=reply-stats]:not([hidden])')?.getBoundingClientRect();Boolean(b?.querySelector('svg[aria-hidden=true]') && r && r.width>=40 && r.height>=40 && r.right<=innerWidth && m && m.width>=40 && m.right<=innerWidth)}"), true, 'mobile message action sizing or statistics control is wrong');
  await screenshot('mobile-actions-light.png');
  await click('.topbar [data-action="chat-menu"]'); await click('[data-action="archive"]');
  await wait("!document.querySelector('.chat-item.active')", 'archived conversation leaves active library');
  await click('[data-action="archive-view"]');
  await wait("document.querySelector('.chat-item.active')!==null", 'archive listing');
  await click('.topbar [data-action="chat-menu"]'); await click('[data-action="archive"]');
  await wait("!document.querySelector('.chat-item.active')", 'restored conversation leaves archive');
  await click('[data-action="archive-view"]');
  await checkNewChatShortcut();
  await click('.topbar [data-action="chat-menu"]'); await click('[data-action="delete"]'); await closeAction('#conversation-submit');
  await wait("document.querySelector('#chat-title').disabled && document.querySelector('#chat-title').textContent==='Chat'", 'confirmed delete');
  await cdp('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
  const createGeometry = await evaluate(`Array.from(document.querySelectorAll('#sidebar .new-chat')).map(button => {
    const label = button.querySelector('.chat-create-label').getBoundingClientRect(); const box = button.getBoundingClientRect();
    return {labelX: label.x, height: box.height, weight: getComputedStyle(button).fontWeight};
  })`);
  assert.equal(createGeometry.length, 1);
  const sidebarTarget = await evaluate(`fetch('/api/chats',{method:'POST',headers:{'Content-Type':'application/json','X-Chat-Request':'1'},body:JSON.stringify({title:'Sidebar target'})}).then(r=>r.json()).then(c=>c.id)`);
  await reloadDocument();
  await wait("document.querySelector('.chat-row-menu')!==null && !document.querySelector('[data-action=\"new\"]').disabled", 'sidebar actions after reload');
  await click('[data-action="new"]');
  await wait("document.querySelector('.chat-item.active')!==null", 'separate selected chat');
  const selectedBeforeMenu = await evaluate("document.querySelector('.chat-item.active').dataset.chat");
  assert.notEqual(selectedBeforeMenu, sidebarTarget);
  const targetMenu = '.chat-row-menu[data-id="' + sidebarTarget + '"]';
  const selectedMenu = '.chat-row.active .chat-row-menu';
  for (const menuSelector of [selectedMenu, targetMenu]) {
    assert.equal(await evaluate(`(()=>{const row=document.querySelector(${JSON.stringify(menuSelector)}).parentElement,m=document.querySelector(${JSON.stringify(menuSelector)}).getBoundingClientRect(),r=row.querySelector('.chat-item').getBoundingClientRect();return Math.abs(m.top-r.top)<1&&Math.abs(m.bottom-r.bottom)<1})()`), true, 'sidebar menu target does not match its chat row bounds');
    const hoverPoint = await evaluate(`(()=>{const r=document.querySelector(${JSON.stringify(menuSelector)}).getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()`);
    await cdp('Input.dispatchMouseEvent', {type: 'mouseMoved', ...hoverPoint});
    assert.equal(await evaluate(`(()=>{const row=document.querySelector(${JSON.stringify(menuSelector)}).parentElement;return getComputedStyle(row.querySelector('.chat-row-menu')).backgroundColor===getComputedStyle(row.querySelector('.chat-item')).backgroundColor})()`), true, 'sidebar menu hover does not match its chat row background');
  }
  const assertSidebarFocusPaint = async context => {
    const paint = await sidebarFocusSamples(targetMenu);
    const expected = paint.expected.match(/\d+/g).map(Number).slice(0, 3);
    assert.equal(paint.pixels.length, 4, 'sidebar focus screenshot could not be sampled');
    assert.equal(paint.pixels.every(pixel => pixel.slice(0, 3).every((channel, index) => channel === expected[index])), true, context + ' sidebar keyboard focus does not fill both row edges: ' + JSON.stringify(paint));
  };
  await evaluate(`document.querySelector(${JSON.stringify(targetMenu)}).focus({focusVisible:true})`);
  for (const theme of ['light', 'dark']) {
    await evaluate(`document.documentElement.dataset.theme=${JSON.stringify(theme)}`);
    await assertSidebarFocusPaint('Desktop ' + theme);
    if (theme === 'light') await screenshot('sidebar-chat-row-focus-desktop.png');
  }
  const drawerKey = async (key, code, modifiers = 0) => {
    await cdp('Input.dispatchKeyEvent', {type:'keyDown', key, code:key, windowsVirtualKeyCode:code, modifiers, ...(key==='Enter'?{text:'\r'}:{})});
    await cdp('Input.dispatchKeyEvent', {type:'keyUp', key, code:key, windowsVirtualKeyCode:code, modifiers});
  };
  await pointerClick(targetMenu);
  await click('#action-menu [data-action="rename"]');
  await wait("document.querySelector('#action-dialog').open", 'desktop rename dialog before resize');
  await evaluate("window.desktopRenameClosed=false;document.querySelector('#action-dialog').addEventListener('close',()=>window.desktopRenameClosed=true,{once:true})");
  await cdp('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: false});
  await wait("document.documentElement.dataset.compact==='true' && getComputedStyle(document.querySelector('#sidebar')).visibility==='hidden'", 'resize hides the desktop dialog origin');
  await drawerKey('Escape', 27);
  await wait("window.desktopRenameClosed", 'desktop rename closes after resize');
  assert.equal(await evaluate("document.activeElement.id"), 'sidebar-toggle', 'closing Rename after resize must focus the visible sidebar opener');
  await click('#drawer-shade'); await wait("document.querySelector('#drawer-shade').hidden", 'narrow drawer shade closes');
  const openKeyboardSidebar = async () => {
    await evaluate("document.querySelector('.mobile-nav').focus()");
    await drawerKey('Enter', 13);
    await wait("document.querySelector('#app').classList.contains('sidebar-open') && document.querySelector('#sidebar').getBoundingClientRect().left>=0", 'narrow sidebar opens and finishes sliding');
  };
  await evaluate(`(() => {
    window.drawerTrace = [];
    const describe = el => el ? [el.id, el.dataset?.action, el.dataset?.view, el.dataset?.id].filter(Boolean).join('/') || el.tagName : '';
    const types=['keydown', 'keypress', 'keyup', 'click', 'focusin', 'close'];
    const record=event=>{
      drawerTrace.push({type:event.type, key:event.key, target:describe(event.target), active:describe(document.activeElement),
        view:document.querySelector('#app').dataset.view, modal:document.querySelector('aside[aria-modal=true]')?.id,
        menu:document.querySelector('#action-menu').matches(':popover-open')});
      if (drawerTrace.length>24) drawerTrace.shift();
    };
    for (const type of types) document.addEventListener(type, record, true);
    window.stopDrawerTrace=()=>{for (const type of types) document.removeEventListener(type, record, true);};
  })()`);
  const sidebarFocusHome = "document.activeElement===document.querySelector('#sidebar [data-action=close-drawers]')";
  const mainIsInert = "document.querySelector('main').inert && document.querySelector('.skip-link').inert";
  await openKeyboardSidebar();
  assert.deepEqual(await evaluate("({focus:"+sidebarFocusHome+", inert:"+mainIsInert+", modal:document.querySelector('#sidebar').getAttribute('aria-modal'), compact:document.documentElement.dataset.compact})"), {focus:true, inert:true, modal:'true', compact:'true'}, 'keyboard sidebar opening must move focus into a modal drawer and make the background inert');
  assert.equal(await evaluate("document.querySelector('#draft').focus();"+sidebarFocusHome), true, 'programmatic focus escaped the sidebar');
  await drawerKey('Tab', 9, 8);
  assert.equal(await evaluate("document.activeElement.id"), 'logout-button', 'sidebar backward Tab did not wrap to the last control');
  await drawerKey('Tab', 9);
  assert.equal(await evaluate(sidebarFocusHome), true, 'sidebar forward Tab did not wrap');
  await pointerClick(targetMenu);
  assert.equal(await evaluate("document.querySelector('#sidebar').contains(document.activeElement) && document.querySelector('#action-menu').matches(':popover-open')"), true, 'open sidebar menu leaves the modal drawer');
  await drawerKey('Escape', 27);
  assert.equal(await evaluate("document.querySelector('#app').classList.contains('sidebar-open') && document.activeElement.dataset.id==="+JSON.stringify(sidebarTarget)), true, 'Escape closed the drawer instead of only its menu');
  await pointerClick(targetMenu);
  await click('#action-menu [data-action="rename"]');
  await wait("document.querySelector('#action-dialog').open && document.activeElement.id==='conversation-value'", 'native dialog over the sidebar');
  await evaluate(`(() => {
    const dialog=document.querySelector('#action-dialog'), close=dialog.onclose;
    window.finishDrawerDialogClose=null;
    dialog.onclose=()=>{dialog.onclose=close;window.finishDrawerDialogClose=close;};
  })()`);
  await drawerKey('Escape', 27);
  await wait("!document.querySelector('#action-dialog').open && document.activeElement.dataset.id==="+JSON.stringify(sidebarTarget)+" && typeof window.finishDrawerDialogClose==='function'", 'native dialog returns to sidebar row before delayed close callback');
  assert.deepEqual(await evaluate(`(() => {
    document.querySelector('.global-nav [data-view=workbench]').focus();
    const before=document.activeElement.dataset.view;
    window.finishDrawerDialogClose();
    return {before, after:document.activeElement.dataset.view};
  })()`), {before:'workbench', after:'workbench'}, 'late dialog close callback stole focus from Workspace');
  await drawerKey('Escape', 27);
  assert.equal(await evaluate("document.activeElement.matches('.mobile-nav') && !document.querySelector('main').inert && document.querySelector('#drawer-shade').hidden"), true, 'Escape did not close the sidebar and restore its opener');
  for (const view of ['monitor', 'workbench', 'preferences']) {
    await openKeyboardSidebar();
    const selector = JSON.stringify('.global-nav [data-view='+view+']');
    await evaluate("document.querySelector("+selector+").focus({preventScroll:true})");
    assert.equal(await evaluate("document.activeElement.matches("+selector+")"), true, view+' navigation did not receive focus');
    await drawerKey('Enter', 13);
    try { await wait("document.querySelector('#app').dataset.view==="+JSON.stringify(view), view+' navigation'); }
    catch (error) { throw new Error(error.message + '\nDrawer events: ' + JSON.stringify(await evaluate('drawerTrace'))); }
    assert.deepEqual(await evaluate("({focus:document.activeElement.id || document.activeElement.outerHTML.slice(0,160), inert:document.querySelector('main').inert, shade:document.querySelector('#drawer-shade').hidden})"), {focus:'shell-title', inert:false, shade:true}, view+' navigation did not release modal state and focus the destination heading');
  }
  await openKeyboardSidebar();
  await fill('#search', 'Sidebar target');
  await wait("document.querySelector("+JSON.stringify(targetMenu)+")!==null", 'sidebar search finds menu row');
  await pointerClick(targetMenu);
  await fill('#search', 'no matching sidebar chat');
  assert.equal(await evaluate("!document.querySelector("+JSON.stringify(targetMenu)+") && document.querySelector('#action-menu').matches(':popover-open')"), true, 'search refresh did not remove the open menu anchor');
  await drawerKey('Escape', 27);
  assert.deepEqual(await evaluate("({focus:document.activeElement.id, modal:document.querySelector('aside[aria-modal=true]')?.id, menu:document.querySelector('#action-menu').matches(':popover-open')})"), {focus:'search', modal:'sidebar', menu:false}, 'sidebar menu Escape must focus Search when the original row is gone');
  await fill('#search', '');
  await evaluate("window.stopDrawerTrace(); document.querySelector('.global-nav [data-view=chat]').focus()");

  await drawerKey('Enter', 13);
  assert.equal(await evaluate("document.querySelector('#app').dataset.view==='chat' && document.activeElement.id==='chat-title'"), true, 'return to Chat did not focus its title');
  await evaluate("document.querySelector('.composer-toolbar [data-action=settings]').focus()");
  await drawerKey('Enter', 13);
  assert.equal(await evaluate("document.activeElement.id==='tab-settings' && document.querySelector('#inspector').getAttribute('aria-modal')==='true' && "+mainIsInert), true, 'composer Settings did not focus the selected tab in a modal Details drawer');
  await drawerKey('Escape', 27);
  assert.equal(await evaluate("document.activeElement.matches('.composer-toolbar [data-action=settings]') && !document.querySelector('main').inert"), true, 'Details did not return focus to its Settings opener');
  await evaluate("document.querySelector('#details-toggle').focus()");
  await drawerKey('Enter', 13);
  assert.equal(await evaluate("document.activeElement.id==='tab-settings' && document.querySelector('#details-toggle').getAttribute('aria-expanded')==='true'"), true, 'header Details did not focus its current tab');
  await drawerKey('Tab', 9, 8);
  assert.equal(await evaluate("document.activeElement.matches('#inspector [aria-label=\"Close details\"]')"), true, 'Details backward Tab visited a roving inactive tab');
  await drawerKey('Escape', 27);
  assert.equal(await evaluate("document.activeElement.id==='details-toggle' && document.querySelector('#drawer-shade').hidden"), true, 'Details Escape did not restore the header opener');
  await openKeyboardSidebar();
  await cdp('Input.dispatchMouseEvent', {type:'mousePressed', x:380, y:420, button:'left', clickCount:1});
  await cdp('Input.dispatchMouseEvent', {type:'mouseReleased', x:380, y:420, button:'left', clickCount:1});
  assert.equal(await evaluate("document.activeElement.id==='sidebar-toggle' && document.querySelector('#drawer-shade').hidden && !document.querySelector('main').inert"), true, 'pointer backdrop did not return focus and release modality');
  await openKeyboardSidebar();
  await evaluate(`document.querySelector(${JSON.stringify(targetMenu)}).scrollIntoView({block:'center'});document.querySelector(${JSON.stringify(targetMenu)}).focus({focusVisible:true})`);
  for (const theme of ['light', 'dark']) {
    await evaluate(`document.documentElement.dataset.theme=${JSON.stringify(theme)}`);
    await assertSidebarFocusPaint('Mobile ' + theme);
    if (theme === 'light') await screenshot('sidebar-chat-row-focus-mobile.png');
  }
  await cdp('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
  await wait("document.documentElement.dataset.compact==='false' && getComputedStyle(document.querySelector('.sidebar')).visibility==='visible'", 'wide sidebar restored');
  assert.equal(await evaluate("document.activeElement.dataset.id==="+JSON.stringify(sidebarTarget)+" && !document.querySelector('#sidebar').hasAttribute('aria-modal') && document.querySelector('#drawer-shade').hidden"), true, 'resizing an open sidebar lost focus or retained modal state');
  await evaluate("if(document.querySelector('#app').classList.contains('inspector-closed')) document.querySelector('#details-toggle').click(); document.querySelector('#draft').focus()");
  assert.equal(await evaluate("!document.querySelector('main').inert && !document.querySelector('#inspector').hasAttribute('aria-modal') && document.activeElement.id==='draft'"), true, 'docked Details trapped desktop focus');
  await evaluate("document.querySelector('#setting-system').focus()");
  assert.equal(await evaluate("document.activeElement.id"), 'setting-system', 'docked system instructions did not receive focus');
  await cdp('Emulation.setDeviceMetricsOverride', {width:390, height:844, deviceScaleFactor:1, mobile:false});
  await wait("document.documentElement.dataset.compact==='true'", 'docked Details becomes a drawer');
  assert.equal(await evaluate("document.activeElement.id==='details-toggle' && document.querySelector('#drawer-shade').hidden"), true, 'narrowing a docked panel left focus on a hidden control');
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440, height:1000, deviceScaleFactor:1, mobile:false});
  await wait("document.documentElement.dataset.compact==='false'", 'restore wide Details preference');
  await evaluate("if(!document.querySelector('#app').classList.contains('inspector-closed')) document.querySelector('#details-toggle').click()");

  await pointerClick(targetMenu);
  assert.equal(await evaluate("document.querySelector('#action-menu-title').textContent"), 'Sidebar target');
  await click('#action-menu [data-action="rename"]');
  await wait("document.querySelector('#action-dialog').open", 'styled rename dialog');
  assert.equal(await evaluate("document.activeElement.id"), 'conversation-value');
  await fill('#conversation-value', 'Research'); await closeAction('#conversation-submit');
  await wait(`document.querySelector('[data-chat="${sidebarTarget}"] .chat-name').textContent==='Research'`, 'rename unselected sidebar chat');
  assert.equal(await evaluate("document.querySelector('.chat-item.active').dataset.chat"), selectedBeforeMenu, 'menu switched the active chat');
  await wait(`document.activeElement?.dataset.id===${JSON.stringify(sidebarTarget)}`, 'rename restores focus to refreshed row');
  await pointerClick(targetMenu);
  await screenshot('sidebar-chat-actions-desktop.png');
  assert.equal(await evaluate("document.querySelector('#action-dialog').open"), false, 'menu opens modal');
  await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: 'End'});
  assert.equal(await evaluate("document.activeElement.dataset.action"), 'delete');
  await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: 'Escape'});
  assert.equal(await evaluate("document.querySelector('#action-menu').matches(':popover-open')"), false);
  assert.equal(await evaluate("document.activeElement.dataset.id"), sidebarTarget, 'menu focus was not restored');
  await pointerClick(targetMenu);
  await cdp('Input.dispatchMouseEvent', {type: 'mousePressed', x: 800, y: 90, button: 'left', clickCount: 1});
  await cdp('Input.dispatchMouseEvent', {type: 'mouseReleased', x: 800, y: 90, button: 'left', clickCount: 1});
  assert.equal(await evaluate("document.querySelector('#action-menu').matches(':popover-open')"), false, 'outside click did not close menu');
  await pointerClick(targetMenu); await click('#action-menu [data-action="folder"]');
  assert.equal(await evaluate("document.querySelector('#conversation-value').maxLength"), 200, 'folder maximum differs from server');
  await fill('#conversation-value', 'Projects'); await closeAction('#conversation-submit');
  await wait(`document.querySelector('#chat-list').textContent.includes('Projects')`, 'move target to folder');
  await pointerClick(targetMenu); await click('#action-menu [data-action="rename"]');
  assert.equal(await evaluate("document.querySelector('#conversation-value').maxLength"), 200, 'rename maximum differs from server');
  const maximumTitle = 'R'.repeat(200);
  await fill('#conversation-value', maximumTitle); await closeAction('#conversation-submit');
  await wait(`document.querySelector('[data-chat="${sidebarTarget}"] .chat-name').textContent===${JSON.stringify(maximumTitle)}`, 'maximum-length original title');
  const countBeforeDuplicate = await evaluate("document.querySelectorAll('.chat-row').length");
  await pointerClick(targetMenu); await click('#action-menu [data-action="duplicate"]');
  await wait(`document.querySelectorAll('.chat-row').length===${countBeforeDuplicate + 1}`, 'duplicate target chat');
  const duplicateTitle = await evaluate("fetch('/api/chats').then(r=>r.json()).then(d=>d.chats.find(c=>c.title.endsWith(' (copy)')).title)");
  assert.equal(duplicateTitle, maximumTitle.slice(0, 193) + ' (copy)', 'maximum-length duplicate title is not persisted within 200 characters');
  assert.equal(duplicateTitle.length, 200);
  assert.equal(await evaluate("document.querySelector('.chat-item.active').dataset.chat"), selectedBeforeMenu);
  await pointerClick(targetMenu); await click('#action-menu [data-action="pin"]');
  await wait(`document.querySelector('[data-chat="${sidebarTarget}"] .pin-mark')!==null`, 'pin target chat');
  await pointerClick(targetMenu); await click('#action-menu [data-action="archive"]');
  await wait(`document.querySelector('[data-chat="${sidebarTarget}"]')===null`, 'archive target chat');
  assert.equal(await evaluate("document.querySelector('.chat-item.active').dataset.chat"), selectedBeforeMenu);
  await click('[data-action="archive-view"]'); await wait(`document.querySelector('${targetMenu}')!==null`, 'archived row menu');
  await pointerClick(targetMenu); await click('#action-menu [data-action="archive"]');
  await click('[data-action="archive-view"]'); await wait(`document.querySelector('${targetMenu}')!==null`, 'restored row menu');
  await cdp('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
  await click('[data-action="close-drawers"]'); await delay(230);
  await pointerClick('.mobile-nav'); await delay(230); await pointerClick(targetMenu);
  assert.equal(await evaluate("{const r=document.querySelector('#action-menu').getBoundingClientRect(); r.left>=0 && r.right<=innerWidth && r.top>=0 && r.bottom<=innerHeight}"), true, 'mobile menu overflows viewport');
  await screenshot('sidebar-chat-actions-mobile.png');
  await click('#action-menu [data-action="rename"]'); await fill('#conversation-value', '   ');
  await evaluate("document.querySelector('#conversation-form').requestSubmit()");
  assert.equal(await evaluate("document.querySelector('#conversation-value').validity.customError"), true);
  await closeAction('#conversation-cancel'); await pointerClick(targetMenu);
  await click('#action-menu [data-action="delete"]');
  assert.equal(await evaluate("document.querySelector('#conversation-value').disabled && !document.querySelector('#conversation-value').validity.customError"), true, 'delete retained invalid rename state');
  await wait("document.querySelector('#action-dialog').open", 'styled delete dialog'); await closeAction('#conversation-submit');
  await wait(`document.querySelector('[data-chat="${sidebarTarget}"]')===null`, 'delete target from mobile sidebar');
  assert.equal(await evaluate("document.querySelector('.chat-item.active').dataset.chat"), selectedBeforeMenu);
  await click('[data-action="close-drawers"]');
  await cdp('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
  await evaluate("document.documentElement.style.zoom='2';window.dispatchEvent(new Event('resize'))");
  await click('.topbar [data-action="chat-menu"]');
  assert.equal(await evaluate("{const r=document.querySelector('#action-menu').getBoundingClientRect(); r.left>=0 && r.right<=innerWidth && r.top>=0 && r.bottom<=innerHeight}"), true, '200% menu overflows viewport');
  await screenshot('chat-actions-200-percent.png');
  await evaluate("window.ChatShell.closeMenu();document.documentElement.style.zoom='1';window.dispatchEvent(new Event('resize'))");
  if (!await detailsOpen()) await click(detailsToggle);
  const navigationChat = await evaluate(`fetch('/api/import',{method:'POST',headers:{'Content-Type':'application/json','X-Chat-Request':'1'},body:JSON.stringify({version:1,chat:{title:'Tool navigation identities',active_leaf:'navigation-message-two'},messages:[{id:'navigation-message-one',parent_id:null,role:'assistant',content:'Two tool calls.',tool_calls:[{id:'nav-call-one',name:'lookup',arguments:'{"query":"first argument"}',result:{content:[{type:'text',text:'first result'}]},status:'completed'},{id:'nav-call-two',name:'lookup',arguments:'{"query":"second argument"}',result:{content:[{type:'text',text:'second result <script>window.injected=true</script>'}]},status:'completed'}]},{id:'navigation-message-two',parent_id:'navigation-message-one',role:'assistant',content:'A later tool call reuses its provider call ID.',tool_calls:[{id:'nav-call-one',name:'lookup',arguments:'{"query":"follow-up argument"}',result:{content:[{type:'text',text:'follow-up result'}]},status:'completed'}]}]})}).then(r=>r.json()).then(c=>c.id)`);
  await reloadDocument(); await wait(`document.querySelector('[data-chat="${navigationChat}"]')!==null`, 'imported tool navigation fixture');
  await click(`[data-chat="${navigationChat}"]`);
  await wait("document.querySelectorAll('.activity-tool-link').length===3", 'Activity lists all tool calls on the selected branch');
  assert.equal(await evaluate("document.querySelectorAll('.message.continuation').length"), 0, 'imported messages without run identities must not be grouped');
  assert.equal(await evaluate("document.querySelector('.activity-history').open"), false, 'old activity expanded by default');
  assert(!(await evaluate("document.querySelector('#panel-activity').innerText")).includes('input /'), 'Activity repeats token statistics');
  await click('.activity-history > summary');
  await click('.activity-history .activity-card > summary');
  await click('.activity-history .activity-tool-link[data-call="nav-call-one"]');
  await wait("document.activeElement?.closest('.tool-call')?.dataset.toolCall==='nav-call-one'", 'first Activity entry reveals matching tool details');
  assert.equal(await evaluate("document.activeElement.closest('.tool-call').innerText.includes('first argument') && document.activeElement.closest('.tool-call').innerText.includes('first result')"), true, 'first call navigation showed the wrong arguments or result');
  const repeatedCallMessage = await evaluate("document.querySelector('#panel-activity > .activity-card .activity-tool-link[data-call=nav-call-one]').dataset.message");
  await click(`.activity-tool-link[data-message="${repeatedCallMessage}"][data-call="nav-call-one"]`);
  await wait("document.activeElement?.closest('.message')?.dataset.message==="+JSON.stringify(repeatedCallMessage), 'reused provider call ID resolves to the requested assistant message');
  assert.equal(await evaluate("document.activeElement.closest('.tool-call').innerText.includes('follow-up argument') && document.activeElement.closest('.tool-call').innerText.includes('follow-up result')"), true, 'message and call identity did not select the correct repeated call ID');
  assert.equal(await evaluate("window.injected===undefined"), true, 'tool result executed as markup');
  await screenshot('activity-tool-navigation-desktop.png');
  await evaluate("document.querySelector('.activity-tool-link[data-call=nav-call-two]').focus()");
  await cdp('Input.dispatchKeyEvent', {type:'keyDown', key:'Enter', code:'Enter', windowsVirtualKeyCode:13, text:'\r', unmodifiedText:'\r'});
  await cdp('Input.dispatchKeyEvent', {type:'keyUp', key:'Enter', code:'Enter', windowsVirtualKeyCode:13});
  await wait("document.activeElement?.closest('.tool-call')?.dataset.toolCall==='nav-call-two'", 'keyboard activation targets the second call identity');
  assert.equal(await evaluate("document.activeElement.closest('.tool-call').innerText.includes('second argument') && document.activeElement.closest('.tool-call').innerText.includes('second result')"), true, 'keyboard navigation showed the wrong call');
  await cdp('Emulation.setDeviceMetricsOverride', {width:390,height:844,deviceScaleFactor:1,mobile:true});
  await wait("document.documentElement.dataset.compact==='true'", 'mobile Activity viewport state');
  await evaluate("if(document.querySelector('#app').classList.contains('inspector-closed')) document.querySelector('.topbar [data-action=inspector]').click()");
  await wait("!document.querySelector('#app').classList.contains('inspector-closed')", 'open mobile inspector');
  await click('.activity-history .activity-tool-link[data-call="nav-call-one"]');
  await wait("document.querySelector('#app').classList.contains('inspector-closed') && document.activeElement?.closest('.tool-call')?.dataset.toolCall==='nav-call-one'", 'mobile navigation closes inspector and focuses matching details');
  await screenshot('activity-tool-navigation-mobile.png');
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  const workspaceChat = await evaluate(`fetch('/api/import',{method:'POST',headers:{'Content-Type':'application/json','X-Chat-Request':'1'},body:JSON.stringify({version:1,chat:{title:'Workspace tool results',active_leaf:'workspace-message'},messages:[{id:'workspace-message',parent_id:null,role:'assistant',content:'Workspace changes are ready.',tool_calls:['list','read','write','run'].map((name,index)=>({id:'workspace-'+index,name:'workspace_'+name,arguments:'{}',result:{content:[]},status:'completed'}))}]})}).then(r=>r.json()).then(c=>c.id)`);
  await reloadDocument(); await wait(`document.querySelector('[data-chat="${workspaceChat}"]')!==null`, 'imported workspace result cards');
  await click(`[data-chat="${workspaceChat}"]`);
  await wait("document.querySelectorAll('.workspace-tool-open').length===3", 'Workspace navigation for list, read and run');
  assert.equal(await evaluate("Boolean(document.querySelector('[data-tool-call=workspace-2] .workspace-tool-open, [data-tool-call=workspace-2] .workspace-tool-view-file'))"), false, 'write without file metadata exposed a file action');
  await pointerClick('.workspace-tool-open');
  assert.equal(await evaluate("document.querySelector('#app').dataset.view"), 'workbench', 'workspace result button did not open workspace');
  const monitorBlock = new URL('/fixture-monitor-load', url).href;
  await cdp('Fetch.enable', {patterns: [{urlPattern: monitorBlock}]});
  const monitorScript = await cdp('Page.addScriptToEvaluateOnNewDocument', {source: `if(location.pathname==='/monitor/view')document.addEventListener('DOMContentLoaded',()=>{const image=new Image();image.hidden=true;image.src=${JSON.stringify(monitorBlock)};document.body.append(image);});`});
  await click('.global-nav [data-view="monitor"]');
  await wait(() => heldMonitorLoad !== null, 'held monitor load resource');
  await wait("Boolean(document.querySelector('#monitor-frame').contentDocument?.querySelector('main'))", 'parsed monitor before load');
  assert.equal(await evaluate("document.querySelector('#monitor-frame').contentDocument.readyState==='complete'"), false, 'held resource did not defer monitor load');
  await cdp('Fetch.fulfillRequest', {requestId: heldMonitorLoad, responseCode: 204});
  await cdp('Fetch.disable'); await cdp('Page.removeScriptToEvaluateOnNewDocument', {identifier: monitorScript.identifier});
  await wait("{const doc=document.querySelector('#monitor-frame').contentDocument; Boolean(doc?.readyState==='complete' && doc.querySelector('main') && doc.documentElement.dataset.theme==='light')}", 'loaded embedded monitor with light theme');
  assert.equal(await evaluate("document.querySelector('#monitor-frame').contentDocument.documentElement.dataset.theme"), 'light', 'monitor theme mismatch');
  assert.equal(await evaluate("getComputedStyle(document.querySelector('#monitor-frame').contentDocument.querySelector('body>header')).display"), 'none', 'duplicate monitor header');
  assert.equal(await evaluate("document.querySelector('#monitor-frame').contentDocument.querySelector('link[rel=icon]').getAttribute('href')"), '/favicon.svg', 'monitor favicon matches chat');
  const lightPanels = await evaluate(`(() => {
    const doc = document.querySelector('#monitor-frame').contentDocument;
    const root = doc.createElement('div'); doc.body.append(root);
    const samples = [['card', 'card-label'], ['lookup-item', ''], ['topology-stages', ''], ['topology-stage-body', ''], ['tooltip', ''], ['search', '']];
    const result = samples.map(([panel, text]) => {
      const box = doc.createElement('div'); box.className = panel;
      const label = doc.createElement('strong'); label.className = text; label.textContent = 'Theme check'; box.append(label); root.append(box);
      const style = getComputedStyle(box);
      return {panel, background: style.backgroundImage === 'none' ? style.backgroundColor : style.backgroundImage, text: getComputedStyle(label).color};
    }); root.remove(); return result;
  })()`);
  const luminance = color => {
    const channels = color.match(/[\d.]+/g).slice(0, 3).map(Number).map(value => { const s = value / 255; return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4; });
    return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
  };
  for (const panel of lightPanels) {
    const backgrounds = panel.background.match(/rgba?\([^)]+\)/g) || [];
    assert(backgrounds.length, 'No rendered panel color: ' + JSON.stringify(panel));
    for (const background of backgrounds) {
      const surface = luminance(background), foreground = luminance(panel.text);
      assert(surface > 0.5, 'Dark surface remains in light mode: ' + JSON.stringify(panel));
      assert((surface + 0.05) / (foreground + 0.05) >= 4.5, 'Light panel text contrast below 4.5: ' + JSON.stringify(panel));
    }
  }
  await screenshot('studio-monitor-desktop.png');
  assert.equal(await evaluate("getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()===getComputedStyle(document.querySelector('#monitor-frame').contentDocument.documentElement).getPropertyValue('--bg').trim()"), true, 'shell does not reuse dashboard theme');
  await click('.global-nav [data-view="preferences"]');
  assert((await evaluate("document.querySelector('#view-preferences').innerText")).includes('qwen3-local'));
  await pointerClick('#view-preferences [data-action="theme"]');
  assert.equal(await evaluate('document.documentElement.dataset.theme'), 'dark');
  await wait("document.querySelector('#monitor-frame').contentDocument.documentElement.dataset.theme==='dark'", 'monitor returns to dark');
  await pointerClick('#view-preferences [data-action="theme"]');
  assert(luminance(await evaluate('getComputedStyle(document.body).backgroundColor')) > 0.5, 'Settings toggle did not render a light page');
  await reloadDocument();
  await wait("window.ChatShell?.config!=null && document.querySelector('#connection').textContent!=='Connecting' && !document.querySelector('[data-action=\"new\"]').disabled", 'theme preference restored after reload');
  assert.equal(await evaluate('document.documentElement.dataset.theme'), 'light', 'saved theme not restored');
  assert(luminance(await evaluate('getComputedStyle(document.body).backgroundColor')) > 0.5, 'Reload left the page dark');
  await click('.global-nav [data-view="preferences"]');

  await screenshot('studio-settings-desktop.png');
  const downloadPath = path.join(profile, 'downloads'); await mkdir(downloadPath, {recursive:true});
  await cdp('Browser.setDownloadBehavior', {behavior: 'allow', downloadPath});
  await click('#backup-download');
  await wait("document.querySelector('#backup-status').textContent.includes('downloaded')", 'backup download control');
  const backupFile = path.join(downloadPath, 'llmx-chat-' + new Date().toISOString().slice(0,10) + '.sqlite3');
  let backup; for(let i=0;i<30;i++){try{backup=await readFile(backupFile);if(backup.length)break;}catch{}await delay(100);}
  assert(backup?.subarray(0,16).toString().startsWith('SQLite format 3'), 'backup is not SQLite');
  const invalidBackup = path.join(profile, 'invalid.sqlite3'); await writeFile(invalidBackup, 'Invalid backup');
  await cdp('Fetch.enable', {patterns: [{urlPattern: new URL('/api/restore', url).href}]});
  await upload('#backup-file', invalidBackup);
  await wait("document.querySelector('#action-dialog').open", 'invalid backup confirmation');
  await pointerClick('#conversation-submit');
  await wait(() => heldRestoreRequest !== null, 'held backup restore request');
  await evaluate("window.restoreCancelProbe=[];document.querySelector('#action-dialog').addEventListener('cancel',e=>restoreCancelProbe.push({cancelable:e.cancelable,prevented:e.defaultPrevented,busy:document.querySelector('#conversation-submit').disabled}),{once:true})");
  assert.equal(await evaluate("document.querySelector('#conversation-submit').disabled && document.querySelector('#conversation-cancel').disabled && document.querySelector('#conversation-submit').getAttribute('aria-busy')==='true'"), true, 'restore dialog is not busy during its request');
  await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27});
  await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27});
  assert.equal(await evaluate("document.querySelector('#action-dialog').open"), true, 'Escape closed a busy restore dialog: '+JSON.stringify(await evaluate('restoreCancelProbe')));
  await cdp('Fetch.continueRequest', {requestId: heldRestoreRequest}); await cdp('Fetch.disable');
  await wait("document.querySelector('#backup-status').classList.contains('error-text') && document.querySelector('#conversation-error').textContent.length>0 && !document.querySelector('#conversation-cancel').disabled", 'invalid backup error');
  await closeAction('#conversation-cancel');
  await upload('#backup-file', backupFile);
  await wait("document.querySelector('#action-dialog').open", 'backup restore dialog');
  const cancelledRestoreMutations = apiMutations.filter(item => item.url.endsWith('/api/restore')).length;
  await closeAction('#conversation-cancel');
  assert.equal(await evaluate("document.querySelector('#action-dialog').open"), false, 'cancel did not close backup restore dialog');
  assert.equal(apiMutations.filter(item => item.url.endsWith('/api/restore')).length, cancelledRestoreMutations, 'cancelled backup restore sent a request');
  const beforeRestore = await documentIdentity(); await upload('#backup-file', backupFile);
  await wait("document.querySelector('#action-dialog').open", 'confirmed backup restore dialog');
  await click('#conversation-submit');
  await waitForDocument(beforeRestore, 'backup restore');
  await wait("document.querySelector('#app').dataset.view==='chat' && window.ChatShell && window.ChatShell.config!==null", 'backup restore reload');
  await click('.global-nav [data-view="workbench"]');
  await wait("document.querySelector('.workspace-revision').textContent==='Revision 0'", 'workspace load');
  const modalOwner = await evaluate("(async()=>{const first=window.ChatShell.conversationForm({title:'Dialog owner probe',label:'Name',value:'first'});document.querySelector('#action-dialog').close();const second=window.ChatShell.conversationForm({title:'Overlapping form',label:'Name',value:'second'});await first;return {settled:true,secondAccepted:(await second)!==undefined};})()");
  assert.deepEqual(modalOwner, {settled: true, secondAccepted: false}, 'shared dialog allowed overlapping ownership');
  await checkEmptyWorkspace();
  assert.equal(await evaluate("document.querySelector('#action-dialog-title').textContent"), 'New workspace file');
  assert.equal(await evaluate("document.activeElement.id"), 'conversation-value', 'new-file dialog did not focus its field');
  await screenshot('studio-new-file-dialog-desktop.png');
  const longestWorkspacePath = 'p'.repeat(236) + '.txt';
  await cdp('Input.insertText', {text: longestWorkspacePath});
  assert.equal(await evaluate("document.querySelector('#conversation-value').value"), longestWorkspacePath, 'new-file field shortened a valid 240-character path');
  await evaluate("{window.escapedForm=new Promise(resolve=>document.querySelector('#action-dialog').addEventListener('close',()=>resolve(true),{once:true}));true}");
  await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27});
  await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27});
  assert.equal(await evaluate('window.escapedForm'), true, 'Escape did not emit the dialog close event');
  await evaluate('delete window.escapedForm');
  assert.equal(await evaluate("document.querySelector('#action-dialog').open"), false, 'Escape cancels new-file dialog');
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), '', 'cancel changed editor contents');
  await click('.workbench-view .primary-button');
  await wait("document.querySelector('#action-dialog').open", 'reopened new-file dialog');
  await fill('#conversation-value', 'Research'); await closeAction('#conversation-submit');
  await wait("!document.querySelector('#action-dialog').open && !document.querySelector('#workspace-editor').disabled && document.querySelector('.workspace-file-path').textContent==='Research'", 'new workspace file ready');
  await fill('#workspace-editor', 'first version\n'); await click('#workspace-save');
  await wait("document.querySelector('.workspace-file-name')?.textContent==='Research' && document.querySelector('#workspace-save').disabled", 'workspace file save');
  assert.equal(await evaluate("document.querySelector('.workspace-checkpoints').checkVisibility()"), false, 'checkpoint history should start outside Files');
  assert.equal(await evaluate("document.querySelector('.workspace-checkpoints details').open"), false, 'checkpoint history should start collapsed');
  const editorTabResults = [];
  for (const [modifiers, movedFocus] of [[0, "!document.activeElement.matches('#workspace-editor')"], [8, "document.activeElement.matches('.workspace-editor-actions .danger')"]]) {
    await evaluate("{const editor=document.querySelector('#workspace-editor');editor.focus();editor.setSelectionRange(0,5)}");
    const tabKey = {key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9, modifiers};
    await cdp('Input.dispatchKeyEvent', {type: 'keyDown', ...tabKey});
    await cdp('Input.dispatchKeyEvent', {type: 'keyUp', ...tabKey});
    editorTabResults.push(await evaluate(`({focus:${movedFocus},
      value:document.querySelector('#workspace-editor').value, clean:document.querySelector('#workspace-save').disabled})`));
  }
  assert.deepEqual(editorTabResults, Array(2).fill({focus: true, value: 'first version\n', clean: true}),
    'Tab and Shift+Tab should move focus out of the Workspace editor without changing the saved buffer');
  await selectWorkspaceSection('history');
  await click('.workspace-checkpoints button');
  await wait("document.querySelector('#action-dialog').open", 'checkpoint label dialog');
  assert.equal(await evaluate("document.querySelector('#conversation-value').maxLength"), 120, 'checkpoint field exceeds server limit');
  const checkpointMutations = apiMutations.filter(item => item.url.endsWith('/api/workspace/checkpoints')).length;
  await closeAction('#conversation-cancel');
  assert.equal(apiMutations.filter(item => item.url.endsWith('/api/workspace/checkpoints')).length, checkpointMutations, 'cancelled checkpoint creation sent a request');
  await click('.workspace-checkpoints button'); await wait("document.querySelector('#action-dialog').open", 'reopened checkpoint label dialog');
  await fill('#conversation-value', 'Workspace checkpoint'); await closeAction('#conversation-submit');
  await wait("document.querySelector('.workspace-checkpoints details').open && document.querySelectorAll('.workspace-checkpoint').length===1 && document.querySelector('.workspace-checkpoint').checkVisibility()", 'created checkpoint appears in history');
  const fileWritesBeforeArtifactCancel = apiMutations.filter(item => item.url.endsWith('/api/workspace/file')).length;
  await evaluate("document.dispatchEvent(new CustomEvent('companion:artifact-edit',{detail:{name:'generated.py',source:'print(1)'}}))");
  await wait("document.querySelector('#action-dialog').open", 'generated-file edit dialog');
  assert.equal(await evaluate("document.querySelector('#action-dialog-title').textContent"), 'Edit in workspace');
  assert.equal(await evaluate("document.querySelector('#conversation-value').value"), 'generated.py');
  await closeAction('#conversation-cancel');
  assert.equal(apiMutations.filter(item => item.url.endsWith('/api/workspace/file')).length, fileWritesBeforeArtifactCancel, 'cancelled generated-file edit wrote a file');
  assert.equal(await evaluate("document.querySelector('.workspace-file-path').textContent"), 'Research', 'cancelled generated-file edit changed the selected file');
  await selectWorkspaceSection('files');
  for (const [failed, revision] of [[false, 2], [true, 3]]) {
    await evaluate(`{
      const originalFetch=window.fetch;
      window.releaseWorkspaceRefresh=null;
      window.fetch=async(...args)=>{
        const response=await originalFetch(...args);
        if(new URL(args[0],location.href).pathname==='/api/workspace'){
          window.fetch=originalFetch;
          await new Promise((resolve,reject)=>{window.releaseWorkspaceRefresh=()=>${failed} ? reject(new Error('Delayed refresh failed')) : resolve();});
        }
        return response;
      };
      document.querySelector('.workspace-actions .secondary-button').click();
    }`);
    await wait("typeof window.releaseWorkspaceRefresh==='function'", 'held stale workspace refresh');
    await fill('#workspace-editor', failed ? 'third version\n' : 'second version\n');
    await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: 's', code: 'KeyS', modifiers: 2});
    await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: 's', code: 'KeyS', modifiers: 2});
    await wait(`fetch('/api/workspace').then(r=>r.json()).then(s=>s.revision===${revision})`, 'keyboard save reached server');
    await evaluate("window.releaseWorkspaceRefresh()");
    await wait(`document.querySelector('.workspace-revision').textContent==='Revision ${revision}' && document.querySelector('.workspace-error').hidden`, failed ? 'saved revision survives failed refresh' : 'saved revision survives delayed refresh');
  }
  await selectWorkspaceSection('history');
  await click('.checkpoint-actions button:first-child');
  await wait("document.querySelector('#artifact-dialog').open", 'checkpoint diff');
  assert((await evaluate("document.querySelector('.artifact-source').textContent")).includes('+third version'));
  assert.equal(await evaluate("document.querySelector('.artifact-source .syntax-add')!==null"), true, 'diff highlighting missing');
  assert.equal(await evaluate("document.querySelector('.artifact-toolbar-actions button:first-child').hidden"), false, 'checkpoint diff still offers generated-file editing');
  await click('#artifact-dialog [aria-label="Close artifact"]');
  await evaluate("Array.from(document.querySelectorAll('.workspace-checkpoint')).find(card=>card.querySelector('strong').textContent==='Workspace checkpoint').querySelector('.checkpoint-actions button:last-child').click()");
  await wait("document.querySelector('#action-dialog').open", 'checkpoint restore dialog');
  assert.equal(await evaluate("document.activeElement.id"), 'conversation-cancel', 'confirmation dialog did not focus its safe action');
  const checkpointRestoreMutations = apiMutations.filter(item => item.url.includes('/api/workspace/checkpoints/') && item.url.endsWith('/restore')).length;
  const checkpointEditorBeforeCancel = await evaluate("document.querySelector('#workspace-editor').value");
  await closeAction('#conversation-cancel');
  assert.equal(apiMutations.filter(item => item.url.includes('/api/workspace/checkpoints/') && item.url.endsWith('/restore')).length, checkpointRestoreMutations, 'cancelled checkpoint restore sent a request');
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), checkpointEditorBeforeCancel, 'cancelled checkpoint restore changed the editor');
  await evaluate("Array.from(document.querySelectorAll('.workspace-checkpoint')).find(card=>card.querySelector('strong').textContent==='Workspace checkpoint').querySelector('.checkpoint-actions button:last-child').click()");
  await wait("document.querySelector('#action-dialog').open", 'reopened checkpoint restore dialog');
  await closeAction('#conversation-submit');
  await wait("document.querySelector('#workspace-editor').value==='first version\\n'", 'checkpoint restore');
  await selectWorkspaceSection('files');
  await fill('#workspace-editor', 'unsaved local edit');
  await evaluate("fetch('/api/workspace').then(r=>r.json()).then(s=>fetch('/api/workspace/file',{method:'POST',headers:{'Content-Type':'application/json','X-Chat-Request':'1'},body:JSON.stringify({path:'external.txt',content:'Concurrent change',revision:s.revision})}))");
  await click('#workspace-save');
  await wait("!document.querySelector('.workspace-error').hidden", 'revision conflict');
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), 'unsaved local edit', 'conflict discarded local text');
  await click('.workspace-file');
  await wait("document.querySelector('#action-dialog').open", 'unsaved edit confirmation');
  const unsavedTextBeforeCancel = await evaluate("document.querySelector('#workspace-editor').value");
  const readsBeforeCancel = apiMutations.filter(item => item.url.includes('/api/workspace/file?')).length;
  const savesBeforeShortcut = apiMutations.filter(item => item.url.endsWith('/api/workspace/file')).length;
  const chatBeforeShortcut = await evaluate("document.querySelector('.chat-item.active')?.dataset.chat");
  await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: 's', code: 'KeyS', modifiers: 2});
  await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: 's', code: 'KeyS', modifiers: 2});
  await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: 'k', code: 'KeyK', modifiers: 2});
  await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: 'k', code: 'KeyK', modifiers: 2});
  assert.equal(apiMutations.filter(item => item.url.endsWith('/api/workspace/file')).length, savesBeforeShortcut, 'Ctrl+S saved while discard dialog was open');
  assert.equal(await evaluate("document.querySelector('.chat-item.active')?.dataset.chat"), chatBeforeShortcut, 'Ctrl+K changed chats while a dialog was open');
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), unsavedTextBeforeCancel, 'keyboard shortcut changed the dirty buffer');
  await closeAction('#conversation-cancel');
  assert.equal(apiMutations.filter(item => item.url.includes('/api/workspace/file?')).length, readsBeforeCancel, 'cancelled unsaved edit confirmation opened another file');
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), unsavedTextBeforeCancel, 'cancelled unsaved edit confirmation discarded text');
  await click('.workspace-file'); await wait("document.querySelector('#action-dialog').open", 'reopened unsaved edit confirmation');
  await closeAction('#conversation-submit');
  await wait("document.querySelector('#workspace-editor').value==='first version\\n'", 'reload after conflict');
  await selectWorkspaceSection('commands');
  assert((await evaluate("document.querySelector('.execution-description').textContent")).includes('Commands use saved workspace files.'), 'command runner should distinguish saved files from the editor buffer');
  await fill('#workspace-command', 'printf browser-job');
  await evaluate("document.querySelector('.workspace-command').requestSubmit()");
  await wait("document.querySelector('.workspace-job-status').textContent.includes('completed')", 'isolated fixture job completion');
  assert((await evaluate("document.querySelector('#workspace-output').textContent")).includes('browser-job complete'));
  await wait("document.querySelector('.workspace-file-list').textContent.includes('generated.txt')", 'completed job files refreshed');
  await fill('#workspace-command', 'sleep-test');
  await evaluate("document.querySelector('.workspace-command').requestSubmit()");
  await wait("document.querySelector('.workspace-job-status').textContent==='running'", 'cancellable container fixture');
  await evaluate("Array.from(document.querySelectorAll('.workspace-command button')).find(b=>b.textContent==='Stop').click()");
  await wait("document.querySelector('.workspace-job-status').textContent.includes('cancelled')", 'cancelled container fixture');
  await selectWorkspaceSection('files');
  await click('#view-workbench .workspace-editor-actions .danger');
  await wait("document.querySelector('#action-dialog').open", 'workspace delete dialog');
  const deletesBeforeCancel = apiMutations.filter(item => item.url.endsWith('/api/workspace/delete')).length;
  await closeAction('#conversation-cancel');
  assert.equal(apiMutations.filter(item => item.url.endsWith('/api/workspace/delete')).length, deletesBeforeCancel, 'cancelled workspace delete sent a request');
  assert.equal(await evaluate("document.querySelector('.workspace-file-list').textContent.includes('Research')"), true, 'cancelled workspace delete removed the file');
  await screenshot('studio-workspace-desktop.png');
  await click('.workspace-file[title="external.txt"]');
  await wait("document.querySelector('#workspace-editor').value==='Concurrent change'", 'source editor before delayed open');
  await evaluate(`(() => {
    const fetch = window.fetch;
    window.workspaceReadHeld = false;
    window.fetch = async (...args) => {
      const response = await fetch(...args);
      if (args[0] !== '/api/workspace/file?path=Research') return response;
      window.fetch = fetch;
      window.workspaceReadHeld = true;
      return new Promise(resolve => { window.releaseWorkspaceRead = () => resolve(response); });
    };
  })()`);
  await click('.workspace-file[title="Research"]');
  await wait('window.workspaceReadHeld', 'workspace file response held');
  await fill('#workspace-editor', 'Typed while another file was loading');
  await evaluate('window.releaseWorkspaceRead()');
  await new Promise(resolve => setTimeout(resolve, 100));
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), 'Typed while another file was loading', 'delayed file read discarded newer editor input');
  assert.equal(await evaluate("document.querySelector('.workspace-file-path').textContent"), 'external.txt', 'delayed file read replaced the edited file');
  await wait("document.querySelector('#action-dialog').open", 'fresh confirmation for new edits');
  await closeAction('#conversation-cancel');
  await fill('#workspace-editor', 'Concurrent change');
  async function holdWorkspaceRead(fail = false) {
    await evaluate(`(() => {
      const fetch = window.fetch;
      window.workspaceReadHeld = false;
      const fail = ${JSON.stringify(fail)};
      window.fetch = async (...args) => {
        const response = await fetch(...args);
        if (args[0] !== '/api/workspace/file?path=Research') return response;
        window.fetch = fetch; window.workspaceReadHeld = true;
        return new Promise(resolve => { window.releaseWorkspaceRead = () => resolve(fail ? new Response(JSON.stringify({error: "Obsolete read failed"}), {status: 503}) : response); });
      };
    })()`);
    await click('.workspace-file[title="Research"]');
    await wait('window.workspaceReadHeld', 'older file read held');
  }
  await holdWorkspaceRead();
  await click('.workspace-file[title="generated.txt"]');
  await wait("document.querySelector('.workspace-file-path').textContent==='generated.txt'", 'newer file read won');
  await evaluate('window.releaseWorkspaceRead()');
  await new Promise(resolve => setTimeout(resolve, 100));
  assert.equal(await evaluate("document.querySelector('.workspace-file-path').textContent"), 'generated.txt', 'older file response replaced a newer selection');
  await holdWorkspaceRead();
  await click('#view-workbench .view-toolbar .primary-button');
  await wait("document.querySelector('#action-dialog').open", 'new file during pending read');
  await fill('#conversation-value', 'pending-read.txt'); await closeAction('#conversation-submit');
  await wait("document.querySelector('.workspace-file-path').textContent==='pending-read.txt'", 'new file selected');
  await evaluate('window.releaseWorkspaceRead()');
  await new Promise(resolve => setTimeout(resolve, 100));
  assert.equal(await evaluate("document.querySelector('.workspace-file-path').textContent"), 'pending-read.txt', 'old read replaced a new unsaved file');
  await click('.workspace-file[title="external.txt"]');
  await wait("document.querySelector('#action-dialog').open", 'discard test new file'); await closeAction('#conversation-submit');
  await wait("document.querySelector('#workspace-editor').value==='Concurrent change'", 'workspace read tests cleaned up');
  await holdWorkspaceRead();
  await fill('#workspace-editor', 'Explicitly discarded after delayed read');
  await evaluate('window.releaseWorkspaceRead()');
  await wait("document.querySelector('#action-dialog').open", 'fresh discard approval');
  await closeAction('#conversation-submit');
  await wait("document.querySelector('.workspace-file-path').textContent==='Research' && document.querySelector('.unsaved-indicator').hidden", 'approved delayed file opens');
  await click('.workspace-file[title="external.txt"]');
  await wait("document.querySelector('.workspace-file-path').textContent==='external.txt'", 'return to source after approval');
  await holdWorkspaceRead(true);
  await click('.workspace-file[title="generated.txt"]');
  await wait("document.querySelector('.workspace-file-path').textContent==='generated.txt'", 'new selection before stale error');
  await evaluate('window.releaseWorkspaceRead()');
  await new Promise(resolve => setTimeout(resolve, 100));
  assert.equal(await evaluate("document.querySelector('.workspace-error').hidden"), true, 'obsolete read failure polluted the current editor');
  await click('.workspace-file[title="external.txt"]');
  await wait("document.querySelector('.workspace-file-path').textContent==='external.txt'", 'return to source before delete intent');
  await holdWorkspaceRead();
  await click('#view-workbench .workspace-editor-actions .danger');
  await wait("document.querySelector('#action-dialog').open", 'delete intent during pending open');
  await evaluate('window.releaseWorkspaceRead()');
  await new Promise(resolve => setTimeout(resolve, 100));
  assert.equal(await evaluate("document.querySelector('.workspace-file-path').textContent"), 'external.txt', 'file changed behind delete confirmation');
  await closeAction('#conversation-cancel');
  await click('.workspace-file[title="external.txt"]');
  await wait("document.querySelector('#workspace-editor').value==='Concurrent change'", 'open distinct file after command');
  await click('.workspace-file[title="Research"]');
  await wait("document.querySelector('#workspace-editor').value==='first version\\n'", 'reload editor base after command');
  async function holdWorkspaceSave() {
  await evaluate(`(() => {
    const original = window.fetch;
    window.releaseWorkspaceSave = null;
    window.fetch = async (...args) => {
      const response = await original(...args);
      if (args[0] !== '/api/workspace/file' || args[1]?.method !== 'POST') return response;
      window.fetch = original;
      window.workspaceSaveResult = await response.clone().json();
      return new Promise(resolve => { window.releaseWorkspaceSave = () => resolve(response); });
    };
  })()`);
  }
  await holdWorkspaceSave();
  await fill('#workspace-editor', 'Submitted workspace text');
  await click('#workspace-save');
  await wait("typeof window.releaseWorkspaceSave==='function'", 'actual workspace save response held');
  await fill('#workspace-editor', 'Newer unsaved workspace text');
  assert.equal(await evaluate("document.querySelector('#workspace-save').disabled && document.querySelector('#workspace-save').getAttribute('aria-busy')==='true'"), true, 'typing re-enabled a pending save');
  assert.equal(await evaluate("document.querySelector('.workspace-editor-actions .danger').disabled"), true, 'file removal must wait for the pending save');
  await evaluate('window.releaseWorkspaceSave()');
  await wait("document.querySelector('.workspace-revision').textContent==='Revision '+window.workspaceSaveResult.revision", 'save metadata refresh');
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), 'Newer unsaved workspace text');
  assert.equal(await evaluate("document.querySelector('.unsaved-indicator').hidden"), false, 'save completion marked newer unsaved text as saved');
  assert.equal(await evaluate("fetch('/api/workspace/file?path=Research').then(r=>r.json()).then(f=>f.content)"), 'Submitted workspace text');
  await click('#workspace-save');
  await wait("document.querySelector('.unsaved-indicator').hidden", 'save newer workspace text');
  assert.equal(await evaluate("fetch('/api/workspace/file?path=Research').then(r=>r.json()).then(f=>f.content)"), 'Newer unsaved workspace text');
  await holdWorkspaceSave();
  await fill('#workspace-editor', 'Saved before switching files'); await click('#workspace-save');
  await wait("typeof window.releaseWorkspaceSave==='function'", 'hold save before file selection');
  await click('.workspace-file[title="external.txt"]');
  await wait("document.querySelector('#action-dialog').open", 'confirm leaving submitted editor');
  await closeAction('#conversation-submit');
  await wait("document.querySelector('#workspace-editor').value==='Concurrent change'", 'destination file opened');
  await fill('#workspace-editor', 'Destination unsaved text');
  await evaluate('window.releaseWorkspaceSave()');
  await wait("document.querySelector('#workspace-save').getAttribute('aria-busy')==='false'", 'original file save settled');
  assert.equal(await evaluate("document.querySelector('.workspace-file-path').textContent"), 'external.txt');
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), 'Destination unsaved text');
  assert.equal(await evaluate("document.querySelector('.unsaved-indicator').hidden"), false, 'old file save marked destination edits clean');
  await click('#workspace-save');
  await wait("document.querySelector('.unsaved-indicator').hidden", 'save destination edits');
  assert.equal(await evaluate("fetch('/api/workspace/file?path=Research').then(r=>r.json()).then(f=>f.content)"), 'Saved before switching files');
  assert.equal(await evaluate("fetch('/api/workspace/file?path=external.txt').then(r=>r.json()).then(f=>f.content)"), 'Destination unsaved text');
  await evaluate(`(() => {
    const original = window.fetch;
    window.fetch = async (...args) => {
      if (args[0] !== '/api/workspace' || (args[1]?.method || 'GET') !== 'GET') return original(...args);
      window.fetch = original;
      return new Response(JSON.stringify({error:'Post-save refresh failed'}),{status:503,headers:{'Content-Type':'application/json'}});
    };
  })()`);
  await fill('#workspace-editor', 'Saved despite failed refresh'); await click('#workspace-save');
  await wait("document.querySelector('.workspace-error').textContent==='Post-save refresh failed'", 'metadata failure reported after accepted write');
  assert.equal(await evaluate("document.querySelector('.unsaved-indicator').hidden"), true, 'accepted save lost its clean baseline on refresh failure');
  await fill('#workspace-editor', 'Follow-up after refresh failure'); await click('#workspace-save');
  await wait("document.querySelector('.unsaved-indicator').hidden && document.querySelector('.workspace-error').hidden", 'follow-up uses accepted write revision');
  assert.equal(await evaluate("fetch('/api/workspace/file?path=external.txt').then(r=>r.json()).then(f=>f.content)"), 'Follow-up after refresh failure');
  const linkedWorkspace = await evaluate(`(async () => {
    let workspace=await ChatShell.api('/workspace');
    const safePath='nested/space & #.txt', removedPath='removed-link.txt', job=workspace.jobs.find(item=>item.command==='printf browser-job');
    const write=(path,content)=>ChatShell.api('/workspace/file','POST',{path,content,revision:workspace.revision});
    workspace=await write(safePath,'Encoded Workspace path.');
    workspace=await write(removedPath,'Previously present.');
    const prior=await ChatShell.api('/workspace/file?path='+removedPath);
    workspace=await ChatShell.api('/workspace/delete','POST',{path:removedPath,revision:workspace.revision});
    const duplicate=(await ChatShell.api('/workspace/jobs','POST',{command:job.command})).job;
    let latest=duplicate;
    for (let attempt=0; latest.status==='running' && attempt<50; attempt++) {
      await new Promise(resolve=>setTimeout(resolve,20));
      latest=(await ChatShell.api('/workspace/jobs/'+duplicate.id)).job;
    }
    if (latest.status!=='completed') throw new Error('Duplicate tool job fixture did not finish');
    const call=(id,name,args,metadata)=>({id,name:'workspace_'+name,arguments:JSON.stringify(args),result:{content:[{type:'text',text:JSON.stringify(metadata)}]},status:'completed'});
    const tool_calls=[
      call('linked-read','read',{path:safePath},await ChatShell.api('/workspace/file?path='+encodeURIComponent(safePath))),
      ...(${workspaceWriteCases})('external.txt',removedPath,workspace.revision,workspace.files),
      call('linked-run','run',{command:job.command},job),
      call('linked-removed','read',{path:removedPath},prior),
      call('linked-pruned','run',{command:'old command'},{id:'00000000000000000000000000000000',command:'old command',status:'completed'}),
      call('linked-malformed','read',{path:'../outside.txt'},{unexpected:'invalid'}),
    ];
    const chat=await ChatShell.api('/import','POST',{version:1,chat:{title:'Direct Workspace links',active_leaf:'linked-message'},messages:[{id:'linked-message',parent_id:null,role:'assistant',content:'Workspace links are ready.',tool_calls}]});
    return {id:chat.id,safePath};
  })()`);
  await reloadDocument();
  await wait("document.querySelector("+JSON.stringify('[data-chat="'+linkedWorkspace.id+'"]')+")!==null", 'direct Workspace link fixture');
  await click('[data-chat="'+linkedWorkspace.id+'"]');
  const link = id => '[data-tool-call="linked-'+id+'"] .workspace-tool-open';
  await wait("document.querySelector("+JSON.stringify(link('read'))+")!==null", 'direct file result action rendered');
  assert.equal(await evaluate("document.querySelector("+JSON.stringify(link('read'))+").textContent"), 'Open file', 'Workspace read result needs a direct file action');
  assert.equal(await evaluate("document.querySelector("+JSON.stringify(link('malformed'))+").textContent"), 'Open saved files', 'malformed result keeps the general saved-files action');
  const linkMutations = apiMutations.filter(item=>item.url.includes('/api/workspace/')).length;
  await checkChatSavedWriteActions('external.txt', 'Follow-up after refresh failure');
  const openWriteEditor = async () => {
    const summary = '[data-tool-call="linked-write"] details:has(.workspace-tool-open) > summary';
    if (!await evaluate(`document.querySelector(${JSON.stringify(link('write'))})?.checkVisibility()`)) await pointerClick(summary);
    await pointerClick(link('write'));
  };
  // Let the newly opened chat finish its scheduled scroll before targeting a tool button.
  await evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))');
  await pointerClick(link('read'));
  await wait("document.querySelector('#workspace-editor').value==='Encoded Workspace path.' && document.activeElement.id==='workspace-editor'", 'open exact escaped file path');
  assert.equal(await evaluate("document.querySelector('.workspace-file-path').textContent"), linkedWorkspace.safePath);
  await fill('#workspace-editor', 'Keep this dirty buffer.');
  await click('.global-nav [data-view="chat"]');
  await openWriteEditor();
  await wait("document.querySelector('#action-dialog').open", 'discard guard for linked file');
  await closeAction('#conversation-cancel');
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), 'Keep this dirty buffer.', 'cancelled link discarded the active buffer');
  await click('.global-nav [data-view="chat"]');
  await openWriteEditor();
  await wait("document.querySelector('#action-dialog').open", 'reopen discard guard for linked file');
  await closeAction('#conversation-submit');
  await wait("document.querySelector('.workspace-file-path').textContent==='external.txt' && document.querySelector('#workspace-editor').value==='Follow-up after refresh failure'", 'write action opens requested file');
  await click('.global-nav [data-view="chat"]');
  await evaluate(`(() => {
    window.releaseToolFile=null;
    const original=window.fetch, path=${JSON.stringify('/api/workspace/file?path='+encodeURIComponent(linkedWorkspace.safePath))};
    window.fetch=async (...args)=>{
      const response=await original(...args);
      if (args[0]!==path) return response;
      window.fetch=original;
      return new Promise(resolve=>window.releaseToolFile=()=>resolve(response));
    };
  })()`);
  await pointerClick(link('read'));
  await wait("typeof window.releaseToolFile==='function'", 'hold tool link file-read response');
  await click('.global-nav [data-view="chat"]');
  await evaluate("window.releaseToolFile()");
  await delay(160);
  assert.equal(await evaluate("document.querySelector('.workspace-file-path').textContent"), 'external.txt', 'leaving Workspace let a stale tool link replace the editor');
  await evaluate("document.querySelector("+JSON.stringify(link('run'))+").focus()");
  await drawerKey('Enter', 13);
  await wait("document.querySelector('.workspace-job.active')?.title==='printf browser-job' && document.querySelector('#workspace-output').textContent.includes('browser-job complete') && document.activeElement.id==='workspace-output'", 'keyboard opens retained job output');
  assert.equal(await evaluate("document.querySelector('#workspace-sections [data-section=commands]').getAttribute('aria-pressed')"), 'true', 'retained job did not open the Commands section');
  assert.equal(await evaluate("Array.from(document.querySelectorAll('.workspace-job')).filter(button=>button.title==='printf browser-job').at(-1).classList.contains('active')"), true, 'job link did not select the older job with the same command');
  for (const [id, reason] of [['removed','file is no longer available'],['pruned','job is no longer available']]) {
    await click('.global-nav [data-view="chat"]');
    await pointerClick(link(id));
    await wait("!document.querySelector('.workspace-error').hidden && document.querySelector('.workspace-error').textContent.includes("+JSON.stringify(reason)+")", 'missing workspace '+id+' feedback');
    assert.equal(await evaluate("document.querySelector('.workspace-file-path').textContent"), 'external.txt', 'unavailable target replaced the editor');
    assert.equal(await evaluate("document.querySelector('.workspace-job.active')?.title"), 'printf browser-job', 'unavailable target switched job output');
  }
  assert.equal(apiMutations.filter(item=>item.url.includes('/api/workspace/')).length, linkMutations, 'tool-result navigation sent a workspace mutation');
  assert.equal(await evaluate("document.querySelector('.workspace-limits')?.matches('details:not([open])')"), true, 'Workspace limits should start collapsed');
  await pointerClick('.workspace-limits summary');
  assert.equal(await evaluate("document.querySelector('.workspace-limits').open && document.querySelector('.workspace-limits').innerText.includes('Per file')"), true, 'Workspace limits disclosure did not show the limits');
  await pointerClick('.workspace-limits summary');
  const linkTheme = await evaluate('document.documentElement.dataset.theme');
  await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  for (const theme of ['dark','light']) {
    await evaluate("document.documentElement.dataset.theme="+JSON.stringify(theme)+"; window.ChatShell.show('chat')");
    assert.equal(await evaluate("document.documentElement.scrollWidth<=innerWidth && Array.from(document.querySelectorAll('.workspace-tool-open')).every(button=>button.getBoundingClientRect().right<=innerWidth)"), true, 'Workspace actions overflow at 390px in '+theme);
    await screenshot('workspace-tool-links-mobile-'+theme+'.png');
  }
  await evaluate("document.documentElement.dataset.theme="+JSON.stringify(linkTheme)+"; window.ChatShell.show('workbench')");
  await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await wait("document.documentElement.dataset.compact==='false' && !document.querySelector('#view-workbench').hidden", 'desktop workspace after viewport change');
  await selectWorkspaceSection('files');
  const previewFiles = {
    'site/styles/site.css': 'h1 { color: rgb(12, 34, 56); }',
    'site/pages/boot.js': 'const title = document.querySelector("h1"); console.log("project-ready", document.compatMode, getComputedStyle(title).color, title.textContent);'
  };
  await evaluate(`(async () => {
    for (const [path, content] of Object.entries(${JSON.stringify(previewFiles)})) {
      const {revision} = await window.ChatShell.api('/workspace');
      await window.ChatShell.api('/workspace/file', 'POST', {path, revision, content});
    }
  })()`);
  await pointerClick('.workspace-actions button');
  await wait("document.querySelector('.workspace-file[title=\"site/pages/boot.js\"]')", 'saved project assets listed');
  await pointerClick('.workspace-file[title="site/pages/boot.js"]');
  await wait("document.querySelector('.workspace-file-path').textContent==='site/pages/boot.js'", 'open a saved source file');
  assert.deepEqual(await evaluate("{const e=document.querySelector('#workspace-editor');[e.selectionStart,e.scrollTop,e.scrollLeft]}"), [0,0,0], 'newly opened file starts at its first character');
  const editorBeforeFilter = await evaluate("document.querySelector('#workspace-editor').value");
  await fill('#workspace-file-filter', 'SITE/PAGES/');
  assert.deepEqual(await evaluate("Array.from(document.querySelectorAll('.workspace-file'), item => item.title)"), ['site/pages/boot.js']);
  await fill('#workspace-file-filter', 'no such file');
  assert.equal(await evaluate("document.querySelector('.workspace-file-list').innerText"), 'No matching files.');
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), editorBeforeFilter, 'filtering replaced the editor buffer');
  await fill('#workspace-file-filter', '');
  const previewSource = '<!doctype html><html><head><link rel="stylesheet" href="../styles/site.css"><script defer src="./boot.js"></script></head><body><h1>Isolated preview</h1><script>console.log("preview-ready");try{parent.document.body.dataset.previewTouched="yes"}catch(e){console.log("parent-blocked")}fetch("/api/session").then(()=>console.error("network-allowed")).catch(()=>console.log("network-blocked"));setTimeout(()=>{throw new Error("preview-error")},20)</script></body></html>';
  await pointerClick('.workspace-actions .primary-button');
  await wait("document.querySelector('#action-dialog').open", 'new HTML file dialog');
  await fill('#conversation-value', 'site/pages/preview.html'); await closeAction('#conversation-submit');
  await wait("document.querySelector('.workspace-file-path').textContent==='site/pages/preview.html'", 'HTML editor');
  await fill('#workspace-editor', previewSource);
  const activeChat = await evaluate("document.querySelector('.chat-item.active')?.dataset.chat");
  const writesBeforeReturn = apiMutations.filter(item=>item.url.endsWith('/api/workspace/file')).length;
  await pointerClick('.workspace-back');
  assert.deepEqual(await evaluate("({view:document.querySelector('#app').dataset.view, focus:document.activeElement.id, chat:document.querySelector('.chat-item.active')?.dataset.chat})"),
    {view: 'chat', focus: 'chat-title', chat: activeChat}, 'Workspace return should focus the current conversation');
  await click('.global-nav [data-view="workbench"]');
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), previewSource, 'return to Chat discarded the unsaved HTML buffer');
  await pointerClick('.workspace-editor-actions button:first-child');
  assert.equal(await evaluate("document.querySelector('#artifact-preview-tab').getAttribute('aria-selected')==='true' && document.querySelector('.artifact-toolbar-actions button:first-child').hidden"), true, 'Workspace HTML should open in Preview without Edit in workspace');
  await wait("document.querySelector('.artifact-console-output').textContent.includes('preview-ready') && document.querySelector('.artifact-console-output').textContent.includes('network-blocked')", 'isolated preview and console bridge');
  assert.equal(await evaluate("document.querySelector('.artifact-frame').getAttribute('sandbox')"), 'allow-scripts');
  assert.equal(await evaluate("document.body.dataset.previewTouched"), undefined, 'preview accessed parent document');
  assert((await evaluate("document.querySelector('.artifact-console-output').textContent")).includes('parent-blocked'));
  await wait("document.querySelector('.artifact-console-output').textContent.includes('preview-error')", 'preview error bridge');
  await wait("document.querySelector('.artifact-console-output').textContent.includes('project-ready CSS1Compat rgb(12, 34, 56) Isolated preview')", 'deferred Workspace script sees the linked stylesheet and current HTML in standards mode');
  const previewRevision = await evaluate("(async () => (await window.ChatShell.api('/workspace')).revision)()");
  const projectNote = '#artifact-dialog .artifact-preview-note details';
  assert((await evaluate("document.querySelector("+JSON.stringify(projectNote)+").innerText")).includes('Workspace revision ' + previewRevision), 'preview did not identify its saved revision');
  await pointerClick(projectNote + ' summary');
  for (const path of Object.keys(previewFiles))
    assert((await evaluate("document.querySelector("+JSON.stringify(projectNote)+").innerText")).includes(path), 'preview did not list ' + path);

  await evaluate("new Promise(resolve=>{function received(event){if(event.data?.token==='forged'){window.removeEventListener('message',received);resolve();}}window.addEventListener('message',received);window.postMessage({channel:'llmx-preview',token:'forged',kind:'error',text:'forged-output'},location.origin);})");
  assert.equal(await evaluate("document.querySelector('.artifact-console-output').textContent.includes('forged-output')"), false, 'preview accepted forged bridge message');
  await screenshot('studio-artifact-preview.png');
  await click('#artifact-source-tab');
  assert.equal(await evaluate("document.querySelector('.artifact-frame')===null"), true, 'source toggle left preview running');
  await pointerClick('#artifact-dialog [aria-label="Close artifact"]');
  await wait("!document.querySelector('#artifact-dialog').open && document.activeElement.matches('.workspace-editor-actions button:first-child')", 'return focus to the Workspace viewer opener');
  assert.equal(apiMutations.filter(item=>item.url.endsWith('/api/workspace/file')).length, writesBeforeReturn, 'viewing or leaving Workspace saved the editor');
  await pointerClick('.workspace-file');
  await wait("document.querySelector('#action-dialog').open", 'leave unsaved preview fixture');
  await closeAction('#conversation-submit');
  await wait("document.querySelector('.workspace-file-path').textContent!=='site/pages/preview.html'", 'return to a saved file');
  await evaluate(`(async()=>{
    for(const path of ['src/index.js','tests/index.js']) {
      const {revision}=await window.ChatShell.api('/workspace');
      await window.ChatShell.api('/workspace/file','POST',{path,revision,content:'Saved '+path});
    }
  })()`);
  await pointerClick('.workspace-back');
  await checkSavedFilesFromChat([['src/index.js','Saved src/index.js'], ['tests/index.js','Saved tests/index.js']]);
  const originalDraft = await evaluate("document.querySelector('#draft').value");
  const bridgePrompt = 'Keep this draft while attaching Workspace files';
  await fill('#draft', bridgePrompt);
  const bridgeMutations = apiMutations.length;
  const showRefreshedWorkspace = async () => {
    await evaluate("window.previousWorkspaceRow=document.querySelector('.workspace-file')");
    await click('.global-nav [data-view="workbench"]');
    await wait("{const row=document.querySelector('.workspace-file'); row && row!==window.previousWorkspaceRow}", 'Workspace file refresh rendered');
    await evaluate("delete window.previousWorkspaceRow");
  };
  await showRefreshedWorkspace();
  await wait("document.querySelector('.workspace-file[title=\"src/index.js\"]')", 'nested Workspace files');
  await pointerClick('.workspace-file[title="src/index.js"]');
  await wait("document.querySelector('#workspace-editor').value==='Saved src/index.js'", 'source file loaded');
  await fill('#workspace-editor', 'Unsaved source edit for Chat');
  await evaluate("document.querySelector('#workspace-attach').focus()");
  const attachKey = {key:'Enter',code:'Enter',windowsVirtualKeyCode:13};
  await cdp('Input.dispatchKeyEvent', {type:'keyDown', ...attachKey, text:'\r', unmodifiedText:'\r'});
  await cdp('Input.dispatchKeyEvent', {type:'keyUp', ...attachKey});
  await wait("document.querySelector('#app').dataset.view==='chat' && document.activeElement.id==='draft' && document.querySelector('[aria-label=\"Remove src:index.js\"]')", 'keyboard attachment returns to composer');
  assert.equal(await evaluate("document.querySelector('#draft').value"), bridgePrompt);
  await showRefreshedWorkspace();
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), 'Unsaved source edit for Chat', 'attaching changed the editor');
  await pointerClick('.workspace-file[title="tests/index.js"]');
  await wait("document.querySelector('#action-dialog').open", 'attached unsaved file still needs a discard decision');
  await closeAction('#conversation-submit');
  await wait("document.querySelector('#workspace-editor').value==='Saved tests/index.js'", 'test file loaded');
  await pointerClick('#workspace-attach');
  await wait("document.querySelector('[aria-label=\"Remove tests:index.js\"]')", 'same basename attachment retains workspace identity');
  const savedBridgeDraft = `(() => {
    const {text,attachments}=JSON.parse(sessionStorage.getItem('llmx-draft:'+localStorage.getItem('llmx-chat'))||'{}');
    return {text,attachments:attachments?.map(({name,text})=>({name,text}))};
  })()`;
  const expectedBridgeDraft = {text:bridgePrompt, attachments:[
    {name:'src:index.js',text:'Unsaved source edit for Chat'}, {name:'tests:index.js',text:'Saved tests/index.js'},
  ]};
  await wait(`JSON.stringify(${savedBridgeDraft})===${JSON.stringify(JSON.stringify(expectedBridgeDraft))}`, 'current draft and exact Workspace attachments persisted');
  assert.deepEqual(await evaluate(savedBridgeDraft), expectedBridgeDraft, 'attachments should contain the exact editor text and distinct relative names');
  assert.equal(apiMutations.length, bridgeMutations, 'attaching from Workspace should neither send nor save');
  await showRefreshedWorkspace();
  await evaluate("{const editor=document.querySelector('#workspace-editor');editor.value='x'.repeat(2*1024*1024+4);editor.dispatchEvent(new Event('input'))}");
  await pointerClick('#workspace-attach');
  await wait("!document.querySelector('.workspace-error').hidden && document.querySelector('.workspace-error').textContent.includes('larger than 2 MiB') && document.activeElement.matches('.workspace-error')", 'Workspace attachment uses composer refusal');
  assert.equal(await evaluate("document.querySelector('#app').dataset.view"), 'workbench', 'refusal left Workspace');
  await fill('#workspace-editor', 'Saved tests/index.js');
  await pointerClick('.workspace-back');
  assert.equal(await evaluate("document.querySelectorAll('#attachments .attachment-pill').length"), 2, 'refusal changed the existing draft attachments');
  await click('[aria-label="Remove src:index.js"]'); await click('[aria-label="Remove tests:index.js"]');
  await fill('#draft', originalDraft);
  await showRefreshedWorkspace();
  await pointerClick('.workspace-file[title="src/index.js"]');
  await wait("document.querySelector('#workspace-editor').value==='Saved src/index.js'", 'source file ready for held attachment');
  await evaluate(`{
    const readFile=File.prototype.arrayBuffer;
    File.prototype.arrayBuffer=function(){
      File.prototype.arrayBuffer=readFile;
      return new Promise(resolve=>{window.releaseWorkspaceAttachment=()=>resolve(readFile.call(this));});
    };
  }`);
  await pointerClick('#workspace-attach');
  await wait("typeof window.releaseWorkspaceAttachment==='function'", 'held Workspace attachment read');
  assert.equal(await evaluate("document.querySelector('#workspace-attach').disabled"), true, 'attachment must disable duplicate activation during read');
  await click('#workspace-attach');
  await pointerClick('.workspace-file[title="tests/index.js"]');
  await wait("document.querySelector('#workspace-editor').value==='Saved tests/index.js' && document.activeElement.id==='workspace-editor'", 'new editor selection during attachment read');
  await evaluate("window.releaseWorkspaceAttachment(); delete window.releaseWorkspaceAttachment;");
  await wait("!document.querySelector('#workspace-attach').disabled && document.querySelector('#attachments .attachment-pill')", 'held attachment finished');
  assert.deepEqual(await evaluate("({view:document.querySelector('#app').dataset.view,focus:document.activeElement.id,count:document.querySelectorAll('#attachments .attachment-pill').length})"), {view:'workbench',focus:'workspace-editor',count:1}, 'held attachment navigated away from the newer file or attached twice');
  await pointerClick('.workspace-back');
  await click('[aria-label="Remove src:index.js"]');
  await showRefreshedWorkspace();
  await fill('#workspace-editor', 'Unsaved file before first project');
  await pointerClick('.workspace-back');
  const draftBeforeProjectSetup = await evaluate("document.querySelector('#draft').value");
  await fill('#draft', 'Unsent draft before first project');
  const mutationsBeforeProjectSetup = apiMutations.length;
  const openEmptyProjectPicker = async (fromAttach = false) => {
    if (fromAttach) await attachmentSource('project');
    else {
      await pointerClick('.composer [data-action="settings"]');
      await pointerClick('#use-workspace-project');
    }
    await wait("document.querySelector('#project-context-status').textContent.includes('Create a project')", 'empty project list');
  };
  await openEmptyProjectPicker(true);
  assert.equal(await evaluate("document.querySelector('#project-context-open-workspace')?.checkVisibility()"), true, 'empty Chat project picker must offer a direct Workspace action');
  assert.equal(await evaluate("document.querySelector('#project-context-select').checkVisibility() || document.querySelector('#project-context-apply').checkVisibility()"), false, 'empty picker still shows unavailable actions');
  await pointerClick('#project-context-open-workspace');
  await wait("document.querySelector('#app').dataset.view==='workbench' && document.querySelector('#workspace-sections [data-section=projects]').getAttribute('aria-pressed')==='true' && document.activeElement.id==='workspace-project-new'", 'first project setup with focus on New project');
  assert.equal(await evaluate("document.querySelector('#project-context-dialog').open"), false);
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), 'Unsaved file before first project');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Unsent draft before first project');
  await pointerClick('.workspace-back');
  await openEmptyProjectPicker();
  await evaluate(`{
    const fetchProjects = window.fetch;
    window.fetch = async function(...args) {
      if (args[0] !== '/api/workspace/projects') return fetchProjects(...args);
      window.fetch = fetchProjects;
      const response = await fetchProjects(...args);
      return new Promise(resolve => { window.releaseProjectSetup = () => resolve(response); });
    };
  }`);
  await pointerClick('#project-context-open-workspace');
  await wait("typeof window.releaseProjectSetup==='function'", 'held project list in Workspace');
  await pointerClick('.workspace-back');
  await pointerClick('.global-nav [data-view="workbench"]');
  await selectWorkspaceSection('projects');
  await evaluate("document.querySelector('#workspace-sections [data-section=projects]').focus(); window.releaseProjectSetup(); delete window.releaseProjectSetup;");
  await wait("!document.querySelector('#workspace-project-new').disabled", 'released project list');
  assert.equal(await evaluate("document.activeElement.matches('#workspace-sections [data-section=projects]')"), true, 'older navigation stole focus from a newer Workspace visit');
  assert.equal(apiMutations.length, mutationsBeforeProjectSetup, 'opening first project setup wrote to the API');
  await fill('#draft', draftBeforeProjectSetup);
  await fill('#workspace-editor', 'Saved tests/index.js');
  await pointerClick('#workspace-project-new');
  await fill('#workspace-project-name', 'Browser project');
  await fill('#workspace-project-instructions', 'Use the selected browser fixtures.');
  await click('#workspace-project-paths input[value="src/index.js"]');
  await click('#workspace-project-paths input[value="tests/index.js"]');
  await pointerClick('#workspace-project-save');
  await wait("document.querySelector('.workspace-project-row')?.textContent.includes('Browser project')", 'created named Workspace project');
  await selectWorkspaceSection('files');
  await pointerClick('.workspace-back');
  await pointerClick('.composer [data-action=attach-project]');
  await wait("document.querySelector('#project-context-files').children.length === 2", 'review saved project directly from composer');
  await pointerClick('#project-context-cancel');
  assert.equal(await evaluate("document.activeElement.matches('.composer [data-action=attach-project]')"), true, 'closing project review should restore its composer shortcut');
  async function reviewProject() {
    await pointerClick('.composer [data-action="settings"]');
    await pointerClick('#use-workspace-project');
    await wait("document.querySelector('#project-context-dialog').open && !document.querySelector('#project-context-apply').disabled", 'saved project ready in Chat');
    assert.equal(await evaluate("document.querySelector('#project-context-files').children.length"), 2, 'project file review count');
    assert((await evaluate("document.querySelector('#project-context-status').textContent")).includes('Workspace revision'));
  }
  async function applyProject() {
    await pointerClick('#project-context-copy');
    await pointerClick('#project-context-apply');
    await wait("!document.querySelector('#project-context-dialog').open && document.activeElement.id==='draft'", 'project returned to draft');
  }
  for (const [width,height] of [[320,720],[360,800],[390,844],[768,1024],[1024,768],[1440,1000],[844,390],[720,500]]) {
    await cdp('Emulation.setDeviceMetricsOverride', {width, height, deviceScaleFactor: width===720 ? 2 : 1, mobile: width<=390});
    await evaluate("window.dispatchEvent(new Event('resize'))");
    for (const section of ['chat','monitor','workbench','preferences']) {
      await evaluate(`window.ChatShell.show(${JSON.stringify(section)})`);
      assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'), true, section+' overflows at '+width+'x'+height);
      if(section==='chat')assert.equal(await evaluate("document.querySelector('.composer').getBoundingClientRect().bottom<=innerHeight"), true, 'composer clipped at '+width+'x'+height);
    }
    await screenshot('studio-settings-'+width+'x'+height+'.png');
    if(width===320){
      await evaluate("window.ChatShell.show('workbench')");
      await selectWorkspaceSection('files');
      await evaluate("document.querySelector('#workspace-attach').scrollIntoView({block:'center',behavior:'instant'})");
      assert.equal(await evaluate("{const button=document.querySelector('#workspace-attach'),box=button.getBoundingClientRect(),parent=button.closest('.workspace-editor').getBoundingClientRect();box.left>=parent.left&&box.right<=parent.right&&button.contains(document.elementFromPoint(box.x+box.width/2,box.y+box.height/2))}"), true, 'Workspace attachment target is clipped or blocked at 320px');
      await screenshot('workspace-attach-mobile.png');
      assert.equal(await evaluate("document.querySelector('.workspace-checkpoints').checkVisibility()"), false, 'checkpoint history appears in Files');
      await selectWorkspaceSection('history');
      assert.equal(await evaluate("document.querySelector('.workspace-checkpoints details').open"), false, 'a reload should start with checkpoint history collapsed');
      await evaluate("document.querySelector('.workspace-checkpoints summary').scrollIntoView({block:'center',behavior:'instant'})");
      await pointerClick('.workspace-checkpoints summary');
      await wait("document.querySelector('.workspace-checkpoint').checkVisibility()", 'open checkpoint history on mobile');
      assert.equal(await evaluate("document.querySelector('.workspace-checkpoints').scrollWidth<=document.querySelector('.workspace-checkpoints').clientWidth"), true, 'checkpoint controls overflow at 320px');
      await screenshot('workspace-checkpoints-mobile.png');
      await pointerClick('.workspace-checkpoints summary');
      await wait("!document.querySelector('.workspace-checkpoints details').open", 'collapse checkpoint history on mobile');
      await selectWorkspaceSection('files');
    }
    if(width===390){await evaluate("window.ChatShell.show('workbench')");await click('.workbench-view .primary-button');await wait("document.querySelector('#action-dialog').open",'mobile new-file dialog');await screenshot('studio-new-file-dialog-mobile.png');await closeAction('#conversation-cancel');}
    if(width===390 || width===768 || width===1024){await evaluate("window.ChatShell.show('workbench')");await screenshot('studio-workspace-'+width+'x'+height+'.png');}
  }
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await evaluate("document.documentElement.style.zoom='2';window.dispatchEvent(new Event('resize'));document.querySelector('[data-action=\"close-drawers\"]').click()");
  for(const section of ['chat','monitor','workbench','preferences']){
    await evaluate(`window.ChatShell.show(${JSON.stringify(section)})`);
    assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true,'CSS 200% zoom overflow: '+section);
    if(section==='chat')assert.equal(await evaluate("document.querySelector('.composer').getBoundingClientRect().bottom<=innerHeight"),true,'composer clipped at 200% zoom');
  }
  await pointerClick('.mobile-nav'); await delay(230);
  await pointerClick('.global-nav [data-view="workbench"]');
  assert.equal(await evaluate("document.querySelector('#app').dataset.view"),'workbench','navigation fails at 200% zoom');
  await screenshot('studio-workspace-200percent.png');
  await evaluate("document.documentElement.style.zoom='1';window.dispatchEvent(new Event('resize'));document.documentElement.dataset.theme='dark';window.ChatShell.show('monitor')");
  await screenshot('studio-monitor-shared-dark.png');
  await evaluate("window.ChatShell.show('chat')");
  const savedBeforeTemporary = await evaluate("fetch('/api/chats').then(r=>r.json()).then(d=>d.chats.length)");
  const lastSavedChat = await evaluate("localStorage.getItem('llmx-chat')");
  await pointerClick('#temporary-button');
  await wait("!document.querySelector('#temporary-notice').hidden && !document.querySelector('#temporary-button').disabled", 'temporary chat mode');
  assert.equal(await evaluate("getComputedStyle(document.querySelector('#temporary-button')).borderTopStyle"), 'solid', 'Normal chat button must use a solid border');
  assert.equal(await evaluate("document.querySelector('#temporary-button .chat-create-label').textContent"), 'Normal chat');
  assert.equal(await evaluate("document.querySelector('#temporary-button').getAttribute('aria-label')"), 'Switch to a normal chat');
  const temporaryId = await evaluate("sessionStorage.getItem('llmx-temporary-chat')");
  assert(temporaryId?.startsWith('tmp_'), 'temporary ID not in tab session');
  assert.equal(await evaluate("localStorage.getItem('llmx-chat')"), lastSavedChat, 'temporary chat replaced saved-chat preference');
  assert.equal(await evaluate("document.querySelector('#tool-settings input')===null"), true, 'temporary tools still selectable');
  await fill('#draft', 'Temporary draft privacy marker.');
  await click('.global-nav [data-view="workbench"]');
  await fill('#workspace-editor', 'Temporary Workspace attachment marker.');
  await pointerClick('#workspace-attach');
  await wait("document.querySelector('#app').dataset.view==='chat' && !document.querySelector('#temporary-notice').hidden && document.querySelector('[aria-label=\"Remove tests:index.js\"]')", 'Workspace file attaches to the temporary draft');
  assert.equal(await evaluate("sessionStorage.getItem('llmx-temporary-chat')"), temporaryId, 'attachment changed temporary mode');
  await delay(300);
  assert.equal(await evaluate("Object.keys(sessionStorage).filter(k=>k.startsWith('llmx-draft:')).some(k=>/Temporary (draft privacy|Workspace attachment) marker/.test(sessionStorage.getItem(k)))"), false, 'temporary draft or Workspace attachment was persisted');
  await click('[aria-label="Remove tests:index.js"]');
  await click('.global-nav [data-view="workbench"]');
  await fill('#workspace-editor', 'Saved tests/index.js');
  await pointerClick('.workspace-back');
  await holdHistoryRefresh();
  const savedDrafts = "Object.fromEntries(Object.keys(sessionStorage).filter(key=>key.startsWith('llmx-draft:')).map(key=>[key,sessionStorage.getItem(key)]))";
  const beforeTemporaryProject = await evaluate(savedDrafts);
  await reviewProject();
  await applyProject();
  assert.deepEqual(await evaluate(savedDrafts), beforeTemporaryProject, 'applying a project in temporary chat wrote a saved draft');
  await send('Temporary browser privacy check');
  await wait("typeof releaseChatHistory==='function' && !document.querySelector('#stop').hidden && document.querySelector('#messages').innerText.includes('Temporary response.')", 'streaming before send admission finishes');
  assert.equal(await evaluate("document.querySelector('#send').disabled"), true, 'delayed history did not hold the send busy state');
  await pointerClick('#temporary-button');
  assert.equal(await evaluate("sessionStorage.getItem('llmx-temporary-chat')"), temporaryId, 'busy send lost temporary chat');
  await evaluate('releaseChatHistory()');
  await wait("!document.querySelector('#send').disabled && !document.querySelector('#stop').hidden", 'send settled while generation stays active');
  await pointerClick('#temporary-button');
  await wait("document.querySelector('#toast').textContent.includes('Stop the response before ending this temporary chat.')", 'temporary exit blocked during generation');
  assert.equal(await evaluate("!document.querySelector('#temporary-notice').hidden && sessionStorage.getItem('llmx-temporary-chat')==="+JSON.stringify(temporaryId)), true, 'active-generation guard ended the temporary chat');
  assert.equal((await fetch(upstreamFixture + '/browser-step')).status, 204);
  await idle();
  assert((await evaluate("document.querySelector('#messages').innerText")).includes('Temporary response.'));
  await reloadDocument();
  await wait("!document.querySelector('#temporary-notice').hidden && document.querySelector('#messages').innerText.includes('Temporary response.')", 'temporary reply survives refresh in memory');
  assert.equal(await evaluate("document.querySelector('#temporary-button .chat-create-label').textContent"), 'Normal chat', 'temporary mode label did not survive reload');
  assert.equal(await evaluate("fetch('/api/chats').then(r=>r.json()).then(d=>d.chats.length)"), savedBeforeTemporary, 'temporary chat leaked into history');
  assert.equal(await evaluate("getComputedStyle(document.querySelector('.composer')).borderTopStyle"), 'dashed', 'temporary composer missing its visual marker');
  await click('.topbar [data-action="chat-menu"]');
  assert.deepEqual(await evaluate("Array.from(document.querySelectorAll('#action-menu button')).map(b=>b.dataset.action)"), ['find-message', 'outline', 'transcript', 'export', 'delete'], 'temporary menu actions');
  await pointerClick('#action-menu [data-action=outline]');
  assert.equal(await evaluate("document.activeElement.dataset.target===document.querySelector('.message.user').dataset.message"), true, 'temporary outline focuses its private prompt');
  await pointerClick('#find-message-dialog button[type=submit]');
  await click('.topbar [data-action="chat-menu"]');
  await pointerClick('#action-menu [data-action=find-message]');
  assert.equal(await evaluate("Array.from(document.querySelectorAll('#find-message-list button')).some(b=>b.dataset.target===document.querySelector('.message.user')?.dataset.message)"), true, 'temporary search found its private prompt');
  await pointerClick('#find-message-dialog button[type=submit]');
  await screenshot('temporary-chat-desktop.png');
  await cdp('Emulation.setDeviceMetricsOverride', {width: 360, height: 800, deviceScaleFactor: 1, mobile: true});
  await click('[data-action="close-drawers"]'); await delay(230);
  assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'), true, 'temporary chat overflows mobile');
  assert.equal(await evaluate("document.querySelector('.composer').getBoundingClientRect().bottom<=innerHeight"), true, 'temporary notice clips composer');
  await screenshot('temporary-chat-mobile.png');
  await cdp('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});

  await click('[data-action="new"]');
  await wait("document.querySelector('#action-dialog').open", 'temporary chat exit confirmation');
  const temporaryDeletes = apiMutations.filter(item => item.method === 'DELETE' && item.url.includes('/api/chats/' + temporaryId)).length;
  await closeAction('#conversation-cancel');
  assert.equal(apiMutations.filter(item => item.method === 'DELETE' && item.url.includes('/api/chats/' + temporaryId)).length, temporaryDeletes, 'cancelled temporary exit sent a delete request');
  assert.equal(await evaluate("sessionStorage.getItem('llmx-temporary-chat')"), temporaryId, 'cancelled temporary exit discarded the chat');
  assert.equal(await evaluate("document.querySelector('#temporary-button .chat-create-label').textContent"), 'Normal chat', 'cancelled exit changed the toggle label');
  await pointerClick('#temporary-button'); await wait("document.querySelector('#action-dialog').open", 'temporary toggle exit confirmation');
  await closeAction('#conversation-submit');
  await wait("document.querySelector('#temporary-notice').hidden && sessionStorage.getItem('llmx-temporary-chat')===null && !document.querySelector('#temporary-button').disabled", 'leaving discards temporary chat');
  assert.equal(await evaluate("document.querySelector('#temporary-button .chat-create-label').textContent"), 'Temporary chat', 'normal conversation label not restored');
  assert.equal(await evaluate("fetch('/api/chats').then(r=>r.json()).then(d=>d.chats.length)"), savedBeforeTemporary, 'temporary toggle created an empty saved conversation');
  assert.equal(await evaluate("document.querySelector('#chat-title').disabled && document.querySelector('#chat-title').textContent==='Chat' && document.querySelector('.chat-item.active')===null"), true, 'leaving temporary mode did not return to the blank normal view');
  assert.equal(await evaluate(`fetch('/api/chats/${temporaryId}').then(r=>r.status)`), 404, 'temporary transcript retained after exit');
  if (await detailsOpen()) await click(detailsToggle);
  const chatHeader = [];
  for (const width of [1440, 390, 320]) for (const theme of ['dark', 'light']) {
    await cdp('Emulation.setDeviceMetricsOverride', {width, height: 1000, deviceScaleFactor: 1, mobile: width < 500});
    await wait(`document.documentElement.dataset.compact==='${width<500}'`, 'mode toggle viewport resize');
    await evaluate("document.documentElement.dataset.theme=" + JSON.stringify(theme));
    const geometry = () => evaluate(`(() => {
      const button = document.querySelector('#temporary-button');
      const menu = document.querySelector('.topbar [data-action="chat-menu"]');
      const box = button.getBoundingClientRect();
      return {width: box.width, height: box.height, x: box.x, menuX: menu.getBoundingClientRect().x,
        titleWidth: document.querySelector('#chat-title').getBoundingClientRect().width};
    })()`);
    const compactLabel = () => evaluate("{const label=document.querySelector('.temporary-short-label'); label?.getClientRects().length ? label.textContent : ''}");
    const {titleWidth, ...before} = await geometry();
    chatHeader.push({viewport:width,theme,titleWidth,...before});
    await screenshot('reading-normal-'+width+'-'+theme+'.png');
    assert.equal(await evaluate("getComputedStyle(document.querySelector('#temporary-button')).borderTopStyle"), 'dashed', 'Temporary chat button lost dashed border');
    if (width < 500) assert.equal(await compactLabel(), 'Temporary', 'mobile temporary mode label is not visible');
    await pointerClick('#temporary-button');
    await wait("!document.querySelector('#temporary-notice').hidden && !document.querySelector('#temporary-button').disabled", 'empty temporary mode');
    assert.equal(await evaluate("getComputedStyle(document.querySelector('#temporary-button')).borderTopStyle"), 'solid', 'Normal chat button must be solid');
    assert.equal(await evaluate("document.querySelector('#recent-conversations').checkVisibility()"), false, 'temporary landing shows saved conversations');
    if (width < 500) assert.equal(await compactLabel(), 'Normal', 'mobile normal mode label is not visible');
    const {titleWidth: temporaryTitleWidth, ...temporaryControls} = await geometry();
    assert.deepEqual(temporaryControls, before, 'mode toggle shifted header controls');
    if (width < 500) assert.equal(temporaryTitleWidth, titleWidth, 'mode toggle shifted the mobile title width');
    assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true, 'toggle overflow');
    await screenshot('toggle-' + width + '-' + theme + '.png');
    await holdHistoryRefresh();
    await pointerClick('#temporary-button');
    await wait("typeof releaseChatHistory==='function' && document.querySelector('#temporary-notice').hidden && sessionStorage.getItem('llmx-temporary-chat')===null", 'normal view before history refresh completes');
    assert.equal(await evaluate("document.querySelector('#temporary-button').disabled"), true, 'pending history did not retain the toggle guard');
    await evaluate('releaseChatHistory()');
    await wait("!document.querySelector('#temporary-button').disabled", 'empty temporary exit fully settled');
    assert.deepEqual(await geometry(), {titleWidth,...before}, 'normal mode shifted header controls');
    assert.equal(await evaluate("fetch('/api/chats').then(r=>r.json()).then(d=>d.chats.length)"), savedBeforeTemporary, 'mode toggle created empty saved chat');
  }
  console.log('Reading layout geometry: '+JSON.stringify({inspectorPreference,replyLayout,chatHeader}));
  assert.deepEqual(inspectorPreference, {defaultOpen:false,rememberedOpen:true,mobileOpen:false,
    restoredOpen:true,rememberedClosed:true,settingsOpened:true,closedAfterSettings:true}, 'docked conversation details preference');
  assert(chatHeader.every(({viewport,titleWidth,width,height})=>viewport===1440 || titleWidth>=viewport-240 && width>=40 && height>=40), 'compact mode toggle crowds the title or is too small');
  assert(replyLayout.every(({width,actionTarget,actionGap,gap})=>actionTarget>=(width===1440?32:40) && actionGap<=8 && gap<=8), 'within-reply spacing or action target');
  await evaluate("window.ChatComposer.attach([new File(['Earlier draft version'], 'src:index.js', {type:'text/plain'})])");
  await fill('#setting-system', 'Keep until I choose to replace it.');
  await reviewProject();
  assert.equal(await evaluate("document.querySelector('#project-context-copy').checked"), false);
  await pointerClick('#project-context-copy');
  await pointerClick('#project-context-apply');
  await wait("document.querySelector('#project-context-error').textContent.includes('Remove the existing src:index.js')", 'conflicting draft file is identified');
  assert.deepEqual(await evaluate("({count:document.querySelectorAll('#attachments .attachment-pill').length,system:document.querySelector('#setting-system').value})"),
    {count:1,system:'Keep until I choose to replace it.'}, 'refusal changed files or instructions');
  await pointerClick('#project-context-cancel');
  await pointerClick('#details-close');
  await wait("document.querySelector('#app').classList.contains('inspector-closed') && !document.querySelector('.composer').closest('[inert]')", 'return from settings to edit the draft');
  await pointerClick('[aria-label="Remove src:index.js"]');
  await reviewProject();
  await applyProject();
  await reviewProject();
  await pointerClick('#project-context-apply');
  await wait("!document.querySelector('#project-context-dialog').open", 'reapplied the named project');
  assert.equal(await evaluate("document.querySelectorAll('#attachments .attachment-pill').length"), 2, 'reapplying the same project duplicated attachments');
  await send('Normal browser persistence check');
  await idle();
  await wait("fetch('/api/chats').then(r=>r.json()).then(d=>d.chats.length===" + (savedBeforeTemporary + 1) + ")", 'normal message appears in saved history');
  assert.equal(await evaluate("document.querySelector('.chat-item.active')?.dataset.chat===localStorage.getItem('llmx-chat')"), true, 'normal send did not select the persisted conversation');
  await cdp('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
  await pointerClick('#temporary-button');
  await wait("!document.querySelector('#temporary-notice').hidden && !document.querySelector('#temporary-button').disabled", 'temporary chat for expiry check');
  const expiredId = await evaluate("sessionStorage.getItem('llmx-temporary-chat')");
  await evaluate(`fetch('/api/chats/${expiredId}',{method:'DELETE',headers:{'X-Chat-Request':'1'}})`);
  await reloadDocument();
  await wait("window.ChatShell?.config!=null && document.querySelector('#connection').textContent!=='Connecting' && sessionStorage.getItem('llmx-temporary-chat')===null && !document.querySelector('[data-action=\"new\"]').disabled", 'missing temporary chat handled after reload');
  assert.equal(await evaluate("document.querySelector('#temporary-notice').hidden"), true);
  const submittedChat = await evaluate("localStorage.getItem('llmx-chat')");
  const sentProjectFiles = await evaluate(`(async () => {
    const {revision} = await window.ChatShell.api('/workspace');
    await window.ChatShell.api('/workspace/file', 'POST', {path:'src/index.js', content:'Later saved contents', revision});
    const snapshot = await window.ChatShell.api('/chats/' + ${JSON.stringify(submittedChat)});
    return snapshot.messages.find(item=>item.content==='Normal browser persistence check').attachments.map(file=>file.text);
  })()`);
  assert.deepEqual(sentProjectFiles, ['Saved src/index.js','Saved tests/index.js'], 'a Workspace edit changed sent project attachments');
  const chatSettings = await evaluate("fetch('/api/chats/'+"+JSON.stringify(submittedChat)+").then(r=>r.json()).then(d=>d.chat.settings)");
  const otherChat = await evaluate("fetch('/api/chats',{method:'POST',headers:{'Content-Type':'application/json','X-Chat-Request':'1'},body:JSON.stringify({settings:"+JSON.stringify(chatSettings)+"})}).then(r=>r.json()).then(d=>d.id)");
  await reloadDocument();
  await wait("document.querySelector('.chat-item.active')?.dataset.chat==="+JSON.stringify(submittedChat), 'restore submitted conversation before delayed send');
  await click('[data-chat="'+otherChat+'"]');
  await wait("document.querySelector('.chat-item.active')?.dataset.chat==="+JSON.stringify(otherChat), 'open the other conversation');
  const otherSettings = await evaluate("fetch('/api/chats/'+"+JSON.stringify(otherChat)+").then(r=>r.json()).then(d=>d.chat.settings)");
  const otherRun = await evaluate("fetch('/api/chats/'+"+JSON.stringify(otherChat)+"+'/runs',{method:'POST',headers:{'Content-Type':'application/json','X-Chat-Request':'1'},body:JSON.stringify({content:'Keep this other response active',settings:"+JSON.stringify(otherSettings)+"})}).then(r=>r.json()).then(d=>d.run.id)");
  await click('[data-chat="'+otherChat+'"]');
  await wait("!document.querySelector('#stop').hidden && document.querySelector('#messages').innerText.includes('The other chat is active.')", 'other conversation response is active');
  await wait(() => runStreams.some(stream => new URL(stream).pathname === '/api/runs/' + otherRun + '/events'), 'other conversation event stream');
  await click('[data-chat="'+submittedChat+'"]');
  await wait("document.querySelector('.chat-item.active')?.dataset.chat==="+JSON.stringify(submittedChat), 'return to submitted conversation');
  const seedRun = await evaluate("fetch('/api/chats/'+"+JSON.stringify(submittedChat)+"+'/runs',{method:'POST',headers:{'Content-Type':'application/json','X-Chat-Request':'1'},body:JSON.stringify({content:'Original editable prompt',parent_id:null,settings:"+JSON.stringify(chatSettings)+"})}).then(r=>r.json()).then(d=>d.run.id)");
  await wait("fetch('/api/chats/'+"+JSON.stringify(submittedChat)+").then(r=>r.json()).then(d=>d.messages.some(m=>m.run_id==="+JSON.stringify(seedRun)+"&&m.role==='assistant'&&m.status==='completed'))", 'editable seed response completes');
  await click('[data-chat="'+submittedChat+'"]');
  await wait("document.querySelector('.message.user')?.innerText.includes('Original editable prompt')", 'refresh editable seed message');
  await click('.message.user [data-action="edit"]');
  const editParent = await evaluate("fetch('/api/chats/'+"+JSON.stringify(submittedChat)+").then(r=>r.json()).then(d=>d.messages.find(m=>m.run_id==="+JSON.stringify(seedRun)+"&&m.role==='user').parent_id)");
  await cdp('Fetch.enable', {patterns: [{urlPattern: url+'/api/chats/'+submittedChat+'/runs', requestStage: 'Response'}]});
  await upload('#file-input', fixture);
  await wait("document.querySelector('#attachments').innerText.includes('notes.txt')", 'submitted edit attachment chip');
  await send('Submitted edit snapshot');
  await wait(() => heldSendResponse !== null, 'delayed run response from the actual server');
  const sentBody = JSON.parse(heldSendResponse.request.postData);
  assert.equal(sentBody.content, 'Submitted edit snapshot', 'delayed edit changed its captured text');
  assert.equal(sentBody.parent_id, editParent, 'delayed edit changed its original parent');
  assert.deepEqual(sentBody.attachments?.map(attachment => attachment.name), ['notes.txt'], 'delayed edit changed its captured attachment');
  assert.equal(sentBody.attachments[0].text, await readFile(fixture, 'utf8'), 'submitted text attachment content changed');
  assert.equal(heldSendResponse.responseStatusCode, 201, 'server did not successfully admit the held request');
  await fill('#draft', 'Newer edit in the original conversation');
  await upload('#file-input', imageFixture);
  await wait("document.querySelector('#attachments').innerText.includes('pixel.png')", 'new image attachment during same-chat edit');
  await cdp('Fetch.continueRequest', {requestId: heldSendResponse.requestId});
  await cdp('Fetch.disable');
  heldSendResponse = null;
  await wait("document.querySelector('#send:not([hidden]):enabled') && document.querySelector('#messages').innerText.includes('The submitted edit completed.')", 'same-chat edited reply completes and Send is available');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Newer edit in the original conversation', 'send cleared a newer edit in the same conversation');
  assert.equal(await evaluate("!document.querySelector('#edit-note').hidden"), true, 'send discarded the newer edit state');
  assert.deepEqual(await evaluate("Array.from(document.querySelectorAll('#attachments .attachment-pill')).map(el=>el.firstChild.textContent)"), ['notes.txt', 'pixel.png'], 'send discarded newer same-chat attachments');
  const firstSubmitted = await evaluate("fetch('/api/chats/'+"+JSON.stringify(submittedChat)+").then(r=>r.json()).then(d=>d.messages.find(m=>m.role==='user'&&m.content==='Submitted edit snapshot'))");
  assert.equal(firstSubmitted?.parent_id, editParent, 'the submitted edit did not retain its original branch parent');
  assert.deepEqual(firstSubmitted?.attachments?.map(attachment => attachment.name), ['notes.txt'], 'the submitted edit did not retain its original attachment');
  await cdp('Fetch.enable', {patterns: [{urlPattern: url+'/api/chats/'+submittedChat+'/runs', requestStage: 'Response'}]});
  await send('Submitted second request snapshot');
  await wait(() => heldSendResponse !== null, 'second delayed run response from the actual server');
  const secondSentBody = JSON.parse(heldSendResponse.request.postData);
  assert.equal(secondSentBody.content, 'Submitted second request snapshot');
  assert.equal(secondSentBody.parent_id, editParent);
  assert.deepEqual(secondSentBody.attachments?.map(attachment => attachment.name), ['notes.txt', 'pixel.png']);
  assert.equal(secondSentBody.attachments[1].data, 'data:image/png;base64,' + (await readFile(imageFixture)).toString('base64'), 'submitted image bytes changed');
  await click('[data-chat="'+otherChat+'"]');
  await wait("document.querySelector('.chat-item.active')?.dataset.chat==="+JSON.stringify(otherChat), 'switch chats while submission response is held');
  await wait(() => runStreams.filter(stream => new URL(stream).pathname === '/api/runs/' + otherRun + '/events').length >= 2, 'other conversation stream resumes after switching back');
  const otherStreamsBeforeRelease = runStreams.filter(stream => new URL(stream).pathname === '/api/runs/' + otherRun + '/events').length;
  await fill('#draft', 'New draft in the other conversation');
  await upload('#file-input', fixture);
  await wait("document.querySelector('#attachments').innerText.includes('notes.txt')", 'new attachment in the other conversation');
  await cdp('Fetch.continueRequest', {requestId: heldSendResponse.requestId});
  await cdp('Fetch.disable');
  heldSendResponse = null;
  await wait("document.querySelector('#send').disabled===false", 'submitted request completion');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'New draft in the other conversation', 'old send cleared a newer draft in the selected chat');
  assert((await evaluate("document.querySelector('#attachments').innerText")).includes('notes.txt'), 'old send cleared the selected chat attachment');
  const otherStreamsAfterRelease = runStreams.filter(stream => new URL(stream).pathname === '/api/runs/' + otherRun + '/events').length;
  assert.equal(otherStreamsAfterRelease, otherStreamsBeforeRelease, 'old send subscribed to the selected chat response again');
  const submitted = await evaluate("fetch('/api/chats/'+"+JSON.stringify(submittedChat)+").then(r=>r.json()).then(d=>d.messages.find(m=>m.role==='user'&&m.content==='Submitted second request snapshot'))");
  assert.deepEqual(submitted?.attachments, secondSentBody.attachments, 'submitted request did not persist its original attachments');
  await click('[data-chat="'+submittedChat+'"]');
  await wait("document.querySelector('.message.assistant')?.innerText.includes('The second submission completed.')", 'open retry source');
  await cdp('Fetch.enable', {patterns: [{urlPattern: url+'/api/chats/'+submittedChat+'/runs', requestStage: 'Response'}]});
  await click('.message.assistant [data-action="retry"]');
  await wait(() => heldSendResponse !== null, 'hold admitted retry response');
  assert.equal(JSON.parse(heldSendResponse.request.postData).regenerate, true);
  assert.equal(heldSendResponse.responseStatusCode, 201);
  await click('[data-chat="'+otherChat+'"]');
  await wait("document.querySelector('.chat-item.active')?.dataset.chat==="+JSON.stringify(otherChat), 'navigate during retry');
  await delay(100);
  const retryStreamsBefore = runStreams.filter(stream => new URL(stream).pathname === '/api/runs/' + otherRun + '/events').length;
  await fill('#draft', 'Keep the destination draft');
  await cdp('Fetch.continueRequest', {requestId: heldSendResponse.requestId});
  await cdp('Fetch.disable'); heldSendResponse = null;
  await delay(300);
  assert.equal(runStreams.filter(stream => new URL(stream).pathname === '/api/runs/' + otherRun + '/events').length, retryStreamsBefore, 'old retry subscribed to the selected chat response again');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Keep the destination draft');
  await click('[data-chat="'+submittedChat+'"]');
  await wait("document.querySelector('.message.assistant')?.innerText.includes('The delayed retry completed.')", 'retry source remains completed');
  await evaluate(`(() => {
    const original = window.fetch;
    window.releaseRetrySnapshot = null;
    window.fetch = async (...args) => {
      const response = await original(...args);
      if (args[0] !== '/api/chats/'+${JSON.stringify(submittedChat)} || (args[1]?.method || 'GET') !== 'GET') return response;
      window.fetch = original;
      return new Promise(resolve => { window.releaseRetrySnapshot = () => resolve(response); });
    };
  })()`);
  await click('.message.assistant [data-action="retry"]');
  await wait("typeof window.releaseRetrySnapshot==='function'", 'hold retry snapshot');
  await click('[data-chat="'+otherChat+'"]');
  await wait("document.querySelector('.chat-item.active')?.dataset.chat==="+JSON.stringify(otherChat), 'leave during retry snapshot');
  await evaluate(`fetch('/api/chats/'+${JSON.stringify(submittedChat)},{method:'PATCH',headers:{'Content-Type':'application/json','X-Chat-Request':'1'},body:JSON.stringify({title:'Fresh reopened conversation'})}).then(r=>{if(!r.ok)throw Error('Fixture rename failed');})`);
  await click('[data-chat="'+submittedChat+'"]');
  await wait("document.querySelector('#chat-title').textContent==='Fresh reopened conversation'", 'reopen current source snapshot');
  await evaluate('window.releaseRetrySnapshot()');
  await delay(300);
  assert.equal(await evaluate("document.querySelector('#chat-title').textContent"), 'Fresh reopened conversation', 'old retry snapshot replaced the reopened conversation');
  await cdp('Fetch.enable', {patterns: [{urlPattern: url+'/api/chats/'+submittedChat+'/runs', requestStage: 'Request'}]});
  await fill('#setting-system', 'Retry with these live settings.');
  await click('.message.assistant [data-action="retry"]');
  await wait(() => heldSendResponse !== null, 'hold retry before server admission');
  assert.equal(await evaluate("document.querySelector('[data-action=save-settings]').disabled"), true, 'Save must wait for the earlier retry admission');
  await click('[data-chat="'+otherChat+'"]');
  await wait("document.querySelector('.chat-item.active')?.dataset.chat==="+JSON.stringify(otherChat), 'leave before retry admission');
  await click('[data-chat="'+submittedChat+'"]');
  await wait("document.querySelector('.chat-item.active')?.dataset.chat==="+JSON.stringify(submittedChat), 'return before retry admission');
  await cdp('Fetch.continueRequest', {requestId: heldSendResponse.requestId});
  await cdp('Fetch.disable'); heldSendResponse = null;
  await wait("document.querySelector('.message.assistant')?.innerText.includes('The returning retry completed.') && document.querySelector('#stop').hidden", 'returned conversation receives accepted retry');
  assert.equal(await evaluate("fetch('/api/chats/'+localStorage.getItem('llmx-chat')).then(r=>r.json()).then(d=>d.chat.settings.system)"), 'Retry with these live settings.', 'Retry without Save did not persist settings');
  await click('[data-chat="'+otherChat+'"]');
  await wait("document.querySelector('.chat-item.active')?.dataset.chat==="+JSON.stringify(otherChat), 'return to waiting destination');
  const browserStep = await fetch(upstreamFixture + '/browser-step');
  assert.equal(browserStep.status, 204, 'could not release the other conversation response');
  await wait("document.querySelector('#stop').hidden && document.querySelector('#messages').innerText.includes('The other chat completed.')", 'other conversation response completes');
  await click('[data-chat="'+submittedChat+'"]');
  await wait("document.querySelector('.chat-item.active')?.dataset.chat==="+JSON.stringify(submittedChat), 'return for admission round trip');
  await cdp('Fetch.enable', {patterns: [{urlPattern: url+'/api/chats/'+submittedChat+'/runs', requestStage: 'Response'}]});
  await send('Round trip accepted draft');
  await wait(() => heldSendResponse !== null, 'hold round trip admission');
  await click('[data-chat="'+otherChat+'"]');
  await wait("document.querySelector('.chat-item.active')?.dataset.chat==="+JSON.stringify(otherChat), 'leave pending round trip');
  await click('[data-chat="'+submittedChat+'"]');
  await wait("document.querySelector('.chat-item.active')?.dataset.chat==="+JSON.stringify(submittedChat), 'return during pending admission');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Round trip accepted draft', 'pending admission lost its draft');
  await cdp('Fetch.continueRequest', {requestId: heldSendResponse.requestId});
  await cdp('Fetch.disable'); heldSendResponse = null;
  await wait("!document.querySelector('#send').disabled && document.querySelector('#draft').value===''", 'accepted restored draft cleared');
  await reloadDocument();
  await wait("document.querySelector('.chat-item.active')?.dataset.chat==="+JSON.stringify(submittedChat), 'accepted draft reload');
  assert.equal(await evaluate("document.querySelector('#draft').value"), '', 'accepted draft resurrected after refresh');
  await cdp('Fetch.enable', {patterns: [{urlPattern: url+'/api/chats/'+submittedChat+'/runs', requestStage: 'Request'}]});
  await fill('#setting-system', 'Send with these live settings.');
  await send('Admission while reopening chat');
  await wait(() => heldSendResponse !== null, 'hold admission before server receives it');
  await fill('#setting-system', 'Save these newer settings afterwards.');
  assert.equal(await evaluate("document.querySelector('[data-action=save-settings]').disabled"), true, 'Save must wait for the earlier admission');
  await click('[data-chat="'+otherChat+'"]');
  await wait("document.querySelector('.chat-item.active')?.dataset.chat==="+JSON.stringify(otherChat), 'other view before stale snapshot');
  await evaluate(`(()=>{const original=window.fetch;window.releaseOldComposeSnapshot=null;window.fetch=async(...args)=>{const response=await original(...args);if(args[0]!=='/api/chats/'+${JSON.stringify(submittedChat)})return response;window.fetch=original;return new Promise(resolve=>window.releaseOldComposeSnapshot=()=>resolve(response));};})()`);
  await click('[data-chat="'+submittedChat+'"]');
  await wait("typeof releaseOldComposeSnapshot==='function'", 'snapshot captured before admission');
  await cdp('Fetch.continueRequest', {requestId: heldSendResponse.requestId});
  await cdp('Fetch.disable'); heldSendResponse = null;
  await wait("!document.querySelector('#send').disabled", 'admission while different chat visible');
  await evaluate('releaseOldComposeSnapshot()');
  await wait("document.querySelector('#messages').innerText.includes('The stale snapshot submission completed.') && document.querySelector('#stop').hidden", 'reopened chat refreshes pre-admission snapshot');
  assert.equal(await evaluate("document.querySelector('#draft').value"), '', 'old snapshot restored accepted draft');
  assert.equal(await evaluate("fetch('/api/chats/'+localStorage.getItem('llmx-chat')).then(r=>r.json()).then(d=>d.chat.settings.system)"), 'Send with these live settings.', 'Send without Save did not persist settings');
  await fill('#setting-system', 'Do not use previous conversation settings.');
  await evaluate("document.querySelector('#tool-settings input').checked=true");
  await click('[data-action="new"]');
  await wait("localStorage.getItem('llmx-chat')!=="+JSON.stringify(submittedChat), 'new chat from populated conversation');
  const defaultSettings = await evaluate("fetch('/api/chats/'+localStorage.getItem('llmx-chat')).then(r=>r.json()).then(d=>d.chat.settings)");
  assert.equal(defaultSettings.system, '', 'new chat inherited unsaved system instructions');
  assert.equal(defaultSettings.tools.length, 0, 'new chat inherited selected tools');
  const availableModel = await evaluate("document.querySelector('#model').value");
  const invalidModelChat = await evaluate("fetch('/api/chats',{method:'POST',headers:{'Content-Type':'application/json','X-Chat-Request':'1'},body:JSON.stringify({settings:{model:'Qwen3.5-0.8B-Q8_0.gguf'}})}).then(r=>r.json()).then(d=>d.id)");
  await evaluate(`localStorage.setItem('llmx-chat',${JSON.stringify(invalidModelChat)})`); await reloadDocument();
  await wait(`document.querySelector('#model').value==='Qwen3.5-0.8B-Q8_0.gguf'`, 'saved unavailable model restored');
  assert.equal(await evaluate("document.querySelector('#model option:checked').textContent"), 'Qwen3.5-0.8B-Q8_0.gguf (unavailable)', 'saved unavailable model is not labeled');
  assert.equal(await evaluate("document.querySelector('#model option:checked').disabled"), true, 'saved unavailable model remains selectable');
  assert.equal(await evaluate("document.querySelector('#send').disabled"), true, 'send remains available with an unavailable model');
  assert((await evaluate("document.querySelector('#banner').textContent")).includes('Choose another model'), 'unavailable model has no recovery guidance');
  await click('[data-action="settings"]');
  await fill('#setting-system', 'Saved while the model was unavailable.');
  await click('[data-action="save-settings"]');
  await wait("fetch("+JSON.stringify('/api/chats/'+invalidModelChat)+").then(r=>r.json()).then(d=>d.chat.settings.system==='Saved while the model was unavailable.')", 'save settings while the model is unavailable');
  const savedUnavailableModel = await evaluate(`fetch('/api/chats/${invalidModelChat}').then(r=>r.json()).then(d=>d.chat.settings.model)`);
  assert.equal(savedUnavailableModel, 'Qwen3.5-0.8B-Q8_0.gguf', 'opening an unavailable model erased saved settings');
  await evaluate(`document.querySelector('#model').value=${JSON.stringify(availableModel)};document.querySelector('#model').dispatchEvent(new Event('change',{bubbles:true}))`);
  assert.equal(await evaluate("document.querySelector('#send').disabled"), false, 'advertised model did not restore sending');
  const catalogOverride = await cdp('Page.addScriptToEvaluateOnNewDocument', {source: `(()=>{const original=window.fetch.bind(window);window.fetch=async(...args)=>{const response=await original(...args);if(new URL(typeof args[0]==='string'?args[0]:args[0].url,location.href).pathname!=='/api/config'||!['empty','missing'].includes(localStorage.getItem('llmx-test-model-catalog')))return response;const config=await response.json();if(localStorage.getItem('llmx-test-model-catalog')==='empty')config.models=[];else delete config.models;return new Response(JSON.stringify(config),{status:response.status,headers:{'Content-Type':'application/json'}})}})();`});
  for (const catalog of ['empty', 'missing']) {
    await evaluate(`localStorage.setItem('llmx-test-model-catalog',${JSON.stringify(catalog)})`); await reloadDocument();
    await wait("document.querySelector('#connection').textContent==='Connected' && document.querySelector('#model').textContent.includes('(unavailable)')", catalog + ' model catalog handled');
    assert.equal(await evaluate("document.querySelector('#model').options.length"), 1, catalog + ' model catalog rendered extra selectable entries');
    assert.equal(await evaluate("document.querySelector('#model option:checked').textContent"), 'Qwen3.5-0.8B-Q8_0.gguf (unavailable)', catalog + ' catalog did not create one unavailable saved-model choice');
    assert.equal(await evaluate("document.querySelector('#send').disabled"), true, catalog + ' catalog allowed sending without an available model');
  }
  await evaluate("localStorage.removeItem('llmx-test-model-catalog')");
  await cdp('Page.removeScriptToEvaluateOnNewDocument', {identifier: catalogOverride.identifier});
  await evaluate(`localStorage.setItem('llmx-chat',${JSON.stringify(submittedChat)})`); await reloadDocument();
  await wait(`document.querySelector('.chat-item.active')?.dataset.chat===${JSON.stringify(submittedChat)}`, 'restore prior chat after unavailable-model check');
  await fill('#draft', 'Keep this draft while the model catalogue changes');
  await fill('#setting-temperature', '0.3');
  await upload('#file-input', fixture);
  await wait("document.querySelector('#attachments').innerText.includes('notes.txt')", 'catalogue refusal attachment');
  await evaluate(`(() => {
    const original = window.fetch.bind(window);
    window.restoreCatalogueFetch = () => { window.fetch = original; };
    window.catalogueStatus = 503;
    window.catalogueModels = [];
    window.catalogueCalls = [];
    window.fetch = async (...args) => {
      const url = new URL(typeof args[0] === 'string' ? args[0] : args[0].url, location.href);
      if (!url.pathname.startsWith('/api/chats/') || !url.pathname.endsWith('/runs') || args[1]?.method !== 'POST') return original(...args);
      window.catalogueCalls.push(JSON.parse(args[1].body));
      const available = window.catalogueStatus !== 503;
      const result = {models: window.catalogueModels.map(id => ({id})), upstream_available: available,
        upstream_error: available ? null : 'The model server is unavailable.'};
      if (window.catalogueStatus === 201) result.run = {id: 'catalogue-browser-fixture'};
      else Object.assign(result, {code: 'model_unavailable', error: available ? 'Choose an available model.' : 'The model server is unavailable. Try again.'});
      return new Response(JSON.stringify(result), {status: window.catalogueStatus, headers: {'Content-Type': 'application/json'}});
    };
  })()`);
  await click('#send');
  await wait("document.querySelector('#connection').textContent==='Unavailable' && !document.querySelector('#send').disabled", 'failed catalogue check allows retry');
  assert.equal(await evaluate("document.querySelector('[data-action=save-settings]').disabled"), false, 'refused admission kept Save disabled');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Keep this draft while the model catalogue changes');
  assert.equal(await evaluate("document.querySelector('#setting-temperature').value"), '0.3');
  assert((await evaluate("document.querySelector('#attachments').innerText")).includes('notes.txt'));
  await evaluate(`(() => {
    const original=window.fetch, url='/api/chats/'+localStorage.getItem('llmx-chat');
    window.releaseOfflineSave=null;
    window.fetch=async (...args) => {
      const response=await original(...args);
      if(args[0]===url && args[1]?.method==='PATCH') {
        window.fetch=original; await new Promise(resolve=>window.releaseOfflineSave=resolve);
      }
      return response;
    };
  })()`);
  await click('[data-action="save-settings"]');
  await wait("typeof window.releaseOfflineSave==='function' && fetch('/api/chats/'+localStorage.getItem('llmx-chat')).then(r=>r.json()).then(d=>d.chat.settings.temperature===0.3)", 'settings saved while upstream connection was unavailable');
  assert.equal(await evaluate("document.querySelector('#send').disabled"), true, 'Send reopened before the settings save response');
  await evaluate('window.releaseOfflineSave()');
  await wait("!document.querySelector('[data-action=save-settings]').disabled && !document.querySelector('#send').disabled", 'save response releases reply submission');
  await evaluate(`window.catalogueStatus=201;window.catalogueModels=[${JSON.stringify(availableModel)}]`);
  await click('#send');
  await wait("window.catalogueCalls.length===2 && document.querySelector('#connection').textContent==='Connected' && !document.querySelector('#send').disabled", 'successful admission restores model connection');
  await fill('#draft', 'Keep the draft after the upstream switches model');
  await fill('#setting-temperature', '0.4');
  await upload('#file-input', fixture);
  await wait("document.querySelector('#attachments').innerText.includes('notes.txt')", 'changed catalogue attachment');
  await evaluate("window.catalogueStatus=409;window.catalogueModels=['replacement-model']");
  await click('#send');
  await wait("document.querySelector('#model').textContent.includes('replacement-model') && document.querySelector('#model option:checked').disabled", 'changed upstream catalogue replaces stale choices');
  assert.equal(await evaluate("document.querySelector('#send').disabled"), true);
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Keep the draft after the upstream switches model');
  assert.equal(await evaluate("document.querySelector('#setting-temperature').value"), '0.4');
  assert((await evaluate("document.querySelector('#attachments').innerText")).includes('notes.txt'));
  assert.equal(await evaluate("window.catalogueCalls[2].settings.model"), availableModel);
  await evaluate("document.querySelector('#model').value='replacement-model';document.querySelector('#model').dispatchEvent(new Event('change',{bubbles:true}))");
  assert.equal(await evaluate("document.querySelector('#send').disabled"), false);
  await evaluate(`window.catalogueModels=[${JSON.stringify(availableModel)}]`);
  await click('.message.assistant [data-action="retry"]');
  await wait("window.catalogueCalls.at(-1)?.regenerate && document.querySelector('#model option:checked').disabled", 'retry refreshes a changed catalogue');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Keep the draft after the upstream switches model');
  assert.equal(await evaluate("document.querySelector('#setting-temperature').value"), '0.4');
  assert((await evaluate("document.querySelector('#attachments').innerText")).includes('notes.txt'));
  await evaluate("window.restoreCatalogueFetch();delete window.restoreCatalogueFetch");
  const noAnswerChat = await evaluate("ChatShell.api('/chats','POST',{title:'Reply without an answer',settings:{model:'qwen3-local',tools:['echo']}}).then(chat=>chat.id)");
  await evaluate("localStorage.setItem('llmx-chat',"+JSON.stringify(noAnswerChat)+")");
  await reloadDocument();
  await wait("document.querySelector('#chat-title').textContent==='Reply without an answer' && !document.querySelector('#send').disabled", 'reasoning-only chat ready');
  await send('Run the tool once.');
  await wait("document.querySelector('.tool-call .approval')", 'intermediate tool call awaiting approval');
  assert.equal(await evaluate("document.querySelector('.reply-no-answer')"), null, 'an intermediate tool call was marked as answerless');
  await click('.tool-call [data-action=approve]');
  await wait("document.querySelector('.message.assistant:last-of-type .reasoning')?.textContent.includes('no answer followed')", 'reasoning-only continuation streaming');
  assert.equal(await evaluate("document.querySelector('.reply-no-answer')"), null, 'streaming reasoning was marked as answerless');
  assert.equal((await fetch(upstreamFixture + '/browser-step')).status, 204);
  const noAnswer = "document.querySelector('.message.assistant:last-child .reply-no-answer')?.textContent==='The model stopped before sending an answer.'";
  await wait(noAnswer, 'completed continuation explains the absent answer');
  assert.equal(await evaluate("document.querySelector('.reply-no-answer').nextElementSibling.querySelector('[data-action=retry]').checkVisibility()"), true, 'Retry is visible beside the notice');
  await reloadDocument();
  await wait(noAnswer, 'saved reasoning-only completion preserves its notice on reload');
  assert.equal(await evaluate("ChatShell.api('/chats/'+localStorage.getItem('llmx-chat')).then(({messages,runs})=>messages.at(-1).finish_reason==='stop'&&runs.length===1)"), true, 'finish reason was lost or reopening retried automatically');
  const searchChat = await evaluate(`fetch('/api/import',{method:'POST',headers:{'Content-Type':'application/json','X-Chat-Request':'1'},body:JSON.stringify({version:1,chat:{title:'Unopened search fixture',active_leaf:'search-message'},messages:[{id:'search-message',role:'assistant',content:'unopened assistant witness'}]})}).then(r=>r.json()).then(chat=>chat.id)`);
  await reloadDocument();
  await wait("!document.querySelector('#login-dialog').open && !document.querySelector('[data-action=new]').disabled", 'reloaded chat search');
  await evaluate(`{
    const fetch = window.fetch;
    window.searchTranscriptRequests = 0;
    window.fetch = (url, options) => {
      if (new URL(url, location.href).pathname === '/api/chats/' + ${JSON.stringify(searchChat)} && (options?.method || 'GET') === 'GET') window.searchTranscriptRequests++;
      return fetch(url, options);
    };
  }`);
  const searchRow = `.chat-row:has([data-chat="${searchChat}"])`;
  const searchResults = condition => wait(condition, 'saved conversation search', 7000);
  await fill('#search', 'UNOPENED ASSISTANT');
  await searchResults(`document.querySelector(${JSON.stringify(searchRow)})!==null && document.querySelectorAll('#chat-list .chat-row').length===1`);
  assert.notEqual(await evaluate("localStorage.getItem('llmx-chat')"), searchChat, 'search opened an unopened conversation');
  await click(searchRow+' .chat-row-menu'); await click('#action-menu [data-action=archive]');
  await searchResults("document.querySelector('#chat-list').innerText.includes('No matching conversations.')");
  await click('[data-action=archive-view]');
  await searchResults(`document.querySelector(${JSON.stringify(searchRow)})!==null`);
  await evaluate(`{
    const fetchSearch = window.fetch;
    window.searchRequests = [];
    window.fetch = async (url, options) => {
      const path = new URL(url, location.href).pathname;
      if (path !== '/api/chats/search') return fetchSearch(url, options);
      const query = JSON.parse(options.body).query;
      window.searchRequests.push(query);
      if (query === 'force failure') return new Response(JSON.stringify({error:'Fixture search failed'}), {status:503});
      const response = await fetchSearch(url, options);
      if (query === 'unopened assistant') return new Promise(resolve => { window.releaseSearch = () => resolve(response); });
      return response;
    };
  }`);
  await fill('#search', 'unopened assistant');
  await searchResults("typeof window.releaseSearch==='function'");
  await fill('#search', 'obsolete query');
  await fill('#search', 'force failure');
  assert.deepEqual(await evaluate('window.searchRequests'), ['unopened assistant'], 'search allowed overlapping requests');
  await evaluate('window.releaseSearch()');
  await searchResults("document.querySelector('#chat-list').innerText.includes('Fixture search failed')");
  assert.deepEqual(await evaluate('window.searchRequests'), ['unopened assistant','force failure'], 'search queued an obsolete query');
  await evaluate('delete window.releaseSearch');
  await fill('#search', 'unopened assistant');
  await searchResults("typeof window.releaseSearch==='function'");
  await fill('#search', '');
  await searchResults(`document.querySelector(${JSON.stringify(searchRow)})!==null`);
  await evaluate('window.releaseSearch()');
  await click('[data-action=archive-view]');
  await searchResults(`!document.querySelector(${JSON.stringify(searchRow)}) && document.querySelector('#chat-list .chat-row')!==null`);
  await click('[data-action=archive-view]');
  await fill('#search', 'UNOPENED ASSISTANT');
  await searchResults(`document.querySelector(${JSON.stringify(searchRow)})!==null`);
  assert.equal(await evaluate('window.searchTranscriptRequests'), 0, 'search downloaded an unopened transcript');
  await click(searchRow+' .chat-item');
  await searchResults("document.querySelector('#messages').innerText.includes('unopened assistant witness')");
  assert.equal(await evaluate('window.searchTranscriptRequests'), 1, 'opening a result did not load its transcript once');
  const background = await evaluate(`fetch('/api/chats',{method:'POST',headers:{'Content-Type':'application/json','X-Chat-Request':'1'},body:JSON.stringify({title:'Background reply fixture'})}).then(r=>r.json()).then(chat=>chat.id)`);
  const backgroundRow = `.chat-row:has([data-chat="${background}"])`;
  const backgroundCue = `${backgroundRow} .chat-run-cue`;
  const expectCue = status => wait(`document.querySelector(${JSON.stringify(backgroundCue)})?.dataset.status===${JSON.stringify(status)}`, 'sidebar reply status '+status);
  const startBackground = () => evaluate(`fetch('/api/chats/'+${JSON.stringify(background)}+'/runs',{method:'POST',headers:{'Content-Type':'application/json','X-Chat-Request':'1'},body:JSON.stringify({content:'Background prompt',settings:{model:'qwen3-local',tools:['echo']}})}).then(r=>r.json()).then(data=>data.run.id)`);
  await reloadDocument();
  await wait(`document.querySelector(${JSON.stringify(backgroundRow)})!==null && !document.querySelector('[data-action=new]').disabled`, 'unopened background chat in catalogue');
  const backgroundRun = await startBackground();
  await expectCue('running');
  await evaluate("document.documentElement.dataset.theme='dark'");
  assert.equal(await evaluate(`getComputedStyle(document.querySelector(${JSON.stringify(backgroundCue)}),'::before').borderTopWidth`), '1px');
  await screenshot('background-status-running-dark.png');
  const streamCountAtBackgroundApproval = runStreams.length;
  await pointerClick(`${backgroundRow} .chat-row-menu`);
  await evaluate(`window.backgroundRow=document.querySelector(${JSON.stringify(backgroundRow)}); window.backgroundFocus=document.activeElement`);
  assert.equal((await fetch(upstreamFixture + '/browser-step')).status, 204);
  await expectCue('approval');
  assert.equal(await evaluate(`document.querySelector(${JSON.stringify(backgroundRow)})===window.backgroundRow && document.activeElement===window.backgroundFocus && !document.querySelector('#action-menu').hidden`), true, 'sidebar status replaced a focused row or open menu');
  assert.equal(runStreams.length, streamCountAtBackgroundApproval, 'sidebar status opened a background transcript stream');
  await evaluate("window.ChatShell.closeMenu()");
  await reloadDocument();
  await expectCue('approval');
  assert.notEqual(await evaluate("localStorage.getItem('llmx-chat')"), background, 'reload selected the background chat');
  await evaluate("document.documentElement.dataset.theme='dark'");
  await screenshot('background-status-desktop-dark.png');
  await click(`${backgroundRow} .chat-row-menu`); await click('#action-menu [data-action=archive]');
  await wait(`!document.querySelector(${JSON.stringify(backgroundRow)})`, 'background row archived');
  await click('[data-action=archive-view]'); await fill('#search', 'Background reply fixture');
  await expectCue('approval');
  await cdp('Emulation.setDeviceMetricsOverride', {width:320,height:844,deviceScaleFactor:1,mobile:true});
  await wait("document.documentElement.dataset.compact==='true'", 'compact background sidebar');
  await evaluate("document.documentElement.dataset.theme='light'");
  await pointerClick('.mobile-nav');
  await wait(`{const cue=document.querySelector(${JSON.stringify(backgroundCue)});const box=cue.getBoundingClientRect();box.left>=0 && box.right<=innerWidth && getComputedStyle(cue).visibility==='visible'}`, 'archived search result cue fits on mobile');
  await screenshot('background-status-mobile-light.png');
  await cdp('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await evaluate("document.documentElement.dataset.theme='dark'");
  assert.equal(await evaluate(`fetch('/api/runs/'+${JSON.stringify(backgroundRun)}+'/cancel',{method:'POST',headers:{'Content-Type':'application/json','X-Chat-Request':'1'},body:'{}'}).then(r=>r.status)`), 200);
  await expectCue('');
  await startBackground();
  await expectCue('failed');
  assert.equal(await evaluate(`document.querySelector(${JSON.stringify(backgroundRow+' .chat-item')}).getAttribute('aria-label').includes('Latest reply failed')`), true, 'failed sidebar status does not name the latest attempt');
  await evaluate(`{
    const fetch = window.fetch;
    const setTimeout = window.setTimeout;
    window.statusFetches = 0;
    window.statusAborts = 0;
    window.setTimeout = (callback, delay, ...args) => {
      const id = setTimeout(callback, delay, ...args);
      if (delay === 9000) window.expireStatus = () => { window.clearTimeout(id); callback(...args); };
      return id;
    };
    window.fetch = (url, options) => {
      if (url === '/api/chats/run-status') {
        const number = ++window.statusFetches;
        return new Promise((resolve, reject) => {
          window.releaseStatus = resolve;
          options.signal.addEventListener('abort', () => {
            window.statusAborts++;
            if (number !== 2) reject(options.signal.reason);
          }, {once:true});
        });
      }
      return fetch(url, options);
    };
  }`);
  await wait("window.statusFetches===1 && typeof window.expireStatus==='function'", 'hold a background status response with its deadline armed');
  await evaluate('window.expireStatus()');
  await wait("window.statusAborts===1 && window.statusFetches===2", 'expired status request aborts and the next poll starts');
  await click('[data-view=preferences]');
  assert.equal(await evaluate('window.statusAborts'), 2, 'leaving Chat did not abort the status request');
  await evaluate("window.releaseStatus(new Response(JSON.stringify({chats:[{id:"+JSON.stringify(background)+",status:'approval'}]})))");
  assert.equal(await evaluate(`document.querySelector(${JSON.stringify(backgroundCue)})?.dataset.status`), '', 'status remains active outside Chat');
  await evaluate("delete window.releaseStatus; Object.defineProperty(document,'hidden',{configurable:true,value:true}); document.dispatchEvent(new Event('visibilitychange'))");
  await click('[data-view=chat]');
  await delay(3150);
  assert.equal(await evaluate('window.statusFetches'), 2, 'hidden Chat started a status request');
  await evaluate("delete document.hidden; document.dispatchEvent(new Event('visibilitychange'))");
  await wait("window.statusFetches===3 && typeof window.releaseStatus==='function'", 'visible Chat refreshes status immediately');
  await evaluate("window.releaseStatus(new Response(JSON.stringify({error:'Sign in to continue'}),{status:401}))");
  await wait("document.querySelector('#login-dialog').open", 'sidebar status detects an expired session');
  assert.equal(await evaluate(`document.querySelector(${JSON.stringify(backgroundCue)})?.dataset.status`), '', 'status leaked across expiry');
  if(originalDashboard){await navigateDocument(originalDashboard);await wait("document.querySelector('main')!==null",'original dashboard');await screenshot('original-dashboard-1440x1000.png');}
  assert.deepEqual(errors, [], 'uncaught browser exceptions');
  assert.deepEqual(nativeDialogs, [], 'native JavaScript dialog was used instead of the shared accessible dialog');
  console.log('PASS: chat regressions, shared monitor/theme, code/diff/HTML isolation, workspace/checkpoints/conflicts/jobs, backup/restore, responsive geometry and CSS 200% zoom. Screenshots: ' + artifacts);
  }
} catch (error) {
  const diagnostic = launchDiagnostic('Browser fixture failed after ' + (Date.now() - startupStarted) + ' ms').message;
  const failure = {error: error.stack || String(error), diagnostic, pending: [...pending.values()].map(call => call.method)};
  console.error(failure.error + '\n' + diagnostic);
  try { await mkdir(artifacts, {recursive: true}); await writeFile(path.join(artifacts, 'failure.json'), JSON.stringify(failure, null, 2)); }
  catch (writeError) { console.error('Could not save browser failure evidence: ' + writeError.message); }
  if (socket?.readyState === 1) { try { await screenshot('failure.png'); } catch (captureError) { console.error('Failure screenshot unavailable: ' + captureError.message); } }
  process.exitCode = 1;
} finally {
  if (socket?.readyState === 1) { try { await cdp('Browser.close'); } catch {} socket.close(); }
  for (const call of pending.values()) { clearTimeout(call.timeout); call.reject(new Error('Browser closed')); }
  await Promise.race([new Promise(resolve => chrome.once('exit', resolve)), delay(2500)]);
  if (chrome.exitCode === null) chrome.kill();
}
