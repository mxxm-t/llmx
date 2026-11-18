/* Real Chrome integration without npm packages. Invoked by test_ui.py. */
import {spawn} from 'node:child_process';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

const [url, executable, profile, artifacts, fixture, imageFixture, originalDashboard, upstreamFixture,
  textMaxFixture, textOverFixture, imageMaxFixture, imageOverFixture, tinyFixture, textRawOverFixture, textBomMaxFixture,
  textRaceAFixture, textRaceBFixture] = process.argv.slice(2);
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const startupTimeoutArg = process.argv.find(arg => arg.startsWith('--startup-timeout-ms='));
const startupTimeoutCandidate = startupTimeoutArg ? Number(startupTimeoutArg.slice('--startup-timeout-ms='.length)) : NaN;
const startupTimeoutMs = Number.isSafeInteger(startupTimeoutCandidate) && startupTimeoutCandidate > 0
  ? startupTimeoutCandidate : 30000;
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
const nativeDialogs = [];
const apiMutations = [];
async function cdp(method, params = {}) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    const timeoutError = new Error('CDP timed out: ' + method);
    const timeout = setTimeout(() => { pending.delete(id); reject(timeoutError); }, 12000);
    pending.set(id, {resolve, reject, timeout});
    socket.send(JSON.stringify({id, method, params}));
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
  let target;
  while (!target && Date.now() < startupDeadline) {
    const remaining = startupDeadline - Date.now();
    try {
      const response = await fetch(startupUrl, {signal: AbortSignal.timeout(Math.min(500, Math.max(1, remaining)))});
      if (response.ok) target = (await response.json()).find(t => t.type === 'page');
    } catch {}
    if (!target) await delay(Math.min(80, Math.max(0, startupDeadline - Date.now())));
  }
  if (!target) throw launchDiagnostic('Chrome DevTools HTTP endpoint did not become ready within ' + (Date.now() - startupStarted) + ' ms (limit ' + startupTimeoutMs + ' ms)');
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    const remaining = startupDeadline - Date.now();
    if (remaining <= 0) { reject(launchDiagnostic('Chrome DevTools WebSocket did not open within ' + (Date.now() - startupStarted) + ' ms (limit ' + startupTimeoutMs + ' ms)')); return; }
    const timeout = setTimeout(() => reject(launchDiagnostic('Chrome DevTools WebSocket did not open within ' + (Date.now() - startupStarted) + ' ms (limit ' + startupTimeoutMs + ' ms)')), remaining);
    socket.onopen = () => { clearTimeout(timeout); resolve(); };
    socket.onerror = () => { clearTimeout(timeout); reject(launchDiagnostic('Chrome DevTools WebSocket connection failed')); };
  });
  console.log('Chrome DevTools ready in ' + (Date.now() - startupStarted) + ' ms');
  socket.onmessage = event => {
    const data = JSON.parse(event.data);
    if (data.id && pending.has(data.id)) { const call = pending.get(data.id); pending.delete(data.id); clearTimeout(call.timeout); if (data.error) call.reject(new Error(JSON.stringify(data.error))); else call.resolve(data.result); }
    if (data.method === 'Runtime.exceptionThrown') { const detail = data.params.exceptionDetails; errors.push(detail.exception?.description || detail.text); }
    if (data.method === 'Network.requestWillBeSent' && /\/api\/runs\/[^/]+\/events(?:\?|$)/.test(data.params.request.url)) runStreams.push(data.params.request.url);
    if (data.method === 'Network.requestWillBeSent' && data.params.request.method !== 'GET') apiMutations.push({url: data.params.request.url, method: data.params.request.method});
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
  };
  await cdp('Page.enable'); await cdp('Runtime.enable'); await cdp('DOM.enable'); await cdp('Network.enable');
  await cdp('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
  await cdp('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: 'dark'}]});
  await navigateDocument(url);
  if (process.argv.includes('--nav-only')) {
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
      await wait("document.querySelector('.workspace-revision').textContent!=='Not loaded'",'live workspace');
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
  await fill('#password', 'browser-test'); await evaluate("document.querySelector('#login-form').requestSubmit()");
  await wait("!document.querySelector('#login-dialog').open && document.querySelector('#model').value==='qwen3-local' && !document.querySelector('[data-action=\"new\"]').disabled", 'authenticated workspace');
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
  const stableMarkdown = await evaluate(String.raw`(() => {
    const samples=['# Title\n\nParagraph with **bold**.\n\n- Item\n  - Nested\n\n\x60\x60\x60python\nprint(1)\n\x60\x60\x60\n\nTail', '| First | Second |\n| --- | --- |\n| A | B |\n\n> Quote\n\n<script>literal</script>'];
    for(const source of samples){
      const current=ChatRender.markdown('');
      for(let i=0;i<=source.length;i++){
        const expected=ChatRender.markdown(source.slice(0,i));
        ChatRender.reconcileMarkdown(current,ChatRender.markdown(source.slice(0,i)));
        if(current.innerHTML!==expected.innerHTML)return {pass:false,stage:'prefix',offset:i};
      }
      for(const suffix of ['\n\n'+source,source,'short','',source]){
        ChatRender.reconcileMarkdown(current,ChatRender.markdown(suffix));
        if(current.innerHTML!==ChatRender.markdown(suffix).innerHTML)return {pass:false,stage:'rewrite'};
      }
    }
    const text='Completed paragraph\n\n\x60\x60\x60python\nprint(1)\n\x60\x60\x60\n\nChanging';
    const current=ChatRender.markdown(text);document.body.append(current);
    const paragraph=current.firstElementChild, button=current.querySelector('.code-heading button');button.textContent='Copied';button.focus();
    const selection=getSelection(),range=document.createRange();range.selectNodeContents(paragraph);selection.removeAllRanges();selection.addRange(range);
    ChatRender.reconcileMarkdown(current,ChatRender.markdown(text+' final block'));
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
    assert.equal(await evaluate("getComputedStyle(document.querySelector('.message.user>.markdown')).borderTopWidth==='0px' && getComputedStyle(document.querySelector('.message.user>.markdown')).backgroundColor!==getComputedStyle(document.querySelector('.message.user button')).backgroundColor"), true, 'user message still styled as a control');
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
  assert.equal((await fetch(upstreamFixture + '/browser-step')).status, 204);
  await wait("document.querySelector('.message.assistant')?.innerText.includes('An alternative approach.') && document.querySelector('#stop').hidden", 'retry branch');
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
          toolHeight:tool.getBoundingClientRect().height,answerHeight:answer.getBoundingClientRect().height,
          overflow:document.documentElement.scrollWidth>innerWidth};
      })()`);
      replyLayout.push({width, theme, ...geometry});
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
  await wait("document.querySelector('#stop').hidden && document.querySelector('#messages').innerText.includes('The attached file is readable.')", 'attachment response');
  await click('[data-tab="stats"]');
  await click('#panel-stats .conversation-totals > summary');
  assert((await evaluate("document.querySelector('#panel-stats').innerText")).includes('960'), 'all-branch totals omitted retries');
  assert.equal(await evaluate("document.querySelectorAll('.attachment-image').length"), 1, 'image attachment missing');
  assert.equal(await evaluate("{const b=document.querySelector('.message.assistant [data-action=retry]'),s=getComputedStyle(b),r=b.getBoundingClientRect(); b.getAttribute('aria-label')==='Retry response' && b.querySelector('svg[aria-hidden=true]') && !b.textContent.trim() && s.backgroundColor!=='rgba(0, 0, 0, 0)' && s.borderTopWidth==='1px' && r.width>=32 && r.height>=30}"), true, 'message actions lack an accessible icon, visible surface or minimum target size');
  await evaluate("{const b=document.querySelector('.message.assistant [data-action=copy]');b.blur();b.focus({preventScroll:true})}");
  await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9});
  await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9});
  await wait("document.activeElement?.getAttribute('aria-describedby')?.startsWith('message-action-tooltip-')", 'keyboard focus tooltip');
  assert.equal(await evaluate("{const b=document.activeElement,t=document.getElementById(b.getAttribute('aria-describedby'));b.closest('.message-actions')!==null && getComputedStyle(b).outlineStyle==='solid' && getComputedStyle(b).outlineWidth==='2px' && t?.getAttribute('role')==='tooltip'}"), true, 'message action keyboard focus or tooltip is not visible');
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
  const userEditPoint = await actionHoverPoint('.message.user [data-action=edit]');
  assert(userEditPoint.hit, 'Edit hover target is obscured: ' + JSON.stringify(userEditPoint));
  await cdp('Input.dispatchMouseEvent', {type: 'mouseMoved', x: userEditPoint.x, y: userEditPoint.y});
  assert.equal(await evaluate("{const b=document.querySelector('.message.user [data-action=edit]'),t=document.getElementById(b.getAttribute('aria-describedby')),r=t?.getBoundingClientRect(); b.matches(':hover') && t?.textContent==='Edit message' && r.left>=0 && r.right<=innerWidth}"), true, 'edge message tooltip is clipped');
  await evaluate("{const s=document.querySelector('#scroll-region');s.scrollTop=s.scrollTop>0?s.scrollTop-1:1}");
  await wait("!document.querySelector('.message-action-tooltip')", 'tooltip dismissal while scrolling');
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
  await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: 'k', code: 'KeyK', modifiers: 2});
  await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: 'k', code: 'KeyK', modifiers: 2});
  await wait("document.querySelectorAll('.message').length===0 && !document.querySelector('#welcome').hidden", 'keyboard new conversation');
  await click('.topbar [data-action="chat-menu"]'); await click('[data-action="delete"]'); await closeAction('#conversation-submit');
  await wait("document.querySelector('#chat-title').textContent==='New conversation'", 'confirmed delete');
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
  await wait("document.querySelectorAll('.workspace-tool-open').length===4", 'workspace tool navigation buttons');
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
  await click('.workbench-view .primary-button');
  await wait("document.querySelector('#action-dialog').open", 'accessible new-file dialog', 700);
  assert.equal(await evaluate("document.querySelector('#action-dialog-title').textContent"), 'New workspace file');
  assert.equal(await evaluate("document.activeElement.id"), 'conversation-value', 'new-file dialog did not focus its field');
  await screenshot('studio-new-file-dialog-desktop.png');
  const longestWorkspacePath = 'p'.repeat(236) + '.txt';
  await cdp('Input.insertText', {text: longestWorkspacePath});
  assert.equal(await evaluate("document.querySelector('#conversation-value').value"), longestWorkspacePath, 'new-file field shortened a valid 240-character path');
  const escapedForm = evaluate("new Promise(resolve=>document.querySelector('#action-dialog').addEventListener('close',()=>resolve(true),{once:true}))");
  await cdp('Input.dispatchKeyEvent', {type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27});
  await cdp('Input.dispatchKeyEvent', {type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27});
  await escapedForm;
  assert.equal(await evaluate("document.querySelector('#action-dialog').open"), false, 'Escape cancels new-file dialog');
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), '', 'cancel changed editor contents');
  await click('.workbench-view .primary-button');
  await wait("document.querySelector('#action-dialog').open", 'reopened new-file dialog');
  await fill('#conversation-value', 'Research'); await closeAction('#conversation-submit');
  await wait("!document.querySelector('#action-dialog').open && !document.querySelector('#workspace-editor').disabled && document.querySelector('.workspace-file-path').textContent==='Research'", 'new workspace file ready');
  await fill('#workspace-editor', 'first version\n'); await click('#workspace-save');
  await wait("document.querySelector('.workspace-file-name')?.textContent==='Research' && document.querySelector('#workspace-save').disabled", 'workspace file save');
  const editorTabResults = [];
  for (const [modifiers, target] of [[0, '[aria-label="Create checkpoint"]'], [8, '.workspace-editor-actions .danger']]) {
    await evaluate("{const editor=document.querySelector('#workspace-editor');editor.focus();editor.setSelectionRange(0,5)}");
    const tabKey = {key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9, modifiers};
    await cdp('Input.dispatchKeyEvent', {type: 'keyDown', ...tabKey});
    await cdp('Input.dispatchKeyEvent', {type: 'keyUp', ...tabKey});
    editorTabResults.push(await evaluate(`({focus:document.activeElement.matches(${JSON.stringify(target)}),
      value:document.querySelector('#workspace-editor').value, clean:document.querySelector('#workspace-save').disabled})`));
  }
  assert.deepEqual(editorTabResults, Array(2).fill({focus: true, value: 'first version\n', clean: true}),
    'Tab and Shift+Tab should move focus out of the Workspace editor without changing the saved buffer');
  await click('[aria-label="Create checkpoint"]');
  await wait("document.querySelector('#action-dialog').open", 'checkpoint label dialog');
  assert.equal(await evaluate("document.querySelector('#conversation-value').maxLength"), 120, 'checkpoint field exceeds server limit');
  const checkpointMutations = apiMutations.filter(item => item.url.endsWith('/api/workspace/checkpoints')).length;
  await closeAction('#conversation-cancel');
  assert.equal(apiMutations.filter(item => item.url.endsWith('/api/workspace/checkpoints')).length, checkpointMutations, 'cancelled checkpoint creation sent a request');
  await click('[aria-label="Create checkpoint"]'); await wait("document.querySelector('#action-dialog').open", 'reopened checkpoint label dialog');
  await fill('#conversation-value', 'Workspace checkpoint'); await closeAction('#conversation-submit');
  await wait("document.querySelectorAll('.workspace-checkpoint').length===1", 'checkpoint creation');
  const fileWritesBeforeArtifactCancel = apiMutations.filter(item => item.url.endsWith('/api/workspace/file')).length;
  await evaluate("document.dispatchEvent(new CustomEvent('companion:artifact-edit',{detail:{name:'generated.py',source:'print(1)'}}))");
  await wait("document.querySelector('#action-dialog').open", 'generated-file edit dialog');
  assert.equal(await evaluate("document.querySelector('#action-dialog-title').textContent"), 'Edit generated file');
  assert.equal(await evaluate("document.querySelector('#conversation-value').value"), 'generated.py');
  await closeAction('#conversation-cancel');
  assert.equal(apiMutations.filter(item => item.url.endsWith('/api/workspace/file')).length, fileWritesBeforeArtifactCancel, 'cancelled generated-file edit wrote a file');
  assert.equal(await evaluate("document.querySelector('.workspace-file-path').textContent"), 'Research', 'cancelled generated-file edit changed the selected file');
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
  await click('.checkpoint-actions button:first-child');
  await wait("document.querySelector('#artifact-dialog').open", 'checkpoint diff');
  assert((await evaluate("document.querySelector('.artifact-source').textContent")).includes('+third version'));
  assert.equal(await evaluate("document.querySelector('.artifact-source .syntax-add')!==null"), true, 'diff highlighting missing');
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
      call('linked-write','write',{path:'external.txt'},{revision:workspace.revision,files:workspace.files}),
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
  assert.equal(await evaluate("document.querySelector("+JSON.stringify(link('malformed'))+").textContent"), 'Open Workspace', 'malformed result keeps the general Workspace action');
  const linkMutations = apiMutations.filter(item=>item.url.includes('/api/workspace/')).length;
  await pointerClick(link('read'));
  await wait("document.querySelector('#workspace-editor').value==='Encoded Workspace path.' && document.activeElement.id==='workspace-editor'", 'open exact escaped file path');
  assert.equal(await evaluate("document.querySelector('.workspace-file-path').textContent"), linkedWorkspace.safePath);
  await fill('#workspace-editor', 'Keep this dirty buffer.');
  await click('.global-nav [data-view="chat"]');
  await pointerClick(link('write'));
  await wait("document.querySelector('#action-dialog').open", 'discard guard for linked file');
  await closeAction('#conversation-cancel');
  assert.equal(await evaluate("document.querySelector('#workspace-editor').value"), 'Keep this dirty buffer.', 'cancelled link discarded the active buffer');
  await click('.global-nav [data-view="chat"]');
  await pointerClick(link('write'));
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
  const previewSource = '<!doctype html><html><body><h1>Isolated preview</h1><script>console.log("preview-ready");try{parent.document.body.dataset.previewTouched="yes"}catch(e){console.log("parent-blocked")}fetch("/api/session").then(()=>console.error("network-allowed")).catch(()=>console.log("network-blocked"));setTimeout(()=>{throw new Error("preview-error")},20)</script></body></html>';
  await evaluate(`window.ChatArtifacts.open({name:'preview.html',language:'html',source:${JSON.stringify(previewSource)}})`);
  await click('#artifact-preview-tab');
  await wait("document.querySelector('.artifact-console-output').textContent.includes('preview-ready') && document.querySelector('.artifact-console-output').textContent.includes('network-blocked')", 'isolated preview and console bridge');
  assert.equal(await evaluate("document.querySelector('.artifact-frame').getAttribute('sandbox')"), 'allow-scripts');
  assert.equal(await evaluate("document.body.dataset.previewTouched"), undefined, 'preview accessed parent document');
  assert((await evaluate("document.querySelector('.artifact-console-output').textContent")).includes('parent-blocked'));
  await wait("document.querySelector('.artifact-console-output').textContent.includes('preview-error')", 'preview error bridge');
  await evaluate("new Promise(resolve=>{function received(event){if(event.data?.token==='forged'){window.removeEventListener('message',received);resolve();}}window.addEventListener('message',received);window.postMessage({channel:'llmx-preview',token:'forged',kind:'error',text:'forged-output'},location.origin);})");
  assert.equal(await evaluate("document.querySelector('.artifact-console-output').textContent.includes('forged-output')"), false, 'preview accepted forged bridge message');
  await screenshot('studio-artifact-preview.png');
  await click('#artifact-source-tab');
  assert.equal(await evaluate("document.querySelector('.artifact-frame')===null"), true, 'source toggle left preview running');
  await click('#artifact-dialog [aria-label="Close artifact"]');
  for (const [width,height] of [[360,800],[390,844],[768,1024],[1024,768],[1440,1000],[844,390],[720,500]]) {
    await cdp('Emulation.setDeviceMetricsOverride', {width, height, deviceScaleFactor: width===720 ? 2 : 1, mobile: width<=390});
    await evaluate("window.dispatchEvent(new Event('resize'))");
    for (const section of ['chat','monitor','workbench','preferences']) {
      await evaluate(`window.ChatShell.show(${JSON.stringify(section)})`);
      assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'), true, section+' overflows at '+width+'x'+height);
      if(section==='chat')assert.equal(await evaluate("document.querySelector('.composer').getBoundingClientRect().bottom<=innerHeight"), true, 'composer clipped at '+width+'x'+height);
    }
    await screenshot('studio-settings-'+width+'x'+height+'.png');
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
  await delay(300);
  assert.equal(await evaluate("Object.keys(sessionStorage).filter(k=>k.startsWith('llmx-draft:')).some(k=>sessionStorage.getItem(k).includes('Temporary draft privacy marker.'))"), false, 'temporary draft was persisted');
  await holdHistoryRefresh();
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
  assert.deepEqual(await evaluate("Array.from(document.querySelectorAll('#action-menu button')).map(b=>b.dataset.action)"), ['export', 'delete'], 'temporary menu exposes unsupported actions');
  await evaluate("window.ChatShell.closeMenu()");
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
  assert.equal(await evaluate("document.querySelector('#chat-title').textContent==='New conversation' && document.querySelector('.chat-item.active')===null"), true, 'leaving temporary mode did not return to the blank normal view');
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
  await wait("document.querySelector('#send').disabled===false", 'same-chat edited request completion');
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
  if(originalDashboard){await navigateDocument(originalDashboard);await wait("document.querySelector('main')!==null",'original dashboard');await screenshot('original-dashboard-1440x1000.png');}
  assert.deepEqual(errors, [], 'uncaught browser exceptions');
  assert.deepEqual(nativeDialogs, [], 'native JavaScript dialog was used instead of the shared accessible dialog');
  console.log('PASS: chat regressions, shared monitor/theme, code/diff/HTML isolation, workspace/checkpoints/conflicts/jobs, backup/restore, responsive geometry and CSS 200% zoom. Screenshots: ' + artifacts);
  }
} catch (error) {
  if (socket?.readyState === 1) { try { await screenshot('failure.png'); } catch {} }
  console.error(error.stack || error); process.exitCode = 1;
} finally {
  if (socket?.readyState === 1) { try { await cdp('Browser.close'); } catch {} socket.close(); }
  for (const call of pending.values()) { clearTimeout(call.timeout); call.reject(new Error('Browser closed')); }
  await Promise.race([new Promise(resolve => chrome.once('exit', resolve)), delay(2500)]);
  if (chrome.exitCode === null) chrome.kill();
}
