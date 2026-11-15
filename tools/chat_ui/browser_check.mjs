/* Real Chrome integration without npm packages. Invoked by test_ui.py. */
import {spawn} from 'node:child_process';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

const [url, executable, profile, artifacts, fixture, imageFixture, originalDashboard, upstreamFixture] = process.argv.slice(2);
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const launchArgs = ['--headless=new', '--no-first-run', '--no-default-browser-check',
  '--disable-background-networking', '--remote-debugging-port=0', '--user-data-dir=' + profile,
  '--window-size=1440,1000', 'about:blank'];
if (process.platform === 'linux') launchArgs.unshift('--no-sandbox');
const chrome = spawn(executable, launchArgs, {stdio: ['ignore', 'ignore', 'pipe'], windowsHide: true});
let launchError = null, launchStderr = '';
chrome.on('error', error => { launchError = error; });
chrome.stderr.on('data', chunk => { launchStderr = (launchStderr + chunk.toString()).slice(-8192); });
function launchDiagnostic(reason) { return new Error(reason + '\nExecutable: ' + executable + '\nPlatform: ' + process.platform + '\nChrome stderr (last 8 KiB):\n' + (launchStderr || '[No stderr received]')); }
let socket;
let nextId = 0;
const pending = new Map();
const errors = [];
const runStreams = [];
let heldMonitorLoad = null;
let heldRestoreRequest = null;
const nativeDialogs = [];
const apiMutations = [];
async function cdp(method, params = {}) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { pending.delete(id); reject(new Error('CDP timed out: ' + method)); }, 12000);
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
  throw new Error('Timed out waiting for ' + description + '\n' + await evaluate('document.body.innerText'));
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
async function idle() { await wait("document.querySelector('#stop').hidden && document.querySelectorAll('.message.assistant').length>0", 'completed response'); }
async function upload(selector, filename) { const root = await cdp('DOM.getDocument'); const target = await cdp('DOM.querySelector', {nodeId: root.root.nodeId, selector}); await cdp('DOM.setFileInputFiles', {nodeId: target.nodeId, files: [filename]}); }
async function screenshot(name) { await delay(240); const result = await cdp('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false}); await writeFile(path.join(artifacts, name), Buffer.from(result.data, 'base64')); }

try {
  let port;
  for (let i = 0; i < 100; i++) {
    if (launchError) throw launchDiagnostic('Chrome failed to spawn: ' + launchError.message);
    if (chrome.exitCode !== null || chrome.signalCode !== null) throw launchDiagnostic('Chrome exited before opening DevTools: exit=' + chrome.exitCode + ', signal=' + chrome.signalCode);
    try { port = Number((await readFile(path.join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]); if (port) break; } catch {}
    await delay(80);
  }
  if (!port) throw launchDiagnostic('Chrome did not open DevTools within 8 seconds');
  const targets = await (await fetch('http://127.0.0.1:' + port + '/json/list')).json();
  const target = targets.find(t => t.type === 'page');
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  socket.onmessage = event => {
    const data = JSON.parse(event.data);
    if (data.id && pending.has(data.id)) { const call = pending.get(data.id); pending.delete(data.id); clearTimeout(call.timeout); if (data.error) call.reject(new Error(JSON.stringify(data.error))); else call.resolve(data.result); }
    if (data.method === 'Runtime.exceptionThrown') errors.push(data.params.exceptionDetails.text);
    if (data.method === 'Network.requestWillBeSent' && /\/api\/runs\/[^/]+\/events(?:\?|$)/.test(data.params.request.url)) runStreams.push(data.params.request.url);
    if (data.method === 'Network.requestWillBeSent' && data.params.request.method !== 'GET') apiMutations.push({url: data.params.request.url, method: data.params.request.method});
    if (data.method === 'Fetch.requestPaused') {
      if (data.params.request.url.endsWith('/api/restore')) heldRestoreRequest = data.params.requestId;
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
  await screenshot('desktop-empty.png');
  await evaluate("for (let i=0;i<12;i++) document.querySelector('[data-action=\"new\"]').click()");
  await wait("document.querySelector('.chat-item.active')!==null", 'new empty conversation');
  await evaluate("new Promise(resolve => setTimeout(resolve, 200))");
  assert.equal(await evaluate("fetch('/api/chats').then(r=>r.json()).then(d=>d.chats.length)"), 1, 'rapid New clicks created duplicate drafts');
  await fill('#draft', 'Keep my unsent draft');
  await click('[data-action="new"]');
  await evaluate("new Promise(resolve => setTimeout(resolve, 200))");
  assert.equal(await evaluate("fetch('/api/chats').then(r=>r.json()).then(d=>d.chats.length)"), 1, 'New on an empty conversation created another draft');
  assert.equal(await evaluate("document.querySelector('#draft').value"), 'Keep my unsent draft');
  await send('Make a practical plan.');
  await wait("document.querySelector('.reasoning')!==null", 'streamed reasoning');
  await reasoningClickDuringDelta('Compare the options.', true);
  await reasoningClickDuringDelta('Choose a clear plan.', false);
  await pointerClick('.reasoning > summary');
  assert.equal(await evaluate("document.querySelector('.reasoning').open"), true, 'reasoning did not reopen while streaming');
  assert.equal((await fetch(upstreamFixture + '/browser-step')).status, 204);
  await idle();
  assert.equal(await evaluate("document.querySelector('.reasoning').open"), true, 'completion discarded open reasoning choice');
  await reloadDocument();
  await wait("document.querySelectorAll('.message.assistant').length===1 && document.querySelector('#stop').hidden", 'refresh and persisted stream completion');
  assert.equal(await evaluate("document.querySelector('.message.assistant').textContent.split('A practical plan.').length-1"), 1, 'snapshot replay duplicated output');
  assert.equal(await evaluate('window.injected===true'), false, 'model HTML executed');
  assert.equal(await evaluate("document.querySelectorAll('.message.assistant table').length"), 1);
  await click('[data-tab="stats"]');
  assert.equal(await evaluate("document.querySelector('.topbar').getBoundingClientRect().top>=0 && document.querySelector('.composer').getBoundingClientRect().bottom<=innerHeight"), true, 'header or composer clipped');
  assert((await evaluate("document.querySelector('#panel-stats').innerText")).includes('240 tok/s'));
  assert((await evaluate("document.querySelector('#panel-stats').innerText")).includes('32'));
  assert((await evaluate("document.querySelector('#panel-stats').innerText")).includes('510 ms'));
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
  await send('Make an edited plan.');
  await wait("document.querySelector('.message.assistant')?.innerText.includes('An edited branch.') && document.querySelector('#stop').hidden", 'edited user branch');
  await click('[data-action="settings"]');
  await click('#tool-settings input'); await click('[data-action="save-settings"]');
  await wait("document.querySelector('#toast').textContent.includes('Settings saved')", 'saved tool selection');
  await send('Use the echo tool.');
  await wait("document.querySelector('.approval')!==null", 'tool approval');
  await reloadDocument();
  await wait("document.querySelector('.approval')!==null", 'persisted approval after refresh');
  await click('[data-action="approve"]');
  await wait("document.querySelector('#stop').hidden && document.querySelector('#messages').innerText.includes('The tool returned hello.')", 'tool continuation');
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
  assert((await evaluate("document.querySelector('#panel-stats').innerText")).includes('960'), 'all-branch totals omitted retries');
  assert.equal(await evaluate("document.querySelectorAll('.attachment-image').length"), 1, 'image attachment missing');
  assert.equal(await evaluate("{const b=document.querySelector('.message.assistant [data-action=retry]'),s=getComputedStyle(b),r=b.getBoundingClientRect(); b.getAttribute('aria-label')==='Retry response' && b.querySelector('svg[aria-hidden=true]') && !b.textContent.trim() && s.backgroundColor!=='rgba(0, 0, 0, 0)' && s.borderTopWidth==='1px' && r.width>=32 && r.height>=30}"), true, 'message actions lack an accessible icon, visible surface or minimum target size');
  await evaluate("{const b=document.querySelector('.message.assistant [data-action=retry]');b.blur();b.focus({preventScroll:true})}");
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
  await screenshot('mobile-light.png');
  assert.equal(await evaluate("{const b=document.querySelector('.message.assistant [data-action=retry]'),r=b?.getBoundingClientRect(),m=document.querySelector('.message.assistant .message-metrics')?.getBoundingClientRect(),a=document.querySelector('.message.assistant .message-actions')?.getBoundingClientRect();Boolean(b?.querySelector('svg[aria-hidden=true]') && b.getAttribute('aria-label')==='Retry response' && r && r.width>=40 && r.height>=40 && r.right<=innerWidth && m && a && m.top>=r.bottom && m.right<=a.right+1)}"), true, 'mobile message action sizing or metric wrapping is wrong');
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
  const createGeometry = await evaluate(`Array.from(document.querySelectorAll('.new-chat')).map(button => {
    const label = button.querySelector('.chat-create-label').getBoundingClientRect(); const box = button.getBoundingClientRect();
    return {labelX: label.x, height: box.height, weight: getComputedStyle(button).fontWeight};
  })`);
  assert.equal(createGeometry.length, 2);
  assert.equal(createGeometry[0].labelX, createGeometry[1].labelX, 'new and temporary labels are misaligned');
  assert.equal(createGeometry[0].height, createGeometry[1].height, 'new and temporary buttons differ in height');
  assert.equal(createGeometry[0].weight, createGeometry[1].weight, 'new and temporary text weights differ');
  const sidebarTarget = await evaluate(`fetch('/api/chats',{method:'POST',headers:{'Content-Type':'application/json','X-Chat-Request':'1'},body:JSON.stringify({title:'Sidebar target'})}).then(r=>r.json()).then(c=>c.id)`);
  await reloadDocument();
  await wait("document.querySelector('.chat-row-menu')!==null && !document.querySelector('[data-action=\"new\"]').disabled", 'sidebar actions after reload');
  await click('[data-action="new"]');
  await wait("document.querySelector('.chat-item.active')!==null", 'separate selected chat');
  const selectedBeforeMenu = await evaluate("document.querySelector('.chat-item.active').dataset.chat");
  assert.notEqual(selectedBeforeMenu, sidebarTarget);
  const targetMenu = '.chat-row-menu[data-id="' + sidebarTarget + '"]';
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
  await wait("!document.querySelector('#temporary-notice').hidden", 'temporary chat mode');
  const temporaryId = await evaluate("sessionStorage.getItem('llmx-temporary-chat')");
  assert(temporaryId?.startsWith('tmp_'), 'temporary ID not in tab session');
  assert.equal(await evaluate("localStorage.getItem('llmx-chat')"), lastSavedChat, 'temporary chat replaced saved-chat preference');
  assert.equal(await evaluate("document.querySelector('#tool-settings input')===null"), true, 'temporary tools still selectable');
  await send('Temporary browser privacy check'); await idle();
  assert((await evaluate("document.querySelector('#messages').innerText")).includes('Temporary response.'));
  await reloadDocument();
  await wait("!document.querySelector('#temporary-notice').hidden && document.querySelector('#messages').innerText.includes('Temporary response.')", 'temporary reply survives refresh in memory');
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
  await click('[data-action="new"]'); await wait("document.querySelector('#action-dialog').open", 'reopened temporary chat exit confirmation');
  await closeAction('#conversation-submit');
  await wait("document.querySelector('#temporary-notice').hidden && sessionStorage.getItem('llmx-temporary-chat')===null", 'leaving discards temporary chat');
  assert.equal(await evaluate(`fetch('/api/chats/${temporaryId}').then(r=>r.status)`), 404, 'temporary transcript retained after exit');
  await wait("!document.querySelector('#temporary-button').disabled && document.querySelector('.chat-item.active')!==null", 'new saved chat finishes before switching again');
  await pointerClick('#temporary-button'); await wait("!document.querySelector('#temporary-notice').hidden", 'second temporary chat');
  const expiredId = await evaluate("sessionStorage.getItem('llmx-temporary-chat')");
  await evaluate(`fetch('/api/chats/${expiredId}',{method:'DELETE',headers:{'X-Chat-Request':'1'}})`);
  await reloadDocument();
  await wait("window.ChatShell?.config!=null && document.querySelector('#connection').textContent!=='Connecting' && sessionStorage.getItem('llmx-temporary-chat')===null && !document.querySelector('[data-action=\"new\"]').disabled", 'missing temporary chat handled after reload');
  assert.equal(await evaluate("document.querySelector('#temporary-notice').hidden"), true);
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
