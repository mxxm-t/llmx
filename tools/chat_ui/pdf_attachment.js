/* Read selected PDF pages in a dedicated browser worker. */
const maxFileBytes = 20 * 1024 * 1024;
const maxDocumentPages = 200;
const maxSelectionPages = 12;
const maxTextBytes = 2 * 1024 * 1024;
const textLimit = new Error('The selected text exceeds 2 MiB. Review fewer pages.');

function within(operation, milliseconds, signal) {
  const timeout = AbortSignal.timeout(milliseconds);
  const guard = signal ? AbortSignal.any([signal, timeout]) : timeout;
  let interrupted;
  return new Promise((resolve, reject) => {
    interrupted = () => reject(guard.reason);
    if (guard.aborted) { interrupted(); return; }
    guard.addEventListener('abort', interrupted, {once: true});
    Promise.resolve().then(() => operation(guard)).then(resolve, reject);
  }).finally(() => guard.removeEventListener('abort', interrupted));
}

function parsePages(value, total) {
  const pages = new Set();
  for (const part of value.split(',')) {
    const match = part.trim().match(/^(\d{1,3})(?:\s*-\s*(\d{1,3}))?$/);
    if (!match) throw new Error('Enter page numbers or ranges separated by commas, such as 1, 3-5.');
    const first = Number(match[1]), last = Number(match[2] || match[1]);
    if (first < 1 || last < first || last > total) throw new Error('Choose pages from 1 to ' + total + ' in ascending ranges.');
    for (let page = first; page <= last; page++) {
      pages.add(page);
      if (pages.size > maxSelectionPages) throw new Error('Choose up to ' + maxSelectionPages + ' pages per attachment.');
    }
  }
  return [...pages].sort((a, b) => a - b);
}

async function readPages(pdf, pages, signal) {
  const labels = await pdf.getPageLabels();
  const encoder = new TextEncoder(), sections = [], empty = [];
  let remaining = maxTextBytes;
  for (const number of pages) {
    signal.throwIfAborted();
    const page = await pdf.getPage(number);
    signal.throwIfAborted();
    const reader = page.streamTextContent().getReader();
    let text = '', bytes = 0, done = false;
    try {
      while (!done) {
        signal.throwIfAborted();
        const chunk = await reader.read();
        done = chunk.done;
        if (done) break;
        for (const item of chunk.value.items) {
          if (typeof item.str !== 'string') continue;
          const part = item.str.replaceAll('\u0000', '') + (item.hasEOL ? '\n' : '');
          if (bytes + part.length > remaining) throw textLimit;
          bytes += encoder.encode(part).length;
          if (bytes > remaining) throw textLimit;
          text += part;
        }
      }
    } finally {
      if (!done) reader.cancel().catch(() => {});
      reader.releaseLock();
    }
    text = text.trim();
    if (!text) { empty.push(number); continue; }
    const label = labels?.[number - 1]?.replace(/[\u0000-\u001f\u007f]/g, ' ').trim();
    const heading = 'Page ' + number + (label && label !== String(number) ? ' (label: ' + label + ')' : '');
    const section = (sections.length ? '\n\n' : '') + heading + '\n' + text;
    if (section.length > remaining) throw textLimit;
    remaining -= encoder.encode(section).length;
    if (remaining < 0) throw textLimit;
    sections.push(section);
  }
  return {text: sections.join(''), empty};
}

export async function selectPdfText(file, {signal} = {}) {
  if (signal?.aborted) return null;
  if (!file.size) throw new Error('The selected PDF is empty.');
  if (file.size > maxFileBytes) throw new Error('Choose a PDF of at most 20 MiB.');

  const {node} = window.ChatRender;
  const dialog = node('dialog'); dialog.id = 'pdf-attachment-dialog';
  const form = node('form', 'dialog-content');
  const title = node('h2', '', 'Select PDF pages'); title.id = 'pdf-attachment-title';
  dialog.setAttribute('aria-labelledby', title.id);
  const name = node('p', '', file.name); name.style.overflowWrap = 'anywhere';
  const status = node('p', '', 'Opening PDF...'); status.id = 'pdf-attachment-status'; status.setAttribute('role', 'status');
  const range = node('input'); range.id = 'pdf-attachment-pages'; range.value = '1'; range.maxLength = 120; range.autocomplete = 'off'; range.disabled = true;
  const rangeField = node('label', 'field', 'Page numbers or ranges (e.g. 1, 3-5)'); rangeField.append(range);
  const preview = node('textarea'); preview.id = 'pdf-attachment-preview'; preview.rows = 7; preview.readOnly = true;
  const previewField = node('label', 'field', 'Text to attach'); previewField.append(preview); previewField.hidden = true;
  const error = node('p', 'error-text'); error.id = 'pdf-attachment-error'; error.setAttribute('role', 'alert'); error.hidden = true;
  const review = node('button', 'secondary-button', 'Review selected pages'); review.id = 'pdf-attachment-review'; review.type = 'submit'; review.disabled = true;
  const cancel = node('button', 'secondary-button', 'Cancel'); cancel.id = 'pdf-attachment-cancel'; cancel.type = 'button';
  const apply = node('button', 'primary-button', 'Add text to draft'); apply.id = 'pdf-attachment-apply'; apply.type = 'button'; apply.disabled = true;
  const actions = node('div', 'conversation-form-actions'); actions.append(cancel, apply);
  form.append(title, name, status, rangeField, review, previewField, error, actions); dialog.append(form);

  const session = new AbortController();
  let finish;
  const selection = new Promise(resolve => { finish = resolve; });
  const cancelSelection = () => { session.abort(); finish(null); };
  signal?.addEventListener('abort', cancelSelection, {once: true});
  cancel.addEventListener('click', cancelSelection);
  dialog.addEventListener('cancel', event => { event.preventDefault(); cancelSelection(); });
  dialog.addEventListener('close', cancelSelection);

  let pdf, task, pdfWorker, worker, reviewedText = '';
  async function dispose() {
    const loading = task, adapter = pdfWorker, port = worker;
    task = pdfWorker = worker = null;
    try { if (loading) await within(() => loading.destroy(), 250); }
    catch (_) { /* Terminate the worker even if it does not acknowledge teardown. */ }
    finally { adapter?.destroy(); port?.terminate(); }
  }
  function showError(message) { error.textContent = message; error.hidden = false; }
  function clearReview() {
    reviewedText = ''; apply.disabled = true; preview.value = ''; previewField.hidden = true; error.hidden = true;
  }
  function fatal(message) {
    status.textContent = ''; range.disabled = true; review.disabled = true;
    cancel.textContent = 'Close'; showError(message); cancel.focus(); void dispose();
  }
  function ready() {
    status.textContent = pdf.numPages + (pdf.numPages === 1 ? ' page' : ' pages') + ' / Review up to ' + maxSelectionPages + ' pages.';
  }
  range.addEventListener('input', () => { clearReview(); ready(); });
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (review.disabled || session.signal.aborted) return;
    clearReview(); ready();
    let pages;
    try { pages = parsePages(range.value, pdf.numPages); }
    catch (problem) { showError(problem.message); range.focus(); return; }
    range.disabled = true; review.disabled = true; status.textContent = 'Reading selected pages...';
    try {
      const result = await within(guard => readPages(pdf, pages, guard), 20000, session.signal);
      range.disabled = false; review.disabled = false;
      const unavailable = 'Pages without extractable text may require OCR.';
      if (!result.text) {
        status.textContent = ''; showError('No extractable text on the selected pages. ' + unavailable);
        if (document.activeElement === document.body) range.focus();
        return;
      }
      status.textContent = result.empty.length
        ? 'Skipped pages without text: ' + result.empty.join(', ') + '. ' + unavailable
        : 'Review the text below before adding it to the draft.';
      preview.value = result.text; reviewedText = preview.value; previewField.hidden = false; apply.disabled = false;
      if (document.activeElement === document.body) preview.focus();
    } catch (problem) {
      if (session.signal.aborted) return;
      if (problem === textLimit) {
        ready(); showError(problem.message); range.disabled = false; review.disabled = false;
        if (document.activeElement === document.body) range.focus();
        return;
      }
      fatal(problem.name === 'TimeoutError' ? 'Reading the selected PDF pages timed out.' : 'Could not read the selected PDF pages.');
    }
  });
  apply.addEventListener('click', () => {
    if (!reviewedText || apply.disabled || session.signal.aborted) return;
    const filename = (file.name.replace(/\.pdf$/i, '') || 'document') + '.txt';
    finish(new File([reviewedText], filename, {type: 'text/plain'}));
  });

  document.body.append(dialog);
  try {
    dialog.showModal(); cancel.focus();
    try {
      pdf = await within(async guard => {
        const [pdfjs, bytes] = await Promise.all([
          import('/pdf-reader/pdf.mjs').catch(() => { throw new Error('The optional PDF reader is unavailable.'); }),
          file.arrayBuffer()
        ]);
        guard.throwIfAborted();
        worker = new Worker('/pdf-reader/pdf.worker.mjs', {type: 'module', name: 'PDF text reader'});
        pdfWorker = new pdfjs.PDFWorker({port: worker});
        task = pdfjs.getDocument({
          data: new Uint8Array(bytes), worker: pdfWorker, cMapUrl: '/pdf-reader/cmaps/', cMapPacked: true,
          standardFontDataUrl: '/pdf-reader/standard_fonts/', useWorkerFetch: true, useWasm: false, disableFontFace: true,
          useSystemFonts: false, disableWorkerRendering: true, isEvalSupported: false, enableXfa: false,
          stopAtErrors: true
        });
        return task.promise;
      }, 15000, session.signal);
      if (pdf.numPages > maxDocumentPages) fatal('Choose a PDF with at most ' + maxDocumentPages + ' pages.');
      else {
        ready(); range.disabled = false; review.disabled = false;
        if (document.activeElement === cancel) range.focus();
      }
    } catch (problem) {
      if (!session.signal.aborted) {
        if (problem.name === 'PasswordException') fatal('This PDF requires a password. Open an unlocked copy to attach its text.');
        else if (problem.name === 'InvalidPDFException') fatal('This file is not a readable PDF.');
        else if (problem.name === 'TimeoutError') fatal('Opening the PDF timed out.');
        else fatal(problem.message === 'The optional PDF reader is unavailable.' ? problem.message : 'Could not open the PDF.');
      }
    }
    return await selection;
  } finally {
    session.abort(); signal?.removeEventListener('abort', cancelSelection);
    dialog.close(); dialog.remove();
    // The next selection owns a fresh worker and can open while this one exits.
    void dispose();
  }
}
