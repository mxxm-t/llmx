/* Safe presentation helpers: untrusted model text only enters text nodes. */
window.ChatRender = (() => {
  'use strict';
  const blockSources = new WeakMap(), markdownSources = new WeakMap();
  function node(tag, cls, text) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined) el.textContent = String(text);
    return el;
  }
  function icon(path, cls) {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg'), shape = document.createElementNS(ns, 'path');
    svg.setAttribute('viewBox', '0 0 24 24'); svg.setAttribute('aria-hidden', 'true');
    if (cls) svg.classList.add(cls);
    shape.setAttribute('d', path); svg.append(shape);
    return svg;
  }
  function link(label, destination, base, allowMail = false) {
    if (typeof destination !== 'string' || !destination) return null;
    try {
      const url = new URL(destination, base);
      if (!['http:', 'https:'].includes(url.protocol) && !(allowMail && url.protocol === 'mailto:')) return null;
      const anchor = node('a', '', label);
      anchor.href = url.href; anchor.target = '_blank'; anchor.rel = 'noopener noreferrer';
      return anchor;
    } catch { return null; }
  }
  function inline(text, target) {
    const re = /`[^`\n]+`|\*\*[^*\n]+\*\*|\*[^*\n]+\*|(!?)\[([^\]\n]*)\]\(((?:[^\s()]|\([^\s()]*\))+)\)/g;
    let pos = 0;
    for (const match of text.matchAll(re)) {
      target.append(document.createTextNode(text.slice(pos, match.index)));
      const token = match[0];
      if (token[0] === '`') target.append(node('code', '', token.slice(1, -1)));
      else if (token.startsWith('**')) target.append(node('strong', '', token.slice(2, -2)));
      else if (token[0] === '*') target.append(node('em', '', token.slice(1, -1)));
      else {
        const [, image, label, destination] = match;
        const caption = image ? label.trim() || 'Open image' : label || token;
        const anchor = (image || label) && link(caption, destination, location.href, !image);
        target.append(anchor || document.createTextNode(caption));
      }
      pos = match.index + token.length;
    }
    target.append(document.createTextNode(text.slice(pos)));
  }
  function columns(text) {
    let width = 0;
    for (const char of text) width += char === '\t' ? 4 - width % 4 : 1;
    return width;
  }
  function listItem(line) {
    const match = line?.match(/^([ \t]*)([-+*]|\d{1,9}[.)])([ \t]+)(.*)$/);
    return match && {indent: columns(match[1]), contentIndent: columns(match[1] + match[2] + match[3]),
      ordered: /^\d/.test(match[2]), start: parseInt(match[2], 10), text: match[4]};
  }
  function removeIndent(line, count) {
    let width = 0, offset = 0;
    while (offset < line.length && width < count && /[ \t]/.test(line[offset])) {
      width += line[offset++] === '\t' ? 4 - width % 4 : 1;
    }
    return ' '.repeat(Math.max(0, width - count)) + line.slice(offset);
  }
  function openingFence(line) {
    const match = /^( {0,3})(`{3,}|~{3,})(.*)$/.exec(line);
    if (!match || (match[2][0] === '`' && match[3].includes('`'))) return null;
    return {indent: match[1].length, marker: match[2], language: match[3].trim().split(/\s+/, 1)[0]};
  }
  function closesFence(line, marker) {
    const match = /^ {0,3}(`{3,}|~{3,})[ \t]*$/.exec(line);
    return match && match[1][0] === marker[0] && match[1].length >= marker.length;
  }
  function codeBlock(source, language) {
    const wrap = node('div', 'code-block');
    const label = node('span', '', language || 'code'); label.dataset.language = language;
    const codeNode = node('code'); if (window.ChatArtifacts) codeNode.append(window.ChatArtifacts.highlight(source, language)); else codeNode.textContent = source;
    const head = node('div', 'code-heading');
    const copy = node('button', 'text-button', 'Copy code');
    copy.type = 'button';
    copy.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(codeNode.textContent); copy.textContent = 'Copied'; setTimeout(() => { copy.textContent = 'Copy code'; }, 1600); }
      catch (_) { copy.textContent = 'Copy unavailable'; }
    });
    const actions = node('div', 'code-actions'); actions.append(copy);
    if (window.ChatArtifacts) {
      const download = node('button', 'text-button', 'Download'); download.type = 'button'; download.addEventListener('click', () => window.ChatArtifacts.download(codeNode.textContent, null, label.dataset.language));
      const open = node('button', 'text-button', 'Open'); open.type = 'button'; open.addEventListener('click', () => window.ChatArtifacts.open({source: codeNode.textContent, language: label.dataset.language})); actions.append(download, open);
    }
    head.append(label, actions);
    const pre = node('pre'); pre.append(codeNode);
    wrap.append(head, pre); return wrap;
  }
  function appendBlock(target, element, start, end) {
    blockSources.set(element, {start, end}); target.append(element);
  }
  function tableCells(line) {
    const cells = [''];
    let separated = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '\\' && line[i + 1] === '|') cells[cells.length - 1] += line[++i];
      else if (ch === '\\' && line[i + 1] === '\\') cells[cells.length - 1] += ch + line[++i];
      else if (ch === '|') { cells.push(''); separated = true; }
      else cells[cells.length - 1] += ch;
    }
    if (separated && !cells[0].trim()) cells.shift();
    if (separated && cells.length && !cells[cells.length - 1].trim()) cells.pop();
    return cells.map(cell => cell.trim());
  }
  function tableColumns(lines, index) {
    const divider = lines[index + 1];
    if (!divider || !lines[index].includes('|') || !/^\s*\|?\s*:?-{3,}/.test(divider)) return null;
    const columns = tableCells(divider);
    if (!columns.length || !columns.every(cell => /^:?-{3,}:?$/.test(cell)) ||
        tableCells(lines[index]).length !== columns.length) return null;
    return columns.map(cell => cell.endsWith(':') ? (cell.startsWith(':') ? 'center' : 'right') : 'left');
  }
  function tableRow(line, alignments, heading) {
    const row = node('tr'), cells = tableCells(line);
    const width = Math.min(cells.length, alignments.length);
    for (let column = 0; column < width; column++) {
      const cell = node(heading ? 'th' : 'td');
      if (heading) cell.scope = 'col';
      if (alignments[column] !== 'left') cell.style.textAlign = alignments[column];
      inline(cells[column], cell);
      row.append(cell);
    }
    if (width < alignments.length) {
      const empty = node('td');
      empty.colSpan = alignments.length - width;
      row.append(empty);
    }
    return row;
  }
  function sourceLines(text) { return String(text || '').replace(/\r/g, '').split('\n'); }
  function markdown(text, depth = 0) {
    const out = node('div', 'markdown');
    const lines = sourceLines(text);
    markdownSources.set(out, lines);
    renderBlocks(out, lines, depth, 0);
    return out;
  }
  function renderBlocks(out, lines, depth, i) {
    while (i < lines.length) {
      const start = i;
      const line = lines[i];
      const fence = openingFence(line);
      if (fence) {
        const code = [];
        i++;
        while (i < lines.length && !closesFence(lines[i], fence.marker)) code.push(removeIndent(lines[i++], fence.indent));
        if (i < lines.length) i++;
        const wrap = codeBlock(code.join('\n'), fence.language); appendBlock(out, wrap, start, i); continue;
      }
      if (!line.trim()) { i++; continue; }
      const alignments = tableColumns(lines, i);
      if (alignments) {
        const table = node('table'); const scroll = node('div', 'table-scroll');
        const thead = node('thead');
        thead.append(tableRow(line, alignments, true)); table.append(thead); i += 2;
        const body = node('tbody');
        while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
          body.append(tableRow(lines[i++], alignments, false));
        }
        table.append(body); scroll.append(table); appendBlock(out, scroll, start, i); continue;
      }
      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) { const h = node('h' + Math.min(heading[1].length + 1, 6)); inline(heading[2], h); appendBlock(out, h, start, ++i); continue; }
      if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) { appendBlock(out, node('hr'), start, ++i); continue; }
      if (/^\s*>/.test(line)) {
        const b = node('blockquote'); while (i < lines.length && /^\s*>/.test(lines[i])) { inline(lines[i++].replace(/^\s*>\s?/, ''), b); b.append(node('br')); } appendBlock(out, b, start, i); continue;
      }
      const firstItem = listItem(line);
      if (firstItem && depth < 8) {
        const list = node(firstItem.ordered ? 'ol' : 'ul');
        if (firstItem.ordered) list.start = firstItem.start;
        let item;
        while ((item = listItem(lines[i])) && item.indent === firstItem.indent && item.ordered === firstItem.ordered) {
          const body = [item.text]; i++;
          let blank = false;
          while (i < lines.length) {
            const next = lines[i];
            if (!next.trim()) { body.push(''); blank = true; i++; continue; }
            if (columns(next.match(/^[ \t]*/)[0]) >= item.contentIndent) {
              body.push(removeIndent(next, item.contentIndent)); blank = false; i++; continue;
            }
            if (!blank && !listItem(next) && !openingFence(next) && !/^\s*(?:#|>|([-*_])(?:\s*\1){2,}\s*$)/.test(next)) {
              body.push(next.trimStart()); i++; continue;
            }
            break;
          }
          const li = node('li'); const rendered = markdown(body.join('\n'), depth + 1);
          if (rendered.firstElementChild?.tagName === 'P') {
            const paragraph = rendered.firstElementChild;
            while (paragraph.firstChild) li.append(paragraph.firstChild);
            paragraph.remove();
          }
          while (rendered.firstChild) li.append(rendered.firstChild);
          list.append(li);
        }
        appendBlock(out, list, start, i); continue;
      }
      const p = node('p'); inline(line, p); i++;
      while (i < lines.length && lines[i].trim() && !openingFence(lines[i]) && !/^\s*(?:#|>|[-+*]\s|\d+[.)]\s)/.test(lines[i]) && !tableColumns(lines, i)) { p.append(node('br')); inline(lines[i++], p); }
      appendBlock(out, p, start, i);
    }
  }
  function updateMarkdown(current, text) {
    const before = markdownSources.get(current), after = sourceLines(text);
    let common = 0;
    while (common < before.length && common < after.length && before[common] === after[common]) common++;
    if (common === before.length && common === after.length) return;
    const previous = Array.from(current.children);
    let keep = 0, restart = 0;
    for (let i = 0; i < previous.length; i++) {
      const source = blockSources.get(previous[i]);
      if (!source || source.end >= common) break;
      // An unchanged blank after a block separates it from later edits. Lists include the blank lines they consume.
      if (!after[source.end].trim()) { keep = i + 1; restart = source.end; }
    }
    const tail = document.createDocumentFragment();
    renderBlocks(tail, after, 0, restart);
    const next = Array.from(tail.children);
    let parsed = 0;
    while (keep < previous.length && parsed < next.length) {
      const source = blockSources.get(previous[keep]), target = blockSources.get(next[parsed]);
      if (!source || !target || previous[keep].tagName !== next[parsed].tagName || source.end - source.start !== target.end - target.start) break;
      let equal = true;
      for (let i = 0; i < source.end - source.start; i++) if (before[source.start + i] !== after[target.start + i]) { equal = false; break; }
      if (!equal) break;
      blockSources.set(previous[keep], target); keep++; parsed++;
    }
    for (let i = previous.length - 1; i >= keep; i--) previous[i].remove();
    for (let i = parsed; i < next.length; i++) current.append(next[i]);
    markdownSources.set(current, after);
  }
  function number(value, suffix = '') { return typeof value === 'number' && Number.isFinite(value) ? new Intl.NumberFormat(undefined, {maximumFractionDigits: 2}).format(value) + suffix : 'Unavailable'; }
  function json(value) { return typeof value === 'string' ? value : JSON.stringify(value, null, 2); }
  function connectionLabel(name) { return name.charAt(0).toUpperCase() + name.slice(1); }
  return {node, icon, link, markdown, updateMarkdown, number, json, connectionLabel};
})();
