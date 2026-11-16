/* Safe presentation helpers: untrusted model text only enters text nodes. */
window.ChatRender = (() => {
  'use strict';
  function node(tag, cls, text) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined) el.textContent = String(text);
    return el;
  }
  function inline(text, target) {
    const re = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\*[^*\n]+\*|\[[^\]\n]+\]\([^\s)]+\))/g;
    let pos = 0;
    for (const match of text.matchAll(re)) {
      target.append(document.createTextNode(text.slice(pos, match.index)));
      const token = match[0];
      if (token[0] === '`') target.append(node('code', '', token.slice(1, -1)));
      else if (token.startsWith('**')) target.append(node('strong', '', token.slice(2, -2)));
      else if (token[0] === '*') target.append(node('em', '', token.slice(1, -1)));
      else {
        const parts = token.match(/^\[([^\]]+)\]\((.*)\)$/);
        const a = node('a', '', parts[1]);
        try {
          const url = new URL(parts[2], location.href);
          if (['http:', 'https:', 'mailto:'].includes(url.protocol)) { a.href = url.href; a.target = '_blank'; a.rel = 'noopener noreferrer'; }
        } catch (_) { /* Invalid links remain plain text. */ }
        target.append(a);
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
  function markdown(text, depth = 0) {
    const out = node('div', 'markdown');
    const lines = String(text || '').replace(/\r/g, '').split('\n');
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (/^\s*```/.test(line)) {
        const language = line.replace(/^\s*```/, '').trim();
        const code = [];
        i++;
        while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
        if (i < lines.length) i++;
        const wrap = node('div', 'code-block');
        const head = node('div', 'code-heading');
        const copy = node('button', 'text-button', 'Copy code');
        copy.type = 'button';
        copy.addEventListener('click', async () => {
          try { await navigator.clipboard.writeText(code.join('\n')); copy.textContent = 'Copied'; setTimeout(() => { copy.textContent = 'Copy code'; }, 1600); }
          catch (_) { copy.textContent = 'Copy unavailable'; }
        });
        const actions = node('div', 'code-actions'); actions.append(copy);
        if (window.ChatArtifacts) {
          const download = node('button', 'text-button', 'Download'); download.type = 'button'; download.addEventListener('click', () => window.ChatArtifacts.download(code.join('\n'), null, language));
          const open = node('button', 'text-button', 'Open'); open.type = 'button'; open.addEventListener('click', () => window.ChatArtifacts.open({source: code.join('\n'), language})); actions.append(download, open);
        }
        head.append(node('span', '', language || 'code'), actions);
        const pre = node('pre'); const codeNode = node('code'); if (window.ChatArtifacts) codeNode.append(window.ChatArtifacts.highlight(code.join('\n'), language)); else codeNode.textContent = code.join('\n'); pre.append(codeNode);
        wrap.append(head, pre); out.append(wrap); continue;
      }
      if (!line.trim()) { i++; continue; }
      if (i + 1 < lines.length && line.includes('|') && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
        const table = node('table'); const scroll = node('div', 'table-scroll');
        const cells = row => row.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
        const thead = node('thead'); const tr = node('tr');
        cells(line).forEach(c => { const th = node('th'); inline(c, th); tr.append(th); });
        thead.append(tr); table.append(thead); i += 2;
        const body = node('tbody');
        while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
          const row = node('tr'); cells(lines[i++]).forEach(c => { const td = node('td'); inline(c, td); row.append(td); }); body.append(row);
        }
        table.append(body); scroll.append(table); out.append(scroll); continue;
      }
      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) { const h = node('h' + Math.min(heading[1].length + 1, 6)); inline(heading[2], h); out.append(h); i++; continue; }
      if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) { out.append(node('hr')); i++; continue; }
      if (/^\s*>/.test(line)) {
        const b = node('blockquote'); while (i < lines.length && /^\s*>/.test(lines[i])) { inline(lines[i++].replace(/^\s*>\s?/, ''), b); b.append(node('br')); } out.append(b); continue;
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
            if (!blank && !listItem(next) && !/^\s*(?:```|#|>|([-*_])(?:\s*\1){2,}\s*$)/.test(next)) {
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
        out.append(list); continue;
      }
      const p = node('p'); inline(line, p); i++;
      while (i < lines.length && lines[i].trim() && !/^\s*(?:```|#|>|[-+*]\s|\d+[.)]\s)/.test(lines[i]) && !(i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1]))) { p.append(node('br')); inline(lines[i++], p); }
      out.append(p);
    }
    return out;
  }
  function number(value, suffix = '') { return typeof value === 'number' && Number.isFinite(value) ? new Intl.NumberFormat(undefined, {maximumFractionDigits: 2}).format(value) + suffix : 'Unavailable'; }
  function json(value) { return typeof value === 'string' ? value : JSON.stringify(value, null, 2); }
  return {node, markdown, number, json};
})();
