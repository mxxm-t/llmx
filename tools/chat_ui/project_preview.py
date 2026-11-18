"""Embed saved CSS and classic JavaScript into an isolated HTML preview."""

from base64 import b64encode
from html import escape
from html.parser import HTMLParser
import posixpath
from urllib.parse import unquote, urlsplit

from workspace import file_path


MAX_PREVIEW_BYTES = 1024 * 1024
CLASSIC_TYPES = ("", "text/javascript", "application/javascript")
RAW_TEXT = ("iframe", "noembed", "noframes", "noscript", "script", "style", "textarea", "title", "xmp")


class ProjectPreview(HTMLParser):
    def __init__(self, path, html, files):
        super().__init__(convert_charrefs=False)
        self.path, self.html, self.files = path, html, files
        self.assets = {}
        self.parts = []
        self.end = self.line_start = self.template_depth = 0
        self.line = 1
        self.raw_text = None
        self.size = len(html.encode())

    def handle_starttag(self, tag, attrs):
        if self.raw_text:
            return
        if tag in RAW_TEXT:
            self.raw_text = tag
        if tag == "template":
            self.template_depth += 1
        if self.template_depth:
            return
        values = dict(reversed(attrs))
        if tag == "link" and "stylesheet" in (values.get("rel") or "").lower().split():
            self.embed(tag, attrs, "href", values.get("href"), ".css", "text/css")
        if tag == "script" and (values.get("type") or "").strip().lower() in CLASSIC_TYPES:
            self.embed(tag, attrs, "src", values.get("src"), ".js", "text/javascript")

    handle_startendtag = handle_starttag

    def handle_endtag(self, tag):
        if self.raw_text == tag:
            self.raw_text = None
        elif not self.raw_text and tag == "template" and self.template_depth:
            self.template_depth -= 1

    def embed(self, tag, attrs, key, reference, suffix, mime):
        if not reference:
            return
        parsed = urlsplit(reference.strip())
        if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith("/"):
            return
        path = posixpath.normpath(posixpath.join(posixpath.dirname(self.path), unquote(parsed.path, errors="strict")))
        path = file_path(path)
        if not path.lower().endswith(suffix) or path not in self.files:
            raise ValueError(f"Workspace preview needs a saved {suffix} asset at {path}")
        original = self.get_starttag_text()
        budget = MAX_PREVIEW_BYTES - self.size + len(original.encode())
        prefix = f"data:{mime};charset=utf-8;base64,"
        if path not in self.assets:
            content = self.files[path].encode()
            if len(prefix) + 4 * ((len(content) + 2) // 3) > budget:
                raise ValueError("The assembled Workspace preview must be at most 1 MiB")
            self.assets[path] = prefix + b64encode(content).decode("ascii")
        asset = self.assets[path]
        if len(asset) > budget:
            raise ValueError("The assembled Workspace preview must be at most 1 MiB")
        attributes = []
        replaced = False
        for name, value in attrs:
            if name == key:
                if replaced:
                    continue
                value, replaced = asset, True
            attributes.append(name if value is None else name + '="' + escape(value, quote=True) + '"')
        replacement = f"<{tag} {' '.join(attributes)}>"
        self.size += len(replacement.encode()) - len(original.encode())
        if self.size > MAX_PREVIEW_BYTES:
            raise ValueError("The assembled Workspace preview must be at most 1 MiB")
        line, column = self.getpos()
        while self.line < line:
            self.line_start = self.html.index("\n", self.line_start) + 1
            self.line += 1
        start = self.line_start + column
        self.parts.extend((self.html[self.end:start], replacement))
        self.end = start + len(original)


def assemble(path, html, files):
    path = file_path(path)
    if not path.lower().endswith((".html", ".htm")):
        raise ValueError("Choose a Workspace HTML file to preview")
    preview = ProjectPreview(path, html, files)
    preview.feed(html)
    preview.close()
    preview.parts.append(html[preview.end:])
    return "".join(preview.parts).encode(), list(preview.assets)
