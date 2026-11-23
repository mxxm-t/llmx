"""Print one section of CHANGELOG.md, the notes of a release (docs/RELEASE.md, The changelog).

The release workflows call it, and the docs check reads the changelog through the same parser.
"""
import os
import re
import sys

CHANGELOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CHANGELOG.md")
UNRELEASED = "Unreleased"
REPO = "https://github.com/mxxm-t/llmx/blob/"
RELATIVE_LINK = re.compile(r"\]\((?![a-z][a-z0-9+.-]*:|#|/)([^)\s]+)\)")
HEADING = re.compile(r"## \[([^\]]+)\](?: - (\d{4}-\d\d-\d\d))?$")


def sections(text):
    """[(name, date, line number, body lines)] of every `## ` heading in file order; a heading not of the form `## [NAME]` or `## [NAME] - YYYY-MM-DD` has the name None."""
    out = []
    for n, line in enumerate(text.split("\n"), 1):
        if line.startswith("## "):
            m = HEADING.match(line.rstrip())
            out.append((m.group(1) if m else None, m.group(2) if m else None, n, []))
        elif out:
            out[-1][3].append(line)
    return out


def notes(text, name):
    """The body of section `name` without its surrounding blank lines, for an Unreleased section without entries a line naming the last release, or None when there is no such section or a version's lists no `- ` entry."""
    found = sections(text)
    for i, (section, _, _, lines) in enumerate(found):
        if section == name:
            body = "\n".join(lines).strip()
            if any(line.startswith("- ") for line in lines):
                return body
            if name != UNRELEASED:
                return None
            last = next((s for s, _, _, _ in found[i + 1:] if s), None)
            return "No changes are listed since %s." % last if last else "No changes are listed yet."
    return None


def absolute_links(body, ref):
    """`body` with each relative link target made a GitHub blob URL at `ref`, since a release page resolves a relative link under /releases/tag/."""
    return RELATIVE_LINK.sub(lambda m: "](%s%s/%s)" % (REPO, ref, m.group(1)), body)


def main(argv):
    if len(argv) == 2 and argv[1] in ("-h", "--help"):
        print("usage: release_notes.py VERSION|%s [CHANGELOG.md]" % UNRELEASED)
        return 0
    if len(argv) not in (2, 3):
        print("usage: release_notes.py VERSION|%s [CHANGELOG.md]" % UNRELEASED, file=sys.stderr)
        return 2
    path = argv[2] if len(argv) == 3 else CHANGELOG
    with open(path, encoding="utf-8") as f:
        body = notes(f.read(), argv[1])
    if body is None:
        print("%s has no section ## [%s] with notes; a release needs its changes listed (docs/RELEASE.md, The changelog)" % (path, argv[1]), file=sys.stderr)
        return 1
    print(absolute_links(body, "main" if argv[1] == UNRELEASED else "v" + argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
