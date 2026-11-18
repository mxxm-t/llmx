# Companion navigation clarity, 2026-11-18

- **Goal:** Distinguish the selected main section from a hovered section and remove decorative shortcut-like numbering.
- **Done:** Hover uses a softer fill in the existing navigation styles. The selected section retains its fill and aria-current, keyboard focus retains its outline, and decorative 01-04 labels and their unused style are removed.
- **Checks:** Chrome reproduced equal active/hover fills in 5.703 seconds before the fix. The combined browser fixture passed in 68.038 seconds after it, covering both themes, mobile navigation and zoom. JavaScript syntax, dead-code (17 expected findings, 20 planted faults) and docs (zero findings, 29 planted faults) passed. Independent source review found no issue. Receipts and screenshots are in `scratch/navigation-clarity/` and `scratch/chat-ui-browser/` in the feature worktree.
- **Landing:** Fast-forward over 2fde5ab75 after the final documentation check. Only static presentation, its browser regression and documentation changed. The Markdown review updated companion README and TESTS and found no other references to decorative navigation numbers. No inference, Python API or unrelated hosted gate applies.
- **Gotchas:** Hover color is a visual distinction, not a new navigation state. Background chat status and workspace links remain separate features.
