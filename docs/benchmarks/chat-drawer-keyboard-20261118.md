# Companion drawer keyboard checks, 2026-11-18

The companion app owns sidebar and Details drawer focus and background inertness.
Narrow drawers are modal; docked desktop Details is not. Native dialogs and the
existing action menu retain their keyboard handling. View navigation focuses its
destination, while dismissal returns to an eligible opener. Resizing does not
leave focus on a hidden pane.

| Regression | Before fix | Final result |
|---|---|---|
| Missing entry focus and background inertness | Chrome failed in 32.035 s | Pass |
| Focus left in a hidden docked pane after resizing | Reproduced in Chrome | Pass |
| Search refresh removes the row that opened a menu | Chrome failed in 33.514 s | Pass |
| Delayed dialog close steals newer focus | Controlled ordering failed in 33.090 s | Pass |
| Rename origin remains connected but becomes hidden | Chrome failed in 32.019 s | Pass |
| Combined browser fixture | Integration failures retained | Pass in 70.118 s |

The final fixture covers native Tab and Shift+Tab traversal, Escape, backdrop
clicks, sidebar menus, Rename dialogs, destination focus, responsive transitions,
search, settings, Workspace and chat behavior. A delayed close now preserves
newer focus, and a disappeared or hidden sidebar row falls back to a visible
search field or sidebar opener. The original spontaneous Workspace navigation
timeout has a retained screenshot but no captured event sequence; the controlled
close-ordering failure is consistent with it, not proof of its sole cause.

The branch lands by fast-forward over published `4902083fe` after JavaScript
syntax, focused browser, docs and dead-code checks. It changes only frontend
assets, the browser fixture and documentation. The API, Python service and
inference source are unchanged, so no new hosted inference or numerical gate
applies. The final docs/static receipts and exact browser input hashes are in
`worktrees/chat-wording-20261118/scratch/search-drawer-landing/`. The earlier
standalone evidence is in `scratch/ui-audit-20261118/drawer-feature/`.

The Markdown reference review updated the companion README, CONTRACT and TESTS.
Completed feature status is the linked Done row in STATUS, not an active work
block. Source checks use the existing CLI because native and build inputs are
unchanged. Chrome emulation does not establish Safari or screen-reader coverage.
