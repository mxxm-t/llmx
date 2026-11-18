# Companion status and refresh focus, 2026-11-18

## Saved conversation status

Saved chat rows expose the latest reply attempt as in progress, awaiting tool
approval, failed or interrupted. Archive and search rows use the same status.
The cue refers to the latest attempt even if an older branch is selected.
Temporary chats, prompt text, tool arguments and error details are excluded.

The Store owns a derived projection of committed run status. It rebuilds the
projection from SQLite on startup and restore and updates it only after a
successful start, state or delete transaction, while retaining the writer lock.
Polling copies the small projection without reading run payloads or taking the
search reader gate. SQLite remains the persistent authority. The engine clears
expired approval state when no other decision remains.

The browser polls only while signed in and viewing visible Chat. Each request
owns a nine-second deadline and abort controller. Navigation, hiding and logout
cancel it. Request epochs reject old responses, snapshots expire, and only cue
contents change so menus and focus remain connected. A static partial ring
avoids a second ellipsis beside the existing menu button.

| Disposable 500-chat plus one live-run fixture | Observed |
|---|---:|
| Latest run JSON | 50.08 MB |
| Paired indexed WAL read per poll, median | 82.209 ms |
| Product projection read, median / maximum | 0.078 / 0.105 ms |
| Concurrent delta persistence, median / maximum | 1.157 / 2.156 ms |
| Warm projection rebuild, median / maximum | 72.047 / 81.933 ms |

The writer was scheduled one millisecond after poll start. System and probe CPU
were recorded, while disk, GPU and individual background processes were not.
These are companion measurements, not inference throughput claims. Raw samples,
query plans and exact Store identity are retained in
`scratch/chat-background-status-20261118/` on the development workstation.

## Message action focus

A real terminal response followed by its persisted snapshot detached a focused
Retry button, moved focus to the page body and removed its tooltip. The same
problem can occur when newly reported metrics change the rendered article.
The existing reconciliation owner now retains a focused action whose action,
target and branch direction still exist, while replacing other controls and
updating metrics. Previous and Next remain distinct with two sibling branches,
where both otherwise share the same target.

| Regression evidence | Result |
|---|---|
| Terminal snapshot before fix | Failed in 28.565 s |
| Two-branch direction collision in first candidate | Failed in 23.416 s |
| Final focus browser fixture | Passed in 72.666 s |
| Final standalone status browser fixture | Passed in 89.832 s |

Coverage includes genuine changing metrics and an equivalent terminal snapshot,
focused action/tooltip identity, branch labels, glyphs, order and targets, plus
reopened statistics showing the newly reported total. Evidence is in
`scratch/ui-audit-20261118/tooltip-race/`. These reproductions prove the focus
problem; they do not establish the cause of every hosted browser timeout.

## Browser failure evidence

Hosted main runs 44313960618 and 44316095627 failed protocol evaluation or keyboard
tooltip focus. Their original logs remain recorded. A diagnostic branch at
614a5c6e5 passed both Windows and Linux in hosted run 44316986784; that later pass
does not explain or erase the earlier failures.

Protocol timeouts now retain the caller stack. Failed waits report focus, view,
visibility and modal state. The hosted workflow retains browser fixture artifacts
for seven days on failure. The green diagnostic run skipped that upload step,
so no exercised upload is claimed from that run.

The outer subprocess limit grows from 100 to 300 seconds for the enlarged
fixture, whose local unittest elapsed time is about 90 seconds. That elapsed time
includes cleanup and is not an exact measure of subprocess headroom. Per-operation
CDP and assertion limits remain 12 seconds and Chrome startup remains 30 seconds.
No assertion, success condition or automatic retry changes.

## Landing scope

The independent changes are stacked over published 9add55959. Store, approval,
route/authentication, rollback and browser checks passed on the feature branches.
The landing head runs the combined browser, syntax, docs/dead-code checks and the
dedicated Windows/Linux Chat workspace workflow before fast-forwarding main.
The source-tested revision and any documentation-only landing delta are recorded
with the qualification receipts. Relevant source identities
and raw results stay beside the development receipts; the final check must use
the integrated source, not substitute a feature-branch pass.

No native runtime, inference, model, kernel, health collector or device source
changes. Numerical and GPU performance gates do not apply. The Markdown impact
review checks references across tracked documentation and updates the companion
README, CONTRACT, TESTS and STATUS, which own the changed behavior. The UI APIs
remain under the companion Store, transport and rendering owners. No schema or
new dependency is added.
