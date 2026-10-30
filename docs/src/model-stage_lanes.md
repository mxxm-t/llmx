# `src/model/stage_lanes.hpp` - a thread for each stage that waits on its devices

`infer::StageLanes` is the one owner of the threads that record a stage whose recording waits on its devices (`Model::stage_waits`: a tensor group's, which waits at every sum, and past the first a device's on a pipelined split, which waits for its rows).
`Model::run_chunks` uses it for a prompt's chunks (`Model::prefill`), and the server's scheduler for its rounds (`Scheduler::Recorder`), so the CLI, the bench and the server record such stages the same way.

- `add(s, m)` makes stage s's lane under the caller's mutex `m` and returns it before its thread starts, so a caller can tie its own state to the lane's condition variable (`Lane::cv`) and stop flag (`Lane::stop`); `start(s, policy)` starts its thread.
- A lane's thread holds `m` and loops: it asks the policy's `idle` how long it may sleep (none for until woken), returns once `stop` is set, runs `step` while `ready` holds, and otherwise waits on its condition variable for that long; `step` and `idle` may unlock `m` and hold it again when they return.
- `notify(s)` wakes a lane after the caller changed what `ready` reads.
- Destruction sets every lane's `stop` under its mutex, wakes it and joins its thread, after the step it is in ends, so a caller declares the lanes after anything they record on.

The lanes hold no pass, pacing or rule: `run_chunks` hands a lane one chunk at a time and commits it on the calling thread, and the scheduler's recorder keeps its holds as its policy (docs/src/server.md).
