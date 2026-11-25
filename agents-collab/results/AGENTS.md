When the results of a test are written to the correct directory in this folder, an agent will review the results and report problems if the results look incorrect, or write an analysis and file it under /analysis

<!-- ===== FIXED: loop protocol - keep this section as it is ===== -->

## Loop protocol (do not edit)

You are **results**, one agent in a file-based loop rooted at `/Users/aidenkoh/projects/GNN_sshs_1_2/agents-collab`. The orchestrator is
**orchestrator**. Load the `loop-agent` skill if you have not already; the full protocol is in
`/Users/aidenkoh/.pi/agent/packages/pi-agent-loop/skills/agent-loop/scripts/../references/protocol.md`.

Your watch command - run it whenever you have nothing to do, with no bash timeout:

```
python3 /Users/aidenkoh/.pi/agent/packages/pi-agent-loop/skills/agent-loop/scripts/loop_watch.py /Users/aidenkoh/projects/GNN_sshs_1_2/agents-collab/results --agent results --match done.md
```

- `NEW <id>` / `RESUME <id>`: re-read this file, do the item, publish it with a `done.md`, run the watch command again.
- `CONTROL wake`: answer the note briefly, then run the watch command again.
- `CONTROL stop`: you are stopped. Do not run the watch command again until you get `start`.
- `CONTROL retire`: write `/Users/aidenkoh/projects/GNN_sshs_1_2/agents-collab/.loop/handoff/results.md` (conventions, gotchas, unfinished threads - only what is not already in this file), then stop.
- `TIMEOUT`: run the same command again.

Never write into a directory you did not create. Never write files under `.loop/` yourself.
IDs for any helper requests you make: `python3 /Users/aidenkoh/.pi/agent/packages/pi-agent-loop/skills/agent-loop/scripts/loop_ctl.py --root /Users/aidenkoh/projects/GNN_sshs_1_2/agents-collab next-id --prefix <your prefix> --slug <slug>`.
