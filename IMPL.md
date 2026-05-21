# IMPL: session break — awaiting next plan

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## Last shipped

**Memory compaction for `recent_runs.md`** (2026-05-21).
`velocity.memory.compact_entry(user_id, file, keep_last_n=10)` bounds a
memory file by keeping its last N H2 blocks and replacing the older
ones with a dated compaction marker. Surfaced as the
`compact_memory(user_id, file, keep_last_n)` MCP tool so the agent can
call it after a busy session. Preserves the audit trail (every prior
`append` recorded a `summary` in `.events.jsonl`) and the structured
run snapshots (queryable via `list_runs` / `db.recent_runs`). 8 new
unit tests; 170 total Python tests pass; lint + ty clean; MCP cache
hash bumped to reflect the new tool in the surface.

The trade-off vs LLM-summarized rollup: this approach keeps the
implementation hermetic and dependency-free, at the cost of a less
narrative-rich "older runs" view. If a narrative summary becomes
desirable later, layer it on top of `.events.jsonl` rather than
fighting the file format.

## Next up (queued, not active)

Per ROADMAP the natural next sessions are:

1. **CodSpeed + crowd-scale (50–100 clients) bench tier** — the
   noise-floor upgrade that makes single-digit-percent regression
   detection meaningful on the WSL2 box; see
   [ROADMAP → Performance](ROADMAP.md#performance).
2. **Prefab `PrefabApp` return types on MCP tools** — `run_demo` and
   siblings return plain dict/list[dict] today; migrate to typed
   Prefab returns so Claude UI can render natively.
3. **`run_demo` real-training sibling** — current `run_demo` calls the
   mock `VelocityServer.run`; add a confirmation-gated tool that
   triggers a real round.

When picking one up, replace this file with a full session plan
(Why / Decisions / Scope / Out of scope / Definition of done) matching
the Trimmed Mean PR template.
