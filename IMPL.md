# IMPL: FEMNIST natural partition

Session-by-session checklist for what's actively in flight. When a PR
ships, its contents get replaced by the next session's plan; between
sessions this file sits as a brief placeholder so it doesn't lie about
being in-flight.

Long-horizon planning lives in [ROADMAP.md](ROADMAP.md). Session-scale
execution lives here.

## In flight

Shipping **FEMNIST natural (writer-keyed) partition** — the ROADMAP calls this
"the first thing missing to make the leaderboard honest across real FL
benchmarks." FEMNIST's realistic non-IID structure comes from *who wrote each
character*, so the federated split must be keyed on the writer, not drawn at
random like iid/dirichlet/shard.

**Why now.** Highest-value unblocked Datasets item; foundational (core library,
not an example); adds no runtime dependency; verifies cleanly via pytest (no
browser, no real download — monkeypatched loader like the CIFAR-100 test).

**Decisions** (research(2026-05); Flower Datasets is the canonical reference):

- **Signature** `velocity.partition.natural(group_ids, num_clients, *, seed)` —
  groups indices by `group_ids` (e.g. `writer_id`), shuffles the distinct groups
  by `seed`, deals them into `num_clients` even chunks; a client is the union of
  its groups' indices. A whole group never splits across clients (the invariant
  that makes it "natural"). `num_clients == #groups` ⇒ one writer per client
  (Flower's `NaturalIdPartitioner`); fewer ⇒ whole writers packed together
  (Flower's `GroupedNaturalIdPartitioner`). `num_clients > #groups` raises.
- **Honor `num_clients`** (vs Flower's `group_size`) because this module's API
  already takes `num_clients` everywhere; practitioners pick a client count, not
  a writers-per-client. Keying on `num_clients` collapses Flower's two
  partitioners into one.
- **No sample-balancing** — uneven client sizes reflect real per-writer sample
  counts; balancing would erase the heterogeneity that's the point. Matches
  Flower (groups by id count, not sample count).
- **Stdlib only** — `dict.fromkeys` for deterministic first-appearance group
  order + seeded shuffle + `divmod` even-chunk, mirroring `iid`. partition.py
  stays torch/numpy/HF-free, so it's still a clean Rust port candidate.
- **Loader threading** — `load_federated(..., partition="natural", group_by=None)`.
  `group_by` names the writer column; when omitted it auto-resolves via a new
  `_GROUP_ALIASES = ("writer_id", "user_id", "client_id", "group_id")`, mirroring
  the image/label alias resolution. Add `"character"` to `_LABEL_ALIASES`
  (FEMNIST's 62-class label column, currently unresolved).
- **Dataset** `flwrlabs/femnist` (814,277 samples; cols `image`, `writer_id`,
  `hsf_id`, `character`/ClassLabel-62). Resolves through the existing loader once
  the two alias additions land.

**Scope**: the `natural` partitioner + its loader plumbing + tests (unit on the
partitioner with synthetic `group_ids`; monkeypatched loader integration test).

**Out of scope** (deliberate, recorded so the boundary is legible):
- MCP `run_experiment` exposure — leave the `iid|dirichlet|shard` validation as
  is; `natural` needs a writer column and there's no FEMNIST experiment/leaderboard
  flow to consume it yet. Wire it when that flow lands.
- A FEMNIST `NORMALIZATION_STATS` entry — its constants aren't as standardized as
  CIFAR; shipping an unverified constant is worse than the `ToTensor` default.
- A runnable `examples/femnist_*.py` — the consumer (leaderboard) isn't built;
  capability + tests is the honest slice.

**Definition of done**: `natural` covers every index exactly once, never splits a
group, is deterministic per seed, rejects `num_clients > #groups`; FEMNIST-shaped
data loads end-to-end through `load_federated(partition="natural")` in a
monkeypatched test; `make lint` + full `make test` green locally; CI green on main.
