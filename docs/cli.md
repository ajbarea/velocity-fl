# CLI Reference

Velocity-FL ships a [Typer](https://typer.tiangolo.com/)-based CLI called `velocity`. After `uv sync` the command is available via `uv run velocity`, or directly on your `PATH` when the venv is activated.

```bash
uv run velocity --help
```

The data subcommands — `run`, `simulate-attack`, and `leaderboard --json` — emit **JSON on stdout** with logs on stderr, so they pipe cleanly into `jq`, files, or downstream tooling. `version`, `strategies`, `sweep`, and the default `leaderboard` table print human-readable text.

---

## `velocity version`

Prints the installed package version.

```bash
uv run velocity version
# 0.1.0
```

---

## `velocity strategies`

Lists supported aggregation strategies. See [Strategies](strategies.md) for when to use each.

```bash
uv run velocity strategies
# FedAvg
# FedProx
# FedMedian
# TrimmedMean
# Krum
# MultiKrum
# Bulyan
# GeometricMedian
# ArKrum
```

---

## `velocity run`

Runs a local orchestrated experiment and prints a JSON array of round summaries.

```bash
uv run velocity run \
    --model-id meta-llama/Llama-3-8B \
    --dataset huggingface/ultrafeedback \
    --strategy FedAvg \
    --rounds 5 \
    --min-clients 10
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--model-id` | `str` | *required* | Hugging Face model identifier. |
| `--dataset` | `str` | *required* | Dataset name or path (HF Hub or local). |
| `--strategy` | `str` | `FedAvg` | `FedAvg`, `FedProx[:mu=…]`, `FedMedian`, `TrimmedMean:k=…`, `Krum:f=…`, `MultiKrum:f=…[,m=…]`, `Bulyan:f=…[,m=…]`, `GeometricMedian[:eps=…,max_iter=…]`, or `ArKrum` (case-insensitive). See [Strategies](strategies.md). |
| `--storage` | `str` | `local://checkpoints` | Checkpoint storage URI. |
| `--min-clients` | `int ≥ 1` | `1` | Minimum clients required per round. |
| `--rounds` | `int ≥ 1` | `1` | Number of federated rounds. |

**Output** — a JSON array; each element has `round`, `num_clients`, `global_loss`, `attack_results`.

---

## `velocity simulate-attack`

Registers one attack and runs a single round so you can observe its impact without standing up a full experiment. See [Attacks](attacks.md) for the full catalog.

```bash
uv run velocity simulate-attack model_poisoning --intensity 0.2
uv run velocity simulate-attack sybil_nodes --count 5
uv run velocity simulate-attack gaussian_noise --intensity 0.1
```

| Argument / Option | Type | Default | Description |
|---|---|---|---|
| `ATTACK_TYPE` (positional) | `str` | *required* | `model_poisoning` \| `sybil_nodes` \| `gaussian_noise`. |
| `--model-id` | `str` | `demo/model` | Model identifier for the one-round probe. |
| `--dataset` | `str` | `demo/dataset` | Dataset identifier for the probe. |
| `--strategy` | `str` | `FedAvg` | Aggregation strategy. |
| `--min-clients` | `int ≥ 1` | `1` | Minimum clients for the probe round. |
| `--intensity` | `float ≥ 0` | `0.1` | Used by `model_poisoning` and `gaussian_noise`. |
| `--count` | `int ≥ 1` | `1` | Used by `sybil_nodes`. |

**Output** — a single JSON object describing the probe round.

> Data-pipeline attacks (label flipping etc.) don't fit the one-shot CLI
> shape — they have to compose with a real data loader. Use
> [`velocity.data_attacks`](attacks.md#data-pipeline-attacks-velocitydata_attacks)
> from a Python script instead.

---

## `velocity leaderboard`

Ranks stored experiment runs from the live run store (`velocity.db`), grouped by config fingerprint and averaged across seeds. See [Leaderboard](leaderboard.md) for the published attack-arena rankings.

```bash
uv run velocity leaderboard
uv run velocity leaderboard --metric rounds-to-target --target 0.9
uv run velocity leaderboard --metric comm-cost
uv run velocity leaderboard --metric robustness
uv run velocity leaderboard --metric pareto-slices
uv run velocity leaderboard --json
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--user` | `str` | `$VFL_USER_ID`, then OS user | Whose stored runs to rank. |
| `--metric` | `str` | `accuracy` | `accuracy` (final-round), `rounds-to-target` (convergence speed), `wall-clock` (aggregation time), `comm-cost` (total bytes communicated, uplink + downlink), `pareto` (accuracy-vs-wall-clock frontier), `pareto-slices` (that frontier per dataset × attack), or `robustness` (accuracy drop under attack). |
| `--target` | `float` | `0.9` | Target accuracy (0–1) for the `rounds-to-target` metric. |
| `--min-runs` | `int ≥ 1` | `1` | Drop config groups with fewer than N runs. |
| `--json` | flag | off | Emit JSON instead of the formatted table. |

**Output** — a ranked table (or JSON with `--json`). An empty store prints a friendly "no runs yet" message rather than failing.

---

## `velocity sweep`

Runs a strategy × attack matrix across seeds and writes a comparison report. Drive it from a TOML experiment file or ad-hoc flags. Full spec: [Sweep spec](sweep-spec.md).

```bash
# From a TOML experiment file
uv run velocity sweep experiments/robust.toml

# Ad-hoc: every strategy × attack (a no-attack baseline is always included)
uv run velocity sweep --strategies FedAvg,FedMedian --attacks gaussian_noise,model_poisoning --rounds 10
```

| Argument / Option | Type | Default | Description |
|---|---|---|---|
| `CONFIG` (positional) | `path` | none | TOML experiment file. Omit when using `--strategies`. |
| `--strategies` | `str` | `""` | Comma-separated strategies for ad-hoc mode. |
| `--attacks` | `str` | `""` | Comma-separated attacks (`model_poisoning` \| `sybil_nodes` \| `gaussian_noise`); a baseline run is always added. |
| `--rounds` | `int ≥ 1` | `5` | Rounds per run (ad-hoc mode). |
| `--min-clients` | `int ≥ 1` | `2` | Min clients per round (ad-hoc mode). |
| `--seed` | `int` | `0` | Base seed; run *i* uses `seed + i`. |
| `--model-id` | `str` | `demo/model` | Model id (ad-hoc mode). |
| `--dataset` | `str` | `demo/dataset` | Dataset (ad-hoc mode). |
| `--out` | `path` | `out/<timestamp>-sweep` | Output directory. |
| `--parallel` | `int` | `min(cpu, #runs)` | Process-pool size. |

**Output** — a timestamped `out/<ts>-sweep/` directory with per-run records and a `comparison.md` ranking report.

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Command completed. |
| `2` | Invalid argument (e.g. unknown strategy, unknown attack). |
| Other | Underlying error — consult stderr. |

## Piping to jq

```bash
uv run velocity run --model-id demo/model --dataset demo/dataset --rounds 3 --min-clients 2 \
    | jq '.[] | {round, loss: .global_loss}'
```
