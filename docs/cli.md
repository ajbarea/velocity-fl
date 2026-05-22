# CLI Reference

VelocityFL ships a [Typer](https://typer.tiangolo.com/)-based CLI called `velocity`. After `uv sync` the command is available via `uv run velocity`, or directly on your `PATH` when the venv is activated.

```bash
uv run velocity --help
```

All subcommands emit **JSON on stdout** and logs on stderr, so they pipe cleanly into `jq`, files, or downstream tooling.

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
# Krum
# MultiKrum
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
