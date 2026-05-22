# Attack-arena dataset

Real-MNIST Byzantine-robust convergence sweep used to drive the Prefab
arena dashboard (`attack_arena` MCP tool, Prefab phase 2).

## How to regenerate

```bash
uv run python scripts/dump_attack_arena.py --rounds 16 --seeds 5
# wall time: ~35 min on CPU
```

## Configuration

Matches `tests/test_paper_attacks_nightly.py` so this corpus stays
directly comparable to the nightly green-light:

| Knob | Value |
| --- | --- |
| Dataset | Hugging Face `ylecun/mnist`, Dirichlet α=1.0 partitioning |
| Clients | 11 total, 2 byzantine (indices 0–1), `min_partition_size=50` |
| Local training | 1 epoch, SGD lr=0.01, batch 64 |
| Rounds | 16 |
| Seeds | 5 per (strategy, attack) cell |
| Strategies | FedAvg, Krum (f=2), MultiKrum (f=2, m=3), Bulyan (f=2), ArKrum |
| Attacks | label_flip · ipm · gaussian (see below) |

## Attacks

Curated paper-cited set vFL implements today (subset of the FLPoison
SoK canonical headliner set — ALIE / Fang / sign-flip / BadNets are
out of scope until those land as native attacks):

- **label_flip** — Tolpegin et al., ESORICS 2020. Bijective class
  permutation on byzantine-client training data.
- **ipm** — Xie et al. 2019, "Fall of Empires". Byzantines emit
  `-1.0 × mean(honest)`, lives near the honest cluster in Euclidean
  distance.
- **gaussian** — Krum-paper canonical gradient poisoning. Byzantines
  emit `randn × 100.0`-scaled noise per layer.

## Output files

### `runs.json`
List of every run. One record per (strategy, attack, seed):

```json
{
  "strategy":         "FedAvg",
  "attack":           "label_flip",
  "seed":             0,
  "elapsed_seconds":  28.7,
  "records": [
    {"round": 1,  "pre_loss": 2.307, "post_acc": 0.7498},
    {"round": 2,  "pre_loss": 0.918, "post_acc": 0.8282},
    …
    {"round": 16, "pre_loss": 0.187, "post_acc": 0.9284}
  ]
}
```

### `aggregated.csv`
Per-(strategy, attack, round) mean + std across the seed sweep. This
is the shape the Prefab `LineChart` band consumes:

```csv
strategy,attack,round,n_seeds,mean_acc,std_acc
FedAvg,gaussian,1,5,0.0928,0.0142
FedAvg,gaussian,16,5,0.0980,0.0011
Krum,gaussian,16,5,0.9247,0.0083
…
```

## Citation framing for the LinkedIn demo

> *Real MNIST, n=11 clients with f=2 byzantine, Dirichlet α=1.0
> non-IID partitioning, 16 rounds, mean ± std over 5 seeds. Three
> paper-cited attacks: Tolpegin 2020 label-flip + Xie 2019 IPM +
> Krum-paper Gaussian. Reproducible via
> `uv run python scripts/dump_attack_arena.py`.*

That sentence ships in the LinkedIn caption verbatim — every adjective
traces to a source.

## What's not here yet

- **ALIE / Fang / sign-flip / BadNets** — implementing these is
  separate research scope (`velocity.attacks` / `velocity.data_attacks`
  extension). They round out the FLPoison SoK canonical headliner
  set; current corpus covers three of the seven.
- **CIFAR-10 / CIFAR-100 corpus** — easier extension once the
  Prefab arena consumes this CSV cleanly. MNIST first because the
  nightly tests already passed there.
- **Vertical-FL / `byzantine fraction > 2 / n` regimes** — Bulyan
  requires n ≥ 4f+3, which pins f=2 at n=11. Pushing f harder needs
  dropping Bulyan from the matrix.
