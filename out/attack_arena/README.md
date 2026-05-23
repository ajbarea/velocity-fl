# Attack-arena dataset

Real-MNIST Byzantine-robust convergence sweep used to drive the Prefab
arena dashboard (`attack_arena` MCP tool, Prefab phase 2).

## How to regenerate

```bash
uv run python scripts/dump_attack_arena.py --rounds 16 --seeds 5
# wall time: ~55 min on CPU
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
| Attacks | gaussian · ipm · label_flip · sign_flip · alie · fang_krum (see below) |

## Attacks

FLPoison canonical headliner set per Liu et al., *SoK: Benchmarking
Poisoning Attacks and Defenses in Federated Learning*, arXiv:2502.03801
(2025). Implementations live in `velocity.paper_attacks`; reference
implementations: https://github.com/vio1etus/FLPoison .

- **gaussian** — Blanchard et al., NeurIPS 2017 (Krum paper).
  Byzantines emit `randn × 100.0`-scaled noise per layer.
- **ipm** — Xie et al., UAI 2020, "Fall of Empires" (arXiv:1903.03936).
  Byzantines emit `-1.0 × mean(honest)`, lives near the honest cluster
  in Euclidean distance.
- **label_flip** — Tolpegin et al., ESORICS 2020. Bijective class
  permutation on byzantine-client training data.
- **sign_flip** — Damaskinos et al., ICML 2018. Each byzantine negates
  its own honestly-trained state. The simplest model-poisoning floor
  every robust aggregator must trivially clear.
- **alie** — Baruch, Baruch, Goldberg, NeurIPS 2019, "A Little Is
  Enough" (arXiv:1902.06156). Byzantines emit
  `mean(honest) + z_max × std(honest)` with `z_max` chosen so the
  perturbation stays within the honest-cluster envelope. Designed to
  evade distance-based defenses.
- **fang_krum** — Fang et al., USENIX Security 2020 (arXiv:1911.11815).
  Aggregator-aware Krum-targeted attack. Binary-searches the lambda
  such that `w_global - lambda × sign(direction)` is selected by Krum
  over the honest cluster.

The set covers six of the seven FLPoison headliner attacks; the
seventh (BadNets-style backdoor) needs an ASR (attack success rate)
metric the arena does not currently report and is out of scope until
the dashboard gains a per-class breakdown.

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
> non-IID partitioning, 16 rounds, mean ± std over 5 seeds. Six
> paper-cited attacks from the FLPoison SoK canonical set:
> Blanchard 2017 Gaussian + Xie 2019 IPM + Tolpegin 2020 label-flip +
> Damaskinos 2018 sign-flip + Baruch 2019 ALIE + Fang 2020 Krum-attack.
> Reproducible via `uv run python scripts/dump_attack_arena.py`.*

That sentence ships in the LinkedIn caption verbatim — every adjective
traces to a source.

## What's not here yet

- **BadNets / DBA / NeuroToxin** — trigger-based backdoor attacks. Out
  of scope until the dashboard supports per-class ASR (attack success
  rate) reporting; the current arena measures only clean-test accuracy.
- **CIFAR-10 / CIFAR-100 corpus** — easier extension once the Prefab
  arena consumes this CSV cleanly. MNIST first because the nightly
  tests already passed there.
- **Vertical-FL / `byzantine fraction > 2 / n` regimes** — Bulyan
  requires n ≥ 4f+3, which pins f=2 at n=11. Pushing f harder needs
  dropping Bulyan from the matrix.
