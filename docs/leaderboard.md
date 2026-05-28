# Byzantine-FL Attack Arena Leaderboard

Final-round test accuracy of each aggregation strategy under the FLPoison canonical attack set, on real MNIST (mean over seeds). Higher is more robust. Data lineage: [`out/attack_arena/aggregated.csv`](https://github.com/ajbarea/velocity-fl/blob/main/out/attack_arena/aggregated.csv) (regenerate via `scripts/dump_attack_arena.py`).

## Worst-case defender ranking

If you must pick one strategy without knowing the attack, this is the order — ranked by each strategy's *weakest* result across all attacks.

| Rank | Strategy | Worst-case accuracy | Weakest under |
| ---: | --- | ---: | --- |
| 1 | Bulyan | 95.7% | Gaussian (Krum-paper) |
| 2 | MultiKrum | 93.8% | Gaussian (Krum-paper) |
| 3 | Krum | 90.9% | Gaussian (Krum-paper) |
| 4 | FedAvg | 9.8% | Gaussian (Krum-paper) |
| 5 | ArKrum | 9.6% | Fang-Krum (Fang 2020) |

## Final accuracy by attack

| Strategy | Gaussian (Krum-paper) | IPM (Fall of Empires) | Label flip (Tolpegin 2020) | Sign flip (Damaskinos 2018) | ALIE (Baruch 2019) | Fang-Krum (Fang 2020) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FedAvg | 9.8% | 89.3% | 93.3% | 86.2% | 96.5% | 13.4% |
| Krum | 90.9% | 90.9% | 90.9% | 90.9% | 96.3% | 90.9% |
| MultiKrum | 93.8% | 93.8% | 93.8% | 93.8% | 95.7% | 93.8% |
| Bulyan | 95.7% | 95.7% | 95.7% | 95.7% | 96.0% | 95.7% |
| ArKrum | 96.2% | 95.8% | 96.2% | 94.5% | 96.4% | 9.6% |
