# Strategies

VelocityFL ships nine aggregation strategies. All nine are implemented in Rust and exposed as frozen Python dataclasses in `velocity.strategy`. Pick one based on your threat model and client heterogeneity.

## Decision guide

| If you have… | Reach for |
|---|---|
| IID clients, no adversary, you want a baseline | **`FedAvg`** |
| Heterogeneous clients, drifting local updates | **`FedProx(mu=…)`** |
| Untrusted clients, possible Byzantine updates, <½ of clients compromised | **`FedMedian`** |
| Untrusted clients, up to `k` compromised per coordinate, want averaged survivors | **`TrimmedMean(k=…)`** |
| Untrusted clients, up to `f` compromised, want a single winner | **`Krum(f=…)`** |
| Untrusted clients, up to `f` compromised, want to average `m` survivors | **`MultiKrum(f=…, m=…)`** |
| Untrusted clients, up to `f` compromised, want the strongest distance-based defense | **`Bulyan(f=…, m=…)`** |
| Untrusted clients, up to ⌊(n−1)/2⌋ compromised, want geometric (not coordinate-wise) robustness | **`GeometricMedian()`** |
| Untrusted clients, unknown / per-round Byzantine count, want parameter-free Krum (not Fang-Krum threat model — see ArKrum's _Known weaknesses_) | **`ArKrum()`** |
| Untrusted clients, aggregator-aware adversary (Fang-style optimisation attacks) | **`GeometricMedian()`** or **`Bulyan(f=…, m=…)`** — *not* a Krum-family selector |

All nine are value objects: compare with `==`, safe to hash, safe to share between threads.

```python
from velocity import (
    FedAvg, FedProx, FedMedian, TrimmedMean,
    Krum, MultiKrum, Bulyan, GeometricMedian, ArKrum,
)

FedAvg() == FedAvg()                  # True
FedProx(mu=0.01) != FedProx(mu=0.1)   # True
```

---

## `FedAvg`

Weighted average by local sample count — the McMahan et al. (2017) baseline.

```text
w_{t+1} = Σ_k (n_k / n) · w_{t+1}^k
```

**Use when** clients are IID and trusted. Fast, stable, easy to reason about.

```python
from velocity import VelocityServer, FedAvg
server = VelocityServer(model_id=..., dataset=..., strategy=FedAvg())
```

---

## `FedProx`

FedAvg-style aggregation with a proximal term that penalizes local updates for drifting too far from the global model. From Li et al. (2020).

```text
minimize over w:   F_k(w) + (μ/2) · ‖w - w_t‖²
```

**Use when** clients are heterogeneous — differing data distributions, compute budgets, or local epoch counts. The proximal term `μ` dampens client drift.

| Field | Default | Effect |
|---|---|---|
| `mu` | `0.01` | Higher → more conservative updates, slower convergence, better stability on non-IID data. |

```python
from velocity import VelocityServer, FedProx
server = VelocityServer(model_id=..., dataset=..., strategy=FedProx(mu=0.05))
```

---

## `FedMedian`

Coordinate-wise median of client updates. From Yin et al. (2018).

```text
w_{t+1}[i] = median( w_{t+1}^k[i]  for k = 1..K )
```

**Use when** you cannot trust every client. Robust against up to ⌊K/2⌋ Byzantine updates — a poisoned client cannot shift the median, only extend its tail.

```python
from velocity import VelocityServer, FedMedian
server = VelocityServer(model_id=..., dataset=..., strategy=FedMedian())
```

> **Pair with attack simulation** — `FedMedian` is the natural companion to the [attacks catalog](attacks.md). Run the same experiment with `FedAvg` and `FedMedian`, then compare `global_loss` trajectories to see resilience in action.

---

## `TrimmedMean`

Coordinate-wise mean after dropping the `k` smallest and `k` largest values. From Yin et al. (2018, [arXiv:1803.01498](https://arxiv.org/abs/1803.01498)).

```text
For each coordinate i:
  s = sort( w_{t+1}^k[i]  for k = 1..K )
  w_{t+1}[i] = mean( s[k : K-k] )
```

**Use when** you want a Byzantine-robust aggregator that is cheaper than `FedMedian` for small `k` and dimension-independent. Tunes the trim budget directly: `k=0` is the uniform mean, `k=⌊(n-1)/2⌋` reduces to the (odd-n) median.

| Field | Constraint | Effect |
|---|---|---|
| `k` | `0 ≤ k`, `2k < n` | Per-coordinate Byzantine tolerance. Raises if the round has fewer than `2k + 1` clients. |

```python
from velocity import VelocityServer, TrimmedMean
server = VelocityServer(model_id=..., dataset=..., strategy=TrimmedMean(k=1))
# Tolerates 1 Byzantine client per coordinate; needs at least 3 clients per round.
```

> **Why uniform, not sample-weighted** — same reason as `MultiKrum`: a Byzantine client can lie about `num_samples` to pull the mean in a weighted average. `TrimmedMean` discards `num_samples` deliberately; pinned by `test_trimmed_mean_k0_is_uniform_mean`.

> **Per-coordinate robustness, not per-client.** Different coordinates trim different client subsets — there is no single "selected" client list. `RoundSummary.selected_client_ids` therefore returns every participating client (same convention as `FedMedian`).

---

## `Krum`

Select the single client whose update looks most like its `n − f − 2` nearest neighbours, by squared Euclidean distance. From Blanchard et al. (2017).

```text
score(i) = Σ_{j ∈ N_i} ‖w_i - w_j‖²     where N_i = the (n-f-2) closest clients
w_{t+1}  = w_{argmin_i score(i)}
```

**Use when** you assume up to `f` of the `n` clients are Byzantine and you want a provably-bounded winner rather than a blended average.

| Field | Constraint | Effect |
|---|---|---|
| `f` | `n ≥ 2f + 3` | Byzantine-tolerance bound. Raises if the round has fewer than `2f + 3` clients. |

```python
from velocity import VelocityServer, Krum
server = VelocityServer(model_id=..., dataset=..., strategy=Krum(f=2))
# Needs at least 2*2 + 3 = 7 clients per round.
```

The round summary exposes the winner's index so you can audit selections:

```python
summaries = server.run(min_clients=7, rounds=1)
summaries[0]["selected_client_ids"]   # e.g. [3]
```

> **Breakdown point** — Krum provably converges when strictly fewer than `n − 2f − 2` of the `n` clients are Byzantine. Falling below that threshold (too many attackers, or too few honest clients) silently degrades robustness; keep `n ≫ 2f + 3` in practice.

---

## `MultiKrum`

Run the Krum scoring, then return the uniform (not sample-weighted) mean of the `m` lowest-scoring updates. From El Mhamdi et al. (2018).

```text
score(i) = same as Krum
S        = indices of the m lowest scores
w_{t+1}  = (1/m) · Σ_{i ∈ S} w_i
```

**Use when** you want Krum's outlier suppression *and* the variance reduction of averaging. Tunes between the two extremes: `m=1` collapses to Krum, `m=n−f` is Multi-Krum's default.

| Field | Default | Constraint |
|---|---|---|
| `f` | — | `n ≥ 2f + 3`. |
| `m` | `n − f` | Must satisfy `1 ≤ m ≤ n − f`. `None` uses the default. |

```python
from velocity import VelocityServer, MultiKrum
server = VelocityServer(model_id=..., dataset=..., strategy=MultiKrum(f=2, m=5))
```

> **Why uniform, not sample-weighted** — Byzantine clients can lie about `num_samples` to amplify their pull in a weighted average. Multi-Krum deliberately discards that signal; this is pinned by `test_multi_krum_m_equals_n_minus_f_is_uniform_mean` and `strategy::tests::multikrum_uniform_weighting_ignores_sample_counts`.

---

## `Bulyan`

Compose Multi-Krum with a coordinate-wise trimmed mean over the survivors. From El Mhamdi et al. (2018, ICML — *The Hidden Vulnerability of Distributed Learning in Byzantium*, Algorithm 2).

```text
Phase 1:  S = top-m clients by Multi-Krum score  (m defaults to n − 2f)
Phase 2:  for each coordinate i:
            sorted = sort(w_i for client in S)
            w_{t+1}[i] = mean(sorted[f : m − f])    # uniform mean of m − 2f survivors
```

**Use when** you want the strongest Byzantine guarantee in the distance-based family. Bulyan inherits Multi-Krum's outlier suppression in Phase 1 and the trimmed-mean's per-coordinate robustness in Phase 2 — bounded contamination of `O(1/√d)` per coordinate vs Multi-Krum's `O(√d)` (the *hidden vulnerability* the paper names). Costs roughly Multi-Krum + TrimmedMean per round.

| Field | Default | Constraint |
|---|---|---|
| `f` | — | `n ≥ 4f + 3`. Strictly tighter than Multi-Krum's `n ≥ 2f + 3`. |
| `m` | `n − 2f` | Must satisfy `2f + 1 ≤ m ≤ n − 2f`. `None` uses the paper's default. |

```python
from velocity import VelocityServer, Bulyan
server = VelocityServer(model_id=..., dataset=..., strategy=Bulyan(f=1, m=None))
# Needs at least 4*1 + 3 = 7 clients per round.
```

> **Cost factor** — Bulyan composes the two slowest robust kernels and lands at their sum minus a small subset discount. At the `large` (10M-param, 10-client) tier it costs ~35× Rust FedAvg in [`docs/benchmarks.md`](benchmarks.md). Worth it when you need the tightest distance-based breakdown bound; otherwise prefer Multi-Krum or GeometricMedian.

---

## `GeometricMedian`

Solve `argmin_y Σ w_i · ‖y − x_i‖` over the flattened client weight vectors using the Weiszfeld iteration. From Pillutla, Kakade, Harchaoui (2022, IEEE TSP — *Robust Aggregation for Federated Learning*, the RFA algorithm).

```text
Initialise:  y_0 = sample-weighted mean (FedAvg estimate)
Iterate:     d_i      = max(eps, ‖y_k − x_i‖)
             y_{k+1}  = Σ (w_i · x_i / d_i) / Σ (w_i / d_i)
Stop:        ‖y_{k+1} − y_k‖ < eps  OR  k ≥ max_iter
```

**Use when** you want geometric (multivariate) robustness rather than coordinate-wise. Coordinate-wise defenses (`FedMedian`, `TrimmedMean`) are robust per-coordinate; the geometric median is robust *as a vector*, so coordinated low-magnitude attacks that hide inside per-coordinate distributions get absorbed instead of rejected. **1/2 breakdown point** — survives up to ⌊(n−1)/2⌋ Byzantine clients with bounded contamination over a constant number of iterations. Sample-weighted (the Weiszfeld update keeps the FedAvg-style `w_i = num_samples / total` weighting).

| Field | Default | Effect |
|---|---|---|
| `eps` | `1e-6` | Numerical floor on per-client distance and convergence threshold on `‖y_{k+1} − y_k‖`. Smaller = more iterations near a fixed point. |
| `max_iter` | `3` | Cap on the Weiszfeld loop. Pillutla et al. recommend a small constant — 3 is a good default; further iterations don't change the breakdown bound. |

```python
from velocity import VelocityServer, GeometricMedian
server = VelocityServer(model_id=..., dataset=..., strategy=GeometricMedian())
# Defaults to eps=1e-6, max_iter=3 — matches the RFA paper's recommendation.
```

> **Empirical edge against label-flipping** — `examples/mnist_label_flipping_vs_robust.py` runs FedAvg / Multi-Krum / FedMedian / GeometricMedian head-to-head with 2 of 10 clients label-flipped under Dirichlet(α=1.0). All three robust aggregators recover most of the attack damage; GeometricMedian narrowly tops the matrix. The new nightly demo asserts the best robust aggregator beats FedAvg by ≥ 0.05 — exactly the "no defense pinned, data picks the winner" framing that respects the literature's known limitation: distance-based defenses degrade against label-flipping in heavy non-IID settings.

---

## `ArKrum`

Parameter-free Krum variant. From Yang, Imam, et al. (2025, arXiv:2505.17226 — *Secure and Private Federated Learning: Achieving Adversarial Resilience through Robust Aggregation*). Removes Krum's requirement that the caller specify `f` (the Byzantine count) in advance — `f̂` is estimated per round from the data.

```text
For each client i, with D_i = sorted ascending distances to other clients:
  1. D'_i ← filter τ = median + (median − min)        # Algorithm 1
  2. f̂_i ← change-point on D'_i                      # rKrum ESTIMATE_F
              (gap-based, with 5× gap-ratio AND 10× magnitude-ratio thresholds)
  3. ScoreKrum_i = sum of (n − f̂_i − 2) smallest distances in D_i
Pick u* with the lowest ScoreKrum.
ū = uniform mean of the (n − f̂_{u*}) updates closest to u* (including u*).
```

**Use when** you don't know the Byzantine count and don't want to over-estimate to be safe. Standard Krum/Multi-Krum/Bulyan need a tight `f` to behave; overestimating wastes honest clients, underestimating lets malicious updates through. ArKrum auto-tunes per round. No parameters. Requires `n ≥ 5` so the median + change-point steps have enough samples.

```python
from velocity import VelocityServer, ArKrum
server = VelocityServer(model_id=..., dataset=..., strategy=ArKrum())
```

### Known weaknesses

**Dense colluding cluster.** Krum-family algorithms break when byzantines outvote honest. ArKrum inherits this from rKrum/Krum: if the malicious cluster is denser (lower intra-cluster distance) than the honest cluster, the Krum score on a byzantine `u_i` becomes lower than on an honest one and `u*` lands on the byzantine side. The paper's evaluation assumes honest is the dominant tight cluster. For an explicitly colluding-safe defense, look at trust-history-based methods (PID/Trust removal — queued under ROADMAP "Client-removal defenses").

**Fang-style aggregator-aware attacks.** Fang et al. (USENIX Security 2020, arXiv:1911.11815) show that an attacker who knows the aggregator's score function can optimise a perturbation that places malicious updates near the honest cluster — close enough that the score function selects them, but pointed at a target poisoning direction. Against Fang-Krum, ArKrum's parameter-free `f̂` estimator typically returns 0 (no big gap on the sorted-distance vector, no outliers above τ), and ArKrum degrades to FedAvg over all clients including the attackers. On vFL's `n=11 / f=2` MNIST nightly, this collapses ArKrum to **9.6% final accuracy** vs. **94-96%** under non-aggregator-aware attacks (label-flip, sign-flip, ALIE, IPM, Gaussian).

This is **not a bug in our implementation** — it is a property of every Krum-family aggregator that scores updates by sum-of-nearest-neighbour distances. SpectralKrum (arXiv:2512.11760, Dec 2025), the newest Krum variant in the literature, explicitly acknowledges *"limited advantage over simpler statistical methods under min-max perturbations where malicious updates remain spectrally indistinguishable from benign ones"* — Fang is min-max perturbation.

If your threat model includes an aggregator-aware adversary, do not use ArKrum, Krum, MultiKrum, Bulyan, or SpectralKrum (Krum-family selectors all share this surface). Reach for `GeometricMedian()` (Weiszfeld-iteration RFA, Pillutla et al. IEEE TSP 2022, 1/2 breakdown point with bounded contamination over a constant number of iterations) or `TrimmedMean(k=f)` / `FedMedian` (coordinate-wise statistics; Fang's transferable Full-Trim variant still degrades these, but less catastrophically than Krum-family). The decision-guide table above flags the Fang-robust choices explicitly.

<!-- research(2026-05): Fang USENIX 2020 + SpectralKrum 2025 confirm Krum-family selectors share aggregator-aware vulnerability; no known parameter-free fix that preserves Krum's score function. Documented as known weakness rather than patched — adding a non-zero minimum f̂ would defeat "parameter-free", and aggregator-aware detection is a separate algorithm (SpectralKrum) not a patch. Bulyan + GeometricMedian remain the Fang-robust choices in this codebase. -->


---

## CLI shorthand

The CLI accepts `Name` for parameter-free strategies and `Name:key=value[,key=value]` for the parameterised ones:

```bash
velocity run  --strategy FedAvg              --model-id demo/m --dataset demo/d
velocity run  --strategy FedProx:mu=0.05     --model-id demo/m --dataset demo/d
velocity run  --strategy TrimmedMean:k=1     --model-id demo/m --dataset demo/d --min-clients 3
velocity run  --strategy Krum:f=2            --model-id demo/m --dataset demo/d --min-clients 7
velocity run  --strategy MultiKrum:f=2,m=5   --model-id demo/m --dataset demo/d --min-clients 7
velocity run  --strategy Bulyan:f=1           --model-id demo/m --dataset demo/d --min-clients 7
velocity run  --strategy GeometricMedian      --model-id demo/m --dataset demo/d
velocity run  --strategy ArKrum               --model-id demo/m --dataset demo/d --min-clients 5
velocity sweep --strategies FedAvg,Krum:f=1  --rounds 5
```

Sweep TOML files accept either the string form or a dict form — see [Sweep spec](sweep-spec.md).

---

## Adding your own

Strategies are defined in `vfl-core/src/strategy.rs`. To add a new one:

1. Add a variant to the `Strategy` enum and implement the aggregation kernel in Rust. Return an `Aggregation { weights, selected_client_ids }`.
2. Expose a PyO3 constructor in `vfl-core/src/lib.rs` (e.g. `Strategy::trimmed_mean(trim_ratio)`).
3. Add a frozen dataclass to `python/velocity/strategy.py` and include it in `ALL_STRATEGIES`.
4. Wire the isinstance dispatch in `VelocityServer._map_strategy`.
5. Extend `parse_strategy` if the new dataclass has non-trivial coercion needs.
6. Add tests in `tests/` and a section to this page.

See [Architecture](architecture.md) for the full layer map.
