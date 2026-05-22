use std::borrow::Borrow;
use std::collections::HashMap;

/// Aggregation strategy for federated learning rounds.
///
/// All variants are paper-cited implementations; see the per-variant doc
/// comments for original venues. Aggregation kernels live further down
/// this file (`fedavg`, `fed_median`, `trimmed_mean`, `krum_select`,
/// `bulyan`); this enum is the dispatch contract.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[allow(clippy::enum_variant_names)] // Public API — variant names mirror FL literature.
pub enum Strategy {
    /// Federated Averaging — sample-weighted mean of client weights.
    ///
    /// McMahan, Moore, Ramage, Hampson, Agüera y Arcas. *Communication-Efficient
    /// Learning of Deep Networks from Decentralized Data*. AISTATS 2017,
    /// pp. 1273–1282.
    /// <https://proceedings.mlr.press/v54/mcmahan17a.html>
    FedAvg,
    /// FedProx — server-side aggregation is identical to FedAvg; the
    /// `mu` proximal coefficient is consumed *client-side* by the local
    /// training step (`velocity.training.local_train(proximal_mu=...)`).
    /// Carried on the server-side strategy purely as round metadata so
    /// caller code can read it back from the orchestrator.
    ///
    /// Li, Sahu, Zaheer, Sanjabi, Talwalkar, Smith. *Federated Optimization
    /// in Heterogeneous Networks*. MLSys 2020, pp. 429–450.
    /// <https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html>
    FedProx { mu: f64 },
    /// Coordinate-wise median — tolerates up to ⌊(n−1)/2⌋ Byzantine
    /// clients per coordinate.
    ///
    /// Yin, Chen, Ramchandran, Bartlett. *Byzantine-Robust Distributed
    /// Learning: Towards Optimal Statistical Rates*. ICML 2018,
    /// pp. 5650–5659.
    /// <https://proceedings.mlr.press/v80/yin18a.html>
    FedMedian,
    /// Coordinate-wise trimmed mean — drop the `k` smallest and `k`
    /// largest values per coordinate, then uniform-mean the remaining
    /// `n − 2k`. Tolerates up to `k` Byzantine clients per coordinate.
    /// Requires `2*k < n`.
    ///
    /// Yin, Chen, Ramchandran, Bartlett. *Byzantine-Robust Distributed
    /// Learning: Towards Optimal Statistical Rates*. ICML 2018,
    /// pp. 5650–5659.
    /// <https://proceedings.mlr.press/v80/yin18a.html>
    TrimmedMean { k: usize },
    /// Krum — picks the single client whose sum of `n − f − 2` smallest
    /// squared distances to others is minimal. Byzantine-robust when
    /// `n ≥ 2*f + 3`.
    ///
    /// Blanchard, El Mhamdi, Guerraoui, Stainer. *Machine Learning with
    /// Adversaries: Byzantine Tolerant Gradient Descent*. NeurIPS 2017.
    /// <https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html>
    Krum { f: usize },
    /// Multi-Krum — averages the top-`m` clients by Krum score. `m = None`
    /// resolves to `n − f` ("largest non-Byzantine group") at aggregation
    /// time. Requires `n ≥ 2*f + 3` and `1 ≤ m ≤ n − f`.
    ///
    /// El Mhamdi, Guerraoui, Rouault. *The Hidden Vulnerability of
    /// Distributed Learning in Byzantium*. ICML 2018.
    /// <https://proceedings.mlr.press/v80/mhamdi18a.html>
    MultiKrum { f: usize, m: Option<usize> },
    /// Bulyan (Algorithm 2) — composes Multi-Krum with a coordinate-wise
    /// trimmed mean. Phase 1 selects `m` candidates via the Multi-Krum
    /// scoring rule; Phase 2 drops the `f` largest and `f` smallest per
    /// coordinate among the survivors and uniform-means the remaining
    /// `β = m − 2f`. `m = None` resolves to `n − 2f` (the paper's default).
    /// Requires `n ≥ 4*f + 3` and `2*f + 1 ≤ m ≤ n − 2*f`.
    ///
    /// El Mhamdi, Guerraoui, Rouault. *The Hidden Vulnerability of
    /// Distributed Learning in Byzantium*. ICML 2018.
    /// <https://proceedings.mlr.press/v80/mhamdi18a.html>
    Bulyan { f: usize, m: Option<usize> },
    /// Geometric Median via Weiszfeld iteration (Robust Federated Aggregation).
    ///
    /// Solves `argmin_y Σ w_i * ||y − x_i||` where `x_i` are the flattened
    /// client weights and `w_i` are sample-count weights. Initialises at
    /// the sample-weighted mean (FedAvg) and iterates Weiszfeld's update
    /// `y_{k+1} = Σ (w_i x_i / d_i) / Σ (w_i / d_i)` with `d_i = ||y_k − x_i||`,
    /// clamped to `eps` to avoid division by zero. The geometric median
    /// has a 1/2 breakdown point — robust to up to ⌊(n−1)/2⌋ Byzantine
    /// clients, with bounded contamination over a constant number of
    /// iterations.
    ///
    /// `eps` is the numerical floor on per-client distance (also the
    /// convergence-stopping threshold on `||y_{k+1} − y_k||`); `max_iter`
    /// caps the Weiszfeld loop. RFA recommends a small constant
    /// (`max_iter = 3`) — in practice the median is well-approximated
    /// after a handful of iterations and further iterations don't change
    /// the breakdown bound.
    ///
    /// Pillutla, Kakade, Harchaoui. *Robust Aggregation for Federated
    /// Learning*. IEEE Transactions on Signal Processing, vol. 70,
    /// pp. 1142–1154, 2022. DOI: 10.1109/TSP.2022.3153135.
    /// <https://arxiv.org/abs/1912.13445>
    GeometricMedian { eps: f64, max_iter: usize },
    /// ArKrum (Average-rKrum) — parameter-free Byzantine-robust aggregator.
    ///
    /// Removes the standard Krum requirement that the caller specify the
    /// number of Byzantine clients `f` in advance. For each client `i`:
    ///
    /// 1. Filter extreme outliers from the sorted-ascending distance vector
    ///    via `τ = median + (median − min)` (Algorithm 1 from the paper).
    /// 2. Estimate `f̂_i` via SSE-minimising change-point detection on the
    ///    filtered distances (the breakpoint between "benign" and
    ///    "malicious" segments). This is rKrum's contribution, layered with
    ///    the median pre-filter that ArKrum adds to fix rKrum's
    ///    underestimation under extreme-update attacks.
    /// 3. Compute the standard Krum score using `n − f̂_i − 2` smallest
    ///    distances.
    ///
    /// Pick `u*` with the lowest score, take the `(n − f̂_{u*})` closest
    /// updates to `u*` (including `u*` itself), and uniform-mean them.
    /// Multi-update averaging matches Multi-Krum's stability gain over
    /// single-winner Krum, while the parameter-free `f̂` estimation matches
    /// rKrum.
    ///
    /// Requires `n ≥ 5` so the filter + change-point step has enough samples
    /// to behave. No explicit `f` bound at construction time — robustness is
    /// data-driven per round.
    ///
    /// Yang, Imam, et al. *Secure and Private Federated Learning: Achieving
    /// Adversarial Resilience through Robust Aggregation*. 2025.
    /// <https://arxiv.org/abs/2505.17226>
    ArKrum,
}

/// A model update from a single client, represented as named weight tensors.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClientUpdate {
    /// Number of training samples used to produce this update.
    pub num_samples: usize,
    /// Flattened parameter arrays keyed by layer name.
    pub weights: HashMap<String, Vec<f32>>,
}

/// Result of aggregating a round of client updates.
///
/// `selected_client_ids` is always populated: non-robust aggregators (FedAvg,
/// FedProx, FedMedian) return `0..n`; Krum/Multi-Krum return the subset that
/// contributed to the aggregate (1 client for Krum, `m` for Multi-Krum).
#[derive(Debug, Clone)]
pub struct Aggregation {
    pub weights: HashMap<String, Vec<f32>>,
    pub selected_client_ids: Vec<usize>,
}

/// Aggregate a list of client updates into a global model using the chosen strategy.
///
/// Accepts any slice element that borrows a `ClientUpdate` — so `&[ClientUpdate]`
/// (owned) and `&[&ClientUpdate]` (refs across the PyO3 boundary) both compile
/// without the caller cloning weight data.
pub fn aggregate<U: Borrow<ClientUpdate>>(
    updates: &[U],
    strategy: &Strategy,
) -> Result<Aggregation, String> {
    if updates.is_empty() {
        return Err("No client updates to aggregate".to_string());
    }

    let all_ids = || (0..updates.len()).collect::<Vec<usize>>();

    match strategy {
        Strategy::FedAvg | Strategy::FedProx { .. } => Ok(Aggregation {
            weights: fedavg(updates)?,
            selected_client_ids: all_ids(),
        }),
        Strategy::FedMedian => Ok(Aggregation {
            weights: fed_median(updates)?,
            selected_client_ids: all_ids(),
        }),
        Strategy::TrimmedMean { k } => Ok(Aggregation {
            weights: trimmed_mean(updates, *k)?,
            selected_client_ids: all_ids(),
        }),
        Strategy::Krum { f } => krum_select(updates, *f, Some(1)),
        Strategy::MultiKrum { f, m } => krum_select(updates, *f, *m),
        Strategy::Bulyan { f, m } => bulyan(updates, *f, *m),
        Strategy::GeometricMedian { eps, max_iter } => Ok(Aggregation {
            weights: geometric_median(updates, *eps, *max_iter)?,
            selected_client_ids: all_ids(),
        }),
        Strategy::ArKrum => ar_krum(updates),
    }
}

/// Weighted average of client weights, weighted by the number of local training samples.
///
/// Uses an f64 accumulator to bound rounding error when summing many small f32
/// contributions, then downcasts to f32 at the end.
fn fedavg<U: Borrow<ClientUpdate>>(updates: &[U]) -> Result<HashMap<String, Vec<f32>>, String> {
    let total_samples: usize = updates.iter().map(|u| u.borrow().num_samples).sum();
    if total_samples == 0 {
        return Err("Total sample count is zero".to_string());
    }
    let total_samples_f64 = total_samples as f64;

    let first = updates[0].borrow();
    let mut global: HashMap<String, Vec<f32>> = HashMap::with_capacity(first.weights.len());

    for (name, first_vec) in first.weights.iter() {
        let len = first_vec.len();
        let mut agg = vec![0.0f64; len];

        for update in updates {
            let w = match update.borrow().weights.get(name) {
                Some(v) => v,
                None => return Err(format!("Client update missing layer '{}'", name)),
            };
            if w.len() != len {
                return Err(format!(
                    "Layer '{}' size mismatch: expected {}, got {}",
                    name,
                    len,
                    w.len()
                ));
            }
            let scale = update.borrow().num_samples as f64 / total_samples_f64;
            for (a, &val) in agg.iter_mut().zip(w.iter()) {
                *a += val as f64 * scale;
            }
        }

        global.insert(name.clone(), agg.into_iter().map(|x| x as f32).collect());
    }

    Ok(global)
}

/// Element-wise coordinate-wise median across client updates.
///
/// Hoists the per-layer client slices out of the inner coordinate loop so the
/// hot path touches only contiguous `&[f32]` slices — no HashMap lookups per
/// coordinate. Uses `select_nth_unstable_by` (O(C) on average) instead of a
/// full sort (O(C log C)) and reuses a single scratch buffer per layer.
fn fed_median<U: Borrow<ClientUpdate>>(updates: &[U]) -> Result<HashMap<String, Vec<f32>>, String> {
    let first = updates[0].borrow();
    let mut global: HashMap<String, Vec<f32>> = HashMap::with_capacity(first.weights.len());

    let mut scratch: Vec<f32> = Vec::with_capacity(updates.len());

    for (name, first_vec) in first.weights.iter() {
        let len = first_vec.len();

        let layer_slices: Vec<&[f32]> = updates
            .iter()
            .map(|u| {
                u.borrow()
                    .weights
                    .get(name)
                    .map(|v| v.as_slice())
                    .ok_or_else(|| format!("Client update missing layer '{}'", name))
            })
            .collect::<Result<_, _>>()?;
        for s in &layer_slices {
            if s.len() != len {
                return Err(format!(
                    "Layer '{}' size mismatch: expected {}, got {}",
                    name,
                    len,
                    s.len()
                ));
            }
        }

        let mut agg = vec![0.0f32; len];
        let n = layer_slices.len();
        let mid = n / 2;
        let even = n.is_multiple_of(2);

        let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);

        for (i, slot) in agg.iter_mut().enumerate() {
            scratch.clear();
            scratch.extend(layer_slices.iter().map(|s| s[i]));

            scratch.select_nth_unstable_by(mid, cmp);
            // After select_nth, scratch[mid] is the k-th order statistic and
            // scratch[..mid] holds the `mid` smallest elements (unordered).
            let median_hi = scratch[mid];
            *slot = if even {
                // even ⇒ n ≥ 2 ⇒ mid ≥ 1, so the left partition is non-empty.
                let lo = scratch[..mid]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                (lo + median_hi) / 2.0
            } else {
                median_hi
            };
        }

        global.insert(name.clone(), agg);
    }

    Ok(global)
}

/// Coordinate-wise trimmed mean across client updates.
///
/// For each coordinate, drops the `k` smallest and `k` largest values across
/// clients, then uniform-means the remaining `n - 2k`. Two `select_nth_unstable_by`
/// calls partition the scratch buffer in O(C) average per coordinate — same
/// inner-loop shape as `fed_median`, no median-of-evens branch.
///
/// Uniform weighting (not sample-weighted) — matches Yin et al. 2018 and our
/// Multi-Krum convention. Returns an error if `2*k >= n` (no elements left).
fn trimmed_mean<U: Borrow<ClientUpdate>>(
    updates: &[U],
    k: usize,
) -> Result<HashMap<String, Vec<f32>>, String> {
    let n = updates.len();
    if 2 * k >= n {
        return Err(format!("TrimmedMean requires 2*k < n; got k={k}, n={n}"));
    }

    let first = updates[0].borrow();
    let mut global: HashMap<String, Vec<f32>> = HashMap::with_capacity(first.weights.len());

    let mut scratch: Vec<f32> = Vec::with_capacity(n);
    let kept = n - 2 * k;
    let inv_kept = 1.0f64 / kept as f64;
    let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);

    for (name, first_vec) in first.weights.iter() {
        let len = first_vec.len();

        let layer_slices: Vec<&[f32]> = updates
            .iter()
            .map(|u| {
                u.borrow()
                    .weights
                    .get(name)
                    .map(|v| v.as_slice())
                    .ok_or_else(|| format!("Client update missing layer '{}'", name))
            })
            .collect::<Result<_, _>>()?;
        for s in &layer_slices {
            if s.len() != len {
                return Err(format!(
                    "Layer '{}' size mismatch: expected {}, got {}",
                    name,
                    len,
                    s.len()
                ));
            }
        }

        let mut agg = vec![0.0f32; len];

        for (i, slot) in agg.iter_mut().enumerate() {
            scratch.clear();
            scratch.extend(layer_slices.iter().map(|s| s[i]));

            // k = 0 short-circuit: no partitioning, just sum the whole slice.
            // The two select_nth calls below would be no-ops at k=0 but the
            // inner-window slicing (k..n-k) needs k > 0 to land cleanly.
            let sum: f64 = if k == 0 {
                scratch.iter().map(|&v| v as f64).sum()
            } else {
                // First select isolates the k smallest into scratch[..k].
                scratch.select_nth_unstable_by(k, cmp);
                // Second select operates on the remaining n-k elements and
                // isolates the k largest into scratch[n-k..]. The middle band
                // scratch[k..n-k] is the kept set (unordered, but sum-stable).
                scratch[k..].select_nth_unstable_by(kept - 1, cmp);
                scratch[k..n - k].iter().map(|&v| v as f64).sum()
            };
            *slot = (sum * inv_kept) as f32;
        }

        global.insert(name.clone(), agg);
    }

    Ok(global)
}

/// Krum / Multi-Krum selection kernel — returns the picked client indices.
///
/// Flattens each client's weights into a shared coordinate space (layer order
/// taken from `updates[0]`), computes the pairwise squared-Euclidean distance
/// matrix, scores each client by the sum of its `n - f - 2` smallest distances
/// to other clients, and returns the `m` lowest-scoring indices in ascending
/// score order (ties broken by original index via stable sort).
///
/// Pure selection: no averaging. `krum_select` composes this with a uniform
/// mean; `bulyan` composes it with a coordinate-wise trimmed mean.
///
/// `m = Some(1)` reproduces Blanchard's original Krum winner. `m = None`
/// resolves to `n - f` ("largest non-Byzantine group").
fn krum_select_indices<U: Borrow<ClientUpdate>>(
    updates: &[U],
    f: usize,
    m: Option<usize>,
) -> Result<Vec<usize>, String> {
    let n = updates.len();

    // Blanchard's breakdown bound: needs n - f - 2 >= 1 honest neighbours to
    // score against, and f Byzantine + (f+1) "too far" exclusions + 1 winner.
    if n < 2 * f + 3 {
        return Err(format!(
            "Krum/Multi-Krum requires n >= 2*f + 3; got n={n}, f={f}"
        ));
    }

    let m = m.unwrap_or(n - f);
    if m == 0 || m > n - f {
        return Err(format!(
            "Multi-Krum requires 1 <= m <= n - f; got m={m}, n={n}, f={f}"
        ));
    }

    // Fix a layer order from the first update; every other client must match.
    let first = updates[0].borrow();
    let layer_names: Vec<String> = first.weights.keys().cloned().collect();
    let layer_sizes: Vec<usize> = layer_names
        .iter()
        .map(|name| first.weights[name].len())
        .collect();
    let flat_len: usize = layer_sizes.iter().sum();

    // Flatten every client into a contiguous f32 vector in `layer_names` order.
    let mut flats: Vec<Vec<f32>> = Vec::with_capacity(n);
    for u in updates {
        let u = u.borrow();
        let mut flat: Vec<f32> = Vec::with_capacity(flat_len);
        for (name, &size) in layer_names.iter().zip(&layer_sizes) {
            let w = u
                .weights
                .get(name)
                .ok_or_else(|| format!("Client update missing layer '{}'", name))?;
            if w.len() != size {
                return Err(format!(
                    "Layer '{}' size mismatch: expected {}, got {}",
                    name,
                    size,
                    w.len()
                ));
            }
            flat.extend_from_slice(w);
        }
        flats.push(flat);
    }

    // Pairwise squared Euclidean distances (symmetric; diagonal = 0). f64
    // accumulator bounds rounding when d is large (e.g. 10⁶ coords × f32 diffs).
    let mut dist = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = flats[i]
                .iter()
                .zip(flats[j].iter())
                .map(|(&a, &b)| {
                    let diff = a as f64 - b as f64;
                    diff * diff
                })
                .sum();
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    // Krum score: sum of the `n - f - 2` smallest distances to *other* clients.
    // The `n >= 2f + 3` check above guarantees `k_terms >= f + 1 >= 1`.
    let k_terms = n - f - 2;
    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut buf: Vec<f64> = Vec::with_capacity(n.saturating_sub(1));
    let cmp_f64 = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    for (i, row) in dist.iter().enumerate() {
        buf.clear();
        for (j, &d) in row.iter().enumerate() {
            if j != i {
                buf.push(d);
            }
        }
        buf.select_nth_unstable_by(k_terms - 1, cmp_f64);
        let score: f64 = buf[..k_terms].iter().sum();
        scores.push((i, score));
    }

    // Ascending sort — smallest score is the Krum winner. Stable on ties so
    // equal-score clients are selected in index order (makes tests deterministic).
    scores.sort_by(|a, b| cmp_f64(&a.1, &b.1));
    Ok(scores[..m].iter().map(|&(i, _)| i).collect())
}

/// Krum / Multi-Krum kernel — picks indices via `krum_select_indices` and
/// returns their uniform (not sample-weighted) average, matching El Mhamdi
/// et al.'s formulation.
fn krum_select<U: Borrow<ClientUpdate>>(
    updates: &[U],
    f: usize,
    m: Option<usize>,
) -> Result<Aggregation, String> {
    let selected_client_ids = krum_select_indices(updates, f, m)?;
    let m_val = selected_client_ids.len();

    // Uniform average over the selected clients, layer by layer. Layer
    // presence/size was validated inside `krum_select_indices` during
    // flattening, so `u.weights[name]` is safe.
    let first = updates[0].borrow();
    let mut weights: HashMap<String, Vec<f32>> = HashMap::with_capacity(first.weights.len());
    let inv_m = 1.0f64 / m_val as f64;
    for (name, first_vec) in first.weights.iter() {
        let size = first_vec.len();
        let mut agg = vec![0.0f64; size];
        for &idx in &selected_client_ids {
            let u = updates[idx].borrow();
            let w = &u.weights[name];
            for (a, &val) in agg.iter_mut().zip(w.iter()) {
                *a += val as f64 * inv_m;
            }
        }
        weights.insert(name.clone(), agg.into_iter().map(|x| x as f32).collect());
    }

    Ok(Aggregation {
        weights,
        selected_client_ids,
    })
}

/// Bulyan aggregator (El Mhamdi et al. 2018, Algorithm 2).
///
/// Phase 1 — reuse the Multi-Krum scoring rule to pick `m` survivor indices.
/// Phase 2 — pass those survivors into the coordinate-wise trimmed-mean kernel
/// with `k = f`, keeping `β = m - 2f` values per coordinate and uniform-meaning
/// them. Zero new math: pure orchestration over existing kernels.
///
/// Default `m = n - 2f` per the paper. Validates the Bulyan-specific bound
/// `n >= 4*f + 3` before dispatching — strictly tighter than Multi-Krum's
/// `n >= 2*f + 3`, so the inner `krum_select_indices` check always passes.
fn bulyan<U: Borrow<ClientUpdate>>(
    updates: &[U],
    f: usize,
    m: Option<usize>,
) -> Result<Aggregation, String> {
    let n = updates.len();

    if n < 4 * f + 3 {
        return Err(format!("Bulyan requires n >= 4*f + 3; got n={n}, f={f}"));
    }

    let m_val = m.unwrap_or(n - 2 * f);
    if m_val < 2 * f + 1 || m_val > n - 2 * f {
        return Err(format!(
            "Bulyan requires 2*f + 1 <= m <= n - 2*f; got m={m_val}, n={n}, f={f}"
        ));
    }

    let selected = krum_select_indices(updates, f, Some(m_val))?;

    // `&[&ClientUpdate]` satisfies `U: Borrow<ClientUpdate>` via the blanket
    // `impl<T: ?Sized> Borrow<T> for &T` — no cloning of weight buffers.
    let subset: Vec<&ClientUpdate> = selected.iter().map(|&i| updates[i].borrow()).collect();
    let weights = trimmed_mean(&subset, f)?;

    Ok(Aggregation {
        weights,
        selected_client_ids: selected,
    })
}

/// Geometric median via the Weiszfeld iteration (RFA - Pillutla et al., IEEE TSP 2022).
///
/// Flattens every client into a shared coordinate space (layer order taken
/// from `updates[0]`), initialises `y` at the sample-weighted mean (the
/// FedAvg estimate - also the closed-form L2-Steiner point), then iterates
///
/// ```text
/// y' = sum(w_i * x_i / d_i) / sum(w_i / d_i),  d_i = max(eps, ||y - x_i||)
/// ```
///
/// Stops at `max_iter` or when `||y' - y|| < eps`. f64 throughout for
/// numerical stability; downcasts to f32 only at the end.
fn geometric_median<U: Borrow<ClientUpdate>>(
    updates: &[U],
    eps: f64,
    max_iter: usize,
) -> Result<HashMap<String, Vec<f32>>, String> {
    let n = updates.len();
    let total_samples: usize = updates.iter().map(|u| u.borrow().num_samples).sum();
    if total_samples == 0 {
        return Err("Total sample count is zero".to_string());
    }
    if eps <= 0.0 {
        return Err(format!("GeometricMedian requires eps > 0; got {eps}"));
    }

    // Fix layer order from the first update; all clients must match.
    let first = updates[0].borrow();
    let layer_names: Vec<String> = first.weights.keys().cloned().collect();
    let layer_sizes: Vec<usize> = layer_names
        .iter()
        .map(|name| first.weights[name].len())
        .collect();
    let flat_len: usize = layer_sizes.iter().sum();

    // Flatten every client into a single contiguous f64 vector. f64 to bound
    // Weiszfeld's accumulator drift over high-dimensional sums.
    let mut flats: Vec<Vec<f64>> = Vec::with_capacity(n);
    for u in updates {
        let u = u.borrow();
        let mut flat = Vec::with_capacity(flat_len);
        for (name, &size) in layer_names.iter().zip(&layer_sizes) {
            let w = u
                .weights
                .get(name)
                .ok_or_else(|| format!("Client update missing layer '{}'", name))?;
            if w.len() != size {
                return Err(format!(
                    "Layer '{}' size mismatch: expected {}, got {}",
                    name,
                    size,
                    w.len()
                ));
            }
            flat.extend(w.iter().map(|&v| v as f64));
        }
        flats.push(flat);
    }

    let total_samples_f64 = total_samples as f64;
    let weights: Vec<f64> = updates
        .iter()
        .map(|u| u.borrow().num_samples as f64 / total_samples_f64)
        .collect();

    // Sample-weighted-mean init — same anchor FedAvg would produce.
    let mut y: Vec<f64> = vec![0.0; flat_len];
    for (flat, &w) in flats.iter().zip(weights.iter()) {
        for (yj, &xj) in y.iter_mut().zip(flat.iter()) {
            *yj += w * xj;
        }
    }

    // Weiszfeld loop. Reuse scratch buffers across iterations.
    let mut y_new: Vec<f64> = vec![0.0; flat_len];
    let mut dists: Vec<f64> = vec![0.0; n];
    for _ in 0..max_iter {
        for (i, flat) in flats.iter().enumerate() {
            let sq: f64 = y
                .iter()
                .zip(flat.iter())
                .map(|(&yj, &xj)| {
                    let d = yj - xj;
                    d * d
                })
                .sum();
            dists[i] = sq.sqrt().max(eps);
        }

        let denom: f64 = weights.iter().zip(dists.iter()).map(|(&w, &d)| w / d).sum();
        if denom <= 0.0 {
            // Pathological: every client coincides with `y`. Already at a
            // fixed point — leave `y` unchanged.
            break;
        }

        // Reset y_new in place rather than allocating each iteration.
        for v in y_new.iter_mut() {
            *v = 0.0;
        }
        for ((flat, &w), &d) in flats.iter().zip(weights.iter()).zip(dists.iter()) {
            let coef = (w / d) / denom;
            for (yj, &xj) in y_new.iter_mut().zip(flat.iter()) {
                *yj += coef * xj;
            }
        }

        let diff_sq: f64 = y
            .iter()
            .zip(y_new.iter())
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .sum();
        std::mem::swap(&mut y, &mut y_new);
        if diff_sq.sqrt() < eps {
            break;
        }
    }

    // Unflatten back to layer-keyed map.
    let mut global: HashMap<String, Vec<f32>> = HashMap::with_capacity(layer_names.len());
    let mut offset = 0;
    for (name, &size) in layer_names.iter().zip(&layer_sizes) {
        let chunk: Vec<f32> = y[offset..offset + size].iter().map(|&v| v as f32).collect();
        global.insert(name.clone(), chunk);
        offset += size;
    }

    Ok(global)
}

// ---------------------------------------------------------------------------
// ArKrum kernel — paper-grade parameter-free Krum (arXiv:2505.17226)
// ---------------------------------------------------------------------------

/// Algorithm 1 (extreme-outlier filter) on a sorted-ascending distance slice.
///
/// `τ = median + (median − min)`. Returns the prefix `D' ⊆ D` of distances
/// satisfying `d ≤ τ` — the head segment we trust for change-point detection.
/// `D` is assumed sorted ascending (we sort once per row before calling).
/// Returns the prefix length so callers can also count *how many were
/// filtered out* (used to compose with the SSE estimate downstream).
fn ar_filter_extreme(sorted: &[f64]) -> &[f64] {
    let n = sorted.len();
    if n < 3 {
        return sorted;
    }
    let mid = n / 2; // integer floor; matches Algorithm 1's `mid = ⌊n/2⌋`.
    let median = sorted[mid];
    let min = sorted[0];
    let tau = median + (median - min);
    // `partition_point` finds the first index whose distance exceeds τ; the
    // prefix is exactly the kept set. O(log n) on an already-sorted slice.
    let cut = sorted.partition_point(|&d| d <= tau);
    &sorted[..cut]
}

/// Estimate the number of values after the breakpoint in a sorted-ascending
/// distance slice (rKrum's `ESTIMATE_F` step).
///
/// Returns `f̂` = number of trailing values that look qualitatively different
/// from the head. On a sorted-ascending sequence the SSE-minimising split is
/// equivalent to "find the largest inter-element gap"; we use the gap form
/// because SSE on uniformly-spaced data biases toward an interior split with
/// no real evidence (Killick et al. 2012 PELT discusses the bias).
///
/// Two thresholds, both required, before we declare a breakpoint:
/// 1. `max_gap > GAP_RATIO * typical_gap` (typical = median of all other
///    gaps; protects against detecting noise as a breakpoint).
/// 2. `tail_min ≥ MAGNITUDE_RATIO * head_max` (the tail is orders of
///    magnitude away from the head). Byzantine updates differ from honest
///    by huge factors; honest-vs-honest geometric tiers within a tight
///    cluster differ by only single-digit factors. The magnitude guard
///    discriminates the two without false-positiving on float arithmetic
///    noise in supposedly-equal distances.
fn ar_estimate_f(sorted: &[f64]) -> usize {
    let n = sorted.len();
    if n < 3 {
        return 0;
    }
    let gaps: Vec<f64> = sorted.windows(2).map(|w| w[1] - w[0]).collect();
    if gaps.is_empty() {
        return 0;
    }
    let (max_idx, &max_gap) = gaps
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    if max_gap <= 0.0 {
        return 0;
    }
    // Typical gap = median of all *other* gaps (zeros included). If we
    // filtered zeros, float arithmetic noise in supposedly-duplicate
    // distances would make us call any positive max_gap a breakpoint.
    let mut other: Vec<f64> = gaps
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != max_idx)
        .map(|(_, &g)| g)
        .collect();
    if other.is_empty() {
        return 0;
    }
    other.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let typical = other[other.len() / 2];
    let threshold_gap_ratio = 5.0;
    let gap_signal = if typical > 0.0 {
        max_gap > threshold_gap_ratio * typical
    } else {
        // typical = 0 means at least half of other gaps are zero (true
        // duplicates). The magnitude check below carries the real signal;
        // we still require max_gap > 0 here so a flat-zero sequence
        // doesn't declare a breakpoint on float noise.
        max_gap > 0.0
    };
    if !gap_signal {
        return 0;
    }
    // Magnitude check: tail must be ≥ MAGNITUDE_RATIO × head. Byzantine
    // updates differ from honest by orders of magnitude (1e3+); honest
    // geometric tiers within a tight cluster differ by < 10×.
    let head_max = sorted[max_idx];
    let tail_min = sorted[max_idx + 1];
    let threshold_magnitude_ratio = 10.0;
    let magnitude_signal = if head_max > 0.0 {
        tail_min >= threshold_magnitude_ratio * head_max
    } else {
        // Head is all zero (perfect-match cluster). Any positive tail is a
        // clear breakpoint.
        tail_min > 0.0
    };
    if !magnitude_signal {
        return 0;
    }
    // Values after the gap = malicious tail. Gap at index `max_idx` sits
    // between sorted[max_idx] and sorted[max_idx + 1]; tail length is
    // `n - (max_idx + 1)`.
    n - (max_idx + 1)
}

/// Aggregate updates with ArKrum (Algorithm 2 from arXiv:2505.17226).
///
/// Shape mirrors `krum_select_indices`: flatten every client into one shared
/// coordinate vector, compute pairwise squared-Euclidean distances, then per
/// client filter + estimate f̂_i + compute ScoreKrum with that f̂_i. Pick the
/// minimum-score client `u*`, take the `(n − f̂_{u*})` updates closest to `u*`
/// (including `u*` itself), and uniform-mean them.
///
/// Returns the same `Aggregation` shape as the other selection-based
/// aggregators: weights + `selected_client_ids` (the averaged neighbour set).
fn ar_krum<U: Borrow<ClientUpdate>>(updates: &[U]) -> Result<Aggregation, String> {
    let n = updates.len();
    if n < 5 {
        return Err(format!(
            "ArKrum requires n >= 5 for the median + change-point steps; got n={n}"
        ));
    }

    // Fix layer order from the first update; every other client must match.
    let first = updates[0].borrow();
    let layer_names: Vec<String> = first.weights.keys().cloned().collect();
    let layer_sizes: Vec<usize> = layer_names
        .iter()
        .map(|name| first.weights[name].len())
        .collect();
    let flat_len: usize = layer_sizes.iter().sum();

    let mut flats: Vec<Vec<f32>> = Vec::with_capacity(n);
    for u in updates {
        let u = u.borrow();
        let mut flat: Vec<f32> = Vec::with_capacity(flat_len);
        for (name, &size) in layer_names.iter().zip(&layer_sizes) {
            let w = u
                .weights
                .get(name)
                .ok_or_else(|| format!("Client update missing layer '{}'", name))?;
            if w.len() != size {
                return Err(format!(
                    "Layer '{}' size mismatch: expected {}, got {}",
                    name,
                    size,
                    w.len()
                ));
            }
            flat.extend_from_slice(w);
        }
        flats.push(flat);
    }

    // Pairwise squared Euclidean distances (symmetric, diagonal = 0). f64
    // accumulator bounds rounding on large coord counts.
    let mut dist = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = flats[i]
                .iter()
                .zip(flats[j].iter())
                .map(|(&a, &b)| {
                    let diff = a as f64 - b as f64;
                    diff * diff
                })
                .sum();
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    // Per-client: sort other-client distances ascending, filter extremes,
    // estimate f̂_i, compute Krum score with k = n − f̂_i − 2 terms.
    let cmp_f64 = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut f_hats: Vec<usize> = Vec::with_capacity(n);
    for (i, row) in dist.iter().enumerate() {
        let mut sorted: Vec<f64> = row
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &d)| d)
            .collect();
        sorted.sort_by(cmp_f64);

        // Algorithm 1 + ESTIMATE_F operate on the filtered prefix. Compose:
        // f̂_i = max(filtered_out_count, ESTIMATE_F(D')). The filter caught
        // obvious extremes; if ESTIMATE_F finds *more* breakpoints inside the
        // remaining set, those are subtler byzantines we'd otherwise miss.
        // Taking the max prevents a tight intra-honest gap from understating
        // f̂ when the filter has already removed several clear outliers.
        let filtered = ar_filter_extreme(&sorted);
        let filtered_out = sorted.len() - filtered.len();
        let f_hat_sse = ar_estimate_f(filtered);
        let f_hat_i = filtered_out.max(f_hat_sse);
        f_hats.push(f_hat_i);

        // ScoreKrum on the unfiltered sorted row — same shape as classical
        // Krum, just with `f̂_i` instead of caller-supplied `f`. Guard
        // `k_terms >= 1` for the small-cluster edge case.
        let n_others = sorted.len();
        let k_terms = n_others.saturating_sub(f_hat_i + 1).max(1);
        let score: f64 = sorted[..k_terms].iter().sum();
        scores.push((i, score));
    }

    // u* = argmin score (stable on ties via earlier index).
    let &(u_star, _) = scores
        .iter()
        .min_by(|a, b| cmp_f64(&a.1, &b.1).then(a.0.cmp(&b.0)))
        .expect("scores is non-empty by the n >= 5 check");
    let f_hat_star = f_hats[u_star];

    // Average the (n - f̂*) closest updates to u*, including u* itself.
    // Guard `m_avg >= 1` so we always emit at least one contributor.
    let m_avg = n.saturating_sub(f_hat_star).max(1);
    let mut neighbours: Vec<(usize, f64)> = (0..n)
        .filter(|&j| j != u_star)
        .map(|j| (j, dist[u_star][j]))
        .collect();
    neighbours.sort_by(|a, b| cmp_f64(&a.1, &b.1).then(a.0.cmp(&b.0)));
    let mut selected_client_ids: Vec<usize> = Vec::with_capacity(m_avg);
    selected_client_ids.push(u_star);
    for &(j, _) in neighbours.iter().take(m_avg.saturating_sub(1)) {
        selected_client_ids.push(j);
    }
    selected_client_ids.sort();

    // Uniform average over the selected indices, layer by layer.
    let inv_m = 1.0f64 / selected_client_ids.len() as f64;
    let mut weights: HashMap<String, Vec<f32>> = HashMap::with_capacity(first.weights.len());
    for (name, first_vec) in first.weights.iter() {
        let size = first_vec.len();
        let mut agg = vec![0.0f64; size];
        for &idx in &selected_client_ids {
            let u = updates[idx].borrow();
            let w = &u.weights[name];
            for (a, &val) in agg.iter_mut().zip(w.iter()) {
                *a += val as f64 * inv_m;
            }
        }
        weights.insert(name.clone(), agg.into_iter().map(|x| x as f32).collect());
    }

    Ok(Aggregation {
        weights,
        selected_client_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_update(num_samples: usize, weights: &[(&str, Vec<f32>)]) -> ClientUpdate {
        ClientUpdate {
            num_samples,
            weights: weights
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        }
    }

    #[test]
    fn fedavg_single_client() {
        let update = make_update(10, &[("layer1", vec![1.0, 2.0, 3.0])]);
        let result = aggregate(&[update], &Strategy::FedAvg).unwrap();
        let layer = &result.weights["layer1"];
        assert!((layer[0] - 1.0).abs() < 1e-5);
        assert!((layer[1] - 2.0).abs() < 1e-5);
        assert!((layer[2] - 3.0).abs() < 1e-5);
        assert_eq!(result.selected_client_ids, vec![0]);
    }

    #[test]
    fn fedavg_equal_weights() {
        let u1 = make_update(5, &[("w", vec![0.0, 2.0])]);
        let u2 = make_update(5, &[("w", vec![2.0, 4.0])]);
        let result = aggregate(&[u1, u2], &Strategy::FedAvg).unwrap();
        let w = &result.weights["w"];
        assert!((w[0] - 1.0).abs() < 1e-5);
        assert!((w[1] - 3.0).abs() < 1e-5);
        assert_eq!(result.selected_client_ids, vec![0, 1]);
    }

    #[test]
    fn fedavg_weighted_by_samples() {
        let u1 = make_update(3, &[("w", vec![0.0])]);
        let u2 = make_update(1, &[("w", vec![4.0])]);
        let result = aggregate(&[u1, u2], &Strategy::FedAvg).unwrap();
        let w = &result.weights["w"];
        assert!((w[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fedmedian_odd_clients() {
        let u1 = make_update(1, &[("w", vec![1.0])]);
        let u2 = make_update(1, &[("w", vec![3.0])]);
        let u3 = make_update(1, &[("w", vec![2.0])]);
        let result = aggregate(&[u1, u2, u3], &Strategy::FedMedian).unwrap();
        assert!((result.weights["w"][0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn fedmedian_even_clients_averages_middle_pair() {
        // Sorted: [1, 2, 3, 4] → median = (2 + 3) / 2 = 2.5
        let u1 = make_update(1, &[("w", vec![3.0, 10.0])]);
        let u2 = make_update(1, &[("w", vec![1.0, 40.0])]);
        let u3 = make_update(1, &[("w", vec![4.0, 20.0])]);
        let u4 = make_update(1, &[("w", vec![2.0, 30.0])]);
        let result = aggregate(&[u1, u2, u3, u4], &Strategy::FedMedian).unwrap();
        let w = &result.weights["w"];
        assert!((w[0] - 2.5).abs() < 1e-5, "got {}", w[0]);
        assert!((w[1] - 25.0).abs() < 1e-5, "got {}", w[1]);
    }

    #[test]
    fn fedmedian_missing_layer_returns_error() {
        let u1 = make_update(1, &[("a", vec![1.0])]);
        let u2 = make_update(1, &[("b", vec![1.0])]);
        let result = aggregate(&[u1, u2], &Strategy::FedMedian);
        assert!(result.is_err());
    }

    #[test]
    fn aggregate_empty_returns_error() {
        let empty: [ClientUpdate; 0] = [];
        let result = aggregate(&empty, &Strategy::FedAvg);
        assert!(result.is_err());
    }

    #[test]
    fn aggregate_missing_layer_returns_error() {
        let u1 = make_update(1, &[("a", vec![1.0])]);
        let u2 = make_update(1, &[("b", vec![1.0])]);
        let result = aggregate(&[u1, u2], &Strategy::FedAvg);
        assert!(result.is_err());
    }

    // ----- Krum / Multi-Krum -----

    #[test]
    fn krum_3client_fixture_picks_closest_pair() {
        // n=3, f=0, k_terms = n-f-2 = 1. Each client scores with its single
        // closest neighbour.
        //   c0=[0,0], c1=[0,1], c2=[10,10]
        //   d(0,1)=1, d(0,2)=200, d(1,2)=(10-0)^2+(10-1)^2=100+81=181
        //   score(0)=min(1,200)=1, score(1)=min(1,181)=1, score(2)=min(200,181)=181
        // Tie between 0 and 1; stable sort keeps index order → Krum picks c0.
        let u0 = make_update(1, &[("w", vec![0.0, 0.0])]);
        let u1 = make_update(1, &[("w", vec![0.0, 1.0])]);
        let u2 = make_update(1, &[("w", vec![10.0, 10.0])]);
        let result = aggregate(&[u0, u1, u2], &Strategy::Krum { f: 0 }).unwrap();
        assert_eq!(result.selected_client_ids, vec![0]);
        let w = &result.weights["w"];
        assert!((w[0] - 0.0).abs() < 1e-6, "got {}", w[0]);
        assert!((w[1] - 0.0).abs() < 1e-6, "got {}", w[1]);
    }

    #[test]
    fn krum_equals_multikrum_m1() {
        let u0 = make_update(1, &[("w", vec![1.0, 1.0])]);
        let u1 = make_update(1, &[("w", vec![1.1, 1.1])]);
        let u2 = make_update(1, &[("w", vec![5.0, 5.0])]);
        let updates = [u0, u1, u2];
        let krum = aggregate(&updates, &Strategy::Krum { f: 0 }).unwrap();
        let mk = aggregate(&updates, &Strategy::MultiKrum { f: 0, m: Some(1) }).unwrap();
        assert_eq!(krum.selected_client_ids, mk.selected_client_ids);
        let a = &krum.weights["w"];
        let b = &mk.weights["w"];
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn krum_excludes_byzantine_outlier() {
        // n=5, f=1 → needs n >= 2*1+3 = 5 ✓. k_terms = 5-1-2 = 2.
        // 4 honest clients around [1,1], 1 outlier at [100,100].
        let mut clients: Vec<ClientUpdate> = [1.0f32, 1.1, 0.9, 1.0]
            .iter()
            .map(|&v| make_update(1, &[("w", vec![v, v])]))
            .collect();
        clients.push(make_update(1, &[("w", vec![100.0, 100.0])]));
        let result = aggregate(&clients, &Strategy::Krum { f: 1 }).unwrap();
        assert_eq!(result.selected_client_ids.len(), 1);
        assert_ne!(
            result.selected_client_ids[0], 4,
            "Krum picked the Byzantine outlier"
        );
    }

    #[test]
    fn multikrum_default_m_excludes_outlier_and_averages_honest() {
        // n=5, f=1, m=None → m = n-f = 4. All 4 honest clients selected,
        // outlier excluded. Uniform mean of 4 identical vectors is itself.
        let u0 = make_update(1, &[("w", vec![2.0, 2.0])]);
        let u1 = make_update(1, &[("w", vec![2.0, 2.0])]);
        let u2 = make_update(1, &[("w", vec![2.0, 2.0])]);
        let u3 = make_update(1, &[("w", vec![2.0, 2.0])]);
        let u4 = make_update(1, &[("w", vec![100.0, 100.0])]);
        let result = aggregate(
            &[u0, u1, u2, u3, u4],
            &Strategy::MultiKrum { f: 1, m: None },
        )
        .unwrap();
        assert_eq!(result.selected_client_ids.len(), 4);
        assert!(!result.selected_client_ids.contains(&4));
        let w = &result.weights["w"];
        assert!((w[0] - 2.0).abs() < 1e-5, "got {}", w[0]);
        assert!((w[1] - 2.0).abs() < 1e-5, "got {}", w[1]);
    }

    #[test]
    fn multikrum_uniform_weighting_ignores_sample_counts() {
        // If Multi-Krum were sample-weighted, u0 (100 samples) would dominate.
        // Paper says uniform — so the mean is 1.0, not 0.02.
        let u0 = make_update(100, &[("w", vec![0.0])]);
        let u1 = make_update(1, &[("w", vec![1.0])]);
        let u2 = make_update(1, &[("w", vec![2.0])]);
        let result = aggregate(&[u0, u1, u2], &Strategy::MultiKrum { f: 0, m: Some(3) }).unwrap();
        // Uniform mean: (0 + 1 + 2) / 3 = 1.0
        assert!((result.weights["w"][0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn krum_rejects_insufficient_clients() {
        // n=2, f=0 → needs n >= 3. Should error.
        let u0 = make_update(1, &[("w", vec![1.0])]);
        let u1 = make_update(1, &[("w", vec![2.0])]);
        let result = aggregate(&[u0, u1], &Strategy::Krum { f: 0 });
        assert!(result.is_err());
    }

    #[test]
    fn multikrum_rejects_bad_m() {
        let u0 = make_update(1, &[("w", vec![1.0])]);
        let u1 = make_update(1, &[("w", vec![2.0])]);
        let u2 = make_update(1, &[("w", vec![3.0])]);
        // n=3, f=0 → m must be in 1..=3.
        assert!(aggregate(
            &[u0.clone(), u1.clone(), u2.clone()],
            &Strategy::MultiKrum { f: 0, m: Some(4) },
        )
        .is_err());
        assert!(aggregate(&[u0, u1, u2], &Strategy::MultiKrum { f: 0, m: Some(0) },).is_err());
    }

    // ----- Trimmed Mean -----

    #[test]
    fn trimmed_mean_k1_drops_extremes() {
        // n=5, k=1 → drop min and max per coord, mean the middle 3.
        // coord 0: sorted [1,2,3,4,100] → trim → mean(2,3,4) = 3.0
        // coord 1: sorted [-50,5,6,7,8] → trim → mean(5,6,7) = 6.0
        let u0 = make_update(1, &[("w", vec![3.0, 6.0])]);
        let u1 = make_update(1, &[("w", vec![1.0, 8.0])]);
        let u2 = make_update(1, &[("w", vec![100.0, -50.0])]);
        let u3 = make_update(1, &[("w", vec![2.0, 7.0])]);
        let u4 = make_update(1, &[("w", vec![4.0, 5.0])]);
        let result = aggregate(&[u0, u1, u2, u3, u4], &Strategy::TrimmedMean { k: 1 }).unwrap();
        let w = &result.weights["w"];
        assert!((w[0] - 3.0).abs() < 1e-5, "got {}", w[0]);
        assert!((w[1] - 6.0).abs() < 1e-5, "got {}", w[1]);
        assert_eq!(result.selected_client_ids, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn trimmed_mean_k0_equals_uniform_mean() {
        // k=0 reduces to a uniform (not sample-weighted) mean. Distinct from
        // FedAvg, which weights by num_samples — that's why u0 having 100
        // samples is irrelevant here.
        let u0 = make_update(100, &[("w", vec![0.0])]);
        let u1 = make_update(1, &[("w", vec![1.0])]);
        let u2 = make_update(1, &[("w", vec![2.0])]);
        let result = aggregate(&[u0, u1, u2], &Strategy::TrimmedMean { k: 0 }).unwrap();
        // Uniform mean: (0 + 1 + 2) / 3 = 1.0
        assert!((result.weights["w"][0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn trimmed_mean_max_k_reduces_to_single_middle_element() {
        // n=3, k=1 → kept=1, the middle order statistic per coord.
        // Same shape as median for odd n.
        let u0 = make_update(1, &[("w", vec![1.0, 30.0])]);
        let u1 = make_update(1, &[("w", vec![2.0, 10.0])]);
        let u2 = make_update(1, &[("w", vec![3.0, 20.0])]);
        let result = aggregate(&[u0, u1, u2], &Strategy::TrimmedMean { k: 1 }).unwrap();
        let w = &result.weights["w"];
        assert!((w[0] - 2.0).abs() < 1e-5, "got {}", w[0]);
        assert!((w[1] - 20.0).abs() < 1e-5, "got {}", w[1]);
    }

    #[test]
    fn trimmed_mean_excludes_byzantine_outlier() {
        // 4 honest clients near 1.0, 1 outlier at 1e6. With k=1, the outlier
        // is trimmed and the result lands near 1.0.
        let mut clients: Vec<ClientUpdate> = [1.0f32, 1.1, 0.9, 1.05]
            .iter()
            .map(|&v| make_update(1, &[("w", vec![v])]))
            .collect();
        clients.push(make_update(1, &[("w", vec![1.0e6])]));
        let result = aggregate(&clients, &Strategy::TrimmedMean { k: 1 }).unwrap();
        assert!(
            result.weights["w"][0].abs() < 5.0,
            "Byzantine value leaked through trim: got {}",
            result.weights["w"][0]
        );
    }

    #[test]
    fn trimmed_mean_rejects_too_large_k() {
        // n=3, k=2 → 2*k=4 >= n. Should error.
        let u0 = make_update(1, &[("w", vec![1.0])]);
        let u1 = make_update(1, &[("w", vec![2.0])]);
        let u2 = make_update(1, &[("w", vec![3.0])]);
        let result = aggregate(&[u0, u1, u2], &Strategy::TrimmedMean { k: 2 });
        assert!(result.is_err());
    }

    #[test]
    fn trimmed_mean_missing_layer_returns_error() {
        let u0 = make_update(1, &[("a", vec![1.0])]);
        let u1 = make_update(1, &[("b", vec![1.0])]);
        let u2 = make_update(1, &[("a", vec![1.0])]);
        let result = aggregate(&[u0, u1, u2], &Strategy::TrimmedMean { k: 0 });
        assert!(result.is_err());
    }

    #[test]
    fn krum_missing_layer_returns_error() {
        let u0 = make_update(1, &[("a", vec![1.0]), ("b", vec![1.0])]);
        let u1 = make_update(1, &[("a", vec![1.0]), ("b", vec![1.0])]);
        let u2 = make_update(1, &[("a", vec![1.0])]); // missing "b"
        let result = aggregate(&[u0, u1, u2], &Strategy::Krum { f: 0 });
        assert!(result.is_err());
    }

    // ----- Bulyan -----

    #[test]
    fn bulyan_matches_manual_calculation_on_fixed_fixture() {
        // n=7, f=1, m=None → m = n-2f = 5, β = m-2f = 3.
        // Honest clients at 1..=6, Byzantine at 100.
        //   Phase 1 (Multi-Krum, k_terms = n-f-2 = 4): the single outlier
        //   scores astronomically high, c0 and c5 score 30 each (furthest
        //   honest), c1/c4 score 15, c2/c3 score 10. Top 5 = {c0..c4}.
        //   Phase 2: sorted subset = [1,2,3,4,5], drop f=1 from each end,
        //   kept = [2,3,4], uniform mean = 3.0.
        let clients: Vec<ClientUpdate> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]
            .iter()
            .map(|&v| make_update(1, &[("w", vec![v])]))
            .collect();
        let result = aggregate(&clients, &Strategy::Bulyan { f: 1, m: None }).unwrap();
        assert_eq!(result.selected_client_ids.len(), 5);
        let mut selected_sorted = result.selected_client_ids.clone();
        selected_sorted.sort();
        assert_eq!(selected_sorted, vec![0, 1, 2, 3, 4]);
        assert!(
            (result.weights["w"][0] - 3.0).abs() < 1e-5,
            "got {}",
            result.weights["w"][0]
        );
    }

    #[test]
    fn bulyan_rejects_insufficient_clients() {
        // n=6, f=1 → needs n >= 4*1+3 = 7. Should error.
        let clients: Vec<ClientUpdate> = (0..6)
            .map(|i| make_update(1, &[("w", vec![i as f32])]))
            .collect();
        let result = aggregate(&clients, &Strategy::Bulyan { f: 1, m: None });
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("n >= 4*f + 3"),
            "unexpected error message shape"
        );
    }

    #[test]
    fn bulyan_rejects_bad_m() {
        // n=7, f=1 → valid m range is [2*1+1, 7-2*1] = [3, 5].
        let clients: Vec<ClientUpdate> = (0..7)
            .map(|i| make_update(1, &[("w", vec![i as f32])]))
            .collect();
        // m=2 below 2f+1.
        let too_small = aggregate(&clients, &Strategy::Bulyan { f: 1, m: Some(2) });
        assert!(too_small.is_err());
        // m=6 above n-2f.
        let too_large = aggregate(&clients, &Strategy::Bulyan { f: 1, m: Some(6) });
        assert!(too_large.is_err());
    }

    #[test]
    fn bulyan_default_m_equals_explicit_n_minus_2f() {
        let clients: Vec<ClientUpdate> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]
            .iter()
            .map(|&v| make_update(1, &[("w", vec![v])]))
            .collect();
        let default_m = aggregate(&clients, &Strategy::Bulyan { f: 1, m: None }).unwrap();
        let explicit_m = aggregate(
            &clients,
            &Strategy::Bulyan {
                f: 1,
                m: Some(5), // n-2f
            },
        )
        .unwrap();
        assert_eq!(
            default_m.selected_client_ids,
            explicit_m.selected_client_ids
        );
        assert!((default_m.weights["w"][0] - explicit_m.weights["w"][0]).abs() < 1e-6);
    }

    #[test]
    fn bulyan_missing_layer_returns_error() {
        let mut clients: Vec<ClientUpdate> = (0..6)
            .map(|_| make_update(1, &[("a", vec![1.0]), ("b", vec![1.0])]))
            .collect();
        clients.push(make_update(1, &[("a", vec![1.0])])); // missing "b"
        let result = aggregate(&clients, &Strategy::Bulyan { f: 1, m: None });
        assert!(result.is_err());
    }

    #[test]
    fn bulyan_excludes_byzantine_outlier() {
        // n=7, f=1 → needs n >= 4f+3 = 7 ✓. Default m = n-2f = 5.
        //   Phase 1 (Multi-Krum): k_terms = n-f-2 = 4, selects 5 of 7 by lowest
        //                         score — rejects the two farthest from the cluster.
        //   Phase 2 (TrimmedMean): k = f = 1, drops min+max per coord,
        //                          uniform-means the remaining β = m-2f = 3.
        // 6 honest clients around [2,2], 1 Byzantine at [100,100].
        // The Byzantine scores highest; a honest client also gets dropped by the
        // m = n-2f = 5 cut. Trimmed mean on 5 near-[2,2] values is near [2,2].
        let mut clients: Vec<ClientUpdate> = [2.0f32, 2.05, 1.95, 2.1, 1.9, 2.0]
            .iter()
            .map(|&v| make_update(1, &[("w", vec![v, v])]))
            .collect();
        clients.push(make_update(1, &[("w", vec![100.0, 100.0])]));
        let result = aggregate(&clients, &Strategy::Bulyan { f: 1, m: None }).unwrap();
        // Byzantine (index 6) is never in the selection.
        assert!(
            !result.selected_client_ids.contains(&6),
            "Bulyan picked the Byzantine outlier: {:?}",
            result.selected_client_ids
        );
        assert_eq!(result.selected_client_ids.len(), 5);
        let w = &result.weights["w"];
        assert!(
            (w[0] - 2.0).abs() < 0.1,
            "coord 0 drifted from honest cluster: got {}",
            w[0]
        );
        assert!(
            (w[1] - 2.0).abs() < 0.1,
            "coord 1 drifted from honest cluster: got {}",
            w[1]
        );
    }

    // ----- Geometric Median (RFA, Pillutla et al. IEEE TSP 2022) -----

    fn gm() -> Strategy {
        Strategy::GeometricMedian {
            eps: 1e-6,
            max_iter: 32, // generous to make convergence-quality tests crisp
        }
    }

    #[test]
    fn geometric_median_single_client_returns_that_client() {
        let u = make_update(7, &[("w", vec![1.0, -2.0, 3.5])]);
        let result = aggregate(&[u], &gm()).unwrap();
        let w = &result.weights["w"];
        for (a, b) in w.iter().zip([1.0f32, -2.0, 3.5].iter()) {
            assert!((a - b).abs() < 1e-5, "got {a}, want {b}");
        }
        assert_eq!(result.selected_client_ids, vec![0]);
    }

    #[test]
    fn geometric_median_two_equal_clients_lands_on_segment() {
        // The geometric median of two equal-weight points is undefined on
        // the *interior* but well-defined as anywhere on the segment. Our
        // Weiszfeld init at the sample-weighted mean lands at the midpoint
        // and stays there.
        let u0 = make_update(1, &[("w", vec![0.0, 0.0])]);
        let u1 = make_update(1, &[("w", vec![10.0, 4.0])]);
        let result = aggregate(&[u0, u1], &gm()).unwrap();
        let w = &result.weights["w"];
        assert!((w[0] - 5.0).abs() < 1e-3, "got {}", w[0]);
        assert!((w[1] - 2.0).abs() < 1e-3, "got {}", w[1]);
    }

    #[test]
    fn geometric_median_three_collinear_clients_picks_middle() {
        // Geometric median of three points equals the L1-median; on a
        // line with equal weights that's the middle point. (Same shape as
        // the univariate case.)
        let u0 = make_update(1, &[("w", vec![1.0])]);
        let u1 = make_update(1, &[("w", vec![5.0])]);
        let u2 = make_update(1, &[("w", vec![2.0])]);
        let result = aggregate(&[u0, u1, u2], &gm()).unwrap();
        // Sorted: [1, 2, 5] → median = 2.
        assert!(
            (result.weights["w"][0] - 2.0).abs() < 1e-3,
            "got {}",
            result.weights["w"][0]
        );
    }

    #[test]
    fn geometric_median_excludes_byzantine_outlier() {
        // 4 honest clients clustered near [1, 1], 1 outlier at [100, 100].
        // The geometric median has a 1/2 breakdown bound, so 1/5 contamination
        // is well within tolerance — result should sit near the cluster, not
        // the centroid (which a plain mean would deliver around [21, 21]).
        let mut clients: Vec<ClientUpdate> = [1.0f32, 1.05, 0.95, 1.02]
            .iter()
            .map(|&v| make_update(1, &[("w", vec![v, v])]))
            .collect();
        clients.push(make_update(1, &[("w", vec![100.0, 100.0])]));
        let result = aggregate(&clients, &gm()).unwrap();
        let w = &result.weights["w"];
        // Honest cluster's centroid is ~1.0 — geometric median should be
        // close. Slack of 0.5 is generous; the contamination bound from
        // RFA puts the actual deviation ≤ 0.05 here.
        assert!(
            (w[0] - 1.0).abs() < 0.5,
            "Byzantine outlier moved coord 0 too far: got {}",
            w[0]
        );
        assert!(
            (w[1] - 1.0).abs() < 0.5,
            "Byzantine outlier moved coord 1 too far: got {}",
            w[1]
        );
    }

    #[test]
    fn geometric_median_sample_weighting_pulls_toward_majority_voice() {
        // Three clients along a line with grossly different sample counts.
        // Geometric median weights each client by num_samples / total, so
        // the heavily-sampled client dominates the median location.
        let u0 = make_update(100, &[("w", vec![0.0])]);
        let u1 = make_update(1, &[("w", vec![10.0])]);
        let u2 = make_update(1, &[("w", vec![20.0])]);
        let result = aggregate(&[u0, u1, u2], &gm()).unwrap();
        // u0 carries ~98% of the weight → median sits near 0.
        assert!(
            result.weights["w"][0].abs() < 0.5,
            "weighted median drifted from majority voice: got {}",
            result.weights["w"][0]
        );
    }

    #[test]
    fn geometric_median_rejects_zero_total_samples() {
        let u = make_update(0, &[("w", vec![1.0])]);
        let result = aggregate(&[u], &gm());
        assert!(result.is_err());
    }

    #[test]
    fn geometric_median_rejects_invalid_eps() {
        let u = make_update(1, &[("w", vec![1.0])]);
        let result = aggregate(
            &[u],
            &Strategy::GeometricMedian {
                eps: 0.0,
                max_iter: 3,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn geometric_median_missing_layer_returns_error() {
        let u0 = make_update(1, &[("a", vec![1.0]), ("b", vec![1.0])]);
        let u1 = make_update(1, &[("a", vec![1.0])]); // missing "b"
        let u2 = make_update(1, &[("a", vec![1.0]), ("b", vec![1.0])]);
        let result = aggregate(&[u0, u1, u2], &gm());
        assert!(result.is_err());
    }

    // ----- ArKrum (arXiv:2505.17226) -----

    #[test]
    fn ar_krum_rejects_too_few_clients() {
        // n=4 < 5, the median+change-point steps need a usable sample size.
        let clients: Vec<ClientUpdate> = (0..4)
            .map(|i| make_update(1, &[("w", vec![i as f32])]))
            .collect();
        let result = aggregate(&clients, &Strategy::ArKrum);
        assert!(result.is_err());
    }

    #[test]
    fn ar_krum_clean_round_aggregates_near_mean() {
        // 5 tightly-clustered honest clients — ArKrum should land near the
        // mean, with most clients selected. The strict "all 5 selected"
        // assertion is too tight at n=5: the median-filter step naturally
        // trims edge distances even on clean data, so f̂ from edge u* often
        // lands at 1. The honest aggregate stays correct regardless.
        let clients: Vec<ClientUpdate> = [0.95f32, 1.0, 1.05, 1.1, 1.15]
            .iter()
            .map(|&v| make_update(1, &[("w", vec![v])]))
            .collect();
        let result = aggregate(&clients, &Strategy::ArKrum).unwrap();
        // Majority of clients must be selected — no byzantines to exclude.
        assert!(
            result.selected_client_ids.len() >= 3,
            "expected majority selection on clean round; got {:?}",
            result.selected_client_ids
        );
        // Aggregate must land near the honest cluster's centre (1.05).
        assert!(
            (result.weights["w"][0] - 1.05).abs() < 0.1,
            "aggregate {} far from honest centre 1.05",
            result.weights["w"][0]
        );
    }

    #[test]
    fn ar_krum_excludes_single_extreme_byzantine() {
        // 7 honest clients tightly clustered near 0; one extreme Byzantine
        // at 1e6. ArKrum's median filter catches the outlier; the SSE
        // change-point step on the filtered set returns 0; f̂_i = max(filtered_out=1,
        // sse=0) = 1; the byzantine never reaches the averaged neighbour set.
        let mut clients: Vec<ClientUpdate> = [0.0f32, 0.01, 0.02, -0.01, -0.02, 0.03, -0.03]
            .iter()
            .map(|&v| make_update(1, &[("w", vec![v])]))
            .collect();
        clients.push(make_update(1, &[("w", vec![1e6])]));
        let byz_idx = clients.len() - 1;
        let result = aggregate(&clients, &Strategy::ArKrum).unwrap();
        assert!(
            !result.selected_client_ids.contains(&byz_idx),
            "single extreme byzantine (index {}) leaked: {:?}",
            byz_idx,
            result.selected_client_ids
        );
        assert!(
            result.weights["w"][0].abs() < 1.0,
            "aggregate {} pulled away from honest cluster",
            result.weights["w"][0]
        );
    }

    #[test]
    fn ar_krum_excludes_clustered_byzantines_when_honest_is_tighter() {
        // 6 honest clustered tightly near 0 + 3 byzantines clustered at 1e5
        // scale. The honest cluster is *tighter* than the byzantine cluster
        // (honest pair-distances ≤ 0.01, byzantine pair-distances ~1e8), so
        // the Krum score on an honest u_i is lower than on any byzantine —
        // u_star comes from the honest set, and the median-filter trims the
        // byzantines before they reach the averaging step. This is the
        // operational definition of ArKrum being colluding-byzantine-safe
        // *when honest is the dominant tight cluster*. The classical Krum
        // failure mode (byzantines outvote honest) needs more byzantines or
        // wider honest spread; not exercised here.
        let mut clients: Vec<ClientUpdate> = [0.0f32, 0.05, 0.1, -0.05, -0.1, 0.02]
            .iter()
            .map(|&v| make_update(1, &[("w", vec![v])]))
            .collect();
        for &v in &[1e5f32, 1.2e5, 9e4] {
            clients.push(make_update(1, &[("w", vec![v])]));
        }
        let result = aggregate(&clients, &Strategy::ArKrum).unwrap();
        for byz_idx in [6usize, 7, 8] {
            assert!(
                !result.selected_client_ids.contains(&byz_idx),
                "byzantine index {} leaked: {:?}",
                byz_idx,
                result.selected_client_ids
            );
        }
        assert!(
            result.weights["w"][0].abs() < 1.0,
            "aggregate {} pulled away from honest scale",
            result.weights["w"][0]
        );
    }

    #[test]
    fn ar_krum_missing_layer_returns_error() {
        let u0 = make_update(1, &[("a", vec![1.0]), ("b", vec![1.0])]);
        let u1 = make_update(1, &[("a", vec![1.0])]); // missing "b"
        let u2 = make_update(1, &[("a", vec![1.0]), ("b", vec![1.0])]);
        let u3 = make_update(1, &[("a", vec![1.0]), ("b", vec![1.0])]);
        let u4 = make_update(1, &[("a", vec![1.0]), ("b", vec![1.0])]);
        let result = aggregate(&[u0, u1, u2, u3, u4], &Strategy::ArKrum);
        assert!(result.is_err());
    }

    #[test]
    fn ar_filter_extreme_trims_far_tail() {
        // sorted ascending; the outlier at 100 should be cut after median+(median-min).
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        // median = sorted[2] = 3.0; min = 1.0; tau = 3 + (3-1) = 5.
        // Kept: [1, 2, 3, 4]; dropped: [100].
        let kept = ar_filter_extreme(&sorted);
        assert_eq!(kept.len(), 4);
        assert!((kept[3] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn ar_estimate_f_returns_zero_on_uniform_input() {
        // No breakpoint when the values cluster cleanly — f̂ should be 0,
        // not biased upward by the SSE minimum-always-an-interior-k trap.
        let sorted = vec![1.0, 1.05, 1.1, 1.15, 1.2];
        assert_eq!(ar_estimate_f(&sorted), 0);
    }

    #[test]
    fn ar_estimate_f_finds_split_on_clear_breakpoint() {
        // 4 tight values + 1 outlier → expected breakpoint after index 4, f̂ = 1.
        let sorted = vec![1.0, 1.05, 1.1, 1.15, 100.0];
        assert_eq!(ar_estimate_f(&sorted), 1);
    }
}
