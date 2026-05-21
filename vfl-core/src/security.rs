use rand::RngExt;
use rand_distr::{Distribution, Normal};

/// Round-level attacks the orchestrator applies during a federated round.
///
/// These operate on weights and client rosters — the data the Rust core
/// actually sees. Data-pipeline attacks (label flipping, backdoor triggers,
/// token replacement) live on the Python side in `velocity.data_attacks`
/// because the Rust core never sees raw labels or input features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AttackType {
    /// Corrupt a fraction of client model weights.
    ModelPoisoning { intensity: f64 },
    /// Inject synthetic Byzantine clients that send random gradients.
    SybilNodes { count: usize },
    /// Gaussian noise added to the global model (data-agnostic perturbation).
    GaussianNoise { std_dev: f64 },
}

/// Result of running an attack simulation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AttackResult {
    pub attack_type: String,
    pub clients_affected: usize,
    pub severity: f64,
    pub description: String,
}

/// Simulate a model poisoning attack by corrupting a fraction of client weights.
///
/// `intensity` ∈ [0, 1] controls how many weights are flipped/zeroed.
pub fn simulate_model_poisoning(
    weights: &mut std::collections::HashMap<String, Vec<f32>>,
    intensity: f64,
) -> AttackResult {
    let mut rng = rand::rng();
    let mut total_params = 0usize;
    let mut corrupted = 0usize;

    for vals in weights.values_mut() {
        total_params += vals.len();
        for v in vals.iter_mut() {
            if rng.random::<f64>() < intensity {
                *v = -*v; // sign flip
                corrupted += 1;
            }
        }
    }

    AttackResult {
        attack_type: "model_poisoning".to_string(),
        clients_affected: 1,
        severity: corrupted as f64 / total_params.max(1) as f64,
        description: format!(
            "Model poisoning: {corrupted}/{total_params} parameters corrupted \
             (intensity={intensity:.2})"
        ),
    }
}

/// Simulate Sybil nodes by generating `count` random Byzantine client updates.
pub fn simulate_sybil_nodes(
    layer_shapes: &std::collections::HashMap<String, usize>,
    count: usize,
) -> (Vec<crate::strategy::ClientUpdate>, AttackResult) {
    let mut rng = rand::rng();
    let updates: Vec<crate::strategy::ClientUpdate> = (0..count)
        .map(|_| {
            let weights = layer_shapes
                .iter()
                .map(|(name, &len)| {
                    let random_weights: Vec<f32> =
                        (0..len).map(|_| rng.random_range(-10.0..10.0)).collect();
                    (name.clone(), random_weights)
                })
                .collect();
            crate::strategy::ClientUpdate {
                num_samples: 1,
                weights,
            }
        })
        .collect();

    let result = AttackResult {
        attack_type: "sybil_nodes".to_string(),
        clients_affected: count,
        severity: 1.0,
        description: format!(
            "Sybil attack: {count} Byzantine clients injected with random gradients"
        ),
    };

    (updates, result)
}

/// Add Gaussian noise to model weights (simulates inference-time perturbation).
pub fn simulate_gaussian_noise(
    weights: &mut std::collections::HashMap<String, Vec<f32>>,
    std_dev: f64,
) -> AttackResult {
    let mut rng = rand::rng();
    let normal = Normal::new(0.0f64, std_dev).expect("std_dev must be finite and non-negative");
    let mut total_params = 0usize;

    for vals in weights.values_mut() {
        total_params += vals.len();
        for v in vals.iter_mut() {
            let noise = normal.sample(&mut rng) as f32;
            *v += noise;
        }
    }

    AttackResult {
        attack_type: "gaussian_noise".to_string(),
        clients_affected: 0,
        severity: std_dev,
        description: format!(
            "Gaussian noise (σ={std_dev:.3}) applied to {total_params} parameters"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_weights() -> HashMap<String, Vec<f32>> {
        let mut m = HashMap::new();
        m.insert("layer1".to_string(), vec![1.0, 2.0, 3.0]);
        m
    }

    #[test]
    fn model_poisoning_high_intensity_corrupts_most() {
        let mut w = make_weights();
        let result = simulate_model_poisoning(&mut w, 1.0);
        // At intensity=1.0 every weight should be flipped
        assert_eq!(result.clients_affected, 1);
        let layer = &w["layer1"];
        assert!(layer[0] < 0.0); // 1.0 → -1.0
        assert!(layer[1] < 0.0);
        assert!(layer[2] < 0.0);
    }

    #[test]
    fn model_poisoning_zero_intensity_leaves_weights_unchanged() {
        let mut w = make_weights();
        simulate_model_poisoning(&mut w, 0.0);
        let layer = &w["layer1"];
        assert!((layer[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sybil_nodes_returns_correct_count() {
        let mut shapes = HashMap::new();
        shapes.insert("w".to_string(), 4);
        let (updates, result) = simulate_sybil_nodes(&shapes, 3);
        assert_eq!(updates.len(), 3);
        assert_eq!(result.clients_affected, 3);
    }

    #[test]
    fn gaussian_noise_changes_weights() {
        let mut w = make_weights();
        let original = w["layer1"].clone();
        simulate_gaussian_noise(&mut w, 5.0);
        // With std_dev=5 it is extremely unlikely all values are identical
        let changed = w["layer1"]
            .iter()
            .zip(original.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed);
    }
}
