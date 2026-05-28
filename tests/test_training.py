"""Tests for :mod:`velocity.training` — the DP-SGD client helper.

Unit tests run DP-SGD on a tiny in-memory dataset so they exercise the real
Opacus path (per-sample clipping + noise + RDP accounting) without a download
or a full training run. opacus + torch are optional; gate at import.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("opacus")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from velocity.training import dp_local_train  # noqa: E402

# Opacus emits benign experimentation notices (secure-RNG off; a torch
# full-backward-hook info message) on every DP step — filter so test output
# stays pristine without masking other warnings.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Secure RNG turned off"),
    pytest.mark.filterwarnings("ignore:Full backward hook is firing"),
]


def _tiny_setup(n: int = 64, in_dim: int = 8, classes: int = 3) -> tuple[nn.Module, DataLoader]:
    model = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, classes))
    x = torch.randn(n, 1, 2, 4)  # flattens to in_dim=8
    y = torch.randint(0, classes, (n,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    return model, loader


def test_returns_finite_positive_epsilon() -> None:
    model, loader = _tiny_setup()
    _, eps = dp_local_train(model, loader, noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5)
    assert eps > 0
    assert eps != float("inf")


def test_returns_trained_plain_module() -> None:
    model, loader = _tiny_setup()
    before = [p.detach().clone() for p in model.parameters()]

    trained, _ = dp_local_train(model, loader, noise_multiplier=0.5, max_grad_norm=1.0)

    # Caller gets a plain nn.Module with clean state_dict keys (no Opacus
    # GradSampleModule "_module." prefix), so it round-trips into a fresh model.
    assert isinstance(trained, nn.Module)
    sd = trained.state_dict()
    assert all("_module" not in k for k in sd)
    nn.Sequential(nn.Flatten(), nn.Linear(8, 3)).load_state_dict(sd)
    # Training actually moved the parameters.
    assert any(not torch.equal(b, a) for b, a in zip(before, trained.parameters(), strict=True))


def test_higher_noise_spends_less_epsilon() -> None:
    # Same steps + sample rate, more noise → stronger privacy → smaller epsilon.
    model_a, loader_a = _tiny_setup()
    model_b, loader_b = _tiny_setup()
    _, eps_low_noise = dp_local_train(model_a, loader_a, noise_multiplier=0.5, max_grad_norm=1.0)
    _, eps_high_noise = dp_local_train(model_b, loader_b, noise_multiplier=2.0, max_grad_norm=1.0)
    assert eps_high_noise < eps_low_noise
