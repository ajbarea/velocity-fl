"""Real federated training utilities for Velocity-FL.

The Rust core (`velocity._core.Orchestrator`) only sees flat layer weights —
it does not know about models, datasets, or losses. This module provides the
PyTorch-side glue that turns "I have N clients with local data" into a real
FedAvg run against the Rust aggregator, with honest per-round evaluation.

Torch is an optional dependency (``velocity-fl[torch]``); importing this
module without it raises a clear error.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "velocity.training requires PyTorch. Install with: pip install 'velocity-fl[torch]'"
    ) from exc


__all__ = [
    "ClientData",
    "dp_local_train",
    "evaluate",
    "layers_to_state_dict",
    "local_train",
    "state_dict_to_layers",
]


@dataclass
class ClientData:
    """One client's view of the federation: their local training loader and sample count."""

    loader: DataLoader
    num_samples: int


def state_dict_to_layers(state_dict: dict[str, Tensor]) -> dict[str, list[float]]:
    """Flatten a torch ``state_dict`` into the Rust core's ``{name: [f32]}`` shape."""
    return {name: tensor.detach().cpu().flatten().tolist() for name, tensor in state_dict.items()}


def layers_to_state_dict(layers: dict[str, Any], reference: dict[str, Tensor]) -> dict[str, Tensor]:
    """Inverse of ``state_dict_to_layers``: reshape flat weights back to tensor shapes.

    Accepts anything ``torch.tensor`` can swallow per layer: lists of float,
    ``numpy.ndarray``, ``tuple[float, ...]``, even another ``Tensor``. The
    Rust core's ``Orchestrator.global_weights()`` returns
    ``dict[str, ndarray[float32]]``, so a strict ``list[float]`` signature
    would be wrong at the actual boundary — ``Any`` reflects what the API
    really is.
    """
    return {
        name: torch.tensor(layers[name], dtype=ref.dtype).reshape(ref.shape)
        for name, ref in reference.items()
    }


def layer_shapes(state_dict: dict[str, Tensor]) -> dict[str, int]:
    """Flat element count per layer — what `Orchestrator.__init__(layer_shapes=...)` wants."""
    return {name: int(tensor.numel()) for name, tensor in state_dict.items()}


def local_train(
    model: nn.Module,
    loader: DataLoader,
    *,
    epochs: int = 1,
    lr: float = 0.01,
    momentum: float = 0.9,
    loss_fn: nn.Module | None = None,
    device: str | torch.device = "cpu",
    proximal_mu: float = 0.0,
    label_attack: Callable[[Tensor], Tensor] | None = None,
) -> nn.Module:
    """Run local SGD on one client's data, returning the trained model in-place.

    With ``proximal_mu > 0`` this implements the FedProx local objective
    (Li et al., MLSys 2020): ``L_prox(w; w_t) = L(w) + (mu/2) * ||w - w_t||^2``,
    where ``w_t`` is the global model at the start of the round. The caller
    is expected to load the global state into ``model`` before calling, so
    the parameters at function entry are the anchor ``w_t``. Setting
    ``proximal_mu = 0`` recovers vanilla FedAvg local SGD.

    With ``label_attack`` set, every minibatch's labels are passed through
    the callable before the loss is computed — this is the data-pipeline
    hook for label-flipping attacks (Biggio et al., ICML 2012; vFL's
    ``velocity.data_attacks`` module). Callable contract: takes a label
    tensor, returns a same-shape tensor. Identity by default (no attack).

    Reference:
        Li, Sahu, Zaheer, Sanjabi, Talwalkar, Smith. *Federated Optimization
        in Heterogeneous Networks*. MLSys 2020.
        https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html
    """
    criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    model.train()
    model.to(device)

    # FedProx proximal anchor: snapshot the parameters the caller just loaded
    # (the global model w_t for this round), detached and frozen on `device`.
    # Iterate `parameters()` not `state_dict()` so we exclude non-trainable
    # buffers like BatchNorm running stats — those don't belong in the
    # proximal term.
    global_params: list[Tensor] | None = None
    if proximal_mu > 0:
        global_params = [p.detach().clone().to(device) for p in model.parameters()]

    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if label_attack is not None:
                y = label_attack(y)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            if global_params is not None:
                prox_sq = sum(
                    ((p - g) * (p - g)).sum()
                    for p, g in zip(model.parameters(), global_params, strict=True)
                )
                loss = loss + (proximal_mu / 2.0) * prox_sq
            loss.backward()
            optimizer.step()
    return model


def dp_local_train(
    model: nn.Module,
    loader: DataLoader,
    *,
    noise_multiplier: float,
    max_grad_norm: float,
    epochs: int = 1,
    lr: float = 0.01,
    momentum: float = 0.9,
    loss_fn: nn.Module | None = None,
    device: str | torch.device = "cpu",
    delta: float = 1e-5,
) -> tuple[nn.Module, float]:
    """Local DP-SGD on one client's data; returns ``(trained_model, epsilon_spent)``.

    The differentially-private counterpart of :func:`local_train`: an Opacus
    ``PrivacyEngine`` wraps the model, optimizer, and loader so each step does
    per-sample gradient clipping at ``max_grad_norm`` then adds Gaussian noise
    scaled by ``noise_multiplier`` (DP-SGD; Abadi et al., CCS 2016). The privacy
    spent over ``epochs`` is returned as ``epsilon`` at ``delta``, via Opacus's
    Rényi-DP accountant (tighter composition than plain (ε, δ)-DP).

    Higher ``noise_multiplier`` ⇒ stronger privacy ⇒ smaller ``epsilon`` and
    lower utility — the trade-off the caller tunes.

    Opacus is an optional dependency (``velocity-fl[dp]``), imported lazily so it
    stays off the base and ``[torch]`` install paths.

    research(2026-05): Opacus 1.6 ``PrivacyEngine.make_private`` is the canonical
    PyTorch DP-SGD path (Fed-BioMed FL reference). This helper is single-round
    (one engine per call); cumulative cross-round accounting + ``secure_mode`` for
    production are demonstrated/discussed in ``examples/mnist_fedavg_dp.py``.
    """
    try:
        from opacus import PrivacyEngine
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "dp_local_train requires Opacus. Install with: pip install 'velocity-fl[dp]'"
        ) from exc

    criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    model.train()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    privacy_engine = PrivacyEngine()
    dp_parts = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    # Opacus's make_private return-type stub overstates the tuple arity (types the
    # with-criterion 4-tuple); the no-criterion call returns the 3 we unpack.
    private_model, private_optimizer, private_loader = dp_parts  # ty: ignore[invalid-assignment]

    for _ in range(epochs):
        for x, y in private_loader:
            x = x.to(device)
            y = y.to(device)
            private_optimizer.zero_grad()
            # Opacus's make_private return-union confuses ty; the wrapper is callable.
            loss = criterion(private_model(x), y)  # ty: ignore[call-non-callable]
            loss.backward()
            private_optimizer.step()

    # `model` was wrapped by reference, so it now holds the trained weights with
    # clean state_dict keys (no GradSampleModule prefix) for aggregation.
    return model, float(privacy_engine.get_epsilon(delta=delta))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    loss_fn: nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> tuple[float, float]:
    """Return ``(mean_loss, accuracy)`` of ``model`` on ``loader``.

    Accuracy assumes a classification head; for regression-style models pass a
    custom ``loss_fn`` and ignore the second return value.
    """
    criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        total_loss += float(criterion(logits, y).item())
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_samples += int(y.numel())
    if total_samples == 0:
        return float("nan"), float("nan")
    return total_loss / total_samples, total_correct / total_samples


def aggregated_loss(per_client: Iterable[tuple[float, int]]) -> float:
    """Sample-weighted mean of per-client losses — useful when a server-side
    eval loader isn't available and you only have client-reported losses."""
    total_loss = 0.0
    total_samples = 0
    for loss, n in per_client:
        total_loss += loss * n
        total_samples += n
    return total_loss / total_samples if total_samples > 0 else float("nan")
