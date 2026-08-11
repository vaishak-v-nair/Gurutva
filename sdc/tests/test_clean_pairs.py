"""Gate 0.2 — clean determinism.

Two clean runs at the same seed must produce bit-identical final weights. This is
the control that makes every divergence measurement in this project meaningful:
without it, an "injected run diverged" result could be nothing but the ambient
nondeterminism of the stack.

Compared by hash rather than tolerance on purpose. A tolerance would hide exactly
the drift this gate exists to catch.
"""

import torch

from sdc.record import fingerprint
from sdc.runner import TrainConfig, make_batches, train


def test_two_clean_runs_are_bit_identical():
    a = train(TrainConfig(steps=10))
    b = train(TrainConfig(steps=10))
    assert a.final_fingerprint == b.final_fingerprint
    assert a.losses == b.losses


def test_clean_run_has_no_injections():
    run = train(TrainConfig(steps=5))
    assert run.n_injections == 0
    assert all(s.n_injections == 0 for s in run.steps)


def test_data_stream_is_identical_across_runs():
    """Batches must not depend on model or injector RNG consumption."""
    cfg = TrainConfig(steps=4)
    first = make_batches(cfg)
    torch.manual_seed(999)
    torch.randn(1000)  # perturb the global stream
    second = make_batches(cfg)
    for (xa, ya), (xb, yb) in zip(first, second):
        assert torch.equal(xa, xb)
        assert torch.equal(ya, yb)


def test_different_seed_changes_the_result():
    """Guards against a fingerprint that is accidentally constant."""
    a = train(TrainConfig(steps=6, seed=1))
    b = train(TrainConfig(steps=6, seed=2))
    assert a.final_fingerprint != b.final_fingerprint


def test_fingerprint_is_sensitive_to_one_bit():
    from sdc.model import ModelConfig, TinyTransformer

    model = TinyTransformer(ModelConfig())
    before = fingerprint(model)
    with torch.no_grad():
        flat = next(model.parameters()).detach().reshape(-1)
        flat.view(torch.int32)[0] ^= 1  # mantissa LSB — the smallest change possible
    assert fingerprint(model) != before


def test_training_actually_learns():
    """A loop that does nothing would pass every determinism gate trivially."""
    run = train(TrainConfig(steps=40))
    assert run.final_loss < run.losses[0], (
        f"loss did not decrease: {run.losses[0]:.4f} -> {run.final_loss:.4f}"
    )
