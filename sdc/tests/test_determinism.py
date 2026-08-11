"""Gate 0.1 — reproducible injection.

The same seed must produce a bit-identical sequence of (step, site, flat_index,
bit) events. Without this, no injected run can be compared against any other and
the whole harness is decorative.
"""

import torch

from sdc.inject import Injector, field_of, max_bit
from sdc.runner import TrainConfig, train


def make(seed=7, bits=(30,), count=2):
    return Injector(seed=seed, bits=bits, count=count, sites=("gemm_out",))


def test_same_seed_same_manifest():
    a = train(TrainConfig(steps=6), make())
    b = train(TrainConfig(steps=6), make())
    assert a.injection_manifest == b.injection_manifest
    assert a.n_injections > 0, "injector never fired; the gate would be vacuous"


def test_different_seed_different_manifest():
    a = train(TrainConfig(steps=6), make(seed=7))
    b = train(TrainConfig(steps=6), make(seed=8))
    assert a.injection_manifest != b.injection_manifest


def test_reset_rewinds_stream():
    inj = make()
    first = [e.key() for e in Injector(seed=7, bits=(30,), count=2).events]
    assert first == []

    t = torch.ones(64)
    inj.corrupt_(t.clone(), "gemm_out")
    before = inj.manifest()
    inj.reset()
    assert inj.manifest() == []
    inj.corrupt_(t.clone(), "gemm_out")
    assert inj.manifest() == before


def test_injected_run_is_itself_reproducible():
    """Not just the manifest — the resulting weights must match too."""
    a = train(TrainConfig(steps=6), make())
    b = train(TrainConfig(steps=6), make())
    assert a.final_fingerprint == b.final_fingerprint


def test_bit_position_is_respected():
    inj = Injector(seed=3, bits=(0,), count=8, sites=("gemm_out",))
    inj.corrupt_(torch.ones(256), "gemm_out")
    assert inj.events, "no events recorded"
    assert {e.bit for e in inj.events} == {0}
    assert {e.field for e in inj.events} == {"mantissa"}


def test_field_labels_match_ieee754_layout():
    assert field_of(31, torch.float32) == "sign"
    assert field_of(30, torch.float32) == "exponent"
    assert field_of(23, torch.float32) == "exponent"
    assert field_of(22, torch.float32) == "mantissa"
    assert field_of(0, torch.float32) == "mantissa"

    assert field_of(15, torch.bfloat16) == "sign"
    assert field_of(7, torch.bfloat16) == "exponent"
    assert field_of(6, torch.bfloat16) == "mantissa"

    assert max_bit(torch.float32) == 31
    assert max_bit(torch.bfloat16) == 15


def test_top_bit_flip_does_not_overflow_narrow_views():
    """The sign bit of a 16-bit view cannot be written as +32768."""
    t = torch.ones(8, dtype=torch.bfloat16)
    Injector(seed=1, bits=(15,), count=8, sites=("gemm_out",)).corrupt_(t, "gemm_out")
    assert torch.all(t < 0), "sign-bit flip did not negate the values"


def test_sites_filter_is_honoured():
    inj = Injector(seed=1, bits=(30,), count=4, sites=("allreduce",))
    inj.corrupt_(torch.ones(64), "gemm_out")
    assert inj.manifest() == []
    inj.corrupt_(torch.ones(64), "allreduce")
    assert len(inj.manifest()) == 4


def test_injector_stream_is_independent_of_global_rng():
    """Injection must not consume the global RNG, or it would shift the data."""
    torch.manual_seed(0)
    Injector(seed=5, bits=(30,), count=4, sites=("gemm_out",)).corrupt_(
        torch.ones(128), "gemm_out"
    )
    after_injection = torch.randn(4)

    torch.manual_seed(0)
    untouched = torch.randn(4)

    assert torch.equal(after_injection, untouched)
