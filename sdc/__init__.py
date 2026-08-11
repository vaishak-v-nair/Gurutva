"""Phase 0 of the compute-integrity work: a fault-injection harness.

Unrelated to the gravity inversion in the rest of this repository — parked here
because a new remote could not be created from the session that wrote it. Imports
nothing from, and is imported by nothing in, the geophysics code. Extractable
with ``git subtree split`` when it earns its own repo.

See ``docs/compute-integrity-plan.md`` for why this exists and what it is for.
"""

from .inject import Injector, InjectionEvent, SITES, field_of, max_bit
from .model import ModelConfig, TinyTransformer
from .record import RunRecord, StepRecord, fingerprint
from .runner import TrainConfig, set_determinism, train

__all__ = [
    "Injector",
    "InjectionEvent",
    "SITES",
    "field_of",
    "max_bit",
    "ModelConfig",
    "TinyTransformer",
    "RunRecord",
    "StepRecord",
    "fingerprint",
    "TrainConfig",
    "set_determinism",
    "train",
]
