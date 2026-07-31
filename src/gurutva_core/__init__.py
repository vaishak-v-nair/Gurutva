"""gurutva_core — the domain-agnostic uncertainty engine.

Gurutva sells one thing at every scale: **gravity, with error bars.** The
mathematics that delivers it comes in two families, and this package keeps
them honestly separate rather than pretending one solver does everything.

    ENGINE 1 — potential-field inversion (infer MASS from its FIELD)
        Poisson's equation is scale-free, so the same machinery serves:
          subsurface voxels (mining/geothermal/CCS)  -> adapters.subsurface
          planetary interfaces (Moho, satellite gravity) -> adapters.moho
          projected mass sheets (lensing, dark matter) -> adapters.lensing
        Shared: forward operator -> declared prior -> posterior -> GATES.

    ENGINE 2 — state/trajectory uncertainty (KNOWN field, uncertain STATE)
        Astrodynamics is NOT a potential-field inversion: the field is
        known, the initial state is not, and chaos amplifies the difference.
        The inference machinery is ensemble propagation, not Poisson
        inversion. Same product, different mathematics — recorded here so
        the unification claim stays true.  -> adapters.astro

WHAT IS GENUINELY SHARED (the actual unification):
    - the gravitational potential model itself,
    - the declared-prior discipline,
    - and above all THE GATE SUITE, which is domain-independent:
        1 licensing   is the observation something this model could produce?
        2 calibration do stated intervals contain truth at their rate?
        3 stability   do independent runs agree at the real observation?
        4 adequacy    can the model reproduce the data at all?
      Gate 4 exists because gates 1-3 all passed for a model that could not
      (S3, 2026-08-04). Every domain adapter inherits all four.
"""

from .gates import GateReport, adequacy, calibration, licensing, stability

__all__ = ["GateReport", "licensing", "calibration", "stability", "adequacy"]
