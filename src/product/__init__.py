"""Gurutva product layer — the verdict a customer buys, built on the
validated posterior engine in `src/`."""

from .verdict import (Verdict, assess, exclusion_limit, functional_sd,
                      informed_fraction, mass_in_region, posterior_mean)

__all__ = ["Verdict", "assess", "exclusion_limit", "functional_sd",
           "informed_fraction", "mass_in_region", "posterior_mean"]
