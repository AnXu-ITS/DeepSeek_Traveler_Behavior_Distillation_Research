"""Reference MATSim Integration Pipeline.

A thin, reproducible wrapper around the project's PRODUCTION code that turns

    traveler population (CSV) + scenario (YAML) + frozen Student checkpoint
    + prepared MATSim supply (network / GTFS-derived schedule)

into

    MATSim-ready population.xml + config.xml + decision manifest + summary

and optionally runs MATSim with the existing preloaded-scenario launcher.

Design rules (see docs/REFERENCE_PIPELINE.md):

- All model/decision logic is the production code under
  ``src/traveler_distillation`` — this package only orchestrates it and adds
  input validation, persistent route/accessibility caching and batch inference
  (which the latency audit verified to be decision-identical to the per-state
  ``decide()`` path).
- No retraining, no teacher calls, no changes to frozen releases or benchmark
  outputs.
- Outputs never land inside a frozen release (``assert_not_frozen_output``).

Scope: this is the *Reference* pipeline for THIS project's Student→MATSim
deployment (real-supply accessibility-aware Student S9). It is NOT a universal
MATSim adapter for arbitrary projects.
"""

from .config import ReferenceConfig, load_reference_config
from .pipeline import ReferencePipeline

__all__ = ["ReferenceConfig", "load_reference_config", "ReferencePipeline"]
