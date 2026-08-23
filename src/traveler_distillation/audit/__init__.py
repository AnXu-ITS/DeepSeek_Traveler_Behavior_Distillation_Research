"""Teacher audit analysis module (v0.1)."""
from .metrics import (
    state_hash,
    prompt_hash,
    l1,
    mean_pairwise_l1,
    max_pairwise_l1,
    aggregate_distribution,
    mean_single_to_aggregate_l1,
    jaccard_similarity,
    mean,
    std,
    value_range,
    spearman_rank,
    direction_reversals,
    selected_mode_agreement,
)
from .repeatability import analyze_repeatability, probability_stability
from .counterfactual import analyze_counterfactual, elasticity_preview
from .persona_sensitivity import analyze_persona
from .flags import compute_flags
from .report import generate_report, decide_recommendation
from .records import RepeatabilityRecord, PersonaContrastRecord

__all__ = [
    "state_hash",
    "prompt_hash",
    "l1",
    "mean_pairwise_l1",
    "max_pairwise_l1",
    "aggregate_distribution",
    "mean_single_to_aggregate_l1",
    "jaccard_similarity",
    "mean",
    "std",
    "value_range",
    "spearman_rank",
    "direction_reversals",
    "selected_mode_agreement",
    "probability_stability",
    "analyze_repeatability",
    "analyze_counterfactual",
    "elasticity_preview",
    "analyze_persona",
    "compute_flags",
    "generate_report",
    "decide_recommendation",
    "RepeatabilityRecord",
    "PersonaContrastRecord",
]
