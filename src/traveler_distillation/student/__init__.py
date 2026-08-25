"""Student model + distillation losses + dataset utilities."""
from .features import FeatureExtractor
from .model import TravelerStudent, masked_softmax
from .losses import (
    kl_divergence,
    mode_cross_entropy,
    departure_huber,
    student_loss,
    elasticity_l1,
    heterogeneity_l1,
    elasticity_direction_loss,
    elasticity_magnitude_loss,
    elasticity_sign_agreement,
)
from .mechanism_losses import mechanism_fidelity_loss, broken_path_fidelity_loss
from .eval import predict_probs, persona_breakdown, counterfactual_sign_agreement
from .dataset import (
    AggregatedTeacherDataset,
    collate_batch,
    CounterfactualPairDataset,
    collate_pairs,
    build_baseline_index,
    make_counterfactual_pairs,
    make_persona_contrast_pairs,
    PersonaContrastPairDataset,
    collate_contrast_pairs,
)
from .mechanism_dataset import (
    MechanismQuadruplet,
    QuadMember,
    load_mechanism_quadruplets,
    encode_quad_member,
    MechanismQuadrupletDataset,
    collate_quadruplets,
)
from .split import group_aware_split, group_overlap

__all__ = [
    "FeatureExtractor",
    "TravelerStudent",
    "masked_softmax",
    "kl_divergence",
    "mode_cross_entropy",
    "departure_huber",
    "student_loss",
    "elasticity_l1",
    "heterogeneity_l1",
    "elasticity_direction_loss",
    "elasticity_magnitude_loss",
    "elasticity_sign_agreement",
    "mechanism_fidelity_loss",
    "broken_path_fidelity_loss",
    "predict_probs",
    "persona_breakdown",
    "counterfactual_sign_agreement",
    "AggregatedTeacherDataset",
    "collate_batch",
    "CounterfactualPairDataset",
    "collate_pairs",
    "build_baseline_index",
    "make_counterfactual_pairs",
    "make_persona_contrast_pairs",
    "PersonaContrastPairDataset",
    "collate_contrast_pairs",
    "MechanismQuadruplet",
    "QuadMember",
    "load_mechanism_quadruplets",
    "encode_quad_member",
    "MechanismQuadrupletDataset",
    "collate_quadruplets",
    "group_aware_split",
    "group_overlap",
]
