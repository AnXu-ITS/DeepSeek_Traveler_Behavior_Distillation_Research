"""S8 real-supply transit accessibility adaptation package.

City-independent accessibility features computed from real OSM + GTFS supply
(never Singapore-specific identities), the Student-S8 architecture (Case B),
and the stratified accessibility dataset builder.
"""
from .gtfs_accessibility import SupplyIndex, classify_accessibility, plan_accessibility
from .accessibility_features import (
    S8_ALT_NUM,
    S8_NEW_ALT_NUM,
    S8FeatureExtractor,
    TravelerStudentS8,
)

__all__ = [
    "SupplyIndex",
    "classify_accessibility",
    "plan_accessibility",
    "S8_ALT_NUM",
    "S8_NEW_ALT_NUM",
    "S8FeatureExtractor",
    "TravelerStudentS8",
]
