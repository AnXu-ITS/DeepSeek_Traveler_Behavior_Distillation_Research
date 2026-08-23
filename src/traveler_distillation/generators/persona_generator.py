"""Reproducible synthetic persona generator."""
from __future__ import annotations

import random

from ..schemas.persona import Persona

_REQUIRED = [
    "age_group",
    "income_group",
    "occupation",
    "household_size",
    "has_children_prob",
    "car_ownership_prob",
    "driving_license_prob",
    "bike_ownership_prob",
    "transit_pass_prob",
    "habitual_mode",
    "schedule_flexibility",
    "mobility_limitation",
]


def _weighted_choice(rng: random.Random, dist: dict) -> str:
    items = list(dist.items())
    if not items:
        raise ValueError("empty categorical distribution")
    total = sum(w for _, w in items)
    if total <= 0:
        raise ValueError("categorical distribution weights must sum to > 0")
    r = rng.random() * total
    acc = 0.0
    for key, weight in items:
        acc += weight
        if r < acc:
            return key
    return items[-1][0]


class PersonaGenerator:
    """Generate reproducible synthetic personas from categorical distributions.

    Distributions live in the config (``generation_v0_1.yaml``), never in code.
    """

    def __init__(self, seed: int = 42, config: dict | None = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.cfg = (config or {}).get("persona", {})

    def generate(self, num_personas: int, id_offset: int = 0) -> list[Persona]:
        """Generate ``num_personas`` personas with ids starting at
        ``P{id_offset+1:06d}`` (used to extend an existing population without
        id collisions)."""
        if num_personas < 0:
            raise ValueError("num_personas must be >= 0")
        self._check_config()
        return [self._generate_one(f"P{id_offset + i + 1:06d}") for i in range(num_personas)]

    def _check_config(self) -> None:
        missing = [k for k in _REQUIRED if k not in self.cfg]
        if missing:
            raise ValueError(
                f"persona config missing keys: {missing}. "
                "Distributions must be provided via config, not hard-coded."
            )

    def _generate_one(self, persona_id: str) -> Persona:
        c = self.cfg
        household_size = self.rng.randint(c["household_size"]["min"], c["household_size"]["max"])
        has_children = self.rng.random() < c["has_children_prob"]
        if household_size < 2:
            has_children = False  # avoid a contradictory 1-person household with children
        return Persona(
            persona_id=persona_id,
            age_group=_weighted_choice(self.rng, c["age_group"]),
            income_group=_weighted_choice(self.rng, c["income_group"]),
            occupation=_weighted_choice(self.rng, c["occupation"]),
            household_size=household_size,
            has_children=has_children,
            car_ownership=self.rng.random() < c["car_ownership_prob"],
            driving_license=self.rng.random() < c["driving_license_prob"],
            bike_ownership=self.rng.random() < c["bike_ownership_prob"],
            transit_pass=self.rng.random() < c["transit_pass_prob"],
            habitual_mode=_weighted_choice(self.rng, c["habitual_mode"]),
            schedule_flexibility=_weighted_choice(self.rng, c["schedule_flexibility"]),
            mobility_limitation=_weighted_choice(self.rng, c["mobility_limitation"]),
        )
