"""Reproducible synthetic trip generator."""
from __future__ import annotations

import random

from ..schemas.trip import Trip
from .persona_generator import _weighted_choice


class TripGenerator:
    """Generate reproducible synthetic trips from purpose templates."""

    def __init__(self, seed: int = 42, config: dict | None = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.cfg = (config or {}).get("trip", {})

    def generate(self, num_trips: int, id_offset: int = 0) -> list[Trip]:
        """Generate ``num_trips`` trips with ids starting at ``T{id_offset+1:06d}``
        (used to extend an existing trip set without id collisions)."""
        if num_trips < 0:
            raise ValueError("num_trips must be >= 0")
        purposes = self.cfg.get("purposes")
        templates = self.cfg.get("templates")
        if not purposes or not templates:
            raise ValueError("trip config requires 'purposes' and 'templates'")
        trips = []
        for i in range(num_trips):
            purpose = _weighted_choice(self.rng, purposes)
            trips.append(self._generate_one(f"T{id_offset + i + 1:06d}", purpose, templates[purpose]))
        return trips

    def _generate_one(self, trip_id: str, purpose: str, tpl: dict) -> Trip:
        d_lo, d_hi = tpl["distance_km"]
        dep_lo, dep_hi = tpl["departure_window_min"]
        arr_lo, arr_hi = tpl["arrival_window_min"]

        distance = self.rng.uniform(d_lo, d_hi)
        departure = self.rng.randint(dep_lo, dep_hi)
        arrival = self.rng.randint(arr_lo, arr_hi)
        # Guarantee arrival is after departure (travel takes some minimum time).
        if arrival <= departure:
            arrival = min(departure + 15, 1439)

        return Trip(
            trip_id=trip_id,
            purpose=purpose,
            origin_type=self.cfg.get("origin_type", "home"),
            destination_type=tpl.get("destination_type", "other"),
            distance_km=round(distance, 3),
            desired_departure_min=departure,
            desired_arrival_min=arrival,
            time_constraint=tpl.get("time_constraint", "soft"),
        )
