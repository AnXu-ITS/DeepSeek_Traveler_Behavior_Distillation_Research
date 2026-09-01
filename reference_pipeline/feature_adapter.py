"""Feature adapter: scenario YAML + population objects -> production states.

This module ONLY composes production functions. The scenario effects applied
to alternatives (transit delay, road disruption) use the exact formulas of the
production Phase C runner (``scripts/singapore/run_phase_c.py``); the
accessibility vectors come from the production PT planner
(``plan_accessibility``); the OD derivation is the production deterministic
hash rule. Nothing behavioral is re-implemented here.
"""
from __future__ import annotations

import hashlib

from traveler_distillation.accessibility.accessibility_dataset import build_real_alternatives
from traveler_distillation.schemas.context import DynamicContext, Weather
from traveler_distillation.schemas.state import UniversalTravelerState

# ---------------------------------------------------------------------------
# Scenario (YAML) -> DynamicContext — mirror run_phase_c.make_context
# ---------------------------------------------------------------------------
def make_context(scenario_cfg) -> DynamicContext:
    w = scenario_cfg.weather
    return DynamicContext(
        context_id=scenario_cfg.context_id,
        weather=Weather(condition=w.condition, intensity=w.intensity),
        road_congestion=scenario_cfg.road_congestion,
        transit_delay_min=scenario_cfg.transit_delay_min,
        transit_disruption=scenario_cfg.transit_disruption,
        road_disruption=scenario_cfg.road_disruption,
        fare_multiplier=scenario_cfg.fare_multiplier,
        parking_cost_multiplier=scenario_cfg.parking_cost_multiplier,
        congestion_charge=scenario_cfg.congestion_charge,
    )


# ---------------------------------------------------------------------------
# Deterministic OD derivation — production formula (MATSimAdapter.build_real_scenario):
#   h = int(sha256(persona_id).hexdigest(), 16)
#   home = activity_nodes[h % len]; dest = activity_nodes[(h + 7919*(t_idx+1)) % len]
# ---------------------------------------------------------------------------
def derive_od(persona_id: str, trip_index: int, activity_nodes: list[dict]) -> tuple[str, str]:
    h = int(hashlib.sha256(persona_id.encode()).hexdigest(), 16)
    home = activity_nodes[h % len(activity_nodes)]["node"]
    dest = activity_nodes[(h + 7919 * (trip_index + 1)) % len(activity_nodes)]["node"]
    return home, dest


def make_alt_factory(idx, context: DynamicContext, cache=None):
    """Production Phase C alt_factory: real-supply alternatives + scenario effects.

    ``cache`` is an optional :class:`RouteCache`; when given, the accessibility
    vectors and mode travel times come from the persistent cache (misses are
    computed with the production functions and stored). Decision outputs are
    identical with or without the cache.
    """
    delay = context.transit_delay_min
    disrupt = context.road_disruption

    def _plan(origin: str, dest: str, dep_sec: float) -> dict:
        if cache is None:
            from traveler_distillation.accessibility.gtfs_accessibility import plan_accessibility
            return plan_accessibility(idx, origin, dest, dep_sec)
        return cache.plan_accessibility(idx, origin, dest, dep_sec)

    # capture the ORIGINAL mode_travel_time before the patch below (the cache
    # must call the original, never the patched wrapper — same capture pattern
    # as run_phase_c.make_shared_idx)
    _orig_mtt = idx.mode_travel_time

    def _mtt(mode: str, src: str, dst: str):
        if cache is None:
            return _orig_mtt(mode, src, dst)
        return cache.mode_travel_time(mode, src, dst, lambda: _orig_mtt(mode, src, dst))

    def factory(persona, trip, ctx, origin, dest):
        acc = _plan(origin, dest, float(trip.desired_departure_min) * 60.0)
        with idx_travel_time(idx, _mtt):
            alts = build_real_alternatives(persona, trip, ctx, idx, origin, dest, acc)
        for alt in alts:
            if alt.mode == "pt" and delay:
                alt.travel_time_min = round(alt.travel_time_min + delay, 3)
                alt.reliability_delay_min = round(alt.reliability_delay_min + delay, 3)
            elif alt.mode == "car" and disrupt:
                alt.travel_time_min = round(alt.travel_time_min + 20.0, 3)
                alt.reliability_delay_min = round(alt.reliability_delay_min + 20.0, 3)
        return alts

    return factory


class idx_travel_time:
    """Temporarily patch ``idx.mode_travel_time`` (same memoization trick as
    run_phase_c.make_shared_idx, but cache-backed). Context manager so the
    production ``build_real_alternatives`` calls the cache-aware method."""

    def __init__(self, idx, fn):
        self.idx = idx
        self.fn = fn
        self._orig = None

    def __enter__(self):
        self._orig = self.idx.mode_travel_time
        self.idx.mode_travel_time = self.fn
        return self.idx

    def __exit__(self, *exc):
        self.idx.mode_travel_time = self._orig
        return False


# ---------------------------------------------------------------------------
# State encoding with static-part memoization (identical outputs to
# ``S8FeatureExtractor.encode``; the audit measured the per-state encode at
# 0.007 ms, so this is a pure convenience, not a correctness change).
# ---------------------------------------------------------------------------
class MemoizedEncoder:
    """Encodes states via the checkpoint extractor; memoizes the parts of
    ``global_cat`` / ``global_num`` that only depend on persona / trip /
    context. The produced lists are fresh copies with IDENTICAL values.

    Safety: on the first few states the memoized output is compared EXACTLY
    against the production ``extractor.encode``; any mismatch permanently
    disables memoization and falls back to the production encoder (parity is
    never traded for speed)."""

    def __init__(self, extractor):
        self.extractor = extractor
        self._persona_cat: dict[str, list] = {}
        self._persona_num: dict[str, list] = {}
        self._trip_cat: dict[str, list] = {}
        self._trip_num: dict[str, list] = {}
        self._context_cat: list | None = None
        self._context_num: list | None = None
        self._checked = 0
        self._memo_ok = True

    def encode(self, state: UniversalTravelerState) -> dict:
        if not self._memo_ok:
            return self.extractor.encode(state)
        out = self._encode_memoized(state)
        if self._checked < 5:
            ref = self.extractor.encode(state)
            if ref != out:
                self._memo_ok = False
                return ref
            self._checked += 1
        return out

    def _encode_memoized(self, state: UniversalTravelerState) -> dict:
        ext = self.extractor
        pid = state.persona.persona_id
        tid = state.trip.trip_id
        if pid not in self._persona_cat:
            self._persona_cat[pid] = [str(v) for v in
                                      (state.persona.age_group, state.persona.income_group,
                                       state.persona.occupation, state.persona.habitual_mode,
                                       state.persona.schedule_flexibility,
                                       state.persona.mobility_limitation)]
            self._persona_num[pid] = [float(v) for v in
                                      (state.persona.household_size, state.persona.has_children,
                                       state.persona.car_ownership, state.persona.driving_license,
                                       state.persona.bike_ownership, state.persona.transit_pass)]
        if tid not in self._trip_cat:
            self._trip_cat[tid] = [str(state.trip.purpose), str(state.trip.time_constraint)]
            self._trip_num[tid] = [float(v) for v in
                                   (state.trip.distance_km, state.trip.desired_departure_min,
                                    state.trip.desired_arrival_min)]
        if self._context_cat is None:
            self._context_cat = [str(state.context.weather.condition)]
            self._context_num = [
                float(state.context.weather.intensity),
                float(state.context.road_congestion),
                float(state.context.transit_delay_min),
                float(state.context.transit_disruption),
                float(state.context.road_disruption),
                float(state.context.fare_multiplier),
                float(state.context.parking_cost_multiplier),
                float(state.context.congestion_charge),
            ]

        cat_names = ["age_group", "income_group", "occupation", "habitual_mode",
                     "schedule_flexibility", "mobility_limitation",
                     "purpose", "time_constraint", "weather_condition"]
        num_names = ["household_size", "has_children", "car_ownership", "driving_license",
                     "bike_ownership", "transit_pass",
                     "distance_km", "desired_departure_min", "desired_arrival_min",
                     "weather_intensity", "road_congestion", "transit_delay_min",
                     "transit_disruption", "road_disruption", "fare_multiplier",
                     "parking_cost_multiplier", "congestion_charge"]

        cat_values = self._persona_cat[pid] + self._trip_cat[tid] + self._context_cat
        num_values = self._persona_num[pid] + self._trip_num[tid] + self._context_num

        global_cat = []
        for name, raw in zip(cat_names, cat_values):
            global_cat.append(ext.cat_vocabs[name].get(raw, 0))
        global_num = []
        for name, raw in zip(num_names, num_values):
            global_num.append((raw - ext.num_mean[name]) / ext.num_std[name])

        alt_mode_idx = []
        alt_num = []
        alt_available = []
        for alt in state.alternatives:
            alt_mode_idx.append(ext.mode_vocab.get(alt.mode, 0))
            alt_num.append([
                (float(getattr(alt, name)) - ext.num_mean[name]) / ext.num_std[name]
                for name in ext.ALT_NUM_ORDER
            ])
            alt_available.append(1.0 if alt.available else 0.0)

        return {
            "global_cat": global_cat,
            "global_num": global_num,
            "alt_mode_idx": alt_mode_idx,
            "alt_num": alt_num,
            "alt_available": alt_available,
        }
