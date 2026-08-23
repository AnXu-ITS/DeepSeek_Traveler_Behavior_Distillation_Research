"""Travel alternative attribute generator."""
from __future__ import annotations

from ..schemas.persona import Persona
from ..schemas.trip import Trip
from ..schemas.context import DynamicContext
from ..schemas.alternative import TravelAlternative


class AlternativeGenerator:
    """Compute transportation option attributes from (persona, trip, context).

    Attributes describe the *option* (time, cost, reliability, exposure), never
    the traveler's preference. Availability uses basic ownership/license rules
    only; behavioral preference is left entirely to the teacher.
    """

    def __init__(self, config: dict | None = None):
        self.cfg = (config or {}).get("alternatives", {})

    def generate(self, persona: Persona, trip: Trip, context: DynamicContext) -> list[TravelAlternative]:
        distance = trip.distance_km
        alternatives = []
        for mode in ["car", "pt", "bike", "walk"]:
            attrs = self._base_attributes(mode, persona, distance)
            attrs = self._apply_context(mode, attrs, context, distance)
            alternatives.append(TravelAlternative(mode=mode, **attrs))
        return alternatives

    # ------------------------------------------------------------------ base
    def _base_attributes(self, mode: str, persona: Persona, d: float) -> dict:
        speeds = self.cfg.get("base_speeds_kmh", {})
        access = self.cfg.get("access_time_min", {})
        transfers = self.cfg.get("transfers", {})
        exposure = self.cfg.get("weather_exposure", {})
        cost = self.cfg.get("base_monetary_cost", {})
        effects = self.cfg.get("effects", {})

        speed = speeds.get(mode, self._default_speed(mode))
        acc = access.get(mode, 0.0)
        n_transfers = transfers.get(mode, 0)
        exp = exposure.get(mode, self._default_exposure(mode))
        transfer_min = effects.get("transfer_time_per_transfer_min", 4.0)

        if mode == "car":
            available = persona.driving_license and persona.car_ownership
            monetary = d * cost.get("car_per_km", 0.6) + cost.get("parking_base", 5.0)
        elif mode == "pt":
            available = True
            monetary = cost.get("pt_base_fare", 2.0) + d * cost.get("pt_per_km", 0.15)
        elif mode == "bike":
            available = persona.bike_ownership
            monetary = 0.0
        else:  # walk
            available = True
            monetary = 0.0

        travel_time = d / speed * 60.0 + acc + n_transfers * transfer_min
        return {
            "available": available,
            "travel_time_min": travel_time,
            "monetary_cost": monetary,
            "access_time_min": acc,
            "transfers": n_transfers,
            "reliability_delay_min": 0.0,
            "weather_exposure": exp,
        }

    # --------------------------------------------------------------- context
    def _apply_context(self, mode: str, attrs: dict, context: DynamicContext, d: float) -> dict:
        effects = self.cfg.get("effects", {})
        cost = self.cfg.get("base_monetary_cost", {})
        exposure = attrs["weather_exposure"]

        cong_factor = effects.get("congestion_travel_time_factor_max", 0.5)
        cong_reliability = effects.get("congestion_reliability_delay_max", 15.0)
        weather_factor = effects.get("weather_travel_time_factor_max", 0.15)

        tt = attrs["travel_time_min"]
        reliability = attrs["reliability_delay_min"]

        if mode == "car":
            tt *= 1.0 + context.road_congestion * cong_factor
            reliability += context.road_congestion * cong_reliability
            if context.road_disruption:
                tt += effects.get("road_disruption_travel_time_min", 20.0)
                reliability += effects.get("road_disruption_reliability_min", 20.0)
            attrs["monetary_cost"] = (
                d * cost.get("car_per_km", 0.6)
                + cost.get("parking_base", 5.0) * context.parking_cost_multiplier
                + context.congestion_charge
            )
        elif mode == "pt":
            tt += context.transit_delay_min
            reliability += context.transit_delay_min
            if context.transit_disruption:
                tt += effects.get("transit_disruption_travel_time_min", 15.0)
                reliability += effects.get("transit_disruption_reliability_min", 30.0)
            attrs["monetary_cost"] = (
                cost.get("pt_base_fare", 2.0) + d * cost.get("pt_per_km", 0.15)
            ) * context.fare_multiplier

        # weather slightly slows exposed modes
        tt *= 1.0 + context.weather.intensity * weather_factor * exposure

        attrs["travel_time_min"] = round(tt, 3)
        attrs["reliability_delay_min"] = round(reliability, 3)
        attrs["monetary_cost"] = round(attrs["monetary_cost"], 3)
        return attrs

    @staticmethod
    def _default_speed(mode: str) -> float:
        return {"car": 32.0, "pt": 18.0, "bike": 14.0, "walk": 4.5}.get(mode, 15.0)

    @staticmethod
    def _default_exposure(mode: str) -> float:
        return {"car": 0.05, "pt": 0.4, "bike": 0.9, "walk": 1.0}.get(mode, 0.5)
