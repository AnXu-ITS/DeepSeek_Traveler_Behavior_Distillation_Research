"""Context perturbation: change exactly one axis of a DynamicContext."""
from __future__ import annotations

from ..schemas.context import DynamicContext


def perturb_context(context: DynamicContext, axis: str, level: float | int | bool) -> DynamicContext:
    """Return a deep copy of ``context`` with only ``axis`` changed to ``level``."""
    new = context.model_copy(deep=True)
    handler = AXIS_HANDLERS.get(axis)
    if handler is None:
        raise ValueError(
            f"unknown perturbation axis '{axis}'. Supported: {sorted(AXIS_HANDLERS)}"
        )
    handler(new, level)
    return new


def _set_weather_intensity(ctx: DynamicContext, level) -> None:
    """Weather intensity is defined as rain intensity.

    intensity == 0.0  -> condition = clear
    intensity >  0.0  -> condition = rain
    """
    intensity = float(level)
    ctx.weather.intensity = intensity
    ctx.weather.condition = "clear" if intensity <= 0.0 else "rain"


def _set_road_congestion(ctx: DynamicContext, level) -> None:
    ctx.road_congestion = float(level)


def _set_transit_delay(ctx: DynamicContext, level) -> None:
    ctx.transit_delay_min = int(level)


def _set_fare_multiplier(ctx: DynamicContext, level) -> None:
    ctx.fare_multiplier = float(level)


def _set_parking_cost_multiplier(ctx: DynamicContext, level) -> None:
    ctx.parking_cost_multiplier = float(level)


def _set_road_disruption(ctx: DynamicContext, level) -> None:
    ctx.road_disruption = bool(level)


AXIS_HANDLERS = {
    "weather_intensity": _set_weather_intensity,
    "road_congestion": _set_road_congestion,
    "transit_delay": _set_transit_delay,
    "fare_multiplier": _set_fare_multiplier,
    "parking_cost_multiplier": _set_parking_cost_multiplier,
    "road_disruption": _set_road_disruption,
}


def perturb_context_multi(
    context: DynamicContext,
    axes: list[tuple[str, float | int | bool]],
) -> DynamicContext:
    """Return a deep copy of ``context`` with MULTIPLE axes changed at once.

    ``axes`` is an ordered list of ``(axis, level)`` pairs, applied in order
    onto successive deep copies. Used by the S5 multi-axis / joint
    counterfactual generator. Unknown axes raise ``ValueError`` (same contract
    as :func:`perturb_context`).
    """
    new = context.model_copy(deep=True)
    for axis, level in axes:
        handler = AXIS_HANDLERS.get(axis)
        if handler is None:
            raise ValueError(
                f"unknown perturbation axis '{axis}'. Supported: {sorted(AXIS_HANDLERS)}"
            )
        handler(new, level)
    return new
