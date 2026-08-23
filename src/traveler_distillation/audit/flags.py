"""Audit flags: flag suspicious teacher behavior (never auto-invalidate)."""
from __future__ import annotations


def compute_flags(
    repeatability: dict | None = None,
    counterfactual: dict | None = None,
    persona: dict | None = None,
    pilot_records: list | None = None,
) -> list[dict]:
    """Produce audit flags from metrics. These are for review only."""
    flags: list[dict] = []

    # --- repeatability flags ---
    for g in (repeatability or {}).get("per_group", []):
        base = {
            "scope": "repeatability",
            "state_hash": g["state_hash"],
            "source_sample_id": g["source_sample_id"],
        }
        if g["selected_mode_agreement"] < 0.67:
            flags.append(
                {
                    **base,
                    "flag": "selected_mode_flip",
                    "detail": f"agreement={g['selected_mode_agreement']} modes={g['selected_modes']}",
                }
            )
        if g["mean_pairwise_l1"] > 0.30:
            flags.append(
                {
                    **base,
                    "flag": "high_repeat_variance",
                    "detail": f"mean_pairwise_l1={g['mean_pairwise_l1']}",
                }
            )
        if g["departure_shift"]["range"] > 15:
            flags.append(
                {
                    **base,
                    "flag": "departure_shift_instability",
                    "detail": f"range={g['departure_shift']['range']} values={g['departure_shift']['values']}",
                }
            )
        if g["confidence"]["std"] > 0.15:
            flags.append(
                {
                    **base,
                    "flag": "confidence_instability",
                    "detail": f"confidence_std={g['confidence']['std']}",
                }
            )
        if any(v["range"] > 0.25 for v in g["probability_stability"].values()):
            flags.append(
                {
                    **base,
                    "flag": "high_repeat_variance",
                    "detail": "some mode probability range > 0.25 across repeats",
                }
            )

    # --- counterfactual direction flags ---
    for axis, res in (counterfactual or {}).get("axes", {}).items():
        for mode, m in res.get("mode_stats", {}).items():
            rho = m["spearman_rho"]
            base = {"scope": "counterfactual", "axis": axis, "mode": mode}
            if axis == "fare_multiplier" and mode == "pt" and rho > 0.30:
                flags.append({**base, "flag": "unexpected_positive_fare_response", "detail": f"pt rho={rho}"})
            if axis == "transit_delay" and mode == "pt" and rho > 0.30:
                flags.append({**base, "flag": "unexpected_positive_transit_delay_response", "detail": f"pt rho={rho}"})
            if axis == "road_congestion" and mode == "car" and rho > 0.30:
                flags.append({**base, "flag": "unexpected_positive_car_congestion_response", "detail": f"car rho={rho}"})
            if axis == "weather_intensity" and mode in ("bike", "walk") and rho > 0.50:
                flags.append({**base, "flag": "unexpected_positive_weather_exposed_mode", "detail": f"{mode} rho={rho}"})
            if m["mean_direction_reversals"] >= 2.0:
                flags.append({**base, "flag": "strong_counterfactual_reversal", "detail": f"{mode} mean reversals={m['mean_direction_reversals']}"})

    # --- low confidence flags from pilot ---
    for s in pilot_records or []:
        if s.teacher is not None and s.teacher.confidence < 0.5:
            flags.append(
                {
                    "scope": "pilot",
                    "sample_id": s.sample_id,
                    "flag": "low_confidence",
                    "detail": f"confidence={s.teacher.confidence}",
                }
            )

    return flags
