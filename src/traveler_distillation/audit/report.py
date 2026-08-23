"""Teacher audit report generation."""
from __future__ import annotations

import json

from .metrics import mean


def decide_recommendation(repeatability: dict, counterfactual: dict, persona: dict) -> str:
    """Choose a recommendation (soft heuristics, not hard thresholds)."""
    agg = repeatability.get("aggregate", {})
    agreement = agg.get("mean_selected_mode_agreement", 0.0)
    l1 = agg.get("mean_pairwise_probability_l1", 0.0)

    persona_l1s = [
        g["mean_pairwise_l1"] for g in persona.get("per_group", []) if g.get("n_members", 0) >= 2
    ]
    persona_mean_l1 = mean(persona_l1s) if persona_l1s else 0.0

    n_flip_groups = sum(
        1 for g in repeatability.get("per_group", []) if g["selected_mode_agreement"] < 0.67
    )
    n_groups = max(len(repeatability.get("per_group", [])), 1)

    if agreement >= 0.85 and l1 <= 0.15 and n_flip_groups / n_groups <= 0.25:
        return "A. SINGLE-CALL TEACHER SUFFICIENT"
    if agreement < 0.60 or l1 > 0.30:
        if persona_mean_l1 < 0.05:
            return "C. TEACHER / PROMPT REQUIRES REVISION"
        return "B. REPEATED-CALL AGGREGATION RECOMMENDED"
    return "B. REPEATED-CALL AGGREGATION RECOMMENDED"


def generate_report(
    api_stats: dict,
    repeatability: dict,
    counterfactual: dict,
    persona: dict,
    elasticity: dict,
    flags: list[dict],
    recommendation: str,
) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Teacher Audit Report v0.1\n")

    # --- Executive summary ---
    agg = repeatability.get("aggregate", {})
    add("## Executive Summary\n")
    add(
        "This audit assesses whether the DeepSeek teacher's behavioral preference "
        "distribution is stable, counterfactually consistent, and persona-sensitive "
        "enough to serve as distillation supervision.\n"
    )
    add(
        f"- **Selected-mode agreement (repeat):** {agg.get('mean_selected_mode_agreement', 'n/a')}\n"
    )
    add(
        f"- **Mean pairwise probability L1 (repeat):** {agg.get('mean_pairwise_probability_l1', 'n/a')}\n"
    )
    add(
        f"- **Mean probability std (repeat):** {agg.get('mean_probability_std', 'n/a')}\n"
    )
    add(
        f"- **Mean confidence std (repeat):** {agg.get('mean_confidence_std', 'n/a')}\n"
    )
    add(
        f"- **Mean departure-shift std (repeat):** {agg.get('mean_departure_shift_std', 'n/a')}\n"
    )
    add(f"- **Total audit flags:** {len(flags)}\n")
    add(f"\n**Verdict:** {recommendation}\n")

    # --- API reliability ---
    add("## API Reliability\n")
    add("```json")
    add(json.dumps(api_stats, ensure_ascii=False, indent=2))
    add("```\n")

    # --- Repeatability ---
    add("## Repeatability\n")
    add("```json")
    add(json.dumps(repeatability, ensure_ascii=False, indent=2))
    add("```\n")

    # --- Counterfactual ---
    add("## Counterfactual Behavior\n")
    for axis, res in sorted(counterfactual.get("axes", {}).items()):
        add(f"\n### axis = {axis}  (groups: {res['n_groups']})\n")
        for mode, m in sorted(res.get("mode_stats", {}).items()):
            add(
                f"- mode **{mode}**: spearman_rho = {m['spearman_rho']}, "
                f"points = {m['n_points']}, mean_direction_reversals = {m['mean_direction_reversals']}"
            )
    add("\nResponse curves (per group):\n")
    add("```json")
    add(json.dumps(counterfactual.get("curves", []), ensure_ascii=False, indent=2))
    add("```\n")

    # --- Elasticity preview ---
    add("## Elasticity Preview\n")
    add("```json")
    add(json.dumps(elasticity, ensure_ascii=False, indent=2))
    add("```\n")

    # --- Persona sensitivity ---
    add("## Persona Sensitivity\n")
    p_agg = persona.get("aggregate", {})
    add(
        f"Mean pairwise L1 across persona groups: {p_agg.get('mean_pairwise_l1_across_groups', 'n/a')}\n"
    )
    add(
        f"Groups with selected-mode difference: {p_agg.get('groups_with_mode_difference', 'n/a')}\n"
    )
    add("```json")
    add(json.dumps(persona.get("per_group", []), ensure_ascii=False, indent=2))
    add("```\n")

    # --- Suspicious cases ---
    add("## Suspicious Cases\n")
    if not flags:
        add("_No suspicious cases flagged._\n")
    else:
        top = flags[:10]
        add(f"Top {len(top)} flagged cases:\n")
        for f in top:
            add(f"- `{f['flag']}` — {json.dumps(f, ensure_ascii=False)}")

    # --- Recommendation ---
    add("\n## Recommendation\n")
    add(recommendation + "\n")
    if recommendation.startswith("A."):
        add("Single-call teacher outputs are stable enough to use directly as distillation targets.\n")
    elif recommendation.startswith("B."):
        add(
            "Teacher outputs show non-trivial variance; aggregate K repeated calls per state "
            "(mean behavioral preference distribution) when building the final teacher dataset.\n"
        )
    else:
        add(
            "Teacher/prompt should be revised before any student training; do not proceed to "
            "distillation with the current outputs.\n"
        )

    return "\n".join(lines)
