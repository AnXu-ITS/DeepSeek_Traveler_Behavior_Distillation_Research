"""Persona sensitivity analysis."""
from __future__ import annotations

from collections import defaultdict

from .metrics import mean_pairwise_l1
from .records import PersonaContrastRecord


def _attr_value(r: PersonaContrastRecord):
    if r.attribute:
        return r.state.persona.model_dump().get(r.attribute)
    return None


def analyze_persona(records: list[PersonaContrastRecord]) -> dict:
    groups: dict[str, list[PersonaContrastRecord]] = defaultdict(list)
    for r in records:
        if r.teacher is not None:
            key = r.persona_group_id or f"unlabeled_{r.audit_state_id}"
            groups[key].append(r)

    per_group = []
    for gid in sorted(groups):
        rs = groups[gid]
        probs = [r.teacher.mode_probabilities for r in rs]
        modes = [r.teacher.selected_mode for r in rs]
        shifts = [r.teacher.departure_time_shift_min for r in rs]
        member_details = []
        for r in rs:
            member_details.append(
                {
                    "audit_state_id": r.audit_state_id,
                    "kind": r.kind,
                    "note": r.note,
                    "attribute": r.attribute,
                    "attribute_value": _attr_value(r),
                    "selected_mode": r.teacher.selected_mode,
                    "probs": r.teacher.mode_probabilities,
                    "departure_time_shift_min": r.teacher.departure_time_shift_min,
                }
            )
        per_group.append(
            {
                "group_id": gid,
                "attribute": rs[0].attribute,
                "kind": rs[0].kind,
                "n_members": len(rs),
                "mean_pairwise_l1": round(mean_pairwise_l1(probs), 4),
                "selected_modes": modes,
                "n_distinct_selected_modes": len(set(modes)),
                "departure_shifts": shifts,
                "member_details": member_details,
            }
        )

    # Aggregate: is the teacher actually using persona info?
    l1s = [g["mean_pairwise_l1"] for g in per_group if g["n_members"] >= 2]
    aggregate = {
        "n_groups": len(per_group),
        "mean_pairwise_l1_across_groups": round(
            sum(l1s) / len(l1s), 4
        ) if l1s else None,
        "groups_with_mode_difference": sum(
            1 for g in per_group if g["n_distinct_selected_modes"] > 1
        ),
    }
    return {"per_group": per_group, "aggregate": aggregate}
