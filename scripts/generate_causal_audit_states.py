#!/usr/bin/env python
"""S6 causal-audit four-state (A/B/C/D) generator.

For each (baseline, audit axis) construct the four causal-audit states:

  A baseline      : context normal, mediator normal   (reuse S3 baseline target)
  B natural       : context high, mediator high       (reuse S3 single-axis CF target)
  C broken-path   : context high, mediator FIXED at baseline  (NEW, teacher TBD)
  D mediator-only : context normal, mediator high     (NEW, teacher TBD)

State C/D decouple the context FLAG from the causal MEDIATOR (the target mode's
alternative attributes), which is the core manipulation for the L4 mechanism
consistency test. C/D teacher targets are left null and filled by
``run_teacher_causal_audit.py`` (K=5); A/B reuse the S3 K=3 targets.

Usage:
    python scripts/generate_causal_audit_states.py --dry-run
    python scripts/generate_causal_audit_states.py --max-personas 6
    python scripts/generate_causal_audit_states.py --output data/causal_audit/states.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from traveler_distillation.config import load_dotenv, load_yaml
from traveler_distillation.dataset.aggregation import AggregatedTeacherTarget
from traveler_distillation.generators import AlternativeGenerator, perturb_context
from traveler_distillation.schemas.state import UniversalTravelerState


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _level_matches(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return a == b


def _override_target_mode(alts_from, alts_ref, target_mode: str, mediator_attrs: list[str]):
    """Copy ``alts_from``, overriding the target mode's mediator attrs to
    ``alts_ref``'s values. Non-target modes are kept as ``alts_from``."""
    ref = next(a for a in alts_ref if a.mode == target_mode)
    out = []
    for a in alts_from:
        if a.mode == target_mode:
            updates = {attr: getattr(ref, attr) for attr in mediator_attrs}
            out.append(a.model_copy(update=updates))
        else:
            out.append(a)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/student_v0_3_s3/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--audit-config", default="configs/causal_audit.yaml")
    ap.add_argument("--output", default="data/causal_audit/states.jsonl")
    ap.add_argument("--max-personas", type=int, default=None, help="limit personas (pilot)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    gen_cfg = load_yaml(args.config)
    a_cfg = load_yaml(args.audit_config)
    axes = a_cfg.get("audit_axes", [])

    records = _load_jsonl(Path(args.dataset))
    by_id = {r["sample_id"]: AggregatedTeacherTarget.model_validate(r) for r in records}

    baselines = [by_id[r["sample_id"]] for r in records
                 if (r.get("perturbation") or {}).get("axis") == "baseline"]
    baselines.sort(key=lambda s: s.sample_id)
    if args.max_personas is not None:
        seen = set()
        keep = []
        for b in baselines:
            pid = b.state.persona.persona_id
            if pid in seen:
                continue
            seen.add(pid)
            keep.append(b)
            if len(seen) >= args.max_personas:
                break
        baselines = keep

    # index single-axis CF by (split_group_id, axis, level)
    s3_cf = defaultdict(list)
    for r in records:
        p = r.get("perturbation") or {}
        if p.get("axis") == "baseline":
            continue
        s3_cf[(r.get("split_group_id"), p.get("axis"))].append(AggregatedTeacherTarget.model_validate(r))

    alt_gen = AlternativeGenerator(gen_cfg)
    out_records: list[dict] = []
    seq = 0

    for base in baselines:
        persona = base.state.persona
        trip = base.state.trip
        A_state = base.state
        A_target = base.teacher_aggregate

        for ax in axes:
            axis_id = ax["axis_id"]
            axis = ax["axis"]
            level = ax["intervention_level"]
            tmode = ax["target_mode"]
            mattrs = ax["mediator_attrs"]

            # State B (natural): perturb context + regenerate alternatives naturally
            B_ctx = perturb_context(A_state.context, axis, level)
            B_ctx = B_ctx.model_copy(deep=True)
            B_ctx.context_id = f"C_S6_{axis_id}_natural_{persona.persona_id}_{trip.trip_id}"
            B_state = UniversalTravelerState(
                persona=persona, trip=trip, context=B_ctx,
                alternatives=alt_gen.generate(persona, trip, B_ctx),
            )

            # State C (broken-path): B's context, A's mediator for target mode
            C_ctx = B_ctx.model_copy(deep=True)
            C_ctx.context_id = f"C_S6_{axis_id}_broken_{persona.persona_id}_{trip.trip_id}"
            C_state = UniversalTravelerState(
                persona=persona, trip=trip, context=C_ctx,
                alternatives=_override_target_mode(B_state.alternatives, A_state.alternatives, tmode, mattrs),
            )

            # State D (mediator-only): A's context, B's mediator for target mode
            D_ctx = A_state.context.model_copy(deep=True)
            D_ctx.context_id = f"C_S6_{axis_id}_mediator_{persona.persona_id}_{trip.trip_id}"
            D_state = UniversalTravelerState(
                persona=persona, trip=trip, context=D_ctx,
                alternatives=_override_target_mode(A_state.alternatives, B_state.alternatives, tmode, mattrs),
            )

            # reuse S3 teacher target for B (natural single-axis CF)
            B_target = None
            for cf in s3_cf.get((base.split_group_id, axis), []):
                if _level_matches(cf.perturbation.level, level):
                    B_target = cf.teacher_aggregate
                    break

            gid = f"{persona.persona_id}::{trip.trip_id}::{axis_id}"
            for stype, state, target in (
                ("baseline", A_state, A_target),
                ("natural", B_state, B_target),
                ("broken", C_state, None),
                ("mediator", D_state, None),
            ):
                seq += 1
                out_records.append({
                    "sample_id": f"S6_{axis_id}_{stype}_{seq:06d}",
                    "audit_group_id": gid,
                    "axis_id": axis_id,
                    "state_type": stype,
                    "persona_group_id": persona.persona_id,
                    "split_group_id": base.split_group_id,
                    "target_mode": tmode,
                    "state": json.loads(state.model_dump_json()),
                    "teacher_aggregate": json.loads(target.model_dump_json()) if target is not None else None,
                    "teacher_k": 3 if target is not None else None,
                })

    print(f"audit states = {len(out_records)}  (baselines={len(baselines)} axes={len(axes)})")
    per_type = defaultdict(int)
    per_axis = defaultdict(int)
    for r in out_records:
        per_type[r["state_type"]] += 1
        per_axis[r["axis_id"]] += 1
    print(f"  by state_type: {dict(per_type)}")
    print(f"  by axis: {dict(per_axis)}")
    new_states = sum(1 for r in out_records if r["teacher_aggregate"] is None)
    print(f"  NEW states needing teacher (C/D) = {new_states}")

    if args.dry_run:
        print("DRY RUN: no file written; verifying decoupling on first group:")
        for r in out_records:
            if r["state_type"] in ("broken", "mediator"):
                s = r["state"]
                t = next(a for a in s["alternatives"] if a["mode"] == r["target_mode"])
                print(f"  {r['sample_id']} [{r['state_type']}] ctx_{r['axis_id']}="
                      f"{s['context'].get('road_congestion', s['context'].get('transit_delay_min', s['context'].get('parking_cost_multiplier')))} "
                      f"-> {r['target_mode']} tt={t['travel_time_min']} rel={t['reliability_delay_min']} cost={t['monetary_cost']}")
                break
        return 0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
