"""Input validation: population CSV schema + field mapping + supply checks.

The canonical field set is EXACTLY what the production schemas
(``schemas/persona.py``, ``schemas/trip.py``) and the production generators
(``PersonaGenerator`` / ``TripGenerator``) consume — nothing is invented here.
Users with differently-named columns provide a ``feature_mapping``
(external name -> canonical name) in the YAML; there is no auto-guessing.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from traveler_distillation.schemas.persona import Persona
from traveler_distillation.schemas.trip import Trip

# ---------------------------------------------------------------------------
# Input specification (canonical names). `default` is only applied to fields
# marked optional=True; required fields have no default and missing rows fail.
# ---------------------------------------------------------------------------
FIELD_SPEC: dict[str, dict[str, Any]] = {
    # --- static traveler attributes (Persona) ---
    "persona_id": {"kind": "persona", "required": True, "type": "str",
                   "unit": "-", "allowed": "any non-empty string", "default": None},
    "age_group": {"kind": "persona", "required": True, "type": "str",
                  "unit": "-", "allowed": "18-24 | 25-34 | 35-44 | 45-64 | 65+", "default": None},
    "income_group": {"kind": "persona", "required": True, "type": "str",
                     "unit": "-", "allowed": "low | medium | high", "default": None},
    "occupation": {"kind": "persona", "required": True, "type": "str",
                   "unit": "-", "allowed": "student | office_worker | service_worker | manual_worker | retired | unemployed | other", "default": None},
    "household_size": {"kind": "persona", "required": True, "type": "int",
                       "unit": "persons", "allowed": ">= 1", "default": None},
    "has_children": {"kind": "persona", "required": True, "type": "bool",
                     "unit": "-", "allowed": "true/false (1/0, yes/no)", "default": None},
    "car_ownership": {"kind": "persona", "required": True, "type": "bool",
                      "unit": "-", "allowed": "true/false (1/0, yes/no)", "default": None},
    "driving_license": {"kind": "persona", "required": True, "type": "bool",
                        "unit": "-", "allowed": "true/false (1/0, yes/no)", "default": None},
    "bike_ownership": {"kind": "persona", "required": True, "type": "bool",
                       "unit": "-", "allowed": "true/false (1/0, yes/no)", "default": None},
    "transit_pass": {"kind": "persona", "required": True, "type": "bool",
                     "unit": "-", "allowed": "true/false (1/0, yes/no)", "default": None},
    "habitual_mode": {"kind": "persona", "required": True, "type": "str",
                      "unit": "-", "allowed": "car | pt | bike | walk | mixed", "default": None},
    "schedule_flexibility": {"kind": "persona", "required": True, "type": "str",
                             "unit": "-", "allowed": "low | medium | high", "default": None},
    "mobility_limitation": {"kind": "persona", "required": True, "type": "str",
                            "unit": "-", "allowed": "none | mild | significant", "default": None},
    # --- trip attributes ---
    "trip_id": {"kind": "trip", "required": True, "type": "str",
                "unit": "-", "allowed": "any non-empty string", "default": None},
    "purpose": {"kind": "trip", "required": True, "type": "str",
                "unit": "-", "allowed": "commute | education | shopping | leisure | healthcare | escort | other", "default": None},
    "origin_type": {"kind": "trip", "required": False, "type": "str",
                    "unit": "-", "allowed": "any non-empty string", "default": "home"},
    "destination_type": {"kind": "trip", "required": False, "type": "str",
                         "unit": "-", "allowed": "any non-empty string",
                         "default": "from purpose (production template convention: commute->work, education/escort->school, shopping->shop, leisure->leisure, healthcare->healthcare, other->other)"},
    "distance_km": {"kind": "trip", "required": True, "type": "float",
                    "unit": "km", "allowed": "> 0", "default": None},
    "desired_departure_min": {"kind": "trip", "required": True, "type": "int",
                              "unit": "minutes after midnight (0-1439)", "allowed": "0 .. 1439", "default": None},
    "desired_arrival_min": {"kind": "trip", "required": True, "type": "int",
                            "unit": "minutes after midnight (0-1439)", "allowed": "0 .. 1439", "default": None},
    "time_constraint": {"kind": "trip", "required": True, "type": "str",
                        "unit": "-", "allowed": "soft | medium | hard", "default": None},
    "origin_node": {"kind": "od", "required": False, "type": "str",
                    "unit": "-", "allowed": "network node id (car-accessible, walk-largest component)",
                    "default": "deterministic hash of persona_id over the activity-node pool (production behavior)"},
    "dest_node": {"kind": "od", "required": False, "type": "str",
                  "unit": "-", "allowed": "network node id (car-accessible, walk-largest component)",
                  "default": "deterministic hash of persona_id + trip index over the activity-node pool (production behavior)"},
}

CANONICAL_COLUMNS: list[str] = list(FIELD_SPEC.keys())
REQUIRED_COLUMNS: list[str] = [n for n, s in FIELD_SPEC.items() if s["required"]]

# production trip-purpose -> destination_type convention (TripGenerator templates)
_PURPOSE_DEST_TYPE = {
    "commute": "work",
    "education": "school",
    "shopping": "shop",
    "leisure": "leisure",
    "healthcare": "healthcare",
    "escort": "school",
    "other": "other",
}

_BOOL_TRUE = {"true", "1", "yes", "y"}
_BOOL_FALSE = {"false", "0", "no", "n"}


class ValidationError(Exception):
    """Raised when the population/config/supply inputs are invalid."""


def _parse_bool(value: str, field: str, row: int) -> bool:
    v = value.strip().lower()
    if v in _BOOL_TRUE:
        return True
    if v in _BOOL_FALSE:
        return False
    raise ValidationError(f"row {row}: field '{field}' must be a boolean, got {value!r}")


def _parse_int(value: str, field: str, row: int) -> int:
    try:
        return int(float(value.strip()))
    except ValueError:
        raise ValidationError(f"row {row}: field '{field}' must be an integer, got {value!r}")


def _parse_float(value: str, field: str, row: int) -> float:
    try:
        return float(value.strip())
    except ValueError:
        raise ValidationError(f"row {row}: field '{field}' must be numeric, got {value!r}")


def read_population_csv(csv_path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read the population CSV (utf-8, BOM-tolerant). Returns (header, rows)."""
    path = Path(csv_path)
    if not path.exists():
        raise ValidationError(f"population input not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValidationError(f"population CSV has no header: {path}")
        header = [h.strip() for h in reader.fieldnames if h.strip()]
        rows = []
        for i, row in enumerate(reader, start=2):
            if None in row:
                raise ValidationError(
                    f"population CSV row {i} has more columns than the header "
                    f"({len(header)} columns)"
                )
            rows.append({k.strip(): (v or "").strip() for k, v in row.items()})
    return header, rows


def apply_feature_mapping(header: list[str], mapping: dict[str, str] | None) -> dict[str, str]:
    """Map external CSV column names -> canonical schema names.

    Unlisted columns keep their own name (identity). No auto-guessing: a
    required canonical column that is missing after mapping is an error.
    """
    mapping = mapping or {}
    for external, canonical in mapping.items():
        if canonical not in CANONICAL_COLUMNS:
            raise ValidationError(
                f"feature_mapping target {canonical!r} is not a known schema field; "
                f"choose from: {', '.join(CANONICAL_COLUMNS)}"
            )
    seen_targets: dict[str, str] = {}
    for external, canonical in mapping.items():
        if canonical in seen_targets:
            raise ValidationError(
                f"feature_mapping maps both {seen_targets[canonical]!r} and {external!r} "
                f"to {canonical!r} — ambiguous"
            )
        seen_targets[canonical] = external
    # apply only to columns that exist (a mapping for an absent column is
    # ignored; a present external column is renamed)
    applied: dict[str, str] = {}
    for col in header:
        applied[col] = mapping.get(col, col)
    return applied


def build_population_objects(
    rows: list[dict[str, str]], col_map: dict[str, str]
) -> tuple[list[Persona], list[Trip], list[list[Trip]], dict[str, tuple[str, str]]]:
    """Turn validated CSV rows into production schema objects.

    Returns (personas, trips, trips_per_persona, explicit_od) where
    explicit_od maps (persona_id, trip_id) -> (origin_node, dest_node) for rows
    that provided OD columns; rows without OD use the production deterministic
    hash derivation instead.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in set(col_map.values())]
    if missing:
        raise ValidationError(
            f"population input is missing required columns: {', '.join(missing)}"
        )

    def _cell(row: dict[str, str], canonical: str) -> str | None:
        # find the external column(s) that map to this canonical name
        cols = [c for c, t in col_map.items() if t == canonical]
        if not cols:
            return None
        return row.get(cols[0])

    errors: list[str] = []
    personas_by_id: dict[str, Persona] = {}
    person_order: list[str] = []
    trips_by_id: dict[str, Trip] = {}
    trip_order: list[str] = []
    groups: dict[str, list[str]] = {}  # persona_id -> trip_ids in row order
    explicit_od: dict[str, tuple[str, str]] = {}

    def require(row_no: int, canonical: str, default=None):
        raw = _cell(rows[row_no], canonical)
        if raw is None or raw == "":
            if default is not None:
                return default
            raise ValidationError(f"row {row_no + 2}: required field '{canonical}' is empty")
        return raw

    for i, row in enumerate(rows):
        rn = i + 2
        try:
            # --- persona part ---
            pid = str(require(i, "persona_id"))
            if pid not in personas_by_id:
                personas_by_id[pid] = Persona(
                    persona_id=pid,
                    age_group=str(require(i, "age_group")),
                    income_group=str(require(i, "income_group")),
                    occupation=str(require(i, "occupation")),
                    household_size=_parse_int(require(i, "household_size"), "household_size", rn),
                    has_children=_parse_bool(require(i, "has_children"), "has_children", rn),
                    car_ownership=_parse_bool(require(i, "car_ownership"), "car_ownership", rn),
                    driving_license=_parse_bool(require(i, "driving_license"), "driving_license", rn),
                    bike_ownership=_parse_bool(require(i, "bike_ownership"), "bike_ownership", rn),
                    transit_pass=_parse_bool(require(i, "transit_pass"), "transit_pass", rn),
                    habitual_mode=str(require(i, "habitual_mode")),
                    schedule_flexibility=str(require(i, "schedule_flexibility")),
                    mobility_limitation=str(require(i, "mobility_limitation")),
                )
                person_order.append(pid)
            # --- trip part ---
            tid = str(require(i, "trip_id"))
            purpose = str(require(i, "purpose"))
            dest_type = require(i, "destination_type", default=None)
            if dest_type is None:
                dest_type = _PURPOSE_DEST_TYPE[purpose]
            if tid not in trips_by_id:
                trips_by_id[tid] = Trip(
                    trip_id=tid,
                    purpose=purpose,
                    origin_type=str(require(i, "origin_type", default="home")),
                    destination_type=str(dest_type),
                    distance_km=_parse_float(require(i, "distance_km"), "distance_km", rn),
                    desired_departure_min=_parse_int(require(i, "desired_departure_min"), "desired_departure_min", rn),
                    desired_arrival_min=_parse_int(require(i, "desired_arrival_min"), "desired_arrival_min", rn),
                    time_constraint=str(require(i, "time_constraint")),
                )
                trip_order.append(tid)
            groups.setdefault(pid, []).append(tid)
            # --- optional explicit OD (both columns or neither) ---
            o_raw = _cell(row, "origin_node")
            d_raw = _cell(row, "dest_node")
            has_o, has_d = bool(o_raw), bool(d_raw)
            if has_o != has_d:
                raise ValidationError(
                    f"row {rn}: origin_node and dest_node must be given together "
                    "(or both omitted for the default deterministic OD)"
                )
            if has_o and has_d:
                explicit_od[(pid, tid)] = (o_raw, d_raw)
        except ValidationError:
            raise
        except Exception as e:  # pydantic validation errors -> row-scoped message
            errors.append(f"row {rn}: {e}")

    if errors:
        raise ValidationError("population validation failed:\n  " + "\n  ".join(errors[:40]))

    personas = [personas_by_id[p] for p in person_order]
    trips = [trips_by_id[t] for t in trip_order]
    trips_per_persona = [[trips_by_id[t] for t in groups[p]] for p in person_order]
    return personas, trips, trips_per_persona, explicit_od


def validate_supply_paths(cfg) -> None:
    """Cheap existence checks for every supply file the build consumes."""
    for label, p in [
        ("matsim_network", cfg.paths.matsim_network),
        ("transit_schedule", cfg.paths.transit_schedule),
        ("transit_vehicles", cfg.paths.transit_vehicles),
        ("transit_stops", cfg.paths.transit_stops),
        ("stop_snapping", cfg.paths.stop_snapping),
        ("trips_by_stop", cfg.paths.trips_by_stop),
        ("activity_nodes", cfg.paths.activity_nodes),
        ("student_checkpoint", cfg.paths.student_checkpoint),
    ]:
        if not Path(p).exists():
            raise ValidationError(f"{label} not found: {p}")


def format_field_spec_markdown() -> str:
    """Markdown table of the input specification (used by docs)."""
    lines = [
        "| field | required | type | unit | allowed values | missing behavior |",
        "|---|---|---|---|---|---|",
    ]
    for name, spec in FIELD_SPEC.items():
        lines.append(
            f"| `{name}` | {'yes' if spec['required'] else 'no'} | {spec['type']} "
            f"| {spec['unit']} | {spec['allowed']} | "
            f"{'error' if spec['required'] else spec['default']} |"
        )
    return "\n".join(lines)
