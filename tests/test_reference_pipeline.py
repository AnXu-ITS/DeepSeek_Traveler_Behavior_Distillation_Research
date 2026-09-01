"""Fast unit tests for the Reference Pipeline package (no supply/model load).

Covered: input spec consistency, CSV validation/mapping, config strictness,
route-cache metadata invalidation. The heavy correctness validation lives in
scripts/reference/validate_reference_pipeline.py (see
docs/REFERENCE_PIPELINE_VALIDATION.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from reference_pipeline.config import load_reference_config  # noqa: E402
from reference_pipeline.route_cache import ROUTING_STATE, RouteCache  # noqa: E402
from reference_pipeline.validators import (  # noqa: E402
    CANONICAL_COLUMNS,
    FIELD_SPEC,
    REQUIRED_COLUMNS,
    ValidationError,
    apply_feature_mapping,
    build_population_objects,
    read_population_csv,
)


def _write_rows(tmp_path: Path, header: list[str], rows: list[list[str]]) -> Path:
    import csv
    p = tmp_path / "population.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return p


_VALID_ROW = [
    "P0001", "25-34", "medium", "office_worker", "2", "false", "true", "true",
    "false", "false", "car", "low", "none",
    "T0001", "commute", "home", "work", "12.4", "480", "540", "hard",
    "", "",  # origin_node / dest_node omitted -> default deterministic OD
]


def _header() -> list[str]:
    return list(FIELD_SPEC.keys())


# ---------------------------------------------------------------------------
def test_field_spec_consistency():
    assert list(FIELD_SPEC.keys()) == CANONICAL_COLUMNS
    assert list(FIELD_SPEC.keys())[:13] == [
        "persona_id", "age_group", "income_group", "occupation", "household_size",
        "has_children", "car_ownership", "driving_license", "bike_ownership",
        "transit_pass", "habitual_mode", "schedule_flexibility", "mobility_limitation",
    ]
    for name, spec in FIELD_SPEC.items():
        assert spec["type"] in ("str", "int", "float", "bool")
        assert spec["required"] in (True, False)


def test_valid_csv_parses(tmp_path):
    p = _write_rows(tmp_path, _header(), [_VALID_ROW])
    header, rows = read_population_csv(p)
    col_map = apply_feature_mapping(header, {})
    personas, trips, tpp, od = build_population_objects(rows, col_map)
    assert len(personas) == 1 and len(trips) == 1
    assert personas[0].persona_id == "P0001"
    assert trips[0].purpose == "commute"
    assert trips[0].destination_type == "work"
    assert len(od) == 0


def test_feature_mapping_renames_columns(tmp_path):
    header = _header()
    header[1] = "AGE_GROUP"  # external name
    row = _VALID_ROW[:]
    p = _write_rows(tmp_path, header, [row])
    _, rows = read_population_csv(p)
    col_map = apply_feature_mapping(header, {"AGE_GROUP": "age_group"})
    personas, _, _, _ = build_population_objects(rows, col_map)
    assert personas[0].age_group == "25-34"


def test_feature_mapping_rejects_unknown_target(tmp_path):
    header = _header()
    with pytest.raises(ValidationError, match="not a known schema field"):
        apply_feature_mapping(header, {"X": "not_a_field"})


def test_missing_required_column_fails(tmp_path):
    header = _header()[1:]  # drop persona_id
    row = _VALID_ROW[1:]
    p = _write_rows(tmp_path, header, [row])
    _, rows = read_population_csv(p)
    with pytest.raises(ValidationError, match="missing required columns"):
        build_population_objects(rows, apply_feature_mapping(header, {}))


def test_bad_enum_fails_with_row_number(tmp_path):
    row = _VALID_ROW[:]
    row[1] = "17-24"  # invalid age_group
    p = _write_rows(tmp_path, _header(), [row])
    _, rows = read_population_csv(p)
    with pytest.raises(ValidationError, match="row 2"):
        build_population_objects(rows, apply_feature_mapping(_header(), {}))


def test_bad_bool_fails(tmp_path):
    row = _VALID_ROW[:]
    row[5] = "maybe"  # has_children
    p = _write_rows(tmp_path, _header(), [row])
    _, rows = read_population_csv(p)
    with pytest.raises(ValidationError, match="must be a boolean"):
        build_population_objects(rows, apply_feature_mapping(_header(), {}))


def test_od_columns_must_be_paired(tmp_path):
    header = _header()[:-1]  # canonical columns minus dest_node
    row = _VALID_ROW[:22]
    row[21] = "12345"        # origin_node given, dest_node column absent
    p = _write_rows(tmp_path, header, [row])
    _, rows = read_population_csv(p)
    with pytest.raises(ValidationError, match="given together"):
        build_population_objects(rows, apply_feature_mapping(header, {}))


def test_row_wider_than_header_fails(tmp_path):
    p = _write_rows(tmp_path, _header()[:-1], [_VALID_ROW])  # 23 values, 22 cols
    with pytest.raises(ValidationError, match="more columns than the header"):
        read_population_csv(p)


def test_config_rejects_unknown_keys(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "paths:\n  population_input: x\n  matsim_network: y\n"
        "  transit_schedule: z\n  transit_vehicles: z\n  transit_stops: z\n"
        "  stop_snapping: z\n  trips_by_stop: z\n  activity_nodes: z\n"
        "  output_dir: o\n  student_checkpoint: s\nbogus_key: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown key.*bogus_key"):
        load_reference_config(cfg, tmp_path)


def test_route_cache_meta_invalidation(tmp_path):
    supply = {}
    for name in ("network", "schedule", "vehicles", "stops", "snapping", "trips_by_stop"):
        f = tmp_path / f"{name}.txt"
        f.write_text(f"v1 {name}", encoding="utf-8")
        supply[name] = f
    cache_path = tmp_path / "route_cache.pkl"
    c1 = RouteCache(cache_path, supply, reuse=True, rebuild=True)
    assert not c1.loaded
    c1._acc[("a", "b", 1.0)] = {"pt_feasible": 1.0}
    c1.flush()
    # same supply -> reuse
    c2 = RouteCache(cache_path, supply, reuse=True)
    assert c2.loaded and ("a", "b", 1.0) in c2._acc
    # changed network -> invalidated
    (tmp_path / "network.txt").write_text("v2 network", encoding="utf-8")
    c3 = RouteCache(cache_path, supply, reuse=True)
    assert not c3.loaded
    # changed routing state -> invalidated
    (tmp_path / "network.txt").write_text("v1 network", encoding="utf-8")
    state2 = dict(ROUTING_STATE)
    state2["walk_speed_ms"] = 9.99
    c4 = RouteCache(cache_path, supply, reuse=True, routing_state=state2)
    assert not c4.loaded


def test_route_cache_rebuild_flag(tmp_path):
    supply = {}
    for name in ("network", "schedule", "vehicles", "stops", "snapping", "trips_by_stop"):
        f = tmp_path / f"{name}.txt"
        f.write_text("v1", encoding="utf-8")
        supply[name] = f
    cache_path = tmp_path / "route_cache.pkl"
    c1 = RouteCache(cache_path, supply, reuse=True, rebuild=True)
    c1._acc[("a", "b", 1.0)] = {"pt_feasible": 0.0}
    c1.flush()
    c2 = RouteCache(cache_path, supply, reuse=True, rebuild=True)
    assert not c2.loaded  # rebuild ignores the existing file
