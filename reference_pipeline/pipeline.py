"""Reference pipeline orchestrator: validate -> build -> (optional) MATSim.

Usage::

    from reference_pipeline import ReferencePipeline
    p = ReferencePipeline(config_path, repo_root)
    p.validate()                    # --validate-only
    summary = p.run(run_matsim=False)   # full build

Everything below is orchestration of the production modules; no behavioral
logic lives here.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from traveler_distillation.student.release_guard import assert_not_frozen_output

from .config import ReferenceConfig, load_reference_config
from .feature_adapter import MemoizedEncoder, make_context
from .matsim_adapter import build_scenario, build_supply_view, parse_events, run_matsim
from .route_cache import RouteCache
from .student_adapter import StudentAdapter
from .validators import (
    apply_feature_mapping,
    build_population_objects,
    read_population_csv,
    validate_supply_paths,
)

logger = logging.getLogger("reference_pipeline")


def _setup_logging(log_path: Path, level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_h = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_h.setFormatter(fmt)
    stream_h = logging.StreamHandler(sys.stdout)
    stream_h.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(getattr(logging, level))
    root.addHandler(file_h)
    root.addHandler(stream_h)


class ReferencePipeline:
    def __init__(self, config_path: str | Path, repo_root: str | Path | None = None):
        self.config_path = Path(config_path)
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[1]
        self.repo_root = Path(repo_root).resolve()
        self.cfg: ReferenceConfig = load_reference_config(self.config_path, self.repo_root)
        self.out_dir = assert_not_frozen_output(Path(self.cfg.paths.output_dir))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _setup_logging(self.out_dir / "pipeline.log", self.cfg.log_level)

    # ------------------------------------------------------------------ input
    def load_population(self):
        """Read + validate the population CSV; returns (personas, trips,
        trips_per_persona, explicit_od, n_rows)."""
        header, rows = read_population_csv(self.cfg.paths.population_input)
        col_map = apply_feature_mapping(header, self.cfg.feature_mapping)
        personas, trips, trips_per_persona, explicit_od = build_population_objects(rows, col_map)
        logger.info(
            "population: %d rows -> %d travelers, %d trips (explicit OD rows: %d)",
            len(rows), len(personas), len(trips), len(explicit_od),
        )
        return personas, trips, trips_per_persona, explicit_od, len(rows)

    # ------------------------------------------------------------- validation
    def validate(self) -> dict:
        """Input/config validation only (no model load, no supply graph)."""
        started = time.perf_counter()
        validate_supply_paths(self.cfg)
        personas, trips, trips_per_persona, explicit_od, n_rows = self.load_population()
        report = {
            "config": str(self.config_path),
            "valid": True,
            "population_input": self.cfg.paths.population_input,
            "num_rows": n_rows,
            "num_travelers": len(personas),
            "num_trips": len(trips),
            "explicit_od_rows": len(explicit_od),
            "scenario": self.cfg.scenario.model_dump(mode="json"),
            "student_checkpoint": self.cfg.paths.student_checkpoint,
            "matsim_run_after_build": self.cfg.matsim.run_after_build,
            "cache_enabled": self.cfg.cache.enabled,
            "validation_seconds": round(time.perf_counter() - started, 3),
        }
        logger.info("validation OK (%s)", report["validation_seconds"])
        return report

    # ------------------------------------------------------------------- run
    def run(self, run_matsim_flag: bool | None = None) -> dict:
        """Full reference build. Returns the run summary dict (also written to
        ``<output_dir>/run_summary.json``)."""
        do_matsim = self.cfg.matsim.run_after_build if run_matsim_flag is None else run_matsim_flag
        t_start = time.perf_counter()
        validate_supply_paths(self.cfg)
        personas, trips, trips_per_persona, explicit_od, n_rows = self.load_population()

        # ---- student (load once; no retraining, no weight modification) ----
        logger.info("loading student checkpoint %s", self.cfg.paths.student_checkpoint)
        t0 = time.perf_counter()
        student = StudentAdapter(self.cfg.paths.student_checkpoint,
                                 None if self.cfg.student.device == "auto" else self.cfg.student.device)
        t_student_load = time.perf_counter() - t0
        logger.info("student loaded (%d params, %s) in %.2fs",
                    student.num_parameters, student.device, t_student_load)

        # ---- supply view (network graph + activity node pool) ----
        logger.info("building supply view (network %s)", self.cfg.paths.matsim_network)
        t0 = time.perf_counter()
        supply_view = build_supply_view(self.cfg.paths.matsim_network,
                                        self.cfg.paths.activity_nodes)
        t_supply = time.perf_counter() - t0
        logger.info("supply view ready in %.2fs (%d activity nodes)",
                    t_supply, len(supply_view[3]))

        # ---- persistent route/accessibility cache ----
        cache = None
        if self.cfg.cache.enabled:
            cache_dir = Path(self.cfg.cache.directory) if self.cfg.cache.directory \
                else self.out_dir / "cache"
            cache_path = cache_dir / "route_cache.pkl"
            cache = RouteCache(
                cache_path,
                {
                    "network": self.cfg.paths.matsim_network,
                    "schedule": self.cfg.paths.transit_schedule,
                    "vehicles": self.cfg.paths.transit_vehicles,
                    "stops": self.cfg.paths.transit_stops,
                    "snapping": self.cfg.paths.stop_snapping,
                    "trips_by_stop": self.cfg.paths.trips_by_stop,
                },
                reuse=self.cfg.cache.reuse,
                rebuild=self.cfg.cache.rebuild,
            )
            logger.info("route cache: %s (%s)", cache_path,
                        "reused" if cache.loaded else "cold / new")

        # ---- context ----
        context = make_context(self.cfg.scenario)
        logger.info("scenario: %s", context.model_dump(mode="json"))

        # ---- build (batch decisions + cached routing + production XML) ----
        supply_paths = {
            "network": self.cfg.paths.matsim_network,
            "schedule": self.cfg.paths.transit_schedule,
            "vehicles": self.cfg.paths.transit_vehicles,
            "stops": self.cfg.paths.transit_stops,
            "snapping": self.cfg.paths.stop_snapping,
            "trips_by_stop": self.cfg.paths.trips_by_stop,
            "activity_nodes": self.cfg.paths.activity_nodes,
        }
        encoder = MemoizedEncoder(student.extractor)  # static-part precompute, parity self-checked
        logger.info("building scenario (%d travelers, %d trips, batch_size=%d)",
                    len(personas), len(trips), self.cfg.student.batch_size)
        result = build_scenario(
            student, personas, trips_per_persona, context, self.out_dir, supply_paths,
            supply_view, batch_size=self.cfg.student.batch_size, cache=cache,
            flow_capacity_factor=self.cfg.matsim.flow_capacity_factor,
            storage_capacity_factor=self.cfg.matsim.storage_capacity_factor,
            explicit_od=explicit_od, encoder=encoder,
        )
        manifest = result["manifest"]
        timings = result["timings"]
        logger.info("build done in %.1fs (feature %.1fs / inference %.2fs / plan+xml %.1fs)",
                    timings["total_build_s"], timings["feature_preparation_s"],
                    timings["student_inference_s"], timings["plan_and_xml_s"])

        if cache is not None:
            cache.flush()
            cache_summary = cache.summary()
            cache_summary["cache_size_bytes"] = cache.size_bytes
            logger.info(
                "cache: hit_rate=%.1f%% (%d/%d calls), entries=%s, size=%.2f MB, "
                "estimated routing time saved=%.1fs",
                cache.hit_rate * 100, cache.total_hits, cache.total_calls,
                cache.entry_counts, cache.size_bytes / 1e6, cache.time_saved_s,
            )
        else:
            cache_summary = None

        # ---- decision manifest (CSV + JSON) ----
        self._write_decision_csv(manifest, result["decisions"])
        mode_stats = self._mode_stats(manifest)

        # ---- optional MATSim execution ----
        matsim_record = None
        if do_matsim:
            logger.info("running MATSim (launcher %s)", self.cfg.matsim.launcher_java)
            t0 = time.perf_counter()
            code, tail = run_matsim(self.out_dir, self.cfg.matsim)
            wall = time.perf_counter() - t0
            t0 = time.perf_counter()
            events = parse_events(self.out_dir) if code == 0 else None
            parse_s = time.perf_counter() - t0
            matsim_record = {
                "exit_code": code,
                "wall_clock_s": round(wall, 1),
                "events_parse_s": round(parse_s, 2),
                "events": events,
                "log_tail": tail,
            }
            logger.info("MATSim exit=%s in %.1fs", code, wall)
            if code != 0:
                logger.error("MATSim log tail:\n%s", tail)

        # ---- run summary ----
        summary = {
            "pipeline": "reference_matsim_integration",
            "config": str(self.config_path),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "inputs": {
                "population_input": self.cfg.paths.population_input,
                "num_rows": n_rows,
                "num_travelers": len(personas),
                "num_trips": len(trips),
                "explicit_od_rows": len(explicit_od),
            },
            "student": {
                "checkpoint": self.cfg.paths.student_checkpoint,
                "checkpoint_sha256": student.checkpoint_sha256(),
                "num_parameters": student.num_parameters,
                "device": student.device,
                "batch_size": self.cfg.student.batch_size,
                "load_seconds": round(t_student_load, 3),
                "supply_view_seconds": round(t_supply, 3),
            },
            "scenario": context.model_dump(mode="json"),
            "mode_distribution": mode_stats,
            "fallback_counts": result["fallback_counts"],
            "timings": timings,
            "cache": cache_summary,
            "matsim": matsim_record,
            "end_to_end_wall_clock_s": round(time.perf_counter() - t_start, 3),
        }
        (self.out_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.out_dir / "decision_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("outputs written to %s", self.out_dir)
        return summary

    # --------------------------------------------------------------- helpers
    def _write_decision_csv(self, manifest: list[dict], decisions: list[dict]) -> None:
        columns = ["persona_id", "trip_id", "student_mode",
                   "p_car", "p_pt", "p_bike", "p_walk",
                   "departure_shift_min", "departure_min",
                   "outbound_mode", "return_mode", "home_node", "dest_node"]
        with (self.out_dir / "decision_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for m, d in zip(manifest, decisions):
                probs = d["mode_probabilities"]
                writer.writerow({
                    "persona_id": m["persona_id"], "trip_id": m["trip_id"],
                    "student_mode": m["student_mode"],
                    "p_car": probs.get("car", ""), "p_pt": probs.get("pt", ""),
                    "p_bike": probs.get("bike", ""), "p_walk": probs.get("walk", ""),
                    "departure_shift_min": m["departure_shift_min"],
                    "departure_min": m["departure_min"],
                    "outbound_mode": m["outbound_mode"],
                    "return_mode": m["return_mode"],
                    "home_node": m["home_node"], "dest_node": m["dest_node"],
                })

    @staticmethod
    def _mode_stats(manifest: list[dict]) -> dict:
        student_modes = Counter(m["student_mode"] for m in manifest)
        executed = Counter(m["outbound_mode"] for m in manifest)
        return {
            "student_mode_counts": dict(sorted(student_modes.items())),
            "student_mode_share": {k: round(v / max(1, len(manifest)), 4)
                                   for k, v in sorted(student_modes.items())},
            "executed_outbound_counts": dict(sorted(executed.items())),
            "n_decisions": len(manifest),
        }
