"""Reference pipeline configuration (strict, validated YAML schema).

Every field maps to something the production pipeline actually consumes — no
invented options. Relative paths are resolved against the repository root
(``<repo>/``), matching the convention used by all existing scripts and
configs. Unknown top-level keys are rejected so typos fail loudly instead of
silently falling back to defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from traveler_distillation.config import load_yaml as _load_yaml


class PathsConfig(BaseModel):
    population_input: str
    matsim_network: str
    transit_schedule: str
    transit_vehicles: str
    transit_stops: str
    stop_snapping: str
    trips_by_stop: str
    activity_nodes: str
    output_dir: str
    student_checkpoint: str


class StudentConfig(BaseModel):
    model_type: Literal["student_s8_v1"] = "student_s8_v1"
    batch_size: int = Field(default=256, ge=1, le=4096)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class ScenarioWeather(BaseModel):
    condition: Literal["clear", "rain", "snow", "heat", "cold", "wind", "other"] = "clear"
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)


class ScenarioConfig(BaseModel):
    """DynamicContext fields (schemas/context.py) — the scenario the Student
    sees. transit_delay / road_disruption are applied to alternatives with the
    SAME formulas as the production Phase C runner."""

    context_id: str = "reference"
    weather: ScenarioWeather = Field(default_factory=ScenarioWeather)
    road_congestion: float = Field(default=0.3, ge=0.0, le=1.0)
    transit_delay_min: int = Field(default=0, ge=0)
    transit_disruption: bool = False
    road_disruption: bool = False
    fare_multiplier: float = Field(default=1.0, gt=0.0)
    parking_cost_multiplier: float = Field(default=1.0, gt=0.0)
    congestion_charge: float = Field(default=0.0, ge=0.0)


class CacheConfig(BaseModel):
    enabled: bool = True
    directory: str = ""  # default: <output_dir>/cache
    reuse: bool = True   # load an existing compatible cache (validated by meta)
    rebuild: bool = False  # ignore any existing cache and overwrite it


class MatsimConfig(BaseModel):
    run_after_build: bool = False
    config_file: str = ""  # default: the config.xml generated in output_dir
    java: str = "java"
    java_xmx: str = "6g"
    timeout_s: int = Field(default=3600, ge=60)
    matsim_dir: str = "tools/matsim-2026.0-release/matsim-2026.0"
    launcher_java: str = "tools/java/RunMatsimPreloaded.java"
    classpath_override: str = ""  # optional full java -cp value (expert)
    flow_capacity_factor: float | None = 0.3
    storage_capacity_factor: float | None = 0.3


class ReferenceConfig(BaseModel):
    pipeline_name: str = "reference_pipeline"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    paths: PathsConfig
    student: StudentConfig = Field(default_factory=StudentConfig)
    scenario: ScenarioConfig = Field(default_factory=ScenarioConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    feature_mapping: dict[str, str] = Field(default_factory=dict)
    matsim: MatsimConfig = Field(default_factory=MatsimConfig)


def _reject_unknown(raw: dict, allowed: dict, where: str) -> None:
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise ValueError(
            f"unknown key(s) in {where}: {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )


def load_reference_config(path: str | Path, repo_root: str | Path) -> ReferenceConfig:
    """Load and validate a reference pipeline YAML.

    Rejects unknown top-level and section keys (typo safety). Relative paths
    are resolved against ``repo_root``.
    """
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"config file {path} did not contain a mapping")

    top_allowed = {
        "pipeline_name", "log_level", "paths", "student", "scenario",
        "cache", "feature_mapping", "matsim",
    }
    _reject_unknown(raw, top_allowed, f"config file {path} (top level)")

    section_allowed = {
        "paths": {"population_input", "matsim_network", "transit_schedule",
                  "transit_vehicles", "transit_stops", "stop_snapping",
                  "trips_by_stop", "activity_nodes", "output_dir",
                  "student_checkpoint"},
        "student": {"model_type", "batch_size", "device"},
        "scenario": {"context_id", "weather", "road_congestion", "transit_delay_min",
                     "transit_disruption", "road_disruption", "fare_multiplier",
                     "parking_cost_multiplier", "congestion_charge"},
        "cache": {"enabled", "directory", "reuse", "rebuild"},
        "matsim": {"run_after_build", "config_file", "java", "java_xmx",
                   "timeout_s", "matsim_dir", "launcher_java",
                   "classpath_override", "flow_capacity_factor",
                   "storage_capacity_factor"},
    }
    for section, allowed in section_allowed.items():
        if section in raw and isinstance(raw[section], dict):
            _reject_unknown(raw[section], allowed, f"{section} section")
        elif section in raw and raw[section] is None:
            raw[section] = {}
    if "scenario" in raw and isinstance(raw.get("scenario"), dict):
        weather = raw["scenario"].get("weather")
        if weather is None:
            weather = {}
        if isinstance(weather, dict):
            _reject_unknown(weather, {"condition", "intensity"}, "scenario.weather")

    cfg = ReferenceConfig(**raw)

    root = Path(repo_root).resolve()
    for name, value in cfg.paths.model_dump().items():
        p = Path(value)
        if not p.is_absolute():
            cfg.paths.__setattr__(name, str((root / p).resolve()))
    if cfg.cache.directory:
        p = Path(cfg.cache.directory)
        cfg.cache.directory = str((p if p.is_absolute() else root / p).resolve())
    if cfg.matsim.matsim_dir and not Path(cfg.matsim.matsim_dir).is_absolute():
        cfg.matsim.matsim_dir = str((root / cfg.matsim.matsim_dir).resolve())
    if cfg.matsim.launcher_java and not Path(cfg.matsim.launcher_java).is_absolute():
        cfg.matsim.launcher_java = str((root / cfg.matsim.launcher_java).resolve())
    return cfg
