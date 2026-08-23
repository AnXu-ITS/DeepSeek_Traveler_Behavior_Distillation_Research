"""Teacher dataset builder: orchestrates generation -> teacher -> validation -> JSONL."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ..schemas.dataset import (
    Perturbation,
    TeacherMetadata,
    TeacherDatasetSample,
    FailedSample,
)
from ..schemas.state import UniversalTravelerState
from ..generators.persona_generator import PersonaGenerator
from ..generators.trip_generator import TripGenerator
from ..generators.baseline_generator import BaselineStateGenerator
from ..generators.counterfactual_generator import CounterfactualContextGenerator
from .writer import JSONLWriter, load_existing_ids
from .statistics import StatisticsAccumulator

logger = logging.getLogger(__name__)


class TeacherDatasetBuilder:
    def __init__(
        self,
        generation_config: dict,
        teacher_config: dict,
        teacher_client,
        parser,
        validator,
        output_dir: str | Path = "data/teacher_v0_1",
        overwrite: bool = False,
        resume: bool = False,
        dataset_filename: str = "teacher_dataset_v0_1.jsonl",
        failed_filename: str = "failed_samples_v0_1.jsonl",
    ):
        self.gen_cfg = generation_config
        self.t_cfg = (teacher_config or {}).get("teacher", {})
        self.teacher = teacher_client
        self.parser = parser
        self.validator = validator
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.resume = resume
        self.dataset_filename = dataset_filename
        self.failed_filename = failed_filename

        self.dataset_version = generation_config.get("dataset", {}).get("version", "unknown")
        self.prompt_version = self.t_cfg.get("prompt_version", "teacher_v0.1")
        self.model = getattr(teacher_client, "model", "unknown")

    # ------------------------------------------------------------------ main
    def build(
        self,
        num_personas: int,
        num_trips: int,
        seed: int,
        max_samples: int | None = None,
        dry_run: bool = False,
    ) -> dict:
        start = datetime.now(timezone.utc).isoformat()

        personas = PersonaGenerator(seed=seed, config=self.gen_cfg).generate(num_personas)
        trips = TripGenerator(seed=seed, config=self.gen_cfg).generate(num_trips)
        perturbations = self.gen_cfg.get("perturbations", {})

        if dry_run:
            return self._dry_run(personas, trips, perturbations, start)

        # Prepare output files.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = self.output_dir / self.dataset_filename
        failed_path = self.output_dir / self.failed_filename

        existing_ids = set()
        if self.resume:
            # Skip samples already attempted (both valid and previously failed)
            # so ``--max-samples`` counts only genuinely new samples and we do
            # not re-burn API calls on external-infra failures.
            existing_ids = load_existing_ids(dataset_path)
            existing_ids |= load_existing_ids(failed_path)

        stats = StatisticsAccumulator()
        manifest = {
            "dataset_version": self.dataset_version,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "software_version": __version__,
            "seed": seed,
            "num_personas": len(personas),
            "num_trips": len(trips),
            "counterfactual_axes": sorted(perturbations.keys()),
            "start_time": start,
        }

        sample_seq = 0
        cf_group_seq = 0
        stopped = False

        with JSONLWriter(dataset_path, overwrite=self.overwrite, resume=self.resume) as dataset_w, \
             JSONLWriter(failed_path, overwrite=self.overwrite, resume=self.resume) as failed_w:

            baseline_gen = BaselineStateGenerator(self.gen_cfg)
            cf_gen = CounterfactualContextGenerator(self.gen_cfg)

            for persona in personas:
                if stopped:
                    break
                for trip in trips:
                    if stopped:
                        break

                    # ----- baseline sample -----
                    baseline_state = baseline_gen.generate(persona, trip)
                    sample_seq += 1
                    baseline_id = f"S{sample_seq:06d}"

                    stopped = self._emit(
                        sample_id=baseline_id,
                        state=baseline_state,
                        perturbation=Perturbation(axis="baseline", level=0.0),
                        cf_group_id=None,
                        pg_group_id=None,
                        baseline_sample_id=None,
                        dataset_w=dataset_w,
                        failed_w=failed_w,
                        stats=stats,
                        existing_ids=existing_ids,
                        max_samples=max_samples,
                    )

                    # ----- counterfactual samples -----
                    for axis, spec in perturbations.items():
                        if stopped:
                            break
                        levels = spec.get("levels", [])
                        cf_group_seq += 1
                        group_id = f"CF_{axis}_{cf_group_seq:06d}"
                        for cs in cf_gen.generate(baseline_state, axis, levels):
                            if stopped:
                                break
                            sample_seq += 1
                            sample_id = f"S{sample_seq:06d}"
                            stopped = self._emit(
                                sample_id=sample_id,
                                state=cs.state,
                                perturbation=Perturbation(axis=cs.axis, level=cs.level),
                                cf_group_id=group_id,
                                pg_group_id=None,
                                baseline_sample_id=baseline_id,
                                dataset_w=dataset_w,
                                failed_w=failed_w,
                                stats=stats,
                                existing_ids=existing_ids,
                                max_samples=max_samples,
                            )

        # Finalize manifest + statistics.
        end = datetime.now(timezone.utc).isoformat()
        stats_dict = stats.to_dict()
        manifest.update(
            {
                "end_time": end,
                "number_requested": stats_dict["total_samples"],
                "number_valid": stats_dict["valid_samples"],
                "number_failed": stats_dict["failed_samples"],
                "stopped_by_max_samples": stopped and max_samples is not None,
            }
        )
        (self.output_dir / "generation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.output_dir / "dataset_statistics.json").write_text(
            json.dumps(stats_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info(
            "Dataset generation complete: valid=%d failed=%d",
            stats_dict["valid_samples"],
            stats_dict["failed_samples"],
        )
        return {"manifest": manifest, "statistics": stats_dict}

    # ----------------------------------------------------------------- helper
    def _emit(
        self,
        sample_id: str,
        state: UniversalTravelerState,
        perturbation: Perturbation,
        cf_group_id: str | None,
        pg_group_id: str | None,
        baseline_sample_id: str | None,
        dataset_w: JSONLWriter,
        failed_w: JSONLWriter,
        stats: StatisticsAccumulator,
        existing_ids: set[str],
        max_samples: int | None,
    ) -> bool:
        """Process one state through the teacher. Returns True if capped by max_samples."""
        if max_samples is not None and stats.total_samples >= max_samples:
            return True
        if self.resume and sample_id in existing_ids:
            return False

        metadata = TeacherMetadata(
            model=self.model,
            prompt_version=self.prompt_version,
            dataset_version=self.dataset_version,
        )

        # 1. teacher call
        try:
            raw = self.teacher.respond(state)
        except Exception as exc:  # TeacherClientError or unexpected
            ft = getattr(exc, "failure_type", "api_error")
            self._record_failure(
                failed_w, stats, sample_id, cf_group_id, pg_group_id, baseline_sample_id,
                perturbation, state, ft, str(exc),
            )
            logger.warning("Teacher sample %s failed (%s)", sample_id, ft)
            return False

        # 2. parse
        try:
            action = self.parser.parse(raw)
        except Exception as exc:
            self._record_failure(
                failed_w, stats, sample_id, cf_group_id, pg_group_id, baseline_sample_id,
                perturbation, state, "parse_failure", str(exc),
            )
            logger.warning("Teacher sample %s parse failed", sample_id)
            return False

        # 3. validate
        result = self.validator.validate(state, action)
        if not result.valid:
            self._record_failure(
                failed_w, stats, sample_id, cf_group_id, pg_group_id, baseline_sample_id,
                perturbation, state, "validation_failure", result.reason,
            )
            logger.warning("Teacher sample %s invalid: %s", sample_id, result.reason)
            return False

        # 4. valid -> write
        sample = TeacherDatasetSample(
            sample_id=sample_id,
            counterfactual_group_id=cf_group_id,
            persona_group_id=pg_group_id,
            baseline_sample_id=baseline_sample_id,
            perturbation=perturbation,
            state=state,
            teacher=action,
            teacher_metadata=metadata,
        )
        dataset_w.append(json.loads(sample.model_dump_json()))
        stats.record_valid(sample)
        logger.info("Teacher sample %s valid", sample_id)
        return False

    @staticmethod
    def _record_failure(
        failed_w: JSONLWriter,
        stats: StatisticsAccumulator,
        sample_id: str,
        cf_group_id: str | None,
        pg_group_id: str | None,
        baseline_sample_id: str | None,
        perturbation: Perturbation,
        state: UniversalTravelerState,
        failure_type: str,
        error: str,
    ) -> None:
        failed = FailedSample(
            sample_id=sample_id,
            counterfactual_group_id=cf_group_id,
            persona_group_id=pg_group_id,
            baseline_sample_id=baseline_sample_id,
            perturbation=perturbation,
            failure_type=failure_type,
            attempts=1,
            error=error[:500],
            state=state,
        )
        failed_w.append(json.loads(failed.model_dump_json()))
        stats.record_failure(failed)

    # ---------------------------------------------------------------- dry run
    def _dry_run(self, personas, trips, perturbations, start: str) -> dict:
        baseline_gen = BaselineStateGenerator(self.gen_cfg)
        cf_gen = CounterfactualContextGenerator(self.gen_cfg)

        total_states = 0
        example = None
        n_cf_groups = 0
        for persona in personas:
            for trip in trips:
                baseline_state = baseline_gen.generate(persona, trip)
                total_states += 1
                if example is None:
                    example = baseline_state
                for axis, spec in perturbations.items():
                    n_cf_groups += 1
                    total_states += len(cf_gen.generate(baseline_state, axis, spec.get("levels", [])))

        print("=" * 50)
        print("DRY RUN")
        print("=" * 50)
        print(f"Personas: {len(personas)}")
        print(f"Trips: {len(trips)}")
        print(f"Persona-trip pairs: {len(personas) * len(trips)}")
        print(f"Counterfactual groups: {n_cf_groups}")
        print(f"Estimated teacher calls: {total_states}")
        print()
        print("Example state (first persona, first trip, baseline):")
        if example is not None:
            print(json.dumps(json.loads(example.model_dump_json()), ensure_ascii=False, indent=2)[:1500])
        print()
        print("DRY RUN COMPLETE")
        print("No API calls made.")
        print("=" * 50)
        return {"dry_run": True, "estimated_teacher_calls": total_states}
