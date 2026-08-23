"""JSONL round-trip and mock end-to-end tests."""
from __future__ import annotations

import json

from traveler_distillation.schemas.dataset import (
    Perturbation,
    TeacherMetadata,
    TeacherDatasetSample,
)
from traveler_distillation.schemas.action import UniversalTravelerAction
from traveler_distillation.teacher import MockTeacherClient, TeacherResponseParser, TeacherResponseValidator
from traveler_distillation.dataset import TeacherDatasetBuilder


def test_jsonl_round_trip(baseline_state):
    sample = TeacherDatasetSample(
        sample_id="S000001",
        counterfactual_group_id=None,
        persona_group_id=None,
        baseline_sample_id=None,
        perturbation=Perturbation(axis="baseline", level=0.0),
        state=baseline_state,
        teacher=UniversalTravelerAction(
            selected_mode="car",
            mode_probabilities={"car": 0.6, "pt": 0.3, "walk": 0.1},
            departure_time_shift_min=0,
            confidence=0.8,
        ),
        teacher_metadata=TeacherMetadata(
            model="mock", prompt_version="teacher_v0.1", dataset_version="test_v0.1"
        ),
    )
    line = sample.model_dump_json()
    assert json.loads(line)["sample_id"] == "S000001"
    restored = TeacherDatasetSample.model_validate(json.loads(line))
    assert restored.model_dump() == sample.model_dump()


def test_perturbation_level_types_round_trip():
    for level in (0.5, 5, True):
        p = Perturbation(axis="x", level=level)
        restored = Perturbation.model_validate(json.loads(p.model_dump_json()))
        assert restored.level == level
        assert type(restored.level) is type(level) or (
            isinstance(restored.level, bool) == isinstance(level, bool)
        )


def test_mock_end_to_end_offline(config, tmp_path):
    """Full pipeline with mock teacher; must complete without network."""
    teacher = MockTeacherClient()
    parser = TeacherResponseParser()
    validator = TeacherResponseValidator()
    builder = TeacherDatasetBuilder(
        generation_config=config,
        teacher_config={"teacher": {"prompt_version": "teacher_v0.1"}},
        teacher_client=teacher,
        parser=parser,
        validator=validator,
        output_dir=tmp_path / "out",
    )
    result = builder.build(num_personas=2, num_trips=2, seed=42)
    stats = result["statistics"]
    assert stats["valid_samples"] > 0
    assert stats["failed_samples"] == 0

    # Read back and validate every row.
    ds_path = tmp_path / "out" / "teacher_dataset_v0_1.jsonl"
    lines = ds_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == stats["valid_samples"]
    for line in lines:
        sample = TeacherDatasetSample.model_validate(json.loads(line))
        assert sample.teacher is not None
        assert validator.validate(sample.state, sample.teacher).valid

    # Manifest + statistics files written.
    assert (tmp_path / "out" / "generation_manifest.json").exists()
    assert (tmp_path / "out" / "dataset_statistics.json").exists()


def test_mock_end_to_end_counterfactual_links(config, tmp_path):
    teacher = MockTeacherClient()
    builder = TeacherDatasetBuilder(
        generation_config=config,
        teacher_config={"teacher": {"prompt_version": "teacher_v0.1"}},
        teacher_client=teacher,
        parser=TeacherResponseParser(),
        validator=TeacherResponseValidator(),
        output_dir=tmp_path / "out",
    )
    builder.build(num_personas=1, num_trips=1, seed=42)

    ds_path = tmp_path / "out" / "teacher_dataset_v0_1.jsonl"
    samples = [TeacherDatasetSample.model_validate(json.loads(l)) for l in ds_path.read_text(encoding="utf-8").splitlines()]
    by_id = {s.sample_id: s for s in samples}

    baseline = [s for s in samples if s.perturbation.axis == "baseline"]
    assert len(baseline) == 1
    baseline_id = baseline[0].sample_id
    assert baseline[0].baseline_sample_id is None

    cf = [s for s in samples if s.perturbation.axis != "baseline"]
    assert len(cf) > 0
    for s in cf:
        assert s.baseline_sample_id == baseline_id
        assert s.counterfactual_group_id is not None
        assert s.counterfactual_group_id.startswith("CF_")
        assert by_id[s.baseline_sample_id].state.persona.model_dump() == s.state.persona.model_dump()
        assert by_id[s.baseline_sample_id].state.trip.model_dump() == s.state.trip.model_dump()
