"""Tests for --resume + --max-samples interaction (audit v0.1, section 3)."""
from __future__ import annotations

from traveler_distillation.dataset import TeacherDatasetBuilder
from traveler_distillation.teacher import (
    MockTeacherClient,
    TeacherClientError,
    TeacherResponseParser,
    TeacherResponseValidator,
)


class _FailingOnceTeacher(MockTeacherClient):
    """Mock teacher that raises on a specific call index (1-based)."""

    def __init__(self, fail_on_call: int):
        super().__init__()
        self.fail_on_call = fail_on_call
        self.calls = 0

    def respond(self, state):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise TeacherClientError("empty_content", "teacher returned empty content")
        return super().respond(state)


def _builder(config, output_dir, resume=False):
    return TeacherDatasetBuilder(
        generation_config=config,
        teacher_config={"teacher": {"prompt_version": "teacher_v0.1"}},
        teacher_client=MockTeacherClient(),
        parser=TeacherResponseParser(),
        validator=TeacherResponseValidator(),
        output_dir=output_dir,
        resume=resume,
    )


def _count_lines(path):
    return len([l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()])


def test_resume_skipped_samples_do_not_consume_max_samples(config, tmp_path):
    out = tmp_path / "out"

    # First run: generate 2 samples.
    r1 = _builder(config, out, resume=False).build(
        num_personas=1, num_trips=1, seed=42, max_samples=2
    )
    assert r1["statistics"]["valid_samples"] == 2
    ds = out / "teacher_dataset_v0_1.jsonl"
    assert _count_lines(ds) == 2

    # Resume with max_samples=1: must generate exactly 1 NEW sample, not 0.
    r2 = _builder(config, out, resume=True).build(
        num_personas=1, num_trips=1, seed=42, max_samples=1
    )
    assert r2["statistics"]["valid_samples"] == 1, (
        "resume must generate 1 new sample; skipped samples must not count "
        "toward max_samples"
    )
    # Total on disk: 2 (previous) + 1 (new) = 3.
    assert _count_lines(ds) == 3


def test_resume_skips_previously_failed_samples(config, tmp_path):
    import json

    out = tmp_path / "out"

    # First run: teacher fails on the 2nd call (S000002).
    _builder2 = TeacherDatasetBuilder(
        generation_config=config,
        teacher_config={"teacher": {"prompt_version": "teacher_v0.1"}},
        teacher_client=_FailingOnceTeacher(fail_on_call=2),
        parser=TeacherResponseParser(),
        validator=TeacherResponseValidator(),
        output_dir=out,
        resume=False,
    )
    _builder2.build(num_personas=1, num_trips=1, seed=42, max_samples=3)

    ds = out / "teacher_dataset_v0_1.jsonl"
    failed = out / "failed_samples_v0_1.jsonl"
    valid_ids = {json.loads(l)["sample_id"] for l in ds.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert valid_ids == {"S000001", "S000003"}
    assert _count_lines(failed) == 1

    # Resume with an always-succeeding teacher and max_samples=1: the failed
    # S000002 must NOT be re-attempted (it is already in the failed file), and
    # the single new sample must be S000004.
    r2 = _builder(config, out, resume=True).build(
        num_personas=1, num_trips=1, seed=42, max_samples=1
    )
    assert r2["statistics"]["valid_samples"] == 1
    valid_ids_after = {
        json.loads(l)["sample_id"]
        for l in ds.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    assert "S000002" not in valid_ids_after
    assert "S000004" in valid_ids_after
    # Failed file unchanged (no double-counting of the same sample_id).
    assert _count_lines(failed) == 1


def test_resume_skips_existing_ids(config, tmp_path):
    out = tmp_path / "out"
    _builder(config, out, resume=False).build(num_personas=1, num_trips=1, seed=42, max_samples=3)
    ds = out / "teacher_dataset_v0_1.jsonl"
    before_ids = set()
    import json
    for l in ds.read_text(encoding="utf-8").splitlines():
        if l.strip():
            before_ids.add(json.loads(l)["sample_id"])
    assert before_ids == {"S000001", "S000002", "S000003"}

    # Resume without max_samples: regenerate everything, skip existing 3,
    # produce the remaining new samples without duplicating ids.
    r2 = _builder(config, out, resume=True).build(num_personas=1, num_trips=1, seed=42)
    after_ids = set()
    for l in ds.read_text(encoding="utf-8").splitlines():
        if l.strip():
            after_ids.add(json.loads(l)["sample_id"])
    # 1 baseline + 18 counterfactuals = 19 total for 1 persona x 1 trip.
    assert len(after_ids) == 19
    assert before_ids <= after_ids
    # No duplicate sample ids.
    assert r2["statistics"]["valid_samples"] == 19 - 3
