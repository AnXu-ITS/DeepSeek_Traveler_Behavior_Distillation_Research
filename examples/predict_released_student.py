"""Infer one synthetic held-out state with the released SA-Student (S9)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from reference_pipeline.student_adapter import StudentAdapter
from traveler_distillation.schemas.state import UniversalTravelerState

def main():
    row = json.loads((ROOT / "outputs/matched_response_v1/bundle/test_endpoints.jsonl").read_text(encoding="utf-8").splitlines()[0])
    state = UniversalTravelerState.model_validate(row["state"])
    adapter = StudentAdapter(ROOT / "releases/s9_supply_aware_v2/checkpoint/model.pt", device="cpu")
    result = adapter.predict([state])[0]
    assert abs(sum(result["mode_probabilities"].values()) - 1.0) < 1e-5
    assert -60 <= result["departure_time_shift_min"] <= 60
    print(json.dumps({"state_id": row["id"], "parameters": adapter.num_parameters,
                      "checkpoint_sha256": adapter.checkpoint_sha256(), "prediction": result}, indent=2))

if __name__ == "__main__":
    main()
