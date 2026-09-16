"""Technical checks for the data-only questionnaire package, not human validity."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from survey_adapter import CARDS, FORMS, MAPPING, MappingError, build_persona, build_state, offered_modes
from reference_pipeline.student_adapter import StudentAdapter
from traveler_distillation.student.features import GLOBAL_CAT

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def checked(condition, message):
    if not condition:
        raise AssertionError(message)

def main():
    checks = []
    text = (ROOT / "docs/plans/Shanghai_Travel_Intention_Survey.md").read_text(encoding="utf-8")
    checked(len(CARDS) == 10, "Expected exactly ten cards")
    checked(len(re.findall(r"^## 情景卡 \d+", text, re.M)) == 10, "Questionnaire/card count mismatch")
    for cid, card in CARDS.items():
        checked([a["mode"] for a in card["alternatives"]] == ["car","pt","bike","walk"], cid)
        pt = card["alternatives"][1]
        total = sum(pt[k] for k in ("access_time_min","wait_time_min","in_vehicle_time_min","transfer_time_min","egress_time_min"))
        checked(total == pt["travel_time_min"], f"{cid}: PT components do not sum to total")
        checked(card["other_car_cost_rmb"] + card["parking_cost_rmb"] == card["alternatives"][0]["monetary_cost"], f"{cid}: cost decomposition")
        checked(pt["pt_feasible"] == 1, f"{cid}: unexpected infeasible PT")
        checked(f"card_id={cid}" in text, f"{cid}: missing displayed card")
        heading = f"<!-- card_id={cid}，"
        fragment = text.split(heading,1)[1].split("\n## ",1)[0]
        labels = {"car":"自己开车", "pt":"公交或地铁",
                  "bike":"普通自行车","walk":"步行"}
        for alt in card["alternatives"]:
            row = f"| {labels[alt['mode']]} | {alt['travel_time_min']} | {alt['monetary_cost']} |"
            checked(row in fragment, f"{cid}: human/model time or cost differs")
        transfer = "无需换乘" if pt["transfers"] == 0 else f"换乘{pt['transfers']}次（{pt['transfer_time_min']}分钟）"
        formula = f"进出站步行共{pt['access_time_min']+pt['egress_time_min']}分钟，首次等车{pt['wait_time_min']}分钟，{transfer}。"
        checked(formula in fragment, f"{cid}: displayed PT components differ")
    checked("C00" not in text and "确认理解" not in text, "Old confirmation remains")
    checked("**您的出发时间会怎么安排？**" not in text, "Removed departure question remains")
    checked(text.count("**您会选择哪种方式？**") == 10, "Mode question count")
    checked("shift_directions" not in MAPPING, "Removed departure response schema remains")
    checks.append("ten_card_content_and_all_displayed_numbers_match_data")

    base = CARDS["B0"]
    checked(CARDS["W1"]["alternatives"] == base["alternatives"], "Rain must hold all alternative numeric attributes fixed")
    checked(CARDS["WD1"]["alternatives"] == CARDS["D1"]["alternatives"], "Joint delay attributes differ")
    for cid, field, new_value in (("F1","fare_multiplier",1.5),("P1","parking_cost_multiplier",3),("R1","road_disruption",True)):
        expected = copy.deepcopy(base["context"])
        expected.update(context_id="SP_"+cid)
        expected[field] = new_value
        checked(CARDS[cid]["context"] == expected, f"{cid}: extra context changes")
    for cid in CARDS:
        checked(CARDS[cid]["trip"] == base["trip"], "All cards must share the same trip")
    checked({CARDS[cid]["alternatives"][1]["travel_time_min"] for cid in ("A_WALK","A_WAIT","A_TRANSFER")} == {45}, "Accessibility time not matched")
    checked(CARDS["D1"]["context"]["transit_delay_min"] == 15, "Delay mismatch")
    checked(CARDS["D1"]["alternatives"][1]["wait_time_min"] == 20, "Delay not included in wait")
    checked(CARDS["R1"]["alternatives"][0]["travel_time_min"] - base["alternatives"][0]["travel_time_min"] == 20, "Road difference")
    checked(CARDS["P1"]["parking_cost_rmb"] == 3 * base["parking_cost_rmb"], "Parking multiplier")
    checks.append("shared_trip_factorial_and_predefined_contrasts_are_consistent")

    positions = Counter()
    transitions = Counter()
    for order in FORMS.values():
        checked(len(order)==10 and set(order)==set(CARDS), "Invalid form permutation")
        positions.update((cid, pos) for pos,cid in enumerate(order))
        transitions.update(zip(order[:-1], order[1:]))
    checked(len(FORMS)==10 and len(positions)==100 and set(positions.values())=={1}, "Position balance")
    checked(len(transitions)==90 and set(transitions.values())=={1}, "First-order transition balance")
    checks.append("ten_Williams_forms_balance_positions_and_ordered_adjacent_pairs")

    raw_base = dict(respondent_id="QA_ONLY", S01="yes", S02="yes",
                    P01="25-34", P02="office_worker", P03="from_5000_to_9999",
                    P04="2", P05="no", P06="yes", P07="yes",
                    P09="yes", P11="no", P12="public_transport",
                    P13="from_6_to_30", P14="none",
                    form_id="F01", survey_version=MAPPING["version"])
    states = []
    mask_sets = set()
    fixtures = []
    for car in (False,True):
        for bike in (False,True):
            raw = dict(raw_base)
            raw.update(P06="yes" if car else "no",
                       P09="yes" if bike else "no")
            fixtures.append(raw)
    raw = dict(raw_base, P07="no")
    fixtures.append(raw)
    raw = dict(raw_base, P06="no", P09="no")
    fixtures.append(raw)
    for idx, raw in enumerate(fixtures):
        raw["respondent_id"] = f"QA_ONLY_{idx}"
        persona = build_persona(raw)
        availability = offered_modes(raw)
        mask_sets.add(tuple(m for m,a in availability.items() if a))
        for cid in CARDS:
            state = build_state(persona, cid, availability)
            checked(state.trip.trip_id == f"QA_ONLY_{idx}::{cid}", "Trip identity collision")
            states.append(state)
    for unsupported in ("ridehail","ebike","car_passenger","other"):
        raw = dict(raw_base, P12=unsupported)
        try:
            build_persona(raw)
        except MappingError as e:
            checked("outside_frozen_vocabulary" in str(e), "Wrong unsupported reason")
        else:
            raise AssertionError("Unsupported habit was silently mapped")
    try:
        build_persona(dict(raw_base, P03="refused"))
    except MappingError:
        pass
    else:
        raise AssertionError("Refused income was silently imputed")
    checked(len({s.trip.trip_id for s in states}) == len(states), "Duplicate trip IDs")
    checked(len(mask_sets)==4, "Four availability sets were not exercised")
    checks.append("sixty_schema_states_cover_four_masks_and_missing_unsupported_boundaries")

    ckpt = ROOT / MAPPING["model_contract"]["checkpoint"]
    before_ckpt = sha(ckpt)
    student = StudentAdapter(ckpt, device="cpu")
    unknown = 0
    for state in states:
        for name in GLOBAL_CAT:
            value = str(student.extractor._get_cat(state, name))
            if student.extractor.cat_vocabs[name].get(value,0)==0:
                unknown += 1
        f = student.extractor.encode(state)
        checked(all(math.isfinite(v) for v in f["global_num"]), "Nonfinite global feature")
        checked(all(math.isfinite(v) for row in f["alt_num"] for v in row), "Nonfinite alternative feature")
        checked(all(idx>0 for idx in f["alt_mode_idx"]), "Unknown mode")
    checked(unknown==0, "Unexpected unknown categorical code in SP fixtures")
    predictions = student.predict(states)
    checked(len(predictions)==len(states), "Wrong inference length")
    for state, result in zip(states, predictions):
        probs = result["mode_probabilities"]
        checked(set(probs)==set(state.available_modes), "Decoded availability mismatch")
        checked(all(math.isfinite(x) and 0<=x<=1 for x in probs.values()), "Invalid probabilities")
        checked(abs(sum(probs.values())-1)<1e-5, "Probability sum")
        checked(math.isfinite(result["departure_time_shift_min"]) and abs(result["departure_time_shift_min"])<=60.0001, "Invalid departure output")
    checked(sha(ckpt)==before_ckpt, "Checkpoint file changed")
    checks.append("read_only_frozen_S9_encoding_and_inference_finite_no_unknown_categories")

    protected = json.loads((HERE/"protected_files_before.json").read_text(encoding="utf-8-sig"))
    after = [{"path":row["path"],"sha256":sha(row["path"])} for row in protected]
    changed = [row["path"] for row,latest in zip(protected,after) if row["sha256"].lower()!=latest["sha256"].lower()]
    checked(not changed, "Protected production files changed")
    (HERE/"protected_files_after.json").write_text(json.dumps(after,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    checks.append("all_protected_release_model_source_and_config_hashes_unchanged")

    report = dict(version=MAPPING["version"], assessment_date="2026-09-12", status="PASS",
                  checks=checks, cards=len(CARDS), order_forms=len(FORMS),
                  unique_ordered_adjacent_pairs=len(transitions),
                  synthetic_QA_personas=len(fixtures), validated_states=len(states),
                  unknown_category_count=unknown, protected_files_checked=len(protected),
                  protected_files_changed=changed, checkpoint_sha256=before_ckpt,
                  questionnaire_sha256=sha(ROOT/"docs/plans/Shanghai_Travel_Intention_Survey.md"),
                  cards_sha256=sha(HERE/"cards.json"),
                  note="Synthetic fixtures only. No real respondent answers, behavior metrics, pilot completion times, training, normalization fit or model selection. Predictions checked only for structural validity and not used to tune cards.")
    (HERE/"validation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__ == "__main__":
    main()
