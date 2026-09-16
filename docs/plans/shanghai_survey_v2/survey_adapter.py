"""Data-only SP adapter using the unchanged production schema.

Reads symbolic questionnaire exports and the fixed card table. Does not generate
routes, alter card costs/times, load a teacher, train a model, or fit normalization.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from traveler_distillation.schemas.persona import Persona
from traveler_distillation.schemas.state import UniversalTravelerState

MAPPING = json.loads((HERE / "field_mapping.json").read_text(encoding="utf-8"))
CARD_PACKAGE = json.loads((HERE / "cards.json").read_text(encoding="utf-8"))
CARDS = {c["card_id"]: c for c in CARD_PACKAGE["cards"]}
FORMS = {f["form_id"]: f["card_order"] for f in json.loads((HERE / "forms.json").read_text(encoding="utf-8"))["forms"]}

class MappingError(ValueError):
    pass

def offered_modes(raw):
    car = all(str(raw.get(k, "")).strip() == "yes" for k in ("P06", "P07", "P08"))
    bike = all(str(raw.get(k, "")).strip() == "yes" for k in ("P09", "P10"))
    return dict(car=car, pt=True, bike=bike, walk=True)

def build_persona(raw):
    errors = []
    values = {"persona_id": str(raw.get("respondent_id", "")).strip()}
    if not values["persona_id"]:
        errors.append("respondent_id:missing")
    if str(raw.get("S01", "")).strip() != "yes":
        errors.append("consent:not_confirmed")
    if str(raw.get("S02", "")).strip() != "yes":
        errors.append("Shanghai_30d:not_confirmed")
    if str(raw.get("C00", "")).strip() != "confirmed":
        errors.append("offered_modes:not_confirmed")
    for target, rule in MAPPING["persona_fields"].items():
        q = rule["question"]
        value = str(raw.get(q, "")).strip()
        if rule.get("kind") == "positive_integer":
            try:
                num = int(value)
                if num < 1:
                    raise ValueError()
                values[target] = num
            except ValueError:
                errors.append(f"{q}/{target}:missing_or_invalid_integer")
        elif value in rule["map"]:
            values[target] = rule["map"][value]
        elif value in rule.get("unsupported", []):
            errors.append(f"{q}/{target}:outside_frozen_vocabulary")
        else:
            errors.append(f"{q}/{target}:missing_or_unrecognized")
    if errors:
        raise MappingError(";".join(errors))
    return Persona(**values)

def build_state(persona, cid, availability):
    card = CARDS[cid]
    trip = copy.deepcopy(card["trip"])
    trip["trip_id"] = f"{persona.persona_id}::{cid}"
    alternatives = copy.deepcopy(card["alternatives"])
    for alt in alternatives:
        alt["available"] = bool(availability[alt["mode"]])
    return UniversalTravelerState(persona=persona, trip=trip,
                                  context=copy.deepcopy(card["context"]),
                                  alternatives=alternatives)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respondents", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/shanghai_survey_v2")
    args = parser.parse_args()
    outdir = args.output_dir.resolve()
    for protected in ("releases", "src", "configs", "reference_pipeline"):
        pp = (ROOT / protected).resolve()
        if outdir == pp or pp in outdir.parents:
            raise SystemExit("Refusing to write survey output into a protected production directory.")
    with args.respondents.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [str(row.get("respondent_id", "")).strip() for row in rows]
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise SystemExit("Every row must have a non-empty, unique respondent_id. No outputs written.")
    encoded = []
    audit = []
    for raw in rows:
        rid = raw["respondent_id"].strip()
        form_id = str(raw.get("form_id", "")).strip()
        try:
            if raw.get("survey_version") != MAPPING["version"]:
                raise MappingError("survey_version:missing_or_mismatch")
            if form_id not in FORMS:
                raise MappingError("form_id:missing_or_unknown")
            person = build_persona(raw)
            available = offered_modes(raw)
            for pos, cid in enumerate(FORMS[form_id], 1):
                state = build_state(person, cid, available)
                encoded.append(dict(respondent_id=rid, survey_version=MAPPING["version"],
                                    form_id=form_id, task_order=pos, card_id=cid,
                                    state=state.model_dump(mode="json")))
            audit.append(dict(respondent_id=rid, input_status="complete", reason="",
                              offered_modes="|".join(k for k,v in available.items() if v)))
        except MappingError as error:
            audit.append(dict(respondent_id=rid, input_status="not_primary_mappable",
                              reason=str(error),
                              offered_modes="|".join(k for k,v in offered_modes(raw).items() if v)))
    outdir.mkdir(parents=True, exist_ok=True)
    targets = [outdir / name for name in ("sp_states.jsonl", "input_coverage.csv", "conversion_summary.json")]
    if any(path.exists() for path in targets):
        raise SystemExit("Output files already exist. Choose a new output directory to preserve prior exports.")
    targets[0].write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in encoded), encoding="utf-8")
    with targets[1].open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["respondent_id","input_status","reason","offered_modes"])
        writer.writeheader()
        writer.writerows(audit)
    summary = dict(version=MAPPING["version"], input_rows=len(rows),
                   complete_input_respondents=sum(x["input_status"]=="complete" for x in audit),
                   generated_states=len(encoded),
                   note="All input rows retained in coverage audit. No choices, predictions or model fitting used in mapping.")
    targets[2].write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
if __name__ == "__main__":
    main()

