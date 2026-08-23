"""MATSim adapter tests: checkpoint decode + scenario XML generation."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import torch

from traveler_distillation.student import FeatureExtractor, TravelerStudent
from traveler_distillation.matsim import MATSimAdapter
from traveler_distillation.schemas.context import DynamicContext, Weather


def _make_checkpoint(tmp_path, state, cfg=None):
    cfg = cfg or {
        "global_hidden_dim": 16, "alternative_hidden_dim": 8,
        "cat_embedding_dim": 4, "mode_embedding_dim": 4,
        "scorer_hidden_dim": 16, "dropout": 0.0,
    }
    extractor = FeatureExtractor().fit([state])
    model = TravelerStudent(cfg, extractor.spec)
    ckpt = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_spec": extractor.spec,
            "extractor_state": extractor.state_dict(),
            "config": cfg,
        },
        ckpt,
    )
    return ckpt


def test_adapter_rejects_checkpoint_without_extractor_state(tmp_path, baseline_state):
    extractor = FeatureExtractor().fit([baseline_state])
    model = TravelerStudent({"global_hidden_dim": 8, "alternative_hidden_dim": 4,
                             "cat_embedding_dim": 4, "mode_embedding_dim": 4,
                             "scorer_hidden_dim": 8, "dropout": 0.0}, extractor.spec)
    old = tmp_path / "old.pt"
    torch.save({"model_state": model.state_dict(), "feature_spec": extractor.spec,
                "config": {}}, old)
    try:
        MATSimAdapter(old)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "extractor_state" in str(exc)


def test_adapter_scenario_xml_valid(tmp_path, baseline_state, persona, trip, config):
    ckpt = _make_checkpoint(tmp_path, baseline_state)
    adapter = MATSimAdapter(ckpt)

    context = DynamicContext(
        context_id="C_test", weather=Weather(condition="clear", intensity=0.0),
        road_congestion=0.3, transit_delay_min=0, transit_disruption=False,
        road_disruption=False, fare_multiplier=1.0, parking_cost_multiplier=1.0,
        congestion_charge=0.0,
    )
    out = tmp_path / "scenario"
    manifest = adapter.build_scenario(
        [persona], [trip], context, out, grid_n=5, spacing_m=1000.0
    )

    # manifest
    assert len(manifest) == 1
    assert manifest[0]["persona_id"] == persona.persona_id
    assert manifest[0]["student_mode"] in ("car", "pt", "bike", "walk")

    # network.xml valid + links reference real nodes
    net_text = (out / "network.xml").read_text(encoding="utf-8")
    assert "network_v1.dtd" in net_text  # MATSim needs the DOCTYPE to pick its parser
    net = ET.parse(out / "network.xml").getroot()
    node_ids = {n.get("id") for n in net.find("nodes").findall("node")}
    link_ids = {l.get("id") for l in net.find("links").findall("link")}
    assert len(node_ids) == 25
    for link in net.find("links").findall("link"):
        assert link.get("from") in node_ids
        assert link.get("to") in node_ids

    # population.xml valid + every activity on a real LINK (MATSim cars park on
    # links), legs use student modes
    pop_text = (out / "population.xml").read_text(encoding="utf-8")
    assert "population_v6.dtd" in pop_text
    pop = ET.parse(out / "population.xml").getroot()
    persons = pop.findall("person")
    assert len(persons) == 1
    plan = persons[0].find("plan")
    acts = plan.findall("activity")
    legs = plan.findall("leg")
    assert len(acts) == 3 and len(legs) == 2  # home -> dest -> home
    for act in acts:
        assert act.get("link") in link_ids
    assert acts[0].get("type") == "home" and acts[-1].get("type") == "home"
    assert legs[0].get("mode") == manifest[0]["student_mode"]

    # attributes written
    attrs = persons[0].find("attributes")
    names = {a.get("name") for a in attrs.findall("attribute")}
    assert "age_group" in names and "habitual_mode" in names

    # config.xml exists and includes teleported mode params
    cfg_text = (out / "config.xml").read_text(encoding="utf-8")
    assert "teleportedModeParameters" in cfg_text
    assert 'value="pt"' in cfg_text

    # adapter manifest json is valid
    manifest2 = json.loads((out / "adapter_manifest.json").read_text(encoding="utf-8"))
    assert manifest2[0]["persona_id"] == persona.persona_id


def test_adapter_departure_shift_applied(tmp_path, baseline_state, persona, trip, config):
    # Random-weight student: shift is whatever tanh head emits; the manifest's
    # departure must equal desired_departure + shift (consistency check).
    ckpt = _make_checkpoint(tmp_path, baseline_state)
    adapter = MATSimAdapter(ckpt)
    context = DynamicContext(
        context_id="C_test", weather=Weather(condition="clear", intensity=0.0),
        road_congestion=0.3, transit_delay_min=0, transit_disruption=False,
        road_disruption=False, fare_multiplier=1.0, parking_cost_multiplier=1.0,
        congestion_charge=0.0,
    )
    manifest = adapter.build_scenario([persona], [trip], context, tmp_path / "s2", grid_n=4)
    m = manifest[0]
    assert abs(m["departure_min"] - (trip.desired_departure_min + m["departure_shift_min"])) < 1e-6
    assert -60.0 <= m["departure_shift_min"] <= 60.0


def test_outgoing_link_covers_all_grid_corners():
    """Every node's outgoing link id must exist in the built network."""
    for n in (2, 3, 10):
        root = MATSimAdapter.build_grid_network(n, 1000.0)
        link_ids = {l.get("id") for l in root.find("links").findall("link")}
        for i in range(n):
            for j in range(n):
                lid = MATSimAdapter._outgoing_link(i, j, n)
                assert lid in link_ids, f"missing link {lid} for node ({i},{j}) n={n}"


def test_outgoing_link_southeast_corner_is_reverse_link():
    # (n-1, n-1) has no outgoing east/north link; must return the REVERSE of
    # the east link from (n-2, n-1): l_{n-2}_{n-1}_er
    assert MATSimAdapter._outgoing_link(9, 9, 10) == "l_8_9_er"
    assert MATSimAdapter._outgoing_link(2, 2, 3) == "l_1_2_er"
