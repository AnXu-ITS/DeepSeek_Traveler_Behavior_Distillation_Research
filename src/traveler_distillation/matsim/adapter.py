"""MATSim adapter: distilled student decisions -> MATSim scenario files.

v0.1 scope (Phase 8 gate):

- loads a trained student checkpoint (model + fitted FeatureExtractor),
- builds a synthetic grid network,
- for each persona+trip, asks the student for its behavioral response under a
  given dynamic context and writes a MATSim daily plan (home -> activity ->
  home) whose outbound leg uses the student's chosen mode and departure shift,
- writes population.xml (with persona attributes) and a config.xml where
  walk/bike/pt run as teleported modes and car runs on the network,
- MATSim runs with ``lastIteration=0`` so its own replanning does NOT overwrite
  the distilled decisions (the student's plan is executed as-is).

Not in v0.1 (documented honestly): return-leg modeling (return uses the same
mode as the outbound leg), transit schedules (pt is teleported), within-day
adaptation.
"""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from ..schemas.context import DynamicContext
from ..schemas.state import UniversalTravelerState
from ..generators.alternative_generator import AlternativeGenerator
from ..student import FeatureExtractor, TravelerStudent
from ..student.dataset import collate_batch

DEFAULT_MODE_SPEEDS = {
    # meters per second, teleported beeline speeds
    "walk": 1.39,   # ~5 km/h
    "bike": 3.9,    # ~14 km/h
    "pt": 5.0,      # ~18 km/h
}

_ACTIVITY_DURATIONS_MIN = {
    "work": 8 * 60,
    "school": 6 * 60,
    "shop": 2 * 60,
    "leisure": 3 * 60,
    "healthcare": 1 * 60,
    "other": 2 * 60,
}

_LEG_MIN = 30  # placeholder travel time for activity end times (teleported)


def _to_hms(minutes: float) -> str:
    total = max(0, int(round(minutes * 60)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class MATSimAdapter:
    """Student checkpoint -> MATSim network + population + config."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        ckpt = torch.load(Path(checkpoint_path), map_location="cpu")
        if "extractor_state" not in ckpt:
            raise ValueError(
                "checkpoint lacks extractor_state (trained with an older script); "
                "retrain or re-save with the current train scripts"
            )
        self.extractor = FeatureExtractor.from_state_dict(ckpt["extractor_state"])
        self.s_cfg = ckpt["config"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TravelerStudent(self.s_cfg, ckpt["feature_spec"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    # ------------------------------------------------------------- decisions
    @torch.no_grad()
    def decide(self, state: UniversalTravelerState) -> dict:
        """Student's behavioral response for one state."""
        feats = self.extractor.encode(state)
        batch = collate_batch([{
            "global_cat": torch.tensor(feats["global_cat"], dtype=torch.long),
            "global_num": torch.tensor(feats["global_num"], dtype=torch.float32),
            "alt_mode_idx": torch.tensor(feats["alt_mode_idx"], dtype=torch.long),
            "alt_num": torch.tensor(feats["alt_num"], dtype=torch.float32),
            "alt_mask": torch.tensor(feats["alt_available"], dtype=torch.float32),
        }])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(batch)
        probs_np = out["mode_probabilities"][0].cpu().numpy()
        probs = {
            alt.mode: float(probs_np[i])
            for i, alt in enumerate(state.alternatives)
            if alt.available
        }
        chosen = max(probs, key=probs.get)
        shift = float(out["departure_time_shift_min"][0].item())
        return {
            "mode": chosen,
            "mode_probabilities": probs,
            "departure_time_shift_min": shift,
        }

    # ---------------------------------------------------------------- network
    @staticmethod
    def build_grid_network(n: int, spacing_m: float, capacity: float = 2000.0) -> ET.Element:
        """n x n grid of nodes with bidirectional links (MATSim network root).

        ``capacity`` is the per-link hourly capacity (veh/h); small values make
        congestion emerge at modest population sizes (feedback-loop demos).
        """
        root = ET.Element("network", name="synthetic-grid")
        nodes = ET.SubElement(root, "nodes")
        for i in range(n):
            for j in range(n):
                ET.SubElement(
                    nodes, "node",
                    id=f"n_{i}_{j}",
                    x=str(round(i * spacing_m, 1)),
                    y=str(round(j * spacing_m, 1)),
                )
        links = ET.SubElement(root, "links")
        for i in range(n):
            for j in range(n):
                for di, dj, d in ((1, 0, "e"), (0, 1, "n")):
                    ni, nj = i + di, j + dj
                    if ni >= n or nj >= n:
                        continue
                    ET.SubElement(
                        links, "link",
                        id=f"l_{i}_{j}_{d}f",
                        **{"from": f"n_{i}_{j}", "to": f"n_{ni}_{nj}",
                           "length": str(spacing_m), "freespeed": "13.89",
                           "capacity": str(capacity), "permlanes": "1.0"},
                    )
                    ET.SubElement(
                        links, "link",
                        id=f"l_{i}_{j}_{d}r",
                        **{"from": f"n_{ni}_{nj}", "to": f"n_{i}_{j}",
                           "length": str(spacing_m), "freespeed": "13.89",
                           "capacity": str(capacity), "permlanes": "1.0"},
                    )
        return root

    @staticmethod
    def _snap(x: float, y: float, spacing_m: float, n: int) -> tuple[str, int, int, float, float]:
        i = max(0, min(n - 1, int(round(x / spacing_m))))
        j = max(0, min(n - 1, int(round(y / spacing_m))))
        return f"n_{i}_{j}", i, j, round(i * spacing_m, 1), round(j * spacing_m, 1)

    @staticmethod
    def _outgoing_link(i: int, j: int, n: int) -> str:
        """A link id leaving node (i, j) that is guaranteed to exist.

        MATSim activities must reference LINK ids (not node ids); cars park on
        the first link of their route. Reverse links are named after the
        FORWARD link's origin node, so the west-bound link leaving (i, j) is
        ``l_{i-1}_{j}_er``, not ``l_{i}_{j}_er``.
        """
        if i < n - 1:
            return f"l_{i}_{j}_ef"
        if j < n - 1:
            return f"l_{i}_{j}_nf"
        if i > 0:
            return f"l_{i-1}_{j}_er"  # reverse of the east link from (i-1, j)
        return f"l_{i}_{j}_nr"

    @staticmethod
    def _home_xy(persona_id: str, spacing_m: float, n: int) -> tuple[float, float]:
        h = int(hashlib.sha256(persona_id.encode()).hexdigest(), 16)
        x = (h % (n * 100)) / 100 * spacing_m * 0.9 + 0.05 * spacing_m
        y = ((h // 1000) % (n * 100)) / 100 * spacing_m * 0.9 + 0.05 * spacing_m
        return x, y

    # -------------------------------------------------------------- scenario
    def build_scenario(
        self,
        personas: list,
        trips: list,
        context: DynamicContext,
        output_dir: str | Path,
        grid_n: int = 20,
        spacing_m: float = 1000.0,
        mode_speeds: dict | None = None,
        trips_per_persona: list[list] | None = None,
        link_capacity: float = 2000.0,
    ) -> dict:
        """Generate network.xml, population.xml, config.xml under output_dir.

        ``trips`` is the trip list used for every persona. When
        ``trips_per_persona`` is given, persona ``i`` instead uses
        ``trips_per_persona[i]`` (each persona has its own trips).

        Returns a manifest with per-person decisions for analysis.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        speeds = {**DEFAULT_MODE_SPEEDS, **(mode_speeds or {})}
        alt_gen = AlternativeGenerator({})

        network_root = self.build_grid_network(grid_n, spacing_m, capacity=link_capacity)

        pop_root = ET.Element("population")
        manifest = []
        for p_idx, persona in enumerate(personas):
            person_trips = trips if trips_per_persona is None else trips_per_persona[p_idx]
            home_x, home_y = self._home_xy(persona.persona_id, spacing_m, grid_n)
            home_node, home_i, home_j, home_sx, home_sy = self._snap(
                home_x, home_y, spacing_m, grid_n
            )
            home_link = self._outgoing_link(home_i, home_j, grid_n)

            person = ET.SubElement(pop_root, "person", id=persona.persona_id)
            attrs = ET.SubElement(person, "attributes")
            for key in (
                "age_group", "income_group", "occupation", "household_size",
                "has_children", "car_ownership", "driving_license",
                "bike_ownership", "transit_pass", "habitual_mode",
                "schedule_flexibility", "mobility_limitation",
            ):
                # population_v6 format: value is TEXT CONTENT, class attribute
                # selects the Java type (NOT type="...").
                el = ET.SubElement(
                    attrs, "attribute", name=key, **{"class": "java.lang.String"}
                )
                el.text = str(getattr(persona, key))

            plan = ET.SubElement(person, "plan", selected="yes")
            for trip in person_trips:
                state = UniversalTravelerState(
                    persona=persona, trip=trip, context=context,
                    alternatives=alt_gen.generate(persona, trip, context),
                )
                decision = self.decide(state)
                dep_min = trip.desired_departure_min + decision["departure_time_shift_min"]

                dest_x = home_x + trip.distance_km * 1000.0
                dest_y = home_y
                dest_node, dest_i, dest_j, dest_sx, dest_sy = self._snap(
                    dest_x, dest_y, spacing_m, grid_n
                )
                dest_link = self._outgoing_link(dest_i, dest_j, grid_n)
                dur_min = _ACTIVITY_DURATIONS_MIN.get(trip.destination_type, 2 * 60)

                ET.SubElement(
                    plan, "activity", type="home", link=home_link,
                    x=str(home_sx), y=str(home_sy), z="0.0", end_time=_to_hms(dep_min),
                )
                ET.SubElement(plan, "leg", mode=decision["mode"])
                ET.SubElement(
                    plan, "activity", type=trip.destination_type, link=dest_link,
                    x=str(dest_sx), y=str(dest_sy), z="0.0",
                    end_time=_to_hms(dep_min + _LEG_MIN + dur_min),
                )
                ET.SubElement(plan, "leg", mode=decision["mode"])  # return, same mode
                # after the return leg the traveler is back home

                manifest.append(
                    {
                        "persona_id": persona.persona_id,
                        "trip_id": trip.trip_id,
                        "purpose": trip.purpose,
                        "destination_type": trip.destination_type,
                        "distance_km": trip.distance_km,
                        "student_mode": decision["mode"],
                        "student_mode_probabilities": decision["mode_probabilities"],
                        "departure_shift_min": round(decision["departure_time_shift_min"], 2),
                        "departure_min": round(dep_min, 2),
                        "home_link": home_link,
                        "dest_link": dest_link,
                    }
                )

            ET.SubElement(
                plan, "activity", type="home", link=home_link,
                x=str(home_sx), y=str(home_sy), z="0.0",
            )

        ET.ElementTree(network_root).write(
            out / "network.xml", encoding="utf-8", xml_declaration=True
        )
        ET.ElementTree(pop_root).write(
            out / "population.xml", encoding="utf-8", xml_declaration=True
        )
        self._insert_doctype(
            out / "network.xml",
            '<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v1.dtd">',
        )
        self._insert_doctype(
            out / "population.xml",
            '<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">',
        )
        self._write_config(out / "config.xml", speeds)

        (out / "adapter_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    @staticmethod
    def _insert_doctype(path: Path, doctype: str) -> None:
        """Insert a DOCTYPE line after the XML declaration.

        MATSim's XML readers use the DOCTYPE to select the parser version
        (v1/v2 network, population_v5/v6); without it the reader's delegate is
        never set and parsing fails with a NullPointerException.
        """
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        # ElementTree writes: <?xml version='1.0' encoding='utf-8'?> first
        out_lines = []
        inserted = False
        for line in lines:
            out_lines.append(line)
            if not inserted and line.strip().startswith("<?xml"):
                out_lines.append(doctype)
                inserted = True
        if not inserted:
            out_lines.insert(0, doctype)
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    @staticmethod
    def _write_config(path: Path, speeds: dict) -> None:
        lines = [
            '<?xml version="1.0" ?>',
            '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">',
            "<config>",
            '\t<module name="global">',
            '\t\t<param name="randomSeed" value="4711" />',
            '\t\t<param name="coordinateSystem" value="Atlantis" />',
            "\t</module>",
            '\t<module name="network">',
            '\t\t<param name="inputNetworkFile" value="network.xml" />',
            "\t</module>",
            '\t<module name="plans">',
            '\t\t<param name="inputPlansFile" value="population.xml" />',
            "\t</module>",
            '\t<module name="controller">',
            '\t\t<param name="outputDirectory" value="./output" />',
            '\t\t<param name="firstIteration" value="0" />',
            '\t\t<param name="lastIteration" value="0" />',
            "\t</module>",
            '\t<module name="linkStats">',
            '\t\t<param name="writeLinkStatsInterval" value="1" />',
            '\t\t<param name="averageLinkStatsOverIterations" value="1" />',
            "\t</module>",
            '\t<module name="qsim">',
            '\t\t<param name="endTime" value="36:00:00" />',
            '\t\t<param name="mainMode" value="car" />',
            '\t\t<param name="trafficDynamics" value="queue" />',
            "\t</module>",
            '\t<module name="routing">',
            '\t\t<param name="networkModes" value="car" />',
        ]
        for mode, speed in sorted(speeds.items()):
            lines += [
                '\t\t<parameterset type="teleportedModeParameters">',
                f'\t\t\t<param name="mode" value="{mode}" />',
                '\t\t\t<param name="beelineDistanceFactor" value="1.3" />',
                f'\t\t\t<param name="teleportedModeSpeed" value="{speed}" />',
                "\t\t</parameterset>",
            ]
        lines += [
            "\t</module>",
            '\t<module name="scoring">',
            '\t\t<parameterset type="scoringParameters">',
        ]
        for mode in ("car", "walk", "bike", "pt"):
            lines += [
                f'\t\t\t<parameterset type="modeParams"><param name="mode" value="{mode}" /></parameterset>',
            ]
        # every activity type that can appear in the population must have
        # utility parameters, otherwise scoring aborts at runtime
        for act_type, dur in [
            ("home", "12:00:00"), ("work", "08:00:00"), ("school", "06:00:00"),
            ("shop", "02:00:00"), ("leisure", "03:00:00"), ("healthcare", "01:00:00"),
            ("other", "02:00:00"),
        ]:
            lines += [
                '\t\t\t<parameterset type="activityParams">',
                f'\t\t\t\t<param name="activityType" value="{act_type}" />',
                f'\t\t\t\t<param name="typicalDuration" value="{dur}" />',
                "\t\t\t</parameterset>",
            ]
        lines += [
            "\t\t</parameterset>",
            "\t</module>",
            "</config>",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
