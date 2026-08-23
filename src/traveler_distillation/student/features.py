"""Feature extraction from UniversalTravelerState into numeric tensors.

Categorical features use learned embeddings (with an <UNK> fallback); numeric
features are z-scored using statistics computed from the TRAINING split only.
"""
from __future__ import annotations

from ..schemas.state import UniversalTravelerState

PERSONA_CAT = [
    "age_group",
    "income_group",
    "occupation",
    "habitual_mode",
    "schedule_flexibility",
    "mobility_limitation",
]
PERSONA_NUM = [
    "household_size",
    "has_children",
    "car_ownership",
    "driving_license",
    "bike_ownership",
    "transit_pass",
]
TRIP_CAT = ["purpose", "time_constraint"]
TRIP_NUM = ["distance_km", "desired_departure_min", "desired_arrival_min"]
CONTEXT_CAT = ["weather_condition"]
CONTEXT_NUM = [
    "weather_intensity",
    "road_congestion",
    "transit_delay_min",
    "transit_disruption",
    "road_disruption",
    "fare_multiplier",
    "parking_cost_multiplier",
    "congestion_charge",
]
ALT_NUM = [
    "travel_time_min",
    "monetary_cost",
    "access_time_min",
    "transfers",
    "reliability_delay_min",
    "weather_exposure",
]

GLOBAL_CAT = PERSONA_CAT + TRIP_CAT + CONTEXT_CAT
GLOBAL_NUM = PERSONA_NUM + TRIP_NUM + CONTEXT_NUM
UNK = "<UNK>"


class FeatureExtractor:
    """Fits vocabularies + normalization stats on train states, then encodes."""

    def __init__(self):
        self.cat_vocabs: dict[str, dict] = {}
        self.mode_vocab: dict[str, int] = {UNK: 0}
        self.num_mean: dict[str, float] = {}
        self.num_std: dict[str, float] = {}
        self.fitted = False

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _get_cat(state: UniversalTravelerState, name: str):
        if name in PERSONA_CAT:
            return getattr(state.persona, name)
        if name in TRIP_CAT:
            return getattr(state.trip, name)
        if name == "weather_condition":
            return state.context.weather.condition
        raise KeyError(name)

    @staticmethod
    def _get_num(state: UniversalTravelerState, name: str) -> float:
        if name in PERSONA_NUM:
            return float(getattr(state.persona, name))
        if name in TRIP_NUM:
            return float(getattr(state.trip, name))
        if name == "weather_intensity":
            return float(state.context.weather.intensity)
        if name in CONTEXT_NUM:
            return float(getattr(state.context, name))
        raise KeyError(name)

    # ---------------------------------------------------------------- fit
    def fit(self, states: list[UniversalTravelerState]) -> "FeatureExtractor":
        cat_values: dict[str, set] = {name: set() for name in GLOBAL_CAT}
        num_values: dict[str, list] = {name: [] for name in GLOBAL_NUM + ALT_NUM}
        modes: set[str] = set()

        for state in states:
            for name in GLOBAL_CAT:
                cat_values[name].add(self._get_cat(state, name))
            for name in GLOBAL_NUM:
                num_values[name].append(self._get_num(state, name))
            for alt in state.alternatives:
                modes.add(alt.mode)
                for name in ALT_NUM:
                    num_values[name].append(float(getattr(alt, name)))

        for name in GLOBAL_CAT:
            vocab = {UNK: 0}
            for v in sorted(str(x) for x in cat_values[name]):
                if v not in vocab:
                    vocab[v] = len(vocab)
            self.cat_vocabs[name] = vocab

        for m in sorted(modes):
            if m not in self.mode_vocab:
                self.mode_vocab[m] = len(self.mode_vocab)

        for name in GLOBAL_NUM + ALT_NUM:
            vals = num_values[name]
            mean = sum(vals) / len(vals) if vals else 0.0
            var = sum((v - mean) ** 2 for v in vals) / len(vals) if vals else 0.0
            std = var ** 0.5
            self.num_mean[name] = mean
            self.num_std[name] = std if std > 1e-8 else 1.0

        self.fitted = True
        return self

    # ---------------------------------------------------------------- encode
    def encode(self, state: UniversalTravelerState) -> dict:
        if not self.fitted:
            raise RuntimeError("FeatureExtractor must be fit before encode")

        global_cat = []
        for name in GLOBAL_CAT:
            v = str(self._get_cat(state, name))
            global_cat.append(self.cat_vocabs[name].get(v, 0))

        global_num = []
        for name in GLOBAL_NUM:
            x = self._get_num(state, name)
            global_num.append((x - self.num_mean[name]) / self.num_std[name])

        alt_mode_idx = []
        alt_num = []
        alt_available = []
        for alt in state.alternatives:
            alt_mode_idx.append(self.mode_vocab.get(alt.mode, 0))
            alt_num.append(
                [
                    (float(getattr(alt, name)) - self.num_mean[name]) / self.num_std[name]
                    for name in ALT_NUM
                ]
            )
            alt_available.append(1.0 if alt.available else 0.0)

        return {
            "global_cat": global_cat,
            "global_num": global_num,
            "alt_mode_idx": alt_mode_idx,
            "alt_num": alt_num,
            "alt_available": alt_available,
        }

    @property
    def spec(self) -> dict:
        return {
            "cat_names": list(GLOBAL_CAT),
            "cat_vocab_sizes": {name: len(v) for name, v in self.cat_vocabs.items()},
            "n_global_num": len(GLOBAL_NUM),
            "n_alt_num": len(ALT_NUM),
            "mode_vocab_size": len(self.mode_vocab),
        }

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        """Full fitted state (vocabularies + normalization stats).

        Needed for deployment: the student checkpoint must be able to encode
        NEW states (e.g. in the MATSim adapter) without re-fitting on training
        data, so the fitted extractor state is saved alongside the model.
        """
        return {
            "cat_vocabs": self.cat_vocabs,
            "mode_vocab": self.mode_vocab,
            "num_mean": self.num_mean,
            "num_std": self.num_std,
        }

    def load_state_dict(self, state: dict) -> "FeatureExtractor":
        self.cat_vocabs = state["cat_vocabs"]
        self.mode_vocab = state["mode_vocab"]
        self.num_mean = state["num_mean"]
        self.num_std = state["num_std"]
        self.fitted = True
        return self

    @classmethod
    def from_state_dict(cls, state: dict) -> "FeatureExtractor":
        return cls().load_state_dict(state)
