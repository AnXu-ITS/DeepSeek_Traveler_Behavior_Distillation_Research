"""Student adapter: production checkpoint loading + BATCH inference.

Wraps the production :class:`S8MATSimAdapter` (which hard-asserts the frozen
S9 checkpoint schema and loads model + fitted extractor + normalization
exactly once). The batch path uses the same encode/collate/forward/decode
steps the latency audit validated (mode decision identical to per-state
``decide()`` on 10,000/10,000 real states; probability deltas at floating
point noise). No model logic is re-implemented.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from traveler_distillation.matsim.s8_adapter import S8MATSimAdapter
from traveler_distillation.student.dataset import collate_batch


class StudentAdapter:
    """Load-once Student S9 wrapper with batched prediction."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"student checkpoint not found: {ckpt}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = ckpt
        self.device = device
        # production load (arch_version / n_alt_num / extractor hard asserts)
        self._adapter = S8MATSimAdapter(ckpt, device=device)

    # ------------------------------------------------------------ properties
    @property
    def extractor(self):
        return self._adapter.extractor

    @property
    def model(self):
        return self._adapter.model

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self._adapter.model.parameters())

    def checkpoint_sha256(self) -> str:
        h = hashlib.sha256()
        with self.checkpoint_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------------ inference
    def decide(self, state) -> dict:
        """Per-state decision (production path, batch size 1)."""
        return self._adapter.decide(state)

    @torch.no_grad()
    def predict(self, states: list, batch_size: int = 256, encoder=None) -> list[dict]:
        """Batch inference over states -> decision dicts identical in structure
        to ``decide()`` (mode = argmax over available alternatives).

        ``encoder`` defaults to the checkpoint extractor; a MemoizedEncoder
        (feature_adapter) may be passed to precompute static persona/trip
        parts — it self-checks parity against the production encoder.
        """
        enc = encoder if encoder is not None else self.extractor
        decisions: list[dict] = []
        for i in range(0, len(states), batch_size):
            chunk = states[i:i + batch_size]
            feats = [enc.encode(s) for s in chunk]
            batch = collate_batch([{
                "global_cat": torch.tensor(f["global_cat"], dtype=torch.long),
                "global_num": torch.tensor(f["global_num"], dtype=torch.float32),
                "alt_mode_idx": torch.tensor(f["alt_mode_idx"], dtype=torch.long),
                "alt_num": torch.tensor(f["alt_num"], dtype=torch.float32),
                "alt_mask": torch.tensor(f["alt_available"], dtype=torch.float32),
            } for f in feats])
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(batch)
            probs_np = out["mode_probabilities"].cpu().numpy()
            shift_np = out["departure_time_shift_min"].cpu().numpy()
            for j, state in enumerate(chunk):
                probs = {
                    alt.mode: float(probs_np[j][k])
                    for k, alt in enumerate(state.alternatives)
                    if alt.available
                }
                chosen = max(probs, key=probs.get)
                decisions.append({
                    "mode": chosen,
                    "mode_probabilities": probs,
                    "departure_time_shift_min": float(shift_np[j]),
                })
        return decisions
