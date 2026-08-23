"""Variable-choice-set Student model.

u_i = f_theta(S, X_i) for each available alternative i,
P_S(i|x) = softmax over available utilities only (masked softmax),
departure shift = 60 * tanh(head(global_emb, pooled_alt_emb)).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over valid alternatives only.

    ``mask`` has 1.0 for available alternatives and 0.0 for padded/unavailable
    entries; masked entries receive -inf logits so they contribute zero mass.
    """
    logits = logits.masked_fill(mask < 0.5, float("-inf"))
    return torch.softmax(logits, dim=-1)


class TravelerStudent(nn.Module):
    def __init__(self, cfg: dict, feature_spec: dict):
        super().__init__()
        self.cat_emb_dim = cfg.get("cat_embedding_dim", 8)
        self.mode_emb_dim = cfg.get("mode_embedding_dim", 8)
        self.global_hidden = cfg.get("global_hidden_dim", 64)
        self.alt_hidden = cfg.get("alternative_hidden_dim", 32)
        self.scorer_hidden = cfg.get("scorer_hidden_dim", 64)
        self.dropout_p = cfg.get("dropout", 0.1)

        self.cat_names = feature_spec["cat_names"]
        self.cat_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(size, self.cat_emb_dim)
                for name, size in feature_spec["cat_vocab_sizes"].items()
            }
        )
        self.mode_embedding = nn.Embedding(feature_spec["mode_vocab_size"], self.mode_emb_dim)

        n_cat_total = len(self.cat_names) * self.cat_emb_dim
        n_global_num = feature_spec["n_global_num"]
        self.n_alt_num = feature_spec["n_alt_num"]

        self.global_encoder = nn.Sequential(
            nn.Linear(n_cat_total + n_global_num, self.global_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.global_hidden, self.global_hidden),
            nn.ReLU(),
        )
        self.alt_encoder = nn.Sequential(
            nn.Linear(self.mode_emb_dim + self.n_alt_num, self.alt_hidden),
            nn.ReLU(),
            nn.Linear(self.alt_hidden, self.alt_hidden),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(self.global_hidden + self.alt_hidden, self.scorer_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.scorer_hidden, 1),
        )
        self.departure_head = nn.Sequential(
            nn.Linear(self.global_hidden + self.alt_hidden, self.scorer_hidden),
            nn.ReLU(),
            nn.Linear(self.scorer_hidden, 1),
        )

    def forward(self, batch: dict) -> dict:
        # global embedding
        cat_embs = [
            self.cat_embeddings[name](batch["global_cat"][:, i])
            for i, name in enumerate(self.cat_names)
        ]
        global_cat = torch.cat(cat_embs, dim=-1)  # (B, n_cat * cat_emb_dim)
        global_in = torch.cat([global_cat, batch["global_num"]], dim=-1)
        global_emb = self.global_encoder(global_in)  # (B, global_hidden)

        # alternative embeddings
        mode_emb = self.mode_embedding(batch["alt_mode_idx"])  # (B, M, mode_emb_dim)
        alt_in = torch.cat([mode_emb, batch["alt_num"]], dim=-1)  # (B, M, mode_emb_dim + n_alt_num)
        alt_emb = self.alt_encoder(alt_in)  # (B, M, alt_hidden)

        # utilities + masked softmax
        B, M, _ = alt_emb.shape
        g_expanded = global_emb.unsqueeze(1).expand(-1, M, -1)  # (B, M, global_hidden)
        scorer_in = torch.cat([g_expanded, alt_emb], dim=-1)  # (B, M, global_hidden + alt_hidden)
        utilities = self.scorer(scorer_in).squeeze(-1)  # (B, M)

        mask = batch["alt_mask"]  # (B, M)
        probs = masked_softmax(utilities, mask)  # (B, M)

        # departure head from masked-mean pooled alternative representation
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (alt_emb * mask.unsqueeze(-1)).sum(dim=1) / mask_sum  # (B, alt_hidden)
        dep_in = torch.cat([global_emb, pooled], dim=-1)
        dep_raw = self.departure_head(dep_in).squeeze(-1)  # (B,)
        departure = 60.0 * torch.tanh(dep_raw)

        return {
            "mode_probabilities": probs,
            "departure_time_shift_min": departure,
            "utilities": utilities,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
