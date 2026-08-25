# S7-W3 Seed Robustness Report (frozen artifact)

Source: `outputs/s7_seed_check/seed_summary.json` (verbatim projection).
W3 config retrained under 4 training seeds (42, 7, 123, 2024), zero API calls,
evaluated on the same test-only pipeline with paired bootstrap CIs.

| seed | parking G_med Δ | congestion G_broken Δ | congestion Gap_shortcut Δ | Δ legacy KL | Δ seen joint KL |
|---|---|---|---|---|---|
| 42 | -0.0054 [-0.0090, -0.0018]* | +0.0017 [-0.0131, +0.0127] | -0.0566 [-0.1164, -0.0051]* | -0.0040 [-0.0052, -0.0027]* | -0.0021 [-0.0035, -0.0006]* |
| 7 | -0.0057 [-0.0096, -0.0021]* | -0.0006 [-0.0153, +0.0111] | -0.0560 [-0.1064, -0.0125]* | -0.0022 [-0.0035, -0.0009]* | -0.0013 [-0.0027, +0.0004] |
| 123 | -0.0060 [-0.0099, -0.0021]* | +0.0012 [-0.0134, +0.0142] | -0.0485 [-0.0792, -0.0221]* | -0.0025 [-0.0037, -0.0012]* | -0.0011 [-0.0025, +0.0004] |
| 2024 | -0.0056 [-0.0093, -0.0021]* | +0.0016 [-0.0126, +0.0127] | -0.0550 [-0.1035, -0.0143]* | -0.0030 [-0.0044, -0.0016]* | -0.0015 [-0.0031, +0.0002] |

**Verdict**: stable_across_seeds = True; regressed_seeds = []
**Interpretation**: S7 improvement > training-seed variance: the repair direction is consistent across all seeds and no seed regresses legacy/seen-joint KL.

CIs are 95% paired bootstrap (B=2000, seed=42); * = CI excludes 0.
