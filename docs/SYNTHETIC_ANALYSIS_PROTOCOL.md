# Synthetic reference analysis, 25 September 2026

The primary comparison and sample-size rule are unchanged from the frozen acquisition protocol. The twelve pilot personas are used only for planning the formal sample; no formal persona is part of the pilot. A complete persona contributes twelve states with three valid responses per state. A failed attempt is retained but never counts as a valid response. No response is excluded because of the direction or magnitude of a model comparison.

Primary outcome: mean absolute discrepancy between Student and reference changes, averaged over union-available modes, eleven baseline-to-intervention contrasts and three training seeds within each persona. The primary difference is signed L1 (weight 1, static selector) minus soft KL (static selector), using full-data models. The sampling unit is the synthetic persona. Planning uses the pilot standard deviation and `max(30, ceil((1.96 * SD / 0.01)^2))`, with a maximum of 120. Pilot and formal outcomes are reported separately. The formal 95% interval uses 10,000 paired persona bootstrap samples, seed 20260925. The planning formula does not guarantee the achieved interval width.

The following descriptive extensions were fixed while pilot acquisition was incomplete, before examining any aggregate model-comparison outcome:

- Report every intervention card separately, preserving both intensities rather than choosing a favorable level.
- Report all three weight-1 neural objectives and the validation-selected MNL baseline for full training; soft KL and signed L1 for delay-family holdout. Keep both validation selectors and average neural seeds within each persona.
- For each method, report response gap, endpoint KL, probability L1 and departure MAE. For service infeasibility, additionally report PT probability mass and top-choice violation; the conceptual availability mask alone does not encode service feasibility.
- Report signed-minus-soft differences for full training and delay holdout, overall and by card. All comparisons except the single full-data/static primary comparison are descriptive, with pointwise intervals and no familywise-significance claim.
- Report Teacher self-disagreement using all three pairs of distinct repeats. Shared endpoints, pairings and repeats are correlated; this is a target-variability diagnostic, not an independent noise ceiling.
- For fare-delay and fare-access, report the second-order interaction discrepancy using the corresponding high-intensity component states and the baseline.

The API reference is the actual returned DeepSeek V4.1 Flash model through OpenCode Go. These comparisons assess transfer to that reference and do not measure the original Pro model's compression error, human agreement, geographic transfer, or human repair. The fixed numeric trip further limits the population being sampled.

All attempts, including transport failures and invalid content, remain in the acquisition ledger. Usage reported for unsuccessful or invalid responses is included in token totals. Calls without provider usage have unknown usage; quota-equivalent estimates are not invoices or verified account debits. Actual peak/off-peak pricing and cache counts are distinguished from the conservative peak/no-cache reservation.

## Transport recovery amendment

After the author enabled Global regions, requests returned the requested model, but the initial pilot had repeated `ConnectError` and `RemoteProtocolError` failures. The first acquisition driver is retained with SHA256 `ae6b1dc81dfcf0a418c1b4c48f20bc104177a055b6e13078a1a3cebc668cf48b`. Its active process finishes under its original bounds. Subsequent launches use a persistent HTTP connection pool with eight workers, a 30-second connection timeout, the unchanged 180-second read timeout, and 2/4/8/16-second bounded backoff.

Only missing valid state-repeat slots are retried. The maximum lifetime attempts per slot is eight, and the maximum cohort attempts is three times the number of required valid replies. The peak/no-cache caps for **reported usage** remain USD 3 for pilot and USD 8 for formal. Unreported transport usage is unknown; these monetary caps do not guarantee total provider debits. Any HTTP 4xx stops new requests. All original failures and invalid replies remain available, including usage on output-truncated replies. No provider, model, prompt, temperature, output cap or state changes; no account balance fallback setting is changed. The extension responds to transport failures, not observed scientific outcomes. The machine-readable amendment records the UTC time and launch hashes.
