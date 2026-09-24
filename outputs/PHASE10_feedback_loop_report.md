# Prototype feedback-loop experiment

This historical prototype explores iteration between a modeled congestion state and Student demand on a simplified environment. Its convergence diagnostics belong to that prototype. The current manuscript experiment holds behavioral predictions fixed during MATSim execution and does not claim that this prototype establishes citywide behavioral equilibrium.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/outputs/PHASE10_feedback_loop_report.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../docs/RESEARCH_DESIGN.md>) · [Training](<../docs/TRAINING.md>) · [Results](<../docs/RESULTS.md>) · [Data and access](<../docs/DATA_SOURCES.md>) · [Model use](<../docs/MODEL_USE.md>)

## Historical numerical table 1

| iter | c_ctx | bike | car | pt | walk | c_obs | mean_delay_ratio |
|---|---|---|---|---|---|---|---|
| 1 | 0.300 | 0.179 | 0.250 | 0.521 | 0.050 | 0.380 | 0.190 |
| 2 | 0.340 | 0.179 | 0.251 | 0.521 | 0.049 | 0.381 | 0.191 |
| 3 | 0.360 | 0.179 | 0.253 | 0.520 | 0.049 | 0.377 | 0.189 |

## Historical numerical table 2

| iter | c_ctx | bike | car | pt | walk | c_obs | mean_delay_ratio |
|---|---|---|---|---|---|---|---|
| 1 | 0.300 | 0.000 | 0.385 | 0.570 | 0.045 | 0.438 | 0.219 |
| 2 | 0.369 | 0.000 | 0.385 | 0.570 | 0.045 | 0.434 | 0.217 |
| 3 | 0.402 | 0.000 | 0.385 | 0.570 | 0.045 | 0.441 | 0.221 |
| 4 | 0.421 | 0.000 | 0.385 | 0.570 | 0.045 | 0.442 | 0.221 |
| 5 | 0.432 | 0.000 | 0.385 | 0.570 | 0.045 | 0.444 | 0.222 |

## Historical numerical table 3

| iter | c_ctx | bike | car | pt | walk | c_obs | mean_delay_ratio |
|---|---|---|---|---|---|---|---|
| 1 | 0.300 | 0.152 | 0.314 | 0.485 | 0.049 | 0.401 | 0.201 |
| 2 | 0.351 | 0.150 | 0.316 | 0.485 | 0.049 | 0.402 | 0.201 |
| 3 | 0.376 | 0.150 | 0.316 | 0.485 | 0.049 | 0.402 | 0.201 |
| 4 | 0.389 | 0.150 | 0.316 | 0.485 | 0.049 | 0.402 | 0.201 |

## Historical numerical table 4

|scenario| initial c | converged c_ctx | equilibrium c_obs | car share change |
|---|---|---|---|---|
| baseline | 0.300 | 0.360 | 0.377 | +0.25pp |
| rain | 0.300 | 0.432 | 0.444 | +0.00pp |
| fare_surge | 0.300 | 0.389 | 0.402 | +0.25pp |

## Retained English technical listings

```powershell
.venv\Scripts\python.exe scripts\run_phase10_loop.py --checkpoint outputs/student_v0_3_c/checkpoints/best.pt --num-personas 800 --trips-per-persona 1 --scenarios baseline,rain,fare_surge --max-iterations 6 --eps 0.02 --grid-n 10 --link-capacity 5 --output data/phase10_loop
```
