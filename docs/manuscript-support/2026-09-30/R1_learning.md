# R1 — Extended learning diagnostics

These are preserved source excerpts from the supplied pre-trim manuscript package. They are extended records, not new experiments. The main article and compact supplement contain the primary evidence.

LaTeX labels and cross-references below retain their source names. They identify the original objects, not the renumbered compact supplement. Source paths that are mentioned in prose are historical descriptions, not a claim that those artifacts were uploaded in this migration.

## Extended timing and feasibility diagnostics

Source: `sections/S2_learning.tex`, lines 8–11.

```latex
MNL-S specifies four linear utilities over 111 explanatory variables. It minimizes cross-entropy against the soft Teacher targets with an $L_2$ penalty, whose weight of $10^{-4}$ was chosen from $\{10^{-4},10^{-3},10^{-2},10^{-1}\}$ by validation macro-source KL. Estimation by L-BFGS-B converges within 2,000 iterations. The ridge departure predictor uses a penalty of $10^{-3}$, chosen by validation departure MAE. Table~\ref{tab:matched_diagnostics} reports feasibility and departure-tail metrics for the models selected on choice.
\input{generated/narrative/table_training_sources.tex}
```

### Source table: `generated/narrative/table_training_sources.tex`

```latex
\begin{table}[pos=htbp]
\centering\small
\caption{Construction of the four behavioral-supervision sources and their contributions to the controlled test set.}
\label{tab:training_sources}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.16}
\begin{tabularx}{\linewidth}{@{}p{0.16\linewidth}Xrr@{}}
\toprule
Source & Comparison represented & States & Pairs \\
\midrule
Single-context & A traveler--trip baseline and single-axis changes in weather, congestion, PT delay, fares, parking or road disruption. & 226 & 214 \\
Accessibility & The same traveler and trip under PT supply profiles, varying access, waiting, transfers and connection feasibility. & 51 & 40 \\
Joint context & Two perturbations applied together, linked to their baseline and single-perturbation counterparts where available. & 96 & 96 \\
Mechanism & A baseline and three context--attribute variants: both changed, context only, or attributes only. These separate different encoded paths of the same intervention. & 64 & 48 \\
\midrule
Total & Held-out persona groups across the four sources. & 437 & 398 \\
\bottomrule
\end{tabularx}
\par\smallskip\begin{minipage}{\linewidth}\footnotesize
A state combines a traveler, a trip, a context, mode attributes and Teacher targets. A response pair links two states, and a state can belong to several pairs. Persona groups are disjoint across training, validation and test (28/6/6). Accessibility data additionally use disjoint OD pools. Teacher probability vectors are averaged over valid repeated queries. The counts are states and pairs, not independent travelers.
\end{minipage}
\end{table}
```

## Ancillary feasibility table

Source: `sections/S2_learning.tex`, lines 33–33.

```latex
\input{generated/narrative/table_matched_diagnostics.tex}
```

### Source table: `generated/narrative/table_matched_diagnostics.tex`

```latex
\begin{table}[pos=htbp]
\centering\small
\caption{Feasibility and departure diagnostics on the same controlled test pool. Neural entries average three training seeds. FVR uses 12 infeasible states per seed. Departure errors are in minutes; response SD describes training-seed variation.}
\label{tab:matched_diagnostics}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Model & FVR & Departure median & Departure P90 & Response SD \\
\midrule
Soft KL & 0.0833 & 5.12 & 15.59 & 0.0032 \\
CE + KL & 0.0833 & 5.16 & 15.74 & 0.0016 \\
Signed response & 0.0278 & 5.18 & 15.37 & 0.0050 \\
Direction + magnitude & 0.0556 & 5.06 & 16.06 & 0.0042 \\
MNL-S & 0.0000 & 10.53 & 25.42 & -- \\
\bottomrule
\end{tabular}
\end{table}
```

## Independent timing predictors and selected fits

Source: `sections/S2_learning.tex`, lines 35–46.

```latex
\subsection{Independent departure predictors}
\label{sec:departure_predictors}
The modular comparison keeps the 444 MNL-S choice coefficients fixed and adds either a bounded linear predictor with 111 parameters or an independent neural predictor with 18,289 parameters. The neural predictor keeps the feature encoders and departure layer of the joint architecture, omits the utility scorer and is trained on the departure Huber loss alone. The two modular Students therefore have 555 and 18,733 parameters in total.

All timing comparisons share the training states, the three seeds, the optimizer and the budget of 6,360 updates. The retained epoch minimizes validation departure MAE, with ties resolved in favour of the earliest epoch, and the four joint objectives are retrained under the same criterion. These timing fits are separate from the fits used for the probability-response comparison, although both follow the same update budget. Table~\ref{tab:departure_runs} lists the selected epochs, the results for every seed and the inference times, and Table~\ref{tab:departure_sources} breaks the errors down by supervision source.
\input{generated/narrative/table_departure_runs.tex}
\input{generated/narrative/table_departure_sources.tex}
Ridge predictions on the test set lie between $-36.63$ and 36.48 minutes, so the $\pm60$-minute bound is never active. Combined with the fixed MNL-S choice probabilities, the independent neural predictor reaches a departure MAE of $7.61\pm0.72$ minutes. It remains $0.180$ minutes above the best joint model, CE+KL selected on departure MAE, with a persona-clustered 95\% interval of $[0.002,0.418]$. Replacing the ridge predictor thus closes most, though not all, of the timing gap to the joint models.

The loss properties and their graphical diagnostic are in Appendix~\ref{main-sec:loss_diagnostics}.
```

### Source table: `generated/narrative/table_departure_runs.tex`

```latex
\begin{table}[pos=htbp]
\centering\footnotesize
\caption{Timing models selected by validation departure MAE, evaluated on the 437 test states.}
\label{tab:departure_runs}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}
\toprule
Model & Seed & Epoch & MAE & Median & P90 & Parameters & Train (s) & Infer. ($\mu$s) \\
\midrule
Soft KL & 42 & 2 & 8.461 & 3.355 & 25.611 & 24562 & 29.9 & 2.00 \\
Soft KL & 2026 & 117 & 7.048 & 5.236 & 15.801 & 24562 & 27.3 & 2.39 \\
Soft KL & 7 & 31 & 7.315 & 5.395 & 16.061 & 24562 & 38.1 & 2.07 \\
CE + KL & 42 & 2 & 8.455 & 3.348 & 25.600 & 24562 & 33.1 & 1.40 \\
CE + KL & 2026 & 108 & 6.719 & 5.452 & 14.114 & 24562 & 38.5 & 2.23 \\
CE + KL & 7 & 36 & 7.103 & 5.110 & 15.388 & 24562 & 34.9 & 1.83 \\
Signed & 42 & 2 & 8.464 & 3.355 & 25.612 & 24562 & 33.8 & 1.91 \\
Signed & 2026 & 35 & 7.441 & 5.399 & 15.143 & 24562 & 33.2 & 1.78 \\
Signed & 7 & 30 & 7.239 & 5.255 & 15.938 & 24562 & 30.9 & 1.75 \\
Dir. + mag. & 42 & 2 & 8.459 & 3.339 & 25.592 & 24562 & 30.2 & 1.67 \\
Dir. + mag. & 2026 & 108 & 6.952 & 4.858 & 14.918 & 24562 & 34.6 & 1.79 \\
Dir. + mag. & 7 & 30 & 7.238 & 5.397 & 16.072 & 24562 & 30.2 & 1.89 \\
MNL + bounded & 42 & 56 & 10.237 & 8.226 & 19.171 & 555 & 2.4 & 0.20 \\
MNL + bounded & 2026 & 31 & 10.737 & 7.321 & 24.947 & 555 & 2.4 & 0.19 \\
MNL + bounded & 7 & 22 & 12.592 & 10.687 & 23.368 & 555 & 2.6 & 0.19 \\
MNL + neural & 42 & 2 & 8.423 & 3.282 & 25.563 & 18733 & 19.7 & 1.30 \\
MNL + neural & 2026 & 116 & 7.348 & 5.248 & 15.745 & 18733 & 19.8 & 1.91 \\
MNL + neural & 7 & 97 & 7.046 & 5.231 & 15.023 & 18733 & 23.7 & 1.82 \\
\bottomrule
\end{tabular*}
\par\smallskip{\footnotesize All timing errors are in minutes. Every run completes 120 epochs and 6,360 updates; Epoch identifies the minimum validation departure-MAE checkpoint. For the modular Students, parameters include the 444 fixed MNL choice coefficients and the timing predictor. Inference time is the median of 30 batches of the 437 encoded states divided by 437, after five untimed calls and with two CPU threads, excluding encoding and transport preparation. Concurrent workload limits cross-run timing comparisons.}
\end{table}
```

### Source table: `generated/narrative/table_departure_sources.tex`

```latex
\begin{table}[pos=htbp]
\centering\footnotesize
\caption{Departure errors by test source under validation departure-MAE selection.}
\label{tab:departure_sources}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}llrrrr@{}}
\toprule
Model & Source & States & MAE & Median & P90 \\
\midrule
Soft KL & single-context & 226 & $6.849\pm0.667$ & 4.226 & 18.489 \\
Soft KL & joint & 96 & $8.923\pm2.010$ & 5.378 & 21.530 \\
Soft KL & mechanism & 64 & $5.921\pm0.466$ & 4.643 & 12.608 \\
Soft KL & accessibility & 51 & $10.614\pm0.732$ & 5.379 & 27.664 \\
CE + KL & single-context & 226 & $6.618\pm0.888$ & 4.282 & 17.240 \\
CE + KL & joint & 96 & $8.729\pm2.170$ & 5.319 & 21.548 \\
CE + KL & mechanism & 64 & $5.911\pm0.558$ & 4.546 & 12.851 \\
CE + KL & accessibility & 51 & $10.453\pm0.769$ & 5.354 & 27.870 \\
Signed & single-context & 226 & $6.964\pm0.553$ & 4.300 & 18.324 \\
Signed & joint & 96 & $9.186\pm1.845$ & 5.317 & 21.257 \\
Signed & mechanism & 64 & $5.843\pm0.352$ & 4.636 & 12.183 \\
Signed & accessibility & 51 & $10.621\pm0.595$ & 5.529 & 28.323 \\
Dir. + mag. & single-context & 226 & $6.765\pm0.748$ & 4.199 & 17.986 \\
Dir. + mag. & joint & 96 & $8.869\pm2.055$ & 5.262 & 21.354 \\
Dir. + mag. & mechanism & 64 & $5.851\pm0.385$ & 4.345 & 11.952 \\
Dir. + mag. & accessibility & 51 & $10.675\pm0.528$ & 5.560 & 28.118 \\
MNL + bounded & single-context & 226 & $10.622\pm1.429$ & 8.019 & 22.556 \\
MNL + bounded & joint & 96 & $12.294\pm1.512$ & 10.099 & 24.505 \\
MNL + bounded & mechanism & 64 & $9.447\pm2.162$ & 8.102 & 17.805 \\
MNL + bounded & accessibility & 51 & $13.805\pm1.021$ & 10.171 & 28.499 \\
MNL + neural & single-context & 226 & $6.807\pm0.672$ & 4.429 & 17.822 \\
MNL + neural & joint & 96 & $8.977\pm1.956$ & 5.312 & 22.015 \\
MNL + neural & mechanism & 64 & $5.806\pm0.517$ & 4.144 & 12.449 \\
MNL + neural & accessibility & 51 & $10.821\pm0.819$ & 4.955 & 27.857 \\
\bottomrule
\end{tabular}
\par\smallskip{\footnotesize All errors are in minutes. MAE reports the mean and sample SD across three training seeds; Median and P90 are means of the corresponding per-seed error quantiles. Source rows cover all test states and do not affect training or model selection.}
\end{table}
```

## Extended source/seed response breakdowns

Source: `sections/S2_learning.tex`, lines 47–51.

```latex
Tables~\ref{tab:response_breakdown}--\ref{tab:response_seed_differences} break the response errors down by supervision source, intervention, persona and seed. When each test persona is removed in turn, direction+magnitude remains better than soft KL, with seed-averaged differences between $-0.00408$ and $-0.00310$ and an improvement in all 18 combinations of seed and removed persona. Signed response improves on the seed average after every removal and in 14 of the 18 individual combinations.
\input{generated/narrative/table_response_breakdown.tex}
\input{generated/narrative/table_response_interventions.tex}
\input{generated/narrative/table_response_seed_differences.tex}
```

### Source table: `generated/narrative/table_response_breakdown.tex`

```latex
\begin{table}[pos=htbp]
\centering\footnotesize
\caption{Where response errors arise in the controlled comparison.}
\label{tab:response_breakdown}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrrr@{}}
\toprule
Grouping & Group & Pairs & Soft KL & CE + KL & Signed & Dir. + mag. \\
\midrule
source & accessibility & 40 & 0.12287 & 0.12834 & 0.11296 & 0.11947 \\
source & joint & 96 & 0.06726 & 0.07674 & 0.06340 & 0.06211 \\
source & single-context & 214 & 0.06008 & 0.06523 & 0.05923 & 0.05762 \\
source & mechanism & 48 & 0.06987 & 0.07292 & 0.06666 & 0.06540 \\
persona & P000002 & 70 & 0.07917 & 0.08605 & 0.07992 & 0.07869 \\
persona & P000008 & 57 & 0.06081 & 0.06934 & 0.05331 & 0.05527 \\
persona & P000009 & 72 & 0.07308 & 0.07613 & 0.07095 & 0.06901 \\
persona & P000015 & 58 & 0.05493 & 0.06483 & 0.04941 & 0.04955 \\
persona & P000016 & 70 & 0.07682 & 0.08440 & 0.07479 & 0.07574 \\
persona & P000018 & 71 & 0.06689 & 0.06810 & 0.06530 & 0.06209 \\
\bottomrule
\end{tabular*}
\par\smallskip{\footnotesize Values are response errors averaged over the three training seeds. Each grouping independently partitions the same 398 response pairs; its rows must not be added to those of another grouping. Mechanism contrasts retain their constructed label/attribute intervention definition.}
\end{table}
```

### Source table: `generated/narrative/table_response_interventions.tex`

```latex
\begin{table}[pos=htbp]
\centering\footnotesize
\caption{Controlled response error by intervention definition.}
\label{tab:response_interventions}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Intervention & Pairs & Soft KL & CE + KL & Signed & Dir. + mag. \\
\midrule
Congestion: context only & 8 & 0.05091 & 0.05784 & 0.05093 & 0.04885 \\
Congestion: attributes only & 8 & 0.04187 & 0.04275 & 0.04199 & 0.04152 \\
Congestion: both & 8 & 0.05253 & 0.06909 & 0.05570 & 0.05503 \\
Fare & 36 & 0.04100 & 0.04289 & 0.04235 & 0.04096 \\
Fare + congestion & 24 & 0.03850 & 0.03984 & 0.03634 & 0.03619 \\
Fare + PT delay & 24 & 0.07456 & 0.07840 & 0.07016 & 0.07083 \\
Parking: context only & 8 & 0.08981 & 0.09016 & 0.08602 & 0.08283 \\
Parking: attributes only & 8 & 0.09207 & 0.09701 & 0.08363 & 0.08431 \\
Parking: both & 8 & 0.09204 & 0.08070 & 0.08167 & 0.07987 \\
Parking price & 36 & 0.05709 & 0.05605 & 0.05375 & 0.05385 \\
Congestion & 48 & 0.04134 & 0.04474 & 0.04077 & 0.03972 \\
Congestion + road disruption & 24 & 0.05745 & 0.07450 & 0.05266 & 0.05111 \\
Congestion + rain & 24 & 0.09853 & 0.11422 & 0.09445 & 0.09030 \\
Road disruption & 12 & 0.06587 & 0.08475 & 0.06310 & 0.06346 \\
Supply profile & 40 & 0.12287 & 0.12834 & 0.11296 & 0.11947 \\
PT delay & 34 & 0.06330 & 0.06899 & 0.06214 & 0.06020 \\
Rain & 48 & 0.09163 & 0.10183 & 0.09141 & 0.08755 \\
\bottomrule
\end{tabular}
\par\smallskip{\footnotesize Each value averages the three validation-KL-selected seeds. Labels follow the definition of each perturbation, not its outcome, and the 17 rows partition all 398 test pairs. Context only, attributes only and both denote mechanism contrasts that change the context description, the level-of-service attributes, or both.}
\end{table}
```

### Source table: `generated/narrative/table_response_seed_differences.tex`

```latex
\begin{table}[pos=htbp]
\centering\footnotesize
\caption{Per-seed paired response-error differences relative to soft KL, using the validation-KL checkpoints.}
\label{tab:response_seed_differences}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}lrrlrr@{}}
\toprule
Objective & Seed & Difference & 95\% interval & Pairs & Personas \\
\midrule
CE + KL & 42 & 0.00742 & [0.00153, 0.01469] & 398 & 6 \\
CE + KL & 2026 & 0.00371 & [-0.00207, 0.01002] & 398 & 6 \\
CE + KL & 7 & 0.00679 & [0.00069, 0.01278] & 398 & 6 \\
Signed & 42 & -0.00524 & [-0.00815, -0.00228] & 398 & 6 \\
Signed & 2026 & -0.00334 & [-0.00941, 0.00038] & 398 & 6 \\
Signed & 7 & 0.00027 & [-0.00252, 0.00304] & 398 & 6 \\
Dir. + mag. & 42 & -0.00499 & [-0.00773, -0.00235] & 398 & 6 \\
Dir. + mag. & 2026 & -0.00441 & [-0.00762, -0.00172] & 398 & 6 \\
Dir. + mag. & 7 & -0.00094 & [-0.00342, 0.00113] & 398 & 6 \\
\bottomrule
\end{tabular}
\par\smallskip{\footnotesize Negative differences favor the listed objective. Intervals use 10,000 paired persona-cluster resamples, retaining all tasks for each of six test personas. They condition on each fitted seed and are exploratory intervals without adjustment across the methods and seeds.}
\end{table}
```
