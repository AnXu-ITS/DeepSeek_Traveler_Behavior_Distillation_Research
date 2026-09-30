# R3 — Synthetic Singapore illustration and journey measures

These are preserved source excerpts from the supplied pre-trim manuscript package. They are extended records, not new experiments. The main article and compact supplement contain the primary evidence.

LaTeX labels and cross-references below retain their source names. They identify the original objects, not the renumbered compact supplement. Source paths that are mentioned in prose are historical descriptions, not a claim that those artifacts were uploaded in this migration.

## Synthetic Singapore illustration and journey summaries

Source: `sections/S5_execution.tex`, lines 16–29.

```latex
\subsection{Singapore population responses and journey measures}
The Singapore scenario experiment simulates 10,000 synthetic travelers on unchanged physical supply. Scenario C0 is the baseline, C1 introduces heavy rain, C2 a 50\% fare increase and C3 a perceived PT delay of 15 minutes. C4 represents a road disruption that adds 20 minutes of car travel time and 20 minutes of reliability delay, and C5 combines rain with PT delay. Populations drawn with seeds 2026, 42 and 7 vary the composition of travelers, and MATSim uses seed 4711. Table~\ref{tab:singapore} gives the exact shares shown in Figure~\ref{main-fig:concept} of the main article, and Table~\ref{tab:seeds} their sensitivity to population composition.
\input{generated/narrative/table_singapore.tex}
\input{generated/narrative/table_population_seeds.tex}
Journey measures for each person run from the first departure to the first activity other than home or an interaction point created by routing, such as a transfer. Table~\ref{tab:singapore_person_counts} reports people, boardings and legs separately. At baseline, 9,741 people complete the outbound journey and 9,062 return home. A completed outbound journey takes 66.43 minutes on average, whereas a completed leg takes 43.67 minutes, because a PT journey can consist of several legs.

Table~\ref{tab:singapore_person_time} also reports, for every departing person, the time elapsed until completion or until the end of the 30-hour simulation, a descriptive measure that includes unfinished journeys. Paired differences between scenarios use people who complete both trips and 4,000 paired-person resamples. All times reflect the network-based speeds and the fixed supply.
\input{generated/narrative/table_sg_person_counts.tex}
\input{generated/narrative/table_sg_person_time.tex}
\FloatBarrier

The feasibility identity and counterexample are in Appendix~\ref{main-sec:feasibility_geometry}.
```

### Source table: `generated/narrative/table_singapore.tex`

```latex
\begin{table}[pos=htbp]
\centering\small
\setlength{\tabcolsep}{3pt}
\caption{Singapore absolute Student decision shares (percent) and MATSim consequences. Time is the completed-leg mean, not the mean journey duration per person. Seed 2026, 10,000 agents.}
\label{tab:singapore}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Scenario & Car & PT & Bike & Walk & Boardings & Car VKT (km) & \shortstack{Completed-leg\\mean (min)} \\
\midrule
C0 & 28.3 & 25.3 & 34.0 & 12.4 & 5613 & 33,939.0 & 43.67 \\
C1 & 37.6 & 45.3 & 13.3 & 3.8 & 9761 & 42,777.5 & 50.86 \\
C2 & 29.9 & 25.7 & 32.6 & 11.8 & 5674 & 35,797.8 & 43.86 \\
C3 & 29.0 & 10.3 & 35.8 & 25.0 & 2474 & 34,684.6 & 32.23 \\
C4 & 1.8 & 37.5 & 41.8 & 19.0 & 8537 & 2,943.4 & 52.71 \\
C5 & 37.7 & 18.6 & 24.5 & 19.2 & 4301 & 42,822.8 & 38.27 \\
\bottomrule
\end{tabular}
\end{table}
```

### Source table: `generated/narrative/table_population_seeds.tex`

```latex
\begin{table}[pos=htbp]
\centering\small
\caption{Paired population-seed responses from the existing multi-seed summary.}
\label{tab:seeds}
\begin{tabular}{@{}lrrr@{}}
\toprule
Scenario & Seed & $\Delta$ PT pp & $\Delta$ car pp \\
\midrule
C1 & 2026 & +20.0 & +9.3 \\
C1 & 42 & +19.7 & +9.2 \\
C1 & 7 & +20.1 & +9.2 \\
C2 & 2026 & +0.4 & +1.6 \\
C2 & 42 & +0.5 & +1.8 \\
C2 & 7 & +0.5 & +1.5 \\
C3 & 2026 & -15.1 & +0.8 \\
C3 & 42 & -15.3 & +0.6 \\
C3 & 7 & -15.7 & +0.6 \\
C4 & 2026 & +12.1 & -26.5 \\
C4 & 42 & +11.7 & -26.1 \\
C4 & 7 & +12.3 & -26.5 \\
C5 & 2026 & -6.7 & +9.4 \\
C5 & 42 & -6.4 & +9.3 \\
C5 & 7 & -6.9 & +9.1 \\
\bottomrule
\end{tabular}
\end{table}
```

### Source table: `generated/narrative/table_sg_person_counts.tex`

```latex
\begin{table}[pos=htbp]
\centering\footnotesize
\caption{Person and leg counts across six Singapore scenarios.}
\label{tab:singapore_person_counts}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Scenario & \shortstack{Outbound\\complete} & \shortstack{Return\\complete} & \shortstack{Outbound\\PT people} & Boardings & \shortstack{Unfinished\\legs} & \shortstack{Stuck\\people} & \shortstack{Completed\\legs} \\
\midrule
C0 & 9,741 & 9,062 & 2,456 & 5,613 & 938 & 211 & 29,476 \\
C1 & 9,546 & 8,367 & 4,442 & 9,761 & 1,633 & 366 & 36,477 \\
C2 & 9,730 & 9,044 & 2,492 & 5,674 & 956 & 221 & 29,567 \\
C3 & 9,884 & 9,600 & 961 & 2,474 & 400 & 102 & 24,219 \\
C4 & 9,647 & 8,619 & 3,671 & 8,537 & 1,381 & 285 & 34,499 \\
C5 & 9,789 & 9,268 & 1,779 & 4,301 & 732 & 183 & 27,266 \\
\bottomrule
\end{tabular}
\par\smallskip\begin{minipage}{0.98\textwidth}\footnotesize Each scenario starts with and departs all 10,000 people. Completion and PT use are person counts. Boardings include transfers and both travel directions; unfinished legs and completed legs are different event-level counts. A stuck person may also have an unfinished leg.\end{minipage}
\end{table}
```

### Source table: `generated/narrative/table_sg_person_time.tex`

```latex
\begin{table}[pos=htbp]
\centering\footnotesize
\caption{Singapore timing with explicit units and completion conditions. All times are in minutes.}
\label{tab:singapore_person_time}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{@{}lrrrrrl@{}}
\toprule
Scenario & \shortstack{Completed-leg\\mean} & \shortstack{Outbound\\mean} & \shortstack{Outbound\\median} & \shortstack{Completion-or-\\horizon mean} & \shortstack{Paired\\people} & \shortstack{Paired outbound change\\versus C0 [95\% CI]} \\
\midrule
C0 & 43.67 & 66.43 & 17.45 & 88.80 & -- & -- \\
C1 & 50.86 & 98.50 & 20.19 & 135.60 & 9,511 & +32.00 [+29.17, +34.69] \\
C2 & 43.86 & 66.88 & 16.97 & 90.01 & 9,708 & +0.60 [-0.45, +1.69] \\
C3 & 32.23 & 38.92 & 15.67 & 49.78 & 9,724 & -27.68 [-29.86, -25.56] \\
C4 & 52.71 & 94.78 & 27.08 & 124.21 & 9,578 & +28.39 [+25.98, +30.83] \\
C5 & 38.27 & 53.33 & 14.35 & 71.97 & 9,695 & -13.87 [-15.83, -11.96] \\
\bottomrule
\end{tabular}
\par\smallskip\begin{minipage}{0.98\textwidth}\footnotesize Outbound means and medians condition on the scenario-specific completers in Table~\ref{tab:singapore_person_counts}; the horizon mean retains all 10,000 departed people until completion or the administrative end at simulation hour 30. Paired changes retain only people completing both outbound journeys and use 4,000 paired-person bootstrap resamples (seed 917). The link-speed implementation limits interpretation as calibrated travel time.\end{minipage}
\end{table}
```
