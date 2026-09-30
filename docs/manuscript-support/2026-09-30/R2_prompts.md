# R2 — Prompt pilots and subset choice metrics

These are preserved source excerpts from the supplied pre-trim manuscript package. They are extended records, not new experiments. The main article and compact supplement contain the primary evidence.

LaTeX labels and cross-references below retain their source names. They identify the original objects, not the renumbered compact supplement. Source paths that are mentioned in prose are historical descriptions, not a claim that those artifacts were uploaded in this migration.

## Prompt and metadata pilot

Source: `sections/S3_human.tex`, lines 31–38.

```latex
\subsection{Teacher prompt and metadata sensitivity}
A small diagnostic uses the first two selected respondents in each city, with all ten tasks and three queries per condition, and keeps the ownership and access rule of the Students. One comparison replaces a generic prompt with the accessibility-aware prompt while metadata are held neutral. The other replaces identifiers that reveal the city with neutral values while the generic prompt is held fixed, without removing all metadata fields.

Across these 40 states, the change of prompt shifts mean PT probability by $+1.04$ points and the neutral metadata shift it by $-1.06$ points (Table~\ref{tab:teacher_prompt_pilot}). Both prompt conditions used the same token budget, whereas some requests in the metadata comparison were repeated with different budgets. Table~\ref{tab:teacher_pilot_sd} reports the variation across repeated queries.
\input{generated/narrative/table_teacher_prompt_pilot.tex}
\input{generated/narrative/table_teacher_pilot_sd.tex}
```

### Source table: `generated/narrative/table_teacher_prompt_pilot.tex`

```latex
\begin{table}[pos=htbp]
\centering\footnotesize
\caption{Descriptive matched diagnostics using the first two preselected people per city (four people, 40 states, three queries per state in each condition). Changes are right minus left under the comparisons defined in the text.}
\label{tab:teacher_prompt_pilot}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrr@{}}
\toprule
Comparison & Scope & Mean state probability $L_1$ & PT change (pp) \\
\midrule
System prompt & Singapore & 0.0767 & +1.55 \\
System prompt & Shanghai & 0.0327 & +0.53 \\
System prompt & Pooled four people & 0.0547 & +1.04 \\
\midrule
Unused metadata & Singapore & 0.0843 & -1.00 \\
Unused metadata & Shanghai & 0.0343 & -1.12 \\
Unused metadata & Pooled four people & 0.0593 & -1.06 \\
\bottomrule
\end{tabular*}
\par\smallskip\begin{minipage}{\linewidth}\footnotesize
All task groups and all three query repeats are complete; no person was replaced and no human answer was used in this diagnostic. Probability vectors are averaged over repeats before calculating the state-level differences. The pooled result weights the two cities equally, with 20 states each. The differences are descriptive and carry no confidence intervals.
\end{minipage}
\end{table}
```

### Source table: `generated/narrative/table_teacher_pilot_sd.tex`

```latex
\begin{table}[pos=htbp]
\centering\footnotesize
\caption{Within-state Teacher query variability in the matched four-person diagnostic. Entries are the mean sample SD of PT probability across three calls, in percentage points.}
\label{tab:teacher_pilot_sd}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr@{}}
\toprule
Condition & Singapore & Shanghai & Pooled \\
\midrule
Generic / city metadata / original support & 4.79 & 1.65 & 3.22 \\
Generic / neutral metadata / original support & 3.99 & 2.36 & 3.18 \\
Accessibility / neutral / original support & 3.94 & 1.84 & 2.89 \\
\bottomrule
\end{tabular*}
\par\smallskip\begin{minipage}{\linewidth}\footnotesize
Variability is computed on successful answers only and describes the repeatability of the Teacher under the recorded request settings.
\end{minipage}
\end{table}
```

## Additional subset choice metrics

Source: `sections/S3_human.tex`, lines 24–24.

```latex
\input{generated/narrative/table_teacher_choices.tex}
```

### Source table: `generated/narrative/table_teacher_choices.tex`

```latex
\begin{table}[pos=htbp]
\centering\footnotesize
\caption{Choice metrics on the Teacher comparison subsets. Each city contributes 24 people and 240 explicit choices. Neural families show the arithmetic mean and sample SD of three separately scored training seeds.}
\label{tab:teacher_subset_choices}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr@{}}
\toprule
Model & Accuracy (\%) & Brier score & PT-share bias (pp) \\
\midrule
\multicolumn{4}{l}{\textit{Singapore}} \\
Current Teacher & $75.42$ & $0.356$ & $-6.21$ \\
SA-Student & $61.67$ & $0.549$ & $-29.70$ \\
MNL-S & $65.00$ & $0.428$ & $0.51$ \\
Soft KL & $65.42\pm 2.32$ & $0.442\pm 0.049$ & $-17.98\pm 2.55$ \\
CE+KL & $63.75\pm 0.72$ & $0.488\pm 0.049$ & $-23.26\pm 2.75$ \\
Signed response & $65.28\pm 2.68$ & $0.437\pm 0.053$ & $-16.32\pm 3.75$ \\
Direction+magnitude & $64.86\pm 4.17$ & $0.443\pm 0.065$ & $-16.68\pm 5.70$ \\
\midrule
\multicolumn{4}{l}{\textit{Shanghai}} \\
Current Teacher & $85.00$ & $0.233$ & $4.32$ \\
SA-Student & $76.67$ & $0.342$ & $-18.71$ \\
MNL-S & $78.33$ & $0.294$ & $-2.53$ \\
Soft KL & $76.39\pm 0.24$ & $0.329\pm 0.027$ & $-12.54\pm 7.59$ \\
CE+KL & $74.44\pm 2.51$ & $0.350\pm 0.015$ & $-14.90\pm 6.75$ \\
Signed response & $77.92\pm 2.73$ & $0.319\pm 0.023$ & $-11.20\pm 6.03$ \\
Direction+magnitude & $77.92\pm 1.10$ & $0.317\pm 0.022$ & $-10.74\pm 5.80$ \\
\bottomrule
\end{tabular*}
\par\smallskip\begin{minipage}{\linewidth}\footnotesize
Teacher choice metrics use the mean probabilities from three successful queries per state. SA-Student and MNL-S are single fitted models; their rows have no training-seed SD. For the four neural objectives, probabilities are not averaged across seeds before computing accuracy or Brier score. PT-share bias is predicted probability minus the human PT indicator. Neither subset contains an answer of the no-suitable-mode type or a choice of a mode the model treats as unavailable.
\end{minipage}
\end{table}
```
