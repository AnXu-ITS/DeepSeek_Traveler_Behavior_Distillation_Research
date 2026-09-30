# R5 — Service provenance and material history

These are preserved source excerpts from the supplied pre-trim manuscript package. They are extended records, not new experiments. The main article and compact supplement contain the primary evidence.

LaTeX labels and cross-references below retain their source names. They identify the original objects, not the renumbered compact supplement. Source paths that are mentioned in prose are historical descriptions, not a claim that those artifacts were uploaded in this migration.

## New-reference service-metadata diagnostic

Source: `sections/D_new_evidence.tex`, lines 16–21.

```latex
\subsection{Returned service metadata}
During formal acquisition, some responses reported zero reasoning tokens and a different usage-detail structure while request settings and the returned model name stayed the same. The diagnostic was documented after observing those metadata and before examining formal method comparisons. Zero reported tokens do not prove absence of internal reasoning or a backend model change. No immutable backend fingerprint was available.

The formal sample has 976 positive-reasoning and 104 zero-reported-reasoning responses. Twenty-four personas have only the former and six have a mixture. The primary paired difference is $-0.00299$ ($[-0.00496,-0.00121]$) in the all-positive group and $-0.00405$ ($[-0.00770,-0.00117]$) in the mixed group. These post-hoc subgroups are confounded with acquisition time and persona order. They neither identify a reasoning-mode effect nor justify excluding data. The primary analysis includes all thirty personas and is conditional on the realized service mix. No samples were excluded because of the metadata diagnostic.
```

## Input-provenance summary table

Source: `sections/A_inputs.tex`, lines 30–44.

```latex
\begin{table}[pos=htbp]
\centering\small
\caption{Sources of information in the stated-choice model inputs.}
\label{tab:input_provenance}
\renewcommand{\arraystretch}{1.13}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.22\linewidth}>{\raggedright\arraybackslash}p{0.43\linewidth}>{\raggedright\arraybackslash}X@{}}
\toprule
Source & Information used & Interpretation \\
\midrule
Shown in the questionnaire & Singapore time and cost tables; the Shanghai PT fare of 6 yuan, parking fee of 30 yuan and one transfer & Attributes presented to respondents, with different numerical detail in the two surveys. \\
Reported by respondents & Personal characteristics supplied by respondents and covered by the model's input mapping & Observed inputs, although not every model attribute was collected. \\
Specified by the researchers & Characteristics not collected, and Shanghai durations and time components not shown in the tasks & Analysis assumptions. The twelve Singapore and eight Shanghai profiles examine selected alternatives rather than a full uncertainty distribution. \\
\bottomrule
\end{tabularx}
\end{table}
```

## Versioned materials and implementation records

Source: `sections/S7_materials.tex`, lines 1–51.

```latex
\section{Model roles, input sources and research materials}
\label{sec:materials_scope}

\subsection{Model roles and input sources}
Table~\ref{main-tab:model_roles} summarizes how the three model families enter the study, and Table~\ref{tab:input_provenance} the sources of information in the stated-choice inputs. Fitting settings and the numerical task specifications are given in Sections~\ref{sec:supp_controlled}, \ref{main-sec:s9_recipe} and~\ref{sec:supp_inputs}.




\FloatBarrier

\subsection{Versioned materials and reproduction tasks}
The expanded experimental evidence is pinned to repository commit
\begin{center}\small\texttt{ba7c5d304b46b46f8f74c451cda2825a3916bf6d}.\end{center}
It adds 27 controlled fits, six delay-family fits, a separate new-persona API reference, repeat and optimization audits, and the verified four-arm supply experiment. The compact article inputs are in \path{data_tables/revision20260925/}, with a source-hash manifest. The original figures can be regenerated with \path{scripts/make_revision_figures.py}; \path{scripts/make_restructured_figures.py} generates the current main-text Figures~\ref{main-fig:response_learning} and~\ref{main-fig:teacher_attribution}. The repository's \path{evidence/experiments_20260925/} contains the full packaged experiment evidence and verification records. Section~\ref{sec:new_protocol} documents the new reference. No independent new-human repair validation has been performed.

The historical material inventory below refers to the edition of 24 September 2026 at commit
\begin{center}\small\texttt{4e67286ce72680a9e3a916a3fb7e97b1548b53f7}.\end{center}
The repository is \url{https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research}. It remains access-controlled. Its \path{docs/REPRODUCIBILITY.md} maps materials to reproduction tasks, and \path{evidence/paper_20260924/} contains the artifact manifest, verification record and compact numerical evidence. The manuscript source package separately contains input definitions, adaptation settings, editable figures and tabulated summaries.

\begin{table}[pos=htbp]
\centering\small
\caption{Teacher targets, reference-model provenance and supporting materials. Frozen-target re-evaluation and live-API replication are distinct tasks.}
\label{tab:teacher_materials}
\renewcommand{\arraystretch}{1.13}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.18\linewidth}>{\raggedright\arraybackslash}p{0.30\linewidth}>{\raggedright\arraybackslash}X@{}}
\toprule
Use & Coverage & Materials and interpretation \\
\midrule
Controlled supervision & 437 test states; repeat analysis covers 373 states and 350 pairs. & Prepared states, aggregate targets, splits, 12 selected neural fits, MNL-S records and evaluation code are included. Individual repeats are unavailable for the 64 mechanism states; the authors identify the service as official DeepSeek V4 Pro, while immutable per-response backend identity is not retained for every target. \\
Supply adaptation & 336 accessibility states; 1,502 valid queries. & Adaptation specification and released SA-Student and predecessor checkpoints are included. Retained response records and training history support the adaptation provenance. \\
Contemporary API reference & 24 respondents per city; ten tasks; three successful elicitations per state. & Configuration, prompt, campaign summary and aggregate comparisons are included. The returned model string is the alias \texttt{deepseek-v4-pro}; participant-linked payloads are retained separately. The alias alone does not establish backend identity. \\
\bottomrule
\end{tabularx}
\end{table}

The authors confirm use of the official DeepSeek V4 Pro direct service for the historical supervision campaign, the same-task survey-reference campaign and the new synthetic-persona reference campaign. The deduplicated supervision archives retain completion creation times from 21 August to 27 August 2026 (UTC), while the final same-task reference completions were created on 21 September 2026 from 16:35:54 to 17:01:40 UTC. These are provider creation times, not selection seeds. The final survey-reference campaign has 1,440 usage-bearing responses, all returning \texttt{deepseek-v4-pro}, and each recorded request specifies 16,384 maximum tokens. Its copied campaign configuration reports 8,192; request-level fields take precedence for this batch. Early repeat records do not retain a response model field, and immutable backend fingerprints are incomplete. Frozen targets remain the basis for exact target-based re-evaluation; service identity alone does not turn a later comparison into a causal estimate of compression error.

\paragraph{Controlled-model evaluation.}
The prepared benchmark is in \path{outputs/matched_response_v1/bundle/}, selected neural fits in \path{outputs/matched_response_v1/train/}, and modular fits in \path{outputs/revision_20260921/baselines/}. Local inference uses a synthetic example and a frozen checkpoint. The archived-checkpoint helper extracts the recorded source and checks its fingerprint before evaluation. Historical fingerprints depend on platform-specific path separators; portability changes should be versioned rather than treated as exact replay. The repository's retained verification record reports local inference and re-evaluation of two selected neural checkpoints, not a new full training campaign.

\paragraph{Survey analyses.}
The repository includes the English questionnaire instruments, complete aggregate contrast tables and analysis implementations. Reproducing respondent-level resampling or participant-linked Teacher comparisons requires the separately retained participant records and appropriate privacy-aware access. Aggregate-table inspection is distinct from re-estimation of participant-level intervals. These restrictions do not apply to the synthetic benchmark targets and released model weights.

\paragraph{Network execution.}
The repository includes the 140-run ledger, 70 paired summaries, experiment driver and event-analysis code. Fresh Helsinki execution requires the fixed population, prepared road/transit supply and Java/MATSim runtime; the full event archive, approximately 57.5 GB, is retained separately. The implementation is specified by \path{scripts/revision_20260921/helsinki_execution.py} and \path{src/traveler_distillation/matsim/adapter.py}. Dependency records are provided, although the historical stages do not share one fully locked environment.

The retained ledgers contain 1,000 people in each completed run and no zero-feasible-mass case. Across conditions, 2,006 outbound PT assignments and 940 car assignments fell back to walking; these are repeated traveler records, not distinct people. No recorded outbound or return walking route consisted only of its origin link. The general adapter can produce an origin-link-only walk after an unsuccessful walking search, so this tested behavior does not establish handling of disconnected networks. These ledger checks assess the completed runs; the repository expansion itself did not rerun all simulations.

Requests for access should be directed to the corresponding author, subject to participant privacy and third-party terms. The included manifest records artifact hashes and the material scope, while the verification record distinguishes executed checks from retained historical outputs. This inventory supersedes earlier availability text that predates the repository expansion.
```
