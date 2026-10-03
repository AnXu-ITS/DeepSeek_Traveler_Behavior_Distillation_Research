# R4 — Detailed computation and acquisition accounting

These are preserved source excerpts from the supplied pre-trim manuscript package. They are extended records, not new experiments. The main article and compact supplement contain the primary evidence.

LaTeX labels and cross-references below retain their source names. They identify the original objects, not the renumbered compact supplement. Source paths that are mentioned in prose are historical descriptions, not a claim that those artifacts were uploaded in this migration.

## New-reference recorded usage and valuation

Source: `sections/D_new_evidence.tex`, lines 21–31.

```latex
\subsection{Recorded usage and retrospective valuation}
\label{sec:cost_summary}
The new pilot and formal campaign made 1,618 attempts, yielding 1,512 valid responses. Usage is available for 1,525 distinct completions, including thirteen invalid responses; 93 attempts have unknown usage. Recorded input and output total 2,206,391 and 4,355,420 tokens, respectively. Cached input is already included in input, and reasoning tokens are already included in output.

Using the official DeepSeek V4 Pro rates checked on 25 September 2026 values the retained usage records at the ranges summarized in Table~\ref{tab:api_cost}. These are retrospective estimates from recorded token counts, not recovered invoices. Neither estimate covers unknown usage or verifies an account debit.

\input{generated/narrative/table_api_cost.tex}
The historical Pro table above includes retained development runs and earlier model versions, not just acquisition for the final Student. It is a current-rate revaluation of observed usage rather than a recovered invoice. It should be interpreted with the same official DeepSeek V4 Pro pricing basis used in Table~\ref{tab:api_cost}. The 27 controlled and six delay-withheld fits sum to 845.69 and 200.17 seconds of recorded fit time; these are neither end-to-end campaign wall times nor CPU-seconds. Hardware rental and electricity charges were not recorded.

For repeated matched workloads, a monetary break-even count would be $N^*=C_0/(c_L-c_S)$ when $c_L>c_S$, where $C_0$ includes acquisition, retries, training and necessary adaptation, and $c_L,c_S$ are comparable per-workload costs. Shared supply and routing work must be counted consistently. Existing measurements supply only part of this accounting. Supplementary Section~\ref{sec:supp_compute} retains the inference, construction and historical metering details without asserting a measured lifecycle break-even point.
```

### Source table: `generated/narrative/table_api_cost.tex`

```latex
\begin{table}[pos=htbp]
\centering\small
\caption{Retained API usage repriced at the official DeepSeek V4 Pro rates checked on 25 September 2026. Costs are estimates in USD, not invoices.}
\label{tab:api_cost}
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\linewidth}{@{}Xrrrr@{}}
\toprule
Acquisition group & Metered calls & Input (M) & Output (M) & USD range \\
\midrule
Primary supervision archives & 8,681 & 10.050 & 32.614 & 65.83--131.66 \\
Final same-task reference & 1,440 & 2.324 & 3.255 & 7.08--14.16 \\
Earlier same-task campaign & 1,475 & 1.969 & 5.575 & 11.58--23.15 \\
Neutral-metadata pilot & 120 & 0.160 & 0.239 & 0.52--1.03 \\
Ownership pilot & 120 & 0.160 & 0.235 & 0.52--1.03 \\
Latency benchmark & 87 & 0.117 & 0.353 & 0.76--1.51 \\
Other historical development & 1,468 & 2.377 & 8.149 & 16.43--32.86 \\
Retained metered total & 13,391 & 17.157 & 50.418 & 102.71--205.42 \\
\bottomrule
\end{tabularx}
\par\smallskip\begin{minipage}{\linewidth}\footnotesize
Completion IDs are deduplicated across archives. Input costs use the recorded cache-hit/miss split; output includes reasoning tokens. The range reprices the same usage entirely at off-peak or peak rates, rather than reconstructing historical deductions. The primary archives contain the legacy, joint and corrected accessibility acquisition pools, including labels outside the final benchmark. Other development includes deprecated S8. A further 576 retained completions lack usage and are excluded from the monetary sum, as are unmetered failures; their cost is unknown. The earlier same-task campaign includes 35 responses rejected by parsing but with recorded token usage. This is neither the marginal cost of one fitted Student nor a full project invoice.
\end{minipage}
\end{table}
```

## Inference, construction and historical token accounting

Source: `sections/S6_compute.tex`, lines 1–26.

```latex
\section{Computational measurements}
\label{sec:supp_compute}
Inference time for choice and timing together is measured on the 437 test states after feature encoding, using two CPU threads. After five untimed calls, 30 batches are timed, and the median batch time is divided by 437 for each seed (Table~\ref{tab:departure_runs}). MNL-S with neural timing requires $1.68\pm0.33$ microseconds per state, compared with $0.19\pm0.002$ with bounded-linear timing and 1.78--2.16 microseconds for the joint neural Students. Feature encoding and route preparation are not included.

The Teacher and SA-Student are compared on the same first 100 states, queried one at a time. The Teacher answers 87 of the 100 requests successfully, taking 86.539 seconds per successful request on average (95\% bootstrap interval 76.745--96.550), whereas SA-Student takes 0.272 milliseconds per state on a local processor. Neither figure includes the construction of a full population. Table~\ref{tab:inference_timing} distinguishes measured values from values scaled to larger populations.
\input{generated/narrative/table_inference.tex}

The nested Singapore populations contain 1,000, 10,000, 20,000 and 50,000 people, simulated with flow and storage capacity factors of 0.3 throughout (Table~\ref{tab:scale_full}). Construction time covers feature preparation, behavioral prediction and plan generation. The value for 1,000 people is the median of three constructions with identical behavioral predictions. Because larger populations also load the network more heavily at unchanged capacity factors, their simulation times do not correspond to calibrated city populations.
\input{generated/narrative/table_scale.tex}

Construction with and without reuse is compared for the same population and scenario. With reuse, route and accessibility calculations that depend only on the network and timetable are taken from an earlier construction, while Student predictions are recomputed. Construction time falls from 3,223.6 to 469.5 seconds in Singapore and from 10,491.8 to 103.2 seconds in Helsinki (Table~\ref{tab:pipeline}). All 10,000 mode and origin--destination assignments are unchanged, and only two Singapore departures and one Helsinki departure differ, by 0.01 minutes after rounding.
\input{generated/narrative/table_pipeline.tex}
All constructions ran on a machine with 24 logical CPU cores and 31.4 GB of memory, using PyTorch without GPU acceleration. Each city was measured once with reuse. The 50,000-person construction partly overlapped a preliminary Helsinki run, and one Helsinki delay construction shared the machine with other work, so these timings describe this machine under its recorded workload. The full cost of distillation would also include acquisition and retry charges that were not recorded in full. The retrospective accounting below uses available usage records and does not fill missing charges with zeros. The largest simulated population has 50,000 people, and the separate measurement on 100,000 states concerns inference alone.
\FloatBarrier

\subsection{Retrospective token accounting}
The accounting script reads retained repeat and API-call records, deduplicates them by completion ID and exports only aggregate counts and file hashes. At peak rates, a completion with $T_h$ cached input tokens, $T_m$ uncached input tokens and $T_o$ output tokens is repriced as
\begin{equation}
 C_{\mathrm{API}}=(0.044T_h+1.32T_m+3.96T_o)/10^6\quad\mathrm{USD}.
\end{equation}
Off-peak rates are half these values. These are the provider's rates checked on 25 September 2026, not a currency conversion or recovered invoice. The separately published CNY rates give CNY 700.30--1,400.61 for the same retained metered total. Promotions, account credits, tax treatment and unmetered requests are not reconstructed. The reproducible aggregate ledger is included with the manuscript under \path{reproduction/cost_audit/}; it contains no participant-level payloads.

The final same-task campaign made 2,880 recorded attempts: 1,440 valid responses and 1,440 HTTP 402 failures without usage. Its metered responses contain 1,409,152 cached and 915,218 uncached input tokens and 3,254,670 output tokens. Unknown charges for failures are not asserted to be zero. The earlier generic-prompt campaign had 1,475 usage-bearing responses, of which 35 were rejected by parsing; their tokens remain in the cost sum. The final campaign consumed an average of 1,614 input and 2,260 output tokens per valid response, which can inform a workload-matched budget but is not a guaranteed token ceiling for other prompts.

The 12 retained controlled-neural training status files report 29.58--32.13 seconds per fit on the recorded CPU setup, or 371.09 seconds summed over the fits. This sum is historical fit time, not elapsed campaign time or the complete training and adaptation history. No hardware rental or electricity charge was recorded. For reuse, the original acquisition cost is sunk; retraining from frozen targets adds no Teacher calls, while collecting new targets must be budgeted separately.
```

### Source table: `generated/narrative/table_inference.tex`

```latex
\begin{table}[pos=htbp]
\centering\small
\caption{Time required for behavioral prediction alone. Teacher latency refers to successful requests, and Student times exclude supply preparation. Values for 10,000 agents are scaled or projected, not measured constructions.}
\label{tab:inference_timing}
\begin{tabularx}{\textwidth}{@{}lXrr@{}}
\toprule
Model & Measurement & Mean latency & 10,000 equivalent \\
\midrule
Teacher (remote) & 100 attempts, 87 successes & 86.539 s & 240.4 h (projected) \\
SA-Student sequential & Same first 100 states, one CPU thread & 0.272 ms & -- \\
SA-Student sequential & 100,000 states, one CPU thread & 0.332 ms & 3.3 s (scaled) \\
SA-Student batch 256 & 20,000 states, 24 CPU threads & 25,399 states/s & 0.4 s (scaled) \\
\bottomrule
\end{tabularx}
\end{table}
```

### Source table: `generated/narrative/table_scale.tex`

```latex
\begin{table}[pos=htbp]
\centering\small
\setlength{\tabcolsep}{3pt}
\caption{Population-scale construction and transport outcomes under unchanged Singapore supply, C0, seed 2026. Leg time averages completed legs, and unfinished legs count departures without a matching arrival. Smaller populations are subsets of larger ones, and the 1,000-agent row is the median of three constructions.}
\label{tab:scale_full}
\begin{tabular}{@{}rrrrrrr@{}}
\toprule
Agents & Build (min) & Decisions (s) & MATSim (min) & Boardings & \shortstack{Unfinished\\legs} & \shortstack{Completed-leg\\mean (min)} \\
\midrule
1,000 & 5.9 & 1.0 & 2.4 & 560 & 98 & 43.82 \\
10,000 & 51.1 & 10.3 & 2.1 & 5,613 & 938 & 43.67 \\
20,000 & 98.9 & 18.8 & 2.4 & 11,357 & 1,956 & 43.90 \\
50,000 & 231.8 & 48.8 & 3.4 & 28,440 & 5,586 & 48.92 \\
\bottomrule
\end{tabular}
\end{table}
```

### Source table: `generated/narrative/table_pipeline.tex`

```latex
\begin{table}[pos=htbp]
\centering\small
\caption{Scenario-construction time for 10,000 agents without (cold) and with (warm) reuse of unchanged network and timetable calculations. Student predictions are recomputed in both cases.}
\label{tab:pipeline}
\begin{tabular}{@{}lrrrr@{}}
\toprule
City & Original (s) & Cold (s) & Warm (s) & Cold/warm \\
\midrule
Singapore & 3,063.1 & 3,223.6 & 469.5 & 6.9 \\
Helsinki & 9,507.0 & 10,491.8 & 103.2 & 101.7 \\
\bottomrule
\end{tabular}
\end{table}
```
