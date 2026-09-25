"""Apply the author-authorized structural revision to the backed-up source."""
from pathlib import Path
import json,re,hashlib,zipfile
R=Path(__file__).resolve().parents[1]
S=R/'sections'
with zipfile.ZipFile(R.parent/'revision_response_20260925/before_narrative_revision_20260925.zip') as z:
 before={n:z.read(n).decode('utf-8-sig').replace('\r\n','\n') for n in z.namelist() if n.endswith('.tex')}
def put(name,text):(S/name).write_text(text.strip()+'\n',encoding='utf-8')

put('01_introduction.tex',r'''
\section{Introduction}
\label{sec:intro}
Suppose a transport planner wants to know how passengers would respond to a less reliable bus service. A model might reproduce today's public-transport share and still predict the wrong change tomorrow. It might also predict a reasonable change that disappears when travelers are assigned to routes and vehicles. Scenario analysis therefore needs more than an accurate baseline: it needs a defensible account of how choices change and how those changes reach the network \citep{horni2016multi}.

Large language models (LLMs) offer a flexible way to describe such choices. They can combine a traveler's circumstances with trip attributes and contextual information in a single prompt. Recent work uses this capacity for travel prediction, route choice and agent-based modeling \citep{mo2026llmtravel,wang2025agentic,liu2025framework}. Repeatedly querying an LLM for a simulated population is costly, however. Behavioral distillation addresses this problem by fitting a compact local model, the \emph{Student}, to judgments elicited from an LLM, the \emph{Teacher}. The Student can then generate travel choices without a live LLM call for every decision.

What should survive this compression? Conventional evaluation asks whether the Student matches the Teacher at individual input states. Transport scenario analysis also asks whether it preserves the change between two states: the same person making the same trip under different conditions. We call the first property \emph{static fidelity} and the second \emph{response fidelity}. Exact agreement at every state would imply exact agreement in every response. With limited data and model capacity, an average static score does not directly assess the particular changes a planner needs.

\begin{figure}[pos=htbp]
\centering\includegraphics[width=0.95\textwidth,keepaspectratio]{figures/narrative/figure1_response_concept.pdf}
\caption{Traveler--trip pairing and illustrative demand responses. Each pair changes travel conditions while holding the traveler and trip fixed. Profiles are illustrative; bars are rounded, pre-routing mode assignments for 10,000 synthetic Singapore travelers, not survey observations. They use the supply-adapted model defined in Section~\ref{sec:architectures}. Poorer accessibility is evaluated separately. Scenario definitions and exact shares are in Supplementary Table~\ref{supp-tab:singapore}.}
\label{fig:concept}
\end{figure}

Preserving a Teacher's response is only the first test. An LLM can differ from people, and a faithful Student can retain that difference. Research on travel choice and synthetic respondents has already shown why predictive success and plausible language require separate behavioral checks \citep{wang2020deep,martinbaos2023prediction,bisbee2024synthetic,liu2026persona}. A further test arises in simulation: predicted probabilities must become assigned modes, feasible routes and actual boardings. Each step can alter the response. Figure~\ref{fig:concept} illustrates the paired traveler--trip comparison and its connection to aggregate demand.

We examine this chain using matched distillation experiments, stated-choice surveys in Singapore and Shanghai, and controlled network execution in Helsinki. Response supervision is tested against the same archived endpoint targets used by ordinary distillation. A new synthetic-persona experiment examines agreement with a later API model, including a setting where delay-related training examples are withheld. The surveys test the resulting models against people's stated responses. Finally, a four-arm simulation distinguishes a perceived delay from a physical reduction in transit service. These settings provide complementary evidence; they are not a cross-city prediction benchmark.

The findings reveal where a favorable comparison stops carrying over. Response supervision helps within covered tasks, but its advantage reverses on withheld delay scenarios. The Student closest to the archived Teacher can predict the opposite direction to a stated human response. In the network, perceived delay and fewer scheduled trips produce markedly different demand changes. Our contribution is an empirical evaluation of these boundaries and a practical way to locate them. The learning objectives and modular models serve as controlled comparisons within that evaluation, rather than as a claim that a new loss solves behavioral validity.
''')
put('02_related_work.tex',r'''
\section{Related work}
\label{sec:related}
\subsection{From plausible traveler agents to empirical agreement}
LLM transport research has progressed from conceptual frameworks to models that predict modes, choose routes and generate travel activities \citep{liu2025framework,nie2025roles,mo2026llmtravel,wang2025agentic,gatsim2026}. These applications exploit the ability to express personal and situational information in language. Their usefulness depends on what the generated decisions reproduce. A coherent account of why a traveler might take a bus is a different object from an empirically supported probability that the traveler will do so.

Alignment studies address this distinction directly. \citet{liu2026persona} infer personas from empirical choices and learn which persona to assign to a traveler. \citet{xu2026satisfaction} use task-specific examples to improve agreement with reported travel satisfaction. Work on synthetic survey respondents also documents sensitivity to prompting and model versions \citep{argyle2023out,aher2023using,bisbee2024synthetic}. These studies motivate a human reference independent of the model that supplies training labels. Our question concerns the additional compression step: whether a compact model preserves elicited responses, and whether doing so improves agreement with people. We reserve the surveys for evaluation; we do not use them to fit an alignment model.

\subsection{Learning a response rather than an isolated prediction}
In choice modeling, predictive accuracy and behavioral interpretation need not favor the same specification \citep{wang2020deep,martinbaos2023prediction}. Random-utility models supply a structured mapping from alternative attributes to probabilities, while representation learning can add flexibility around a specified utility core \citep{mcfadden1974conditional,train2009discrete,sifringer2020enhancing}. When the target is an LLM distribution, a logit Student may benefit from this restricted form without identifying real travelers' preferences. We therefore compare neural and logit specifications under common inputs and examine their behavioral discrepancies separately.

The idea of supervising relations is established in machine learning. Knowledge distillation commonly transfers individual predictions \citep{hinton2015distilling,gou2021knowledge}; relational distillation transfers relationships between examples \citep{park2019relational}, and Sobolev training incorporates derivatives of the target function \citep{czarnecki2017sobolev}. Our response penalty operates on finite changes between specified travel conditions. It uses endpoint targets already available to every controlled Student and requires no derivative access to the API. The issue tested here is whether prioritizing those changes helps under the coverage available, including when an intervention family is absent from training.

\subsection{Behavioral surrogates within transport simulation}
A surrogate can replace different parts of a simulator. \citet{natterer2025surrogates} predict network-level traffic changes from an agent-based model, enabling faster policy exploration. We replace the traveler decision function while retaining route construction and simulation. This makes the conversion from probabilities to executed travel an observable part of the evaluation.

The concurrent CALM preprint integrates calibrated choice, optional LLM planning, network feedback and replayable evaluation \citep{cheng2026calm}. Its system-level tests are relevant to the same broader problem of validating generative transport models. Our study concentrates on finite paired responses across three references: archived LLM judgments, stated human choices and executed demand. In particular, we separate changing the behavioral description of service from changing its physical timetable. The distinction is necessary before an apparent response to disruption can be interpreted as a consequence of network supply.
''')
put('03_methodology.tex',r'''
\section{Evaluating and learning traveler responses}
\label{sec:method}
\subsection{The response as the object of comparison}
\label{sec:states}
A traveler state $\mathbf x$ describes a person, a trip, contextual conditions and attributes of four alternatives: private car, public transport (PT), bicycle and walking. Attributes include time, cost, weather exposure and the components of a PT journey. A model returns a normalized choice distribution $\mathbf p(\mathbf x)$ and a departure-time adjustment. The distribution is a model output, not an established human choice frequency. Appendix~\ref{sec:supp_inputs} explains the inputs and the survey encodings.

For two states that hold the person and trip fixed, the response is
\begin{equation}
\Delta\mathbf p_j=\mathbf p_j(\mathbf x')-\mathbf p_j(\mathbf x),\qquad j\in\{T,S\}.
\label{eq:response}
\end{equation}
Here $T$ and $S$ denote Teacher and Student. We evaluate the absolute difference between their responses, averaged over modes available in either state and then over paired states:
\begin{equation}
E_{\Delta p}=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{|\mathcal A_i|}
\sum_{m\in\mathcal A_i}|\Delta p_{S,im}-\Delta p_{T,im}|.
\label{eq:response_metric_main}
\end{equation}
Lower error means closer agreement in the specified change. This finite contrast does not by itself estimate a population causal effect or an elasticity. Static KL divergence measures endpoint agreement, departure MAE measures timing agreement, and an interaction contrast asks whether two perturbations combine in the same way. Supplementary Section~\ref{supp-sec:metric_definitions} gives the additional definitions.

\subsection{Controlled and adapted Students}
\label{sec:architectures}
The joint neural Student encodes the traveler and the attributes of each alternative, scores the alternatives and normalizes over the available modes. It shares these representations with a departure predictor. The multinomial logit Student (MNL-S) instead uses linear choice utilities on the same structured information. Separate ridge, bounded-linear and neural timing predictors allow choice and departure adjustment to favor different specifications. The neural outputs are bounded to an hour in either direction. Appendix~\ref{sec:model_details} explains the model roles; detailed dimensions and fitting settings are in Supplementary Section~\ref{supp-sec:supp_controlled}.

We also evaluate a frozen supply-adapted Student (SA-Student). This model was adapted with additional PT attributes and contrasts between contexts, travelers and accessibility levels. Its training history and availability treatment differ from those of the controlled Students. It therefore provides a deployment case, not an isolated test of the response penalty (Appendix~\ref{sec:s9_recipe}).

\subsection{Response-aware fitting}
\label{sec:loss}
The soft-KL baseline fits the Teacher distribution and departure adjustment at both endpoints of every training pair. Signed-response fitting adds a weighted penalty on the difference between the Teacher and Student changes:
\begin{equation}
\mathcal L=\mathcal L_{\mathrm{endpoint}}+\lambda\mathcal L_{\mathrm{response}}.
\label{eq:training_objective}
\end{equation}
The response penalty is a batch average over available mode--pair entries. We compare direct signed absolute error with a penalty that separates direction and magnitude. An additional CE+KL comparator fits the Teacher's most likely mode as a hard label. All controlled neural objectives receive the same endpoint information. Thus paired supervision changes what the optimizer prioritizes; it does not supply new Teacher knowledge. Appendix~\ref{sec:loss_diagnostics} gives the exact losses and their limitations, including a flat region in the direction-plus-magnitude penalty.

\subsection{Human responses and network execution}
\label{sec:execution_method}
For a survey contrast, the human response is the change in the share selecting the relevant mode or mode group. The corresponding model response is the change in its predicted probability, averaged over the same respondents. We report their difference as \emph{response bias}: positive values mean a more positive model response, and negative values a more negative one. Full task sets are retained together when resampling respondents.

An API reference queried on the same survey tasks provides a further descriptive comparison. On the same people and contrasts,
\begin{equation}
S-H=(T_{\mathrm{now}}-H)+(S-T_{\mathrm{now}}),
\label{eq:attribution}
\end{equation}
where each term is a mean response. This identity separates two observed discrepancies. Because $T_{\mathrm{now}}$ was queried later, it does not identify the historical error introduced by compression.

In simulation, we trace PT use from predicted probabilities through feasibility adjustment, mode assignment, routing and boarding. A feasible route is one returned by the specified direct-or-one-transfer search, rather than proof that all other connections are impossible. Deterministic assignment takes the most probable mode; stochastic assignment samples the probabilities; feasibility-constrained sampling first removes unavailable or unroutable alternatives and renormalizes. Without that correction, a failed car, bicycle or PT route triggers a walking fallback. Predictions remain fixed during each simulation run.

Every stage uses the initial population as denominator. If $R$ denotes a change from baseline, the predicted-to-simulated gap is
\begin{equation}
G=R^{\mathrm{sim}}-R^{\mathrm{pred}}
=(R^{\mathrm{adj}}-R^{\mathrm{pred}})+(R^{\mathrm{sim}}-R^{\mathrm{adj}}).
\label{eq:execution_decomposition}
\end{equation}
This separates a change to the sampling distribution from subsequent execution. Boarding and journey completion remain distinct outcomes. The departure adjustment is applied before route feasibility is checked, so timing can change the itinerary even if it leaves the binary feasibility indicator unchanged. Appendix~\ref{sec:execution_support} explains the execution controls and the geometry of the feasibility correction.
''')

# Further sections are applied below; the exact old content is retained in the backup.
put('04_experimental_design.tex',r'''
\section{Evidence and experimental design}
\label{sec:design}
Table~\ref{tab:study_design} distinguishes the data used to learn a model from the evidence used to evaluate it. Neither human survey is used for training or selection. Training seeds, repeated API responses and repeated assignments describe different sources of variability and are never counted as additional people.
\input{generated/revision20260925/table_design.tex}

\subsection{Archived targets and matched training}
\label{sec:controlled_design}
The archived benchmark contains disjoint training, validation and test groups of synthetic traveler profiles, here called personas. The test set has 398 response pairs from six personas. Within each training seed, controlled neural models share initialization, architecture, endpoint targets, pair order and optimization budget. The primary choice-model selector minimizes validation macro-source KL, giving each supervision source a role in selection. A response-based selector and a grid of response-loss weights test sensitivity. MNL-S regularization is selected on the same validation criteria. Departure predictors are selected separately by departure MAE, so timing comparisons do not silently change the choice-selection rule. Appendix~\ref{sec:optimization_support} and Supplementary Section~\ref{supp-sec:supp_controlled} document these comparisons.

The original split also withholds a fare--congestion combination while retaining its constituent factors. We audited endpoints, references and linked contrast units to confirm that the held-out combination does not enter training or validation. This is a compositional test. To test a stronger omission, we additionally remove positive-delay endpoints and entire delay-mechanism groups from training and validation, refit feature statistics, and retrain soft-KL and signed-response models. Budgets are matched between methods within each setting. The full and delay-withheld settings have different retained data and update counts, so their difference does not isolate removal of the intervention family as a pure causal factor.

Archived Teacher targets were collected through the official DeepSeek V4 Pro service, according to the authors' collection records and confirmation. The later same-task survey campaign also requested and returned \texttt{deepseek-v4-pro}. Repeated valid vectors are averaged within state. We preserve these targets as the reference for compression; service names alone do not certify an immutable backend across collection periods.

\subsection{A new synthetic-persona reference test}
\label{sec:new_reference_design}
A separate experiment evaluates the frozen Students on new persona profiles, one fixed numerical trip and a common set of travel conditions. Twelve pilot personas are used only for sample-size planning; thirty different personas form the formal sample. Each has a baseline and eleven changes covering delay, fare, access, combinations, rain and PT infeasibility. Three valid API responses are averaged per state. Profiles are excluded from all prior persona sets. The fixed trip and inherited feasible supply profile limit interpretation to these inputs, rather than geographic or population generalization.

This new reference was supplied by OpenCode Go with requested and returned model \texttt{deepseek-v4.1-flash}. It therefore measures agreement with a different API model, not compression error against the original Pro Teacher. The primary contrast, defined before formal collection, is signed-response minus soft-KL error under full training and KL selection. We average the paired differences over contrasts and training seeds within persona, then bootstrap personas. Delay-family and individual-condition comparisons are descriptive, with pointwise intervals. The pilot is excluded from formal estimation. Appendix~\ref{sec:new_protocol} gives the precision rule, acquisition settings and a post-hoc service-metadata diagnostic. No samples were excluded because of that diagnostic.

\subsection{Independent stated-choice evaluation}
\label{sec:human_design}
The Singapore analysis includes 332 residents and the Shanghai model comparison includes 321 respondents. Each completed ten stated-choice tasks. Recruitment through personal contacts and onward sharing produced convenience and snowball samples; the target is agreement within these samples, not citywide demand. Singapore data were collected from 20 August to 5 September 2026 and Shanghai data from 12 to 18 September 2026. Appendix~\ref{sec:supp_inputs} gives eligibility, sample flow and the tasks.

Singapore respondents saw numerical time-and-cost tables. Most Shanghai conditions were qualitative, so unreported durations require numerical reference values for model evaluation. Those assumptions were defined without reference to the human answers. We examine alternative input specifications and a separate network-derived-input diagnostic. All explicit choices are scored, including choices unavailable under the model's ownership rule. The two Shanghai answers indicating no suitable mode are excluded from four-mode scoring. Human-only sensitivity retains eligible respondents outside the model's input mapping.

Within-person contrasts compare stated mode changes with probability changes. Respondent bootstraps retain each person's complete task set and use shared resamples across models. We report nominal 95\% intervals and family-adjusted intervals over the contrasts in each city. Seed variability is separate. An additional same-input API comparison uses 24 respondents per city, with three valid queries per task and no access to their answers. This later Pro reference supports Eq.~\ref{eq:attribution}; it is distinct from both archived supervision and the new Go synthetic sample.

\paragraph{Ethics and consent statement.} The stated-choice questionnaires were administered as anonymous, minimal-risk surveys. No direct personal identifiers were collected. Before beginning the questionnaire, all participants were shown an electronic information and consent statement describing the study purpose, voluntary participation, intended research use of the responses, and confidentiality protections; only respondents who provided consent proceeded to the survey. Formal institutional ethics approval was not obtained. The study procedures were designed to protect participant privacy and to limit data collection to information necessary for the stated research purpose.

\subsection{Perceived conditions and physical transit supply}
\label{sec:execution_design}
The Helsinki experiments hold a synthetic population of 1,000 travelers fixed and use OSM roads and HSL timetables. The original experiment compares assignment rules under an added perceived PT delay while leaving physical supply unchanged. A four-arm extension distinguishes this input manipulation from an actual service reduction: baseline; perceived delay with the original timetable; reduced service with baseline behavioral predictions frozen; and reduced service with predictions recomputed from its level of service. The reduction removes alternating departures within line-and-stop-sequence groups, updating both the route-search index and the MATSim timetable. Delay is not added again to the reduced-service arms.

The extension uses feasibility-constrained assignment, SA-Student and MNL-S with a fixed independent neural timing predictor, and five paired assignment seeds. Its 40 configurations include verified reuse of 12 historical runs and 28 new runs. Responses are paired within person and seed; assignment seeds are averaged within person before person-level bootstrap resampling. This is a single-iteration execution experiment, with no feedback from experienced conditions to a subsequent behavioral decision. Network-derived active-mode speeds are retained, so journey times are diagnostic outputs, not calibrated predictions of the benefits of a service change. Appendix~\ref{sec:execution_support} and Supplementary Section~\ref{supp-sec:supp_execution} provide the stage definitions and additional controls.
''')
put('05_results.tex',r'''
\section{Results}
\label{sec:results}
\subsection{Response learning helps within coverage, but does not remove its limits}
Response supervision improves agreement with the archived Teacher under matched training (Figure~\ref{fig:response_learning}a). The gain is modest relative to the effect of model specification: MNL-S has the lowest response error, while adding a hard-label objective produces the largest error. Separating direction and magnitude does not show a clear additional benefit over direct signed-response fitting. Neither response objective improves the interaction metric over soft KL. The detailed comparison is in Appendix~\ref{sec:core_tables}.

\begin{figure}[pos=htbp]
\centering\includegraphics[width=\linewidth]{figures/revision20260925/figure2_learning.pdf}
\caption{Learning gains depend on task coverage. (a) Archived-Teacher response error: means over three neural seeds, with seed SD; MNL-S is a single fit. (b) Signed-response minus soft-KL error on thirty new personas, relative to the Go Flash reference. Bars average seeds and within-person contrasts; whiskers are paired-persona 95\% bootstrap intervals. Negative values favor response supervision. Full-training and delay-withheld fits have matched budgets within, but different budgets between, settings. The overall full-training comparison is primary; delay-specific intervals are descriptive and unadjusted for multiplicity. The two panels use different references and are not directly comparable error scales.}
\label{fig:response_learning}
\end{figure}

The new synthetic-persona sample supports a small improvement under full training. Its primary paired error difference is $-0.0032$, with a 95\% interval of $[-0.0049,-0.0016]$ (Table~\ref{tab:primary_results}). Because the API reference differs from the training Teacher, this is evidence of cross-model response agreement at the fixed trip. It is not an estimate of human accuracy. Variation across personas and the remaining conditions is shown in Appendix~\ref{sec:new_protocol}.

The delay-family omission changes the conclusion. With delay examples withheld, signed-response fitting has larger errors than soft KL on both new delay intensities and on the combined fare--delay condition (Figure~\ref{fig:response_learning}b). The retrospective archived-delay comparison points in the same direction. The overall mean across all new conditions partly conceals this failure. Thus an objective that improves covered responses does not establish that the Student has learned a transferable behavioral rule for an omitted intervention.
\input{generated/revision20260925/table_primary.tex}

Weight and selection sensitivity reinforce this interpretation. Increasing response weight can lower the archived response error, but selecting by validation response does not consistently improve test performance. The direction-plus-magnitude objective is particularly sensitive to that selection change at unit weight. These are sensitivity comparisons, not grounds for choosing a favorable test result after evaluation (Appendix~\ref{sec:optimization_support}). Repeated-Teacher diagnostics also show material variation in the elicited target; they do not provide a precise noise ceiling.

The choice and timing tasks favor different specifications. MNL-S preserves probability responses well, whereas an independent neural timing predictor substantially reduces departure error relative to linear timing and approaches the jointly trained models. This motivates modular estimation. It does not confer an economic interpretation on unconstrained logit coefficients: in the feasible-state diagnostic, some fare increases raise the fitted PT probability (Appendix~\ref{sec:optimization_support}).

\subsection{The most faithful Student need not be the most human-like}
\label{sec:human_results}
The strongest ranking reversal concerns poorer PT accessibility in Singapore. MNL-S, the best model against the archived Teacher, predicts an increase of about 14 percentage points in PT use, while the stated share falls by about 24 points. CE+KL, the least faithful controlled objective, comes closest to that stated change. The task combines longer total travel time, a longer access walk and more transfers, so the reversal concerns the whole access burden rather than a single walking-time coefficient.

\begin{figure}[pos=p]
\centering\includegraphics[width=\linewidth]{figures/revision20260925/figure3_human.pdf}
\caption{Agreement with stated responses. (a,b) Two contrasts that expose different model failures: bars show model response minus human response; whiskers are family-adjusted respondent-bootstrap intervals. (c,d) All retained contrasts on a common bias scale. A dot marks an interval excluding zero after adjustment within city and model; an unmarked cell does not establish equivalence. Neural responses average three seeds. Singapore has 332 respondents; Shanghai has 321, except the interaction and walking-versus-waiting contrasts, which have 320 complete respondents. Positive bias means a more positive model response, not better performance. Exact responses and intervals are retained in Supplementary Tables~\ref{supp-tab:responses_singapore_0}--\ref{supp-tab:responses_shanghai_1}.}
\label{fig:human_response}
\end{figure}

SA-Student exhibits a different limitation. Its delay response is close to the Singapore point estimate, but it overstates the Shanghai decline under the reference encoding (Figure~\ref{fig:human_response}). It also reverses the Shanghai fare and walking-versus-waiting responses. These discrepancies persist across the specified alternative inputs. Because the Shanghai delay description is qualitative and its intensity is held fixed in that sensitivity analysis, the result remains conditional on the numerical interpretation of the task.

Matching levels does not resolve the problem. MNL-S is closer to Singapore's overall PT share, yet SA-Student has the smaller mean absolute response bias there. A model can match how many people choose PT while missing how that share changes. Choice accuracy offers another ranking rather than a general resolution (Appendix~\ref{sec:core_tables}).

Replacing reference values with constructed network inputs improves Shanghai choice accuracy, but most of the gain remains after whole supply profiles are exchanged between respondents. The improvement therefore appears to come largely from a general change in numerical inputs, not from matching observed personal journeys. Those journeys were not collected. This sensitivity check limits what can be claimed about supply-based human alignment.

\subsection{A later API reference helps locate discrepancies}
\label{sec:teacher_results}
The same-task API subsets reveal different patterns beneath a Student--human discrepancy (Figure~\ref{fig:teacher_attribution}). For Singapore accessibility, the later Pro reference also understates the human response, and SA-Student lies close to that reference. Moving the Student toward the reference alone would leave substantial human disagreement. For Shanghai delay, the API reference is close to the human point estimate, while SA-Student responds much more strongly. The latter case motivates checking the Student and its input representation, but does not identify compression as the historical cause.

\begin{figure}[pos=htbp]
\centering\includegraphics[width=\linewidth]{figures/revision20260925/figure4_decomposition.pdf}
\caption{Three patterns in the matched survey subsets, each with 24 respondents. The later official Pro API reference and SA-Student are evaluated on the same inputs. Bars and nominal 95\% respondent-bootstrap intervals show the two components of Eq.~\ref{eq:attribution} and their total. Component point estimates add to the total; interval endpoints need not. The walking-versus-waiting case illustrates cancellation. These descriptive contrasts do not reconstruct the original Teacher backend. The complete family-adjusted decomposition is in Appendix~\ref{sec:core_tables}.}
\label{fig:teacher_attribution}
\end{figure}

In the walking-versus-waiting task, the two components oppose one another. The Student's modest total bias hides a larger API--human discrepancy offset by a Student--API discrepancy. This explains why a small net error should not be interpreted as evidence that both components are sound.

\subsection{Changing perceived service is different from changing the timetable}
\label{sec:execution_results}
The four-arm Helsinki experiment makes this distinction visible (Figure~\ref{fig:execution_response}). Adding perceived delay produces a substantial fall in predicted PT use and boarding. Removing alternate scheduled departures produces a much smaller demand change when the behavioral inputs are recomputed. With baseline predictions frozen, the reduced timetable leaves the boarding share unchanged in this population, although simulated journey times change. This result does not mean supply is irrelevant; it shows that a perceived-delay input is not an interchangeable substitute for the specified physical disruption.

\begin{figure}[pos=htbp]
\centering\includegraphics[width=\linewidth]{figures/revision20260925/figure5_supply.pdf}
\caption{Four-arm physical-supply experiment with a fixed population of 1,000 Helsinki travelers. (a,b) Mean PT shares across the decision-to-boarding sequence. (c,d) Changes relative to baseline, with paired-person 95\% bootstrap intervals after averaging five common assignment seeds within person. ``Fewer trips, fixed'' retains baseline behavior; ``updated'' recomputes behavior from disrupted level of service. All arms use feasibility-constrained assignment. Both route-search and simulation timetables are changed in the reduced-service arms. All outbound journeys complete in these runs; boarding and completion are nevertheless different outcomes.}
\label{fig:execution_response}
\end{figure}

For SA-Student, perceived delay reduces boarding by about 14 percentage points, compared with less than one point after the reduced timetable is translated into new behavioral inputs. MNL-S shows the same qualitative separation, with a smaller perceived-delay response. The contrast remains a property of these fitted models, supply inputs and fixed population. There is no iterative behavioral adaptation or equilibrium claim.

Assignment introduces a further distinction. In the original perceived-delay experiment, removing infeasible PT probability makes assigned demand routable but does not necessarily move the executed response closer to the raw prediction. Baseline PT shares and changes from baseline can also behave differently. Appendix~\ref{sec:execution_support} retains the assignment comparison and the stagewise explanation. All stage differences use the initial population, so changes in the denominator cannot masquerade as response preservation.

Journey-time conclusions require more caution than mode counts. The earlier active-mode speed check changes the sign of the simulated time response while leaving assigned modes fixed. The physical-supply extension retains those historical network-speed conventions. Its negative time differences therefore cannot establish that reducing transit service improves real travel times.

\subsection{After distillation, supply preparation dominates the measured workload}
Behavioral inference accounts for less than one percent of the larger measured scenario builds. Reusing unchanged network and timetable calculations saves much more construction time than choosing between the compact Students. The practical gain is local reuse of a fitted decision model within a broader computational pipeline, not elimination of the cost of building or executing a transport scenario.

Existing usage logs also support a bounded acquisition-cost estimate without new API experiments. The new synthetic pilot and formal sample record about 6.56 million tokens, valued at approximately US\$3.50 of Go plan allowance under the documented request-time and cache rates. Unreported usage remains unknown; this is not an invoice. Historical Pro acquisition is accounted for separately in Appendix~\ref{sec:cost_summary}. Missing acquisition charges and local-compute tariffs prevent a measured lifecycle break-even claim.
''')
put('06_discussion.tex',r'''
\section{Discussion}
\label{sec:discussion}
\subsection{A useful surrogate needs a stated domain of response}
Response fidelity turns a general modeling concern into an explicit test: for the same traveler and trip, does the Student preserve the Teacher's change? The controlled results support giving this target attention during fitting. They do not establish a universal advantage for response supervision. The delay-withheld results show that improving contrasts represented in training can coexist with poorer responses to an omitted family. A favorable overall average cannot resolve that failure.

This boundary follows the information available to the learner. Our paired penalty reweights known endpoint targets; it does not reveal the response of the Teacher in an unsampled part of the input space. Relational and derivative-based learning provide precedents for supervising more than isolated outputs \citep{park2019relational,czarnecki2017sobolev}, but the benefits of a particular relation depend on its coverage and the fitted model. The appropriate model-selection question is therefore tied to an intended intervention and its plausible range, rather than to the presence of a response term in the loss.

Model specification matters as much as the objective in this study. Linear choice utilities preserve the archived responses better than the neural alternatives, while neural timing substantially improves the modular model's departure prediction. This suggests a useful separation of outputs. It does not establish economically identified preferences: the targets are LLM judgments, the utilities are unconstrained, and the local fare diagnostic exposes inconsistent probability directions. The logit comparison is an informative baseline and a possible basis for later recalibration, not an automatic certificate of behavioral validity.

\subsection{Human evidence changes the meaning of success}
The accessibility reversal is consequential because it changes the substantive conclusion a planner would draw. A Teacher-faithful model can point in the wrong direction relative to the people in the evaluation sample. Conversely, agreement on one response can result from opposing discrepancies that cancel. Reporting static fit, paired response agreement and the same-task reference comparison makes these cases distinguishable.

Empirical alignment must therefore remain separate from compression. Persona learning and task-specific examples offer ways to improve alignment \citep{liu2026persona,xu2026satisfaction}, but any repair should be evaluated on people and tasks not used to construct it. No such independent repair validation is available here. The existing questionnaires diagnose mismatches; they do not demonstrate that those mismatches have been corrected. A new human sample is the next step for that claim.

The survey format also constrains interpretation. These are stated choices from convenience samples, not observed reactions to actual service changes. Singapore presents numerical alternatives, whereas Shanghai requires additional numerical assumptions for qualitative tasks. Sensitivity profiles examine some of those assumptions without exhausting them. Future validation should use numerically matched randomized tasks, explicit alternative definitions and independent respondents before extending these findings to population demand or field behavior.

\subsection{Transport supply is part of the behavioral test}
The four-arm design separates two mechanisms often merged in a disruption scenario: changing what a behavioral model is told, and changing the service that travelers encounter. Their effects differ markedly here. Recomputing level of service connects the physical change to the decision inputs, but does not guarantee an adequate response. Both the input construction and the resulting behavior need validation for the intervention of interest.

Execution adds another layer. Feasibility correction necessarily redistributes excluded probability mass, while routing and boarding determine how much of the assigned demand is realized. A common population denominator and a stagewise account reveal those changes. This complements network-outcome surrogates \citep{natterer2025surrogates} and integrated traveler-day systems \citep{cheng2026calm}: the component being replaced determines which downstream discrepancies remain to be tested. Our single-iteration runs isolate transmission with fixed predictions; they do not capture adaptation to experienced congestion or an equilibrium after a disruption.

\subsection{What the present evidence can support}
The archived benchmark offers a controlled comparison but contains few independent test personas. The new synthetic sample expands persona coverage at one numerical trip and uses a different API model. Some returned usage metadata changed during acquisition without an immutable backend fingerprint, so that comparison is conditional on the realized service mix. The surveys add human evidence within their task encodings and recruitment limits. Helsinki adds an execution test, with uncalibrated journey-time assumptions that preclude claims about real service benefits.

Within these boundaries, the study supports a sequence of decisions for model use: define the required intervention response, test its fidelity to frozen targets, compare it with independent human evidence, and follow it through the network. A failure identifies where additional data or model changes are needed. Local inference can make repeated analysis economical, but computational speed cannot substitute for any of these behavioral checks.
''')
put('07_conclusion.tex',r'''
\section{Conclusion}
\label{sec:conclusion}
A traveler model can imitate its LLM Teacher, disagree with people and change again during network execution. Evaluating paired responses makes these distinctions visible. Matched training shows a modest benefit from explicit response supervision, while the delay-family holdout exposes its coverage limit. Human surveys reveal a reversal between Teacher fidelity and a substantive accessibility response. Physical-supply experiments show why a perceived delay and a reduced timetable must be tested separately. These results support response-specific validation of compact LLM-derived models, with model provenance, human agreement and execution treated as distinct evidence. They do not yet support independent human repair, generalization to arbitrary disruptions or calibrated predictions of real-world travel-time benefits.
''')

G=R/'generated'/'revision20260925';G.mkdir(parents=True,exist_ok=True)
(G/'table_design.tex').write_text(r'''
\begin{table}[pos=htbp]
\caption{Evidence sources and the unit behind each comparison.}\label{tab:study_design}
\small\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\linewidth}{@{}p{.23\linewidth}p{.28\linewidth}X@{}}\toprule
Question & Independent units & What is compared \\\midrule
Compression fidelity & Six held-out personas; 398 pairs & Matched objectives and specifications against archived Pro targets \\
Response outside training coverage & Thirty new personas; pilot kept separate & Full and delay-withheld Students against a later Go Flash reference \\
Agreement with people & 332 Singapore and 321 Shanghai respondents & Within-person stated responses; later Pro reference on 24 people per city \\
Transmission through supply & 1,000 fixed synthetic travelers & Assignment rules and four supply arms; paired assignment seeds \\\bottomrule
\end{tabularx}
\end{table}
'''.strip()+'\n',encoding='utf-8')
import csv
d=list(csv.DictReader((R/'data_tables/revision20260925/formal_paired_contrasts.csv').open(encoding='utf-8-sig')))
rows=[]
for fam,card,label in [('full','ALL','Full training: all contrasts (primary)'),('delay_holdout','ALL','Delay withheld: all contrasts'),('delay_holdout','delay_8','Delay withheld: 8-minute delay'),('delay_holdout','delay_45','Delay withheld: 45-minute delay'),('delay_holdout','fare_delay','Delay withheld: fare + delay')]:
 q=next(x for x in d if x['family']==fam and x['card']==card and x['selection']=='static')
 rows.append(label+f" & ${float(q['mean']):+.4f}$ & $[{float(q['ci_low']):+.4f}, {float(q['ci_high']):+.4f}]$ "+r'\\')
(G/'table_primary.tex').write_text(r'''
\begin{table}[pos=htbp]
\caption{Formal new-persona comparison: response error of signed-response fitting minus soft KL.}\label{tab:primary_results}
\small\begin{tabularx}{\linewidth}{@{}Xrr@{}}\toprule
Setting and contrast & Difference & 95\% paired interval \\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabularx}
\par\smallskip\footnotesize All rows use thirty personas, KL-selected fits and a mean over three training seeds within persona. Negative differences favor signed-response fitting. Intervals use 10,000 persona resamples; all rows except the designated primary comparison are descriptive and unadjusted for multiplicity. The reference is the later Go Flash model at one fixed trip.
\end{table}
''',encoding='utf-8')

# Appendix A: move the reader-facing questionnaire and encoding explanation intact.
a=before['sections/S1_inputs.tex']
a=a[:a.index('Table~\\ref{tab:input_dictionary} lists')]
put('A_inputs.tex',a)
put('S1_inputs.tex',r'''
\section{Encoded field dictionary}
\label{sec:field_dictionary}
The questionnaire tasks, sample flow, route-search definitions and input sensitivity are in Appendix~\ref{main-sec:supp_inputs} of the article. This section retains the field-level dictionary for reproduction.
\input{generated/narrative/table_input_dictionary.tex}
\FloatBarrier
''')

# Model-role and provenance tables are reader-facing; move them out of the supplement.
mat=before['sections/S7_materials.tex'];tables=re.findall(r'\\begin\{table\}.*?\\end\{table\}',mat,re.S)
roles=next(t for t in tables if r'\label{tab:model_roles}' in t)
prov=next(t for t in tables if r'\label{tab:input_provenance}' in t)
put('A_inputs.tex',(S/'A_inputs.tex').read_text()+ '\n'+prov+'\n\\FloatBarrier\n')
mat=mat.replace(roles,'').replace(prov,'')
put('S7_materials.tex',mat)

oldmethod=before['sections/03_methodology.tex']
loss=oldmethod[oldmethod.index(r'\subsection{Response-aware learning objectives}'):oldmethod.index(r'\subsection{From predicted behavior')]
loss=loss.replace(r'\label{sec:loss}',r'\label{sec:loss_details}')
oldlearn=before['sections/S2_learning.tex']
diag=oldlearn[oldlearn.index(r'\subsection{Properties of the learning objectives}'):oldlearn.index(r'\subsection{Response robustness}')]
put('S2_learning.tex',oldlearn.replace(diag,r'The loss properties and their graphical diagnostic are in Appendix~\ref{main-sec:loss_diagnostics}.'+'\n\n'))
s9=before['sections/S4_s9.tex'].replace(r'\section{Supply-adapted Student}',r'\subsection{Supply-adapted Student}')
put('B_models.tex',r'''
\section{Model specifications and learning sensitivity}
\label{sec:model_details}
'''+roles+r'''
The neural model normalizes utilities over the available set $\mathcal A(\mathbf x)$:
\begin{equation}
p_{S,m}(\mathbf x)=\frac{\exp(u_m)}{\sum_{k\in\mathcal A(\mathbf x)}\exp(u_k)},\quad m\in\mathcal A(\mathbf x).
\label{eq:utility}
\end{equation}
Unavailable alternatives receive zero probability. MNL-S replaces the neural utilities with linear utilities on the same encoded information. Its coefficients are not constrained to have behavioral signs.
'''+loss+diag+r'''
\subsection{Weight, selection and specification checks}
\label{sec:optimization_support}
The response-weight grid contains $\lambda\in\{0.25,0.5,1,2\}$ for signed-response and direction-plus-magnitude fitting, plus soft KL, each with three seeds. All 27 fits run for 120 epochs and 6,360 updates. Two validation selectors retain checkpoints from the same trajectory: macro-source KL and mean response error. Nine historical unit-weight configurations reproduce their parameters and selected epochs. The grid is a sensitivity analysis; the formal primary comparison retains unit-weight signed response and the KL selector.
\begin{figure}[pos=htbp]
\centering\includegraphics[width=\linewidth]{figures/revision20260925/figureA_weights.pdf}
\caption{Response-weight and validation-selector sensitivity on the archived benchmark. Points are means over three seeds on the same six test personas, not independent replications. Lines connect discrete fitted weights. Test minima are not used for post-hoc primary model selection.}
\label{fig:weight_sensitivity}
\end{figure}
The delay-withheld setting retains 1,562 training and 307 validation endpoints, after removing 277 and 58 respectively. Whole linked delay groups are removed, including zero-delay members. Its six fits use 120 epochs and 5,280 updates. On the 58 archived delay pairs, signed-response minus soft-KL error is $+0.00313$ ($[0.00182,0.00494]$) under KL selection and $+0.00532$ ($[0.00362,0.00715]$) under response selection. These retrospective intervals are conditional on only six personas. They agree in direction with the formal delayed-state result but are not an independent human test.

All four MNL regularization candidates converge; both choice selectors choose $10^{-4}$. A finite-perturbation diagnostic at the 39 explicitly feasible test endpoints increases fare by one unit, total PT time by five minutes, or access and total time jointly by five minutes. The latter two reduce PT probability at every endpoint; fare increases raise it at one third. These local directions do not identify willingness to pay or a value of travel time. Fixed-batch gradient records separate KL, departure and response terms along training; they are optimization diagnostics, not evidence of a causal mechanism for the loss advantage.

\input{sections/A_s9}
''')
put('A_s9.tex',s9)

# The compact main comparisons move into the article appendix; full cells remain reproducible.
def float_table(name):
 t=(R/'generated/narrative'/name).read_text()
 t=t.replace(r'\begin{resulttable}',r'\begin{table}[pos=htbp]\centering\small').replace(r'\end{resulttable}',r'\end{table}')
 (G/name).write_text(t,encoding='utf-8');return r'\input{generated/revision20260925/'+name+'}\n'
core=''.join(float_table(n) for n in ['table_response_main.tex','table_timing_main.tex','table_human_main.tex','table_teacher_decomp_s9.tex','table_execution_main.tex'])
olds5=before['sections/S5_execution.tex'];sp=olds5[olds5.index(r'\subsection{Active-mode speed sensitivity'):olds5.index(r'\subsection{',olds5.index(r'\subsection{Active-mode speed sensitivity')+12)]
geom=olds5[olds5.index(r'\subsection{Feasibility correction and response geometry}'):]
put('S5_execution.tex',olds5.replace(sp,'The active-mode speed diagnostic is in Appendix~\\ref{main-sec:speed_sensitivity}.\n\n').replace(geom,'The feasibility identity and counterexample are in Appendix~\\ref{main-sec:feasibility_geometry}.\n'))
put('C_comparisons.tex',r'''
\section{Supporting comparisons and execution interpretation}
\label{sec:core_tables}
The following tables retain the exact core comparisons behind the main narrative. Seed standard deviations measure training variability; they are not respondent confidence intervals. The same-task API decomposition uses common respondent resamples and refers to a later Pro service, not a recovered training backend.
'''+core+r'''
\FloatBarrier
\subsection{Assignment, routing and completion}
\label{sec:execution_support}
The original perceived-delay experiment contains 140 model--condition--assignment configurations, all with the full population retained. Its sampling policies use three paired assignment seeds; this differs from the five-seed four-arm extension. No run encountered zero retained feasible mass. If that mass is zero, the implementation stops rather than silently discarding the traveler. Behavioral predictions are computed once, followed by a departure adjustment, feasibility checks, assignment and routing at the adjusted time.

For SA-Student, the original feasibility-constrained result shifts the PT response by approximately $+0.27$ percentage points during probability adjustment and by $-1.34$ points between adjusted probabilities and boarding. The total gap is about $-1.07$ points. These values describe three assignment seeds and should not be interchanged with the five-seed extension. The new reanalysis averages paired seeds within person before estimating intervals for every stage increment; its full aggregate ledger is supplied with the article data. Signed averages can hide opposing model-level gaps, which is why the supplement also reports absolute gaps after averaging assignment repetitions within each fitted model.

The physical disruption groups scheduled trips by line and ordered stops, retains alternating departures chronologically, and keeps singleton groups. It retains 6,924 of 13,655 trips across 344 groups. The baseline level-of-service reconstruction matches all 1,000 inputs. Every one of the 40 extension configurations passes checks on population size, assignment transformation, common random numbers, supply paths and retained trip identifiers. Each completes all outbound journeys. Completion-conditioned and completion-or-horizon means therefore coincide in these runs, although neither is a calibrated real-world time estimate.
'''+geom+sp)

put('D_new_evidence.tex',r'''
\section{New-reference protocol, service provenance and cost}
\label{sec:new_protocol}
\subsection{Persona sampling and analysis}
The acquisition manifest fixes 132 new persona profiles excluded from earlier datasets. The first twelve are pilot-only; the next thirty form the formal sample. A baseline and eleven perturbations share one numeric trip. The conditions cover delay of 8 and 45 minutes, fare multipliers of 1.25 and 2.5, access increases of 5 and 20 minutes, fare--delay and fare--access combinations, PT infeasibility, and rain intensities of 0.25 and 0.9. These input values define scenarios, not all-independent unseen intervention families. The archived training-support audit is retained alongside the state manifest.

The primary planning statistic is the persona-level signed-response minus soft-KL error, averaging three training seeds and eleven contrasts. The pilot standard deviation is 0.006306. The prespecified rule $\max\{30,\lceil(1.96s/0.01)^2\rceil\}$, with a maximum of 120, yields thirty formal personas. It is a precision planning rule, not a guarantee of interval width. Formal analysis uses 10,000 paired-persona bootstrap draws (seed 20260925), retaining the complete contrasts within persona. All other method, selector and condition comparisons are descriptive, with pointwise intervals. The pilot is never pooled with the formal sample.

\begin{figure}[pos=htbp]
\centering\includegraphics[width=\linewidth]{figures/revision20260925/figureA_generalization.pdf}
\caption{Additional formal-sample views under full training and KL selection. (a) All thirty persona-level primary paired differences. (b) The other eight retained contrasts, complementing the delay conditions in the main text. Whiskers are descriptive pointwise 95\% persona-bootstrap intervals. All comparisons use the same Go Flash reference and fixed trip.}
\label{fig:new_personas}
\end{figure}

Requests used OpenCode Go, model \texttt{deepseek-v4.1-flash}, temperature 0.2, an 8,192-token output budget and the provider's default reasoning setting. The requested and returned model strings agree. The pilot supplies 432 valid responses and the formal sample 1,080. Missing valid state--repeat units were reacquired after recorded connection or parsing failures without changing the prompt, state manifest or primary analysis. Acquisition attempts, completion identifiers, usage and prompt hashes are preserved in the experiment package. This campaign is distinct from the historical official Pro training and survey-reference campaigns.

\subsection{Returned service metadata}
During formal acquisition, some responses reported zero reasoning tokens and a different usage-detail structure while request settings and the returned model name stayed the same. The diagnostic was documented after observing those metadata and before examining formal method comparisons. Zero reported tokens do not prove absence of internal reasoning or a backend model change. No immutable backend fingerprint was available.

The formal sample has 976 positive-reasoning and 104 zero-reported-reasoning responses. Twenty-four personas have only the former and six have a mixture. The primary paired difference is $-0.00299$ ($[-0.00496,-0.00121]$) in the all-positive group and $-0.00405$ ($[-0.00770,-0.00117]$) in the mixed group. These post-hoc subgroups are confounded with acquisition time and persona order. They neither identify a reasoning-mode effect nor justify excluding data. The primary analysis includes all thirty personas and is conditional on the realized service mix.

\subsection{Recorded usage and retrospective valuation}
\label{sec:cost_summary}
The new pilot and formal campaign made 1,618 attempts, yielding 1,512 valid responses. Usage is available for 1,525 distinct completions, including thirteen invalid responses; 93 attempts have unknown usage. Recorded input and output total 2,206,391 and 4,355,420 tokens, respectively. Cached input is already included in input, and reasoning tokens are already included in output.

Using the Go rate card checked on 25 September 2026, with request-time peak/off-peak classification and reported cache usage, values the recorded total at US\$3.4952 of plan allowance \citep{opencode2026go}. Revaluing the same known tokens at all-peak, no-cache rates gives US\$5.8884. Neither estimate covers unknown usage or verifies an account debit. The US\$10 monthly subscription is a separate fixed expense and is not added to every request.

\input{generated/narrative/table_api_cost.tex}
The historical Pro table above includes retained development runs and earlier model versions, not just acquisition for the final Student. It is a current-rate revaluation of observed usage rather than a recovered invoice. It must not be priced with the new Go Flash rates. The 27 controlled and six delay-withheld fits sum to 845.69 and 200.17 seconds of recorded fit time; these are neither end-to-end campaign wall times nor CPU-seconds. Hardware rental and electricity charges were not recorded.

For repeated matched workloads, a monetary break-even count would be $N^*=C_0/(c_L-c_S)$ when $c_L>c_S$, where $C_0$ includes acquisition, retries, training and necessary adaptation, and $c_L,c_S$ are comparable per-workload costs. Shared supply and routing work must be counted consistently. Existing measurements supply only part of this accounting. Supplementary Section~\ref{supp-sec:supp_compute} retains the inference, construction and historical metering details without asserting a measured lifecycle break-even point.
''')

# Update the article wrapper and the linked supplement, retaining authors and ethics.
main=before['cas-sc-template.tex']
abstract=r'''Transport scenario analysis needs models that reproduce how choices change when travel conditions change. Distilling a large language model (LLM) into a compact traveler model can reduce repeated inference costs, but agreement with the LLM does not establish agreement with people or with executed travel. We evaluate this chain through paired Teacher--Student responses, independent stated-choice surveys and controlled transit-supply experiments. Matched training shows a modest benefit from explicit response supervision, while a logit Student achieves greater fidelity to the archived Teacher. A new synthetic-persona test supports the response-learning gain under full training, but the advantage reverses on delay scenarios when that intervention family is withheld. In the human surveys, the Teacher-best Student predicts an increase in public-transport use under poorer accessibility where respondents report a decline. Network experiments further distinguish a perceived-delay input from an actual reduction in scheduled service: the two changes produce markedly different demand responses, even with the same population and assignment rule. These findings show why compression fidelity, human agreement and execution must be evaluated separately. Response-specific comparisons help identify the data or model component requiring improvement before a fast behavioral surrogate is used for transport scenario analysis.'''
main=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',main,flags=re.S)
availability=r'''\section*{Data and code availability}
The archived benchmark, selected models, analysis scripts and compact new-experiment evidence are retained in the research repository, \url{https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research}. The new experimental evidence is pinned to commit \texttt{ba7c5d3}; the complete identifier and source hashes accompany the figure data. The article source includes the aggregate CSV/JSON inputs and scripts needed to regenerate its figures. Frozen API targets permit re-evaluation without new calls. Participant-level survey records and participant-linked payloads remain separately held subject to privacy and applicable permissions. Access to restricted materials should be requested from the corresponding author. Fresh network execution also requires the documented supply files and MATSim environment. Supplementary Section~\ref{supp-sec:materials_scope} distinguishes these dependencies from locally reproducible analysis.
\FloatBarrier
\appendix
\input{sections/A_inputs}
\input{sections/B_models}
\input{sections/C_comparisons}
\input{sections/D_new_evidence}
\FloatBarrier
'''
main=main[:main.index(r'\section*{Data and code availability}')]+availability+main[main.index(r'\bibliographystyle'):]
(R/'cas-sc-template.tex').write_text(main,encoding='utf-8')
supp=before['supplement.tex'].replace(r'\input{sections/S4_s9}','')
supp=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:r'\begin{abstract}'+'\n'+r'This supplement retains field dictionaries, complete training and evaluation tables, repeat-query diagnostics, computational measurements and material inventories for reproduction or further analysis. Questionnaire interpretation, core comparison tables, loss properties and new-reference provenance are included in the article appendices.'+'\n'+r'\end{abstract}',supp,flags=re.S)
(R/'supplement.tex').write_text(supp,encoding='utf-8')

# Rewrite references to material moved between the article and its supplement.
moved=set()
for n in ['A_inputs.tex','B_models.tex','A_s9.tex','C_comparisons.tex','D_new_evidence.tex']:
 txt=(S/n).read_text();moved.update(re.findall(r'\\label\{([^}]+)\}',txt))
 for path in re.findall(r'\\input\{([^}]+)\}',txt):
  p=R/(path+'.tex')
  if p.exists():moved.update(re.findall(r'\\label\{([^}]+)\}',p.read_text()))
for p in list(S.glob('*.tex'))+list((R/'generated').rglob('*.tex'))+[R/'cas-sc-template.tex']:
 t=p.read_text();is_supp=p.name.startswith('S')
 for label in moved:
  t=t.replace('{supp-'+label+'}', '{'+('main-' if is_supp else '')+label+'}')
  if is_supp:t=t.replace('{'+label+'}', '{main-'+label+'}') if r'\label{'+label+'}' not in t else t
 # Reader-facing sections now refer to supplement-only labels through xr.
 if p.name in ['A_inputs.tex','B_models.tex','A_s9.tex','C_comparisons.tex','D_new_evidence.tex']:
  for label in ['sec:supp_controlled','sec:materials_scope','tab:departure_runs','tab:departure_sources','sec:supp_human']:
   t=t.replace('{'+label+'}','{supp-'+label+'}')
 # Correct the prose describing destinations as well as their label prefix.
 for label in moved:
  t=re.sub(r'Supplementary (Section|Table)~\\ref\{'+re.escape(label)+r'\}',lambda m:('Appendix~' if m.group(1)=='Section' else 'Table~')+r'\ref{'+label+'}',t)
 p.write_text(t,encoding='utf-8')

bib=R/'references.bib';bt=bib.read_text()
if 'opencode2026go' not in bt:bt+=r'''
@misc{opencode2026go,
 author={{OpenCode}}, title={Go}, year={2026},
 url={https://opencode.ai/docs/go/},
 note={Provider plan and rate documentation; accessed 25 September 2026}
}
'''
bib.write_text(bt,encoding='utf-8')
(R/'highlights.txt').write_text('Paired responses expose errors hidden by static imitation scores.\nResponse supervision helps covered tasks but can fail on withheld delays.\nTeacher fidelity does not determine agreement with stated human responses.\nPerceived delay and physical service reduction produce different demand changes.\n',encoding='utf-8')
changed=[]
for rel,old in before.items():
 p=R/rel
 if p.exists() and p.read_text()!=old:changed.append({'file':rel,'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(R/'editorial_notes/narrative_apply_report.json').write_text(json.dumps({'mode':'full_reemission_escalated','authorization':'User explicitly requested narrative, figure and appendix restructuring on 2026-09-25','changes':changed},indent=2),encoding='utf-8')
print('Applied',len(changed),'changed pre-existing TeX files; added appendices and evidence tables.')
