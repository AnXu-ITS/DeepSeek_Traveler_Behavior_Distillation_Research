# Stated-choice questionnaires and results

The two questionnaires evaluate whether fixed models agree with human responses to changed travel conditions. Human answers are not used to train, select or tune the Students.

| Resource | Singapore | Shanghai |
|---|---|---|
| Complete English instrument | [Questionnaire](SINGAPORE_QUESTIONNAIRE.md) | [Questionnaire](SHANGHAI_QUESTIONNAIRE.md) |
| Platform / period | Google Forms; 20 August–5 September 2026 | Wenjuanxing; 12–18 September 2026 |
| Consent / screening | 334 consented/completed; two short-term visitors screened out | 341 consented; eight had not traveled in Shanghai during the preceding 30 days |
| Eligible completions | 332 residents | 333 people |
| Model-evaluation cohort | 332 people | 321 people; 12 outside the input mapping retained for human-only sensitivity |
| Scored choices | 3,320 | 3,208 explicit four-mode choices; two “none suitable” responses excluded from four-mode scoring |
| Paired responses | Five contrasts, n=332 each | Eight contrasts; n=321 except interaction and walk-versus-wait, n=320 |
| Numerical task detail | Explicit time/cost tables | Mostly qualitative; explicitly CNY 6 fare, CNY 30 parking, one transfer |

Recruitment used convenience and snowball sharing through WeChat, WhatsApp and personal contacts. The samples are not representative city samples. Both instruments recorded consent and displayed a privacy statement. The study was conducted without a formal institutional ethics review or exemption determination. The [current author-supplied ethics and consent statement](../manuscript-support/2026-10-03/DECLARATIONS.md#ethics-and-consent-statement) records anonymous, voluntary participation and electronic consent.

## Results and analysis choices

- [Complete human/model response results](RESULTS.md): all 78 city/model/contrast rows, adjusted intervals and denominators.
- [Project results](../RESULTS.md): choice metrics, same-task Teacher decomposition, input sensitivity and simulation comparisons.
- [Aggregate model scores](../../evidence/paper_20260924/survey/model_scores.csv): filter **`profile == ownership`** for the paper's primary setting.
- [Authoritative mapping note](../../evidence/paper_20260924/survey/authoritative_mapping_revision.json): distinguishes the original ownership-conditioned contract from all-displayed-options diagnostics.

The model treats car as private car requiring ownership and a licence, while the displayed survey choice includes driving/taxi wording. Bicycle requires access. All explicit labels are nevertheless scored, including eight Singapore and two Shanghai choices of masked alternatives. NLL uses a floor of 1e-8; an unavailable chosen mode would otherwise yield infinite NLL.

Respondent bootstrap intervals retain each person's complete set of tasks and use shared resamples for paired comparisons (10,000 draws, seed 917). Family-adjusted intervals cover five Singapore or eight Shanghai contrasts separately by model/city. Training-seed SD is a different uncertainty measure. An interval covering zero does not establish equivalence.

## Preserving the original evidence

These English documents translate the retained questionnaire content. They do not change raw responses, respondent eligibility or the original administered instruments. For Shanghai, the live instrument extraction was captured on 18 September 2026. Earlier design drafts in `docs/plans/` are historical and should not be mistaken for the final field questionnaire.

Only aggregate response results are included in this repository update. Individual response workbooks and participant-linked inputs/Teacher payloads remain separately retained. The [Shanghai instrument](https://v.wjx.cn/vm/YJnKeRu.aspx) identifies the survey; access to the instrument is not access to individual responses. The Singapore source link in the manuscript is an owner/editor URL and may require permission.
