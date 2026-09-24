# Shanghai Travel Mode Survey — English translation

Documentation translation of the field instrument captured from [Wenjuanxing](https://v.wjx.cn/vm/YJnKeRu.aspx) on 18 September 2026 at 13:36 (UTC+8). All 23 questions are single-choice and required when displayed. This is the final field wording, not one of the earlier design drafts. The translation does not modify the administered questionnaire or its responses.

## Consent and eligibility

**1. Are you at least 18 years old, and are you willing to participate in this anonymous survey?**

- A. Yes
- B. No

**2. Have you traveled in Shanghai during the past 30 days?**

- A. Yes
- B. No

Question 2 is shown only after “Yes” to Question 1. Questions 3–23 require “Yes” to both Questions 1 and 2. Selecting “No” ends the applicable survey path.

## Personal characteristics and travel habits

**3. What is your age?**

- A. 18–24 years
- B. 25–34 years
- C. 35–44 years
- D. 45–64 years
- E. 65 years or older

**4. What is your main current occupation?**

- A. Student
- B. Office work
- C. Service-sector work
- D. Manufacturing, construction or manual labor
- E. Retired
- F. Unemployed
- G. Homemaker
- H. Other

**5. What is your approximate personal monthly income?**

- A. Below CNY 5,000
- B. CNY 5,000–9,999
- C. CNY 10,000–19,999
- D. CNY 20,000 or more
- E. Prefer not to answer

**6. Does your household include any children under 18?**

- A. Yes
- B. No

**7. Does your household have a car available for your use?**

- A. Yes
- B. No

**8. Do you hold a valid car driving licence?**

- A. Yes
- B. No

**9. Do you own a conventional bicycle? (Exclude electric bicycles and shared bicycles.)**

- A. Yes
- B. No

**10. Do you hold a currently valid bus or metro period pass? (Exclude ordinary stored-value travel cards and ride-payment codes.)**

- A. Yes
- B. No

**11. Which mode do you use most often?**

- A. Drive a car myself
- B. Ride in a family member's or friend's car
- C. Bus or metro
- D. Conventional bicycle
- E. Taxi or ride-hailing
- F. Electric bicycle
- G. Walking
- H. Several modes about equally
- I. Other

**12. How much can you usually adjust your departure time?**

- A. It is essentially fixed; by at most five minutes
- B. By 6–30 minutes
- C. By more than 30 minutes
- D. I do not have a fixed departure time

**13. Does your physical condition affect your ability to walk or cycle?**

- A. Hardly at all
- B. To some extent
- C. Considerably

## Ten scenario choices

Each of Questions 14–23 asks **“Which mode would you choose?”**, with the same options:

- A. Drive myself or take a taxi
- B. Bus or metro
- C. Conventional bicycle
- D. Walking
- E. None of these is suitable

The question-specific conditions are:

| Question | Scenario wording |
|---|---|
| 14 | In clear weather with normal traffic conditions, which mode would you choose? |
| 15 | In rainy weather with normal traffic conditions, which mode would you choose? |
| 16 | In clear weather when the bus or metro is delayed, which mode would you choose? |
| 17 | In rainy weather when the bus or metro is delayed, which mode would you choose? |
| 18 | In clear weather when the bus or metro fare has risen to CNY 6, which mode would you choose? |
| 19 | In clear weather when the parking fee has risen to CNY 30, which mode would you choose? |
| 20 | In clear weather with a temporary road obstruction, which mode would you choose? |
| 21 | In clear weather when the bus or metro stop is relatively far away, which mode would you choose? |
| 22 | In clear weather when bus or metro services are less frequent, which mode would you choose? |
| 23 | In clear weather when the bus or metro journey requires one transfer, which mode would you choose? |

No travel durations, OD locations, quantitative delay or headway values are stated in these question texts. The numerical values used to encode them for the models are analysis assumptions, documented in the manuscript. “None suitable” is retained as a distinct raw answer; it is not silently mapped to walking or PT.

## Instrument structure and analysis

The captured page uses one page (`pg=1`); question type `3` is single choice and `req=1` marks required questions. Display relations are `1,1` for Question 2 and `1,1|2,1` for Questions 3–23. The instrument has 13 consent/eligibility/attribute questions and ten scenario questions. Static questionnaire banners are not substantive questions.

The English presentation consolidates the repeated response options above for readability; the same five options apply independently to every task. The administered questionnaire's car label remains broader than the model's ownership-conditioned private-car alternative.

[Sample flow and results](README.md) · [Complete response table](RESULTS.md)
