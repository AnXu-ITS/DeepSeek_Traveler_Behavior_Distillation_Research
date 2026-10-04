# Survey design history

For the administered instruments, use the complete English [Shanghai questionnaire](../surveys/SHANGHAI_QUESTIONNAIRE.md) and [Singapore questionnaire](../surveys/SINGAPORE_QUESTIONNAIRE.md), with the [sample flow](../surveys/README.md) and [survey results](../surveys/RESULTS.md). The final Shanghai instrument contains 23 single-choice questions, required when displayed, including ten scenario tasks.

The Shanghai materials below record the development of respondent attributes, scenario cards, display logic and data mappings. Proposed questions and numerical settings are historical design records, not evidence of what was administered or newly executed results. Complete original records remain linked at the immutable historical commit `de906ac8a3efae692797bd08e340375107b8e52a`.

## Original design

- [Original v1 questionnaire](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/plans/shanghai_survey_v2/Shanghai_Travel_Intention_Survey_v1_original.md)
- [Retained early design tables](Shanghai_Travel_Intention_Survey.md)

## Version 2 design package

- Original records: [change history](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/plans/shanghai_survey_v2/CHANGELOG.md) and [participant sections](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/plans/shanghai_survey_v2/participant_sections.md)
- Retained records: [full questionnaire design tables](shanghai_survey_v2/Shanghai_Travel_Intention_Survey_v2_full.md) and [implementation tables and commands](shanghai_survey_v2/IMPLEMENTATION.md)
- Definitions: [scenario cards](shanghai_survey_v2/cards.json), [forms](shanghai_survey_v2/forms.json) and [field mapping](shanghai_survey_v2/field_mapping.json)
- [Bilingual participant-section source template](shanghai_survey_v2/participant_sections.source.txt): historical input to the version 2 package builder, not the administered final questionnaire
- Scripts: [package builder](shanghai_survey_v2/build_package.py), [survey adapter](shanghai_survey_v2/survey_adapter.py) and [package validator](shanghai_survey_v2/validate_package.py)

## Version 3 design package

- [Original implementation record](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/plans/shanghai_survey_v3/IMPLEMENTATION.md)
- Definitions: [scenario cards](shanghai_survey_v3/cards.json), [forms](shanghai_survey_v3/forms.json), [field mapping](shanghai_survey_v3/field_mapping.json) and [question options](shanghai_survey_v3/question_options.json)
- Scripts: [survey adapter](shanghai_survey_v3/survey_adapter.py) and [package validator](shanghai_survey_v3/validate_package.py)

## Field instrument source

- Original records: [field documentation](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/plans/shanghai_survey_v3/field/README.md) and [Chinese questionnaire](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/plans/shanghai_survey_v3/field/%E4%B8%8A%E6%B5%B7%E5%87%BA%E8%A1%8C%E6%96%B9%E5%BC%8F%E8%B0%83%E6%9F%A5_%E9%97%AE%E5%8D%B7.md)
- [Retained source HTML](shanghai_survey_v3/field/%E4%B8%8A%E6%B5%B7%E5%87%BA%E8%A1%8C%E6%96%B9%E5%BC%8F%E8%B0%83%E6%9F%A5_%E5%8E%9F%E5%A7%8B%E9%A1%B5%E9%9D%A2.html)
- [Complete English field questionnaire](../surveys/SHANGHAI_QUESTIONNAIRE.md): final wording captured on 18 September 2026, including consent, eligibility, attributes and all ten scenario choices

[Research design](../RESEARCH_DESIGN.md) · [Training](../TRAINING.md) · [Project results](../RESULTS.md) · [Data and access](../DATA_SOURCES.md) · [Model use](../MODEL_USE.md)
