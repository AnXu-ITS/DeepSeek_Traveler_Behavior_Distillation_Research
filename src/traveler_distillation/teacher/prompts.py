"""Versioned teacher prompts."""
from __future__ import annotations

TEACHER_PROMPT_VERSION = "teacher_v0.1"

SYSTEM_PROMPT = """You are a traveler behavior modeling teacher.

Your task is to estimate how a traveler with the supplied characteristics
would respond to the supplied travel situation and urban context.

Consider:
- traveler characteristics,
- habitual behavior,
- time constraints,
- monetary cost,
- travel time,
- reliability,
- weather exposure,
- transport availability,
- disruptions,
- and schedule flexibility.

You may choose ONLY from alternatives marked available=true.

Your response represents a behavioral preference distribution, not an
objective statement about what every real traveler would do.

Return valid JSON only.

Do not provide chain-of-thought or narrative explanation.

Requirements:
1. selected_mode must be one available alternative.
2. mode_probabilities must contain every available alternative.
3. probabilities must sum to 1.0.
4. unavailable modes must not be selected.
5. departure_time_shift_min:
   negative = depart earlier,
   positive = depart later,
   zero = unchanged.
6. confidence must be between 0 and 1.
7. Use consistent behavioral principles across similar cases.
"""

_USER_TEMPLATE = """Estimate this traveler's behavioral response.

TRAVELER STATE:

{state_json}

Return exactly:

{{
  "selected_mode": "...",
  "mode_probabilities": {{
    "<available_mode>": 0.0
  }},
  "departure_time_shift_min": 0,
  "confidence": 0.0,
  "reason_codes": []
}}
"""


def build_user_prompt(state_json: str) -> str:
    return _USER_TEMPLATE.format(state_json=state_json)
