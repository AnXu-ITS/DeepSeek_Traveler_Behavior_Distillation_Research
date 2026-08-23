"""Teacher integration: client, prompts, parser, validator."""
from .prompts import TEACHER_PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from .client import DeepSeekTeacherClient, MockTeacherClient, TeacherClientError
from .parser import TeacherResponseParser, TeacherParseError
from .validator import TeacherResponseValidator, ValidationResult

__all__ = [
    "TEACHER_PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "build_user_prompt",
    "DeepSeekTeacherClient",
    "MockTeacherClient",
    "TeacherClientError",
    "TeacherResponseParser",
    "TeacherParseError",
    "TeacherResponseValidator",
    "ValidationResult",
]
