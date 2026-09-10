"""Prompt 模板集中管理。"""

from .classify import (
    CATEGORIZE_PROMPT,
    FALLBACK_MESSAGE,
    INTERVIEW_SUB_PROMPT,
    LEARNING_SUB_PROMPT,
)
from .personas import LANG_RULE, PERSONAS, get_persona

__all__ = [
    "CATEGORIZE_PROMPT",
    "LEARNING_SUB_PROMPT",
    "INTERVIEW_SUB_PROMPT",
    "FALLBACK_MESSAGE",
    "PERSONAS",
    "get_persona",
    "LANG_RULE",
]
