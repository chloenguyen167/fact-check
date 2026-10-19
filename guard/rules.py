"""Tầng rule-based của Guard: loại sớm input không phải factual claim."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

QUESTION_CUES = [
    "có phải",
    "liệu rằng",
    "liệu có",
    "tại sao",
    "vì sao",
    "bao nhiêu",
    "như thế nào",
    "thế nào",
    "là gì",
    "có nên",
    "cho hỏi",
    "cho mình hỏi",
    "cho em hỏi",
    "xin hỏi",
]

QUESTION_END = re.compile(r"\?\s*$")

# Đuôi câu hỏi chỉ tính khi đứng cuối câu ("... có hiệu quả hay không" giữa câu
# trần thuật vẫn là claim)
QUESTION_TAIL = re.compile(
    r"(nên không|được không|có đúng không|đúng không|phải không|hay không|không ạ)"
    r"[\s.!…]*$"
)

OPINION_CUES = [
    "tôi thấy",
    "mình thấy",
    "em thấy",
    "tôi nghĩ",
    "mình nghĩ",
    "em nghĩ",
    "theo tôi",
    "theo mình",
    "theo em",
    "tôi cảm thấy",
    "mình cảm thấy",
    "hình như",
    "có vẻ",
    "cảm giác",
    "tôi tin",
    "mình tin",
    "chắc là",
    "tôi cho rằng",
    "mình cho rằng",
]

SELF_CUES = [
    "tôi bị",
    "mình bị",
    "em bị",
    "tôi đang bị",
    "mình đang bị",
    "tôi mắc",
    "mình mắc",
    "bác sĩ bảo tôi",
    "bác sĩ nói tôi",
    "kết quả của tôi",
    "xét nghiệm của tôi",
    "bệnh của tôi",
    "mẹ tôi bị",
    "bố tôi bị",
    "người nhà tôi",
]

COMMAND_CUES = [
    "giúp tôi",
    "giúp mình",
    "giúp em",
    "khuyên tôi",
    "chỉ cho tôi",
    "làm ơn",
    "vui lòng",
]

# Động từ mệnh lệnh chỉ tính ở đầu câu ("cần được bác sĩ tư vấn" vẫn là claim)
COMMAND_START = re.compile(r"^(hãy|tư vấn|hướng dẫn|gợi ý|kê)\s")

MIN_TOKENS = 4


def normalize(text: str) -> str:
    """Chuẩn hóa unicode NFC, lower, gọn khoảng trắng."""
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class RuleResult:
    verdict: str  # "pass" | "reject"
    reason: str | None = None
    matched: str | None = None


def _find_cue(text: str, cues: list[str]) -> str | None:
    for cue in cues:
        if cue in text:
            return cue
    return None


def rule_filter(text: str) -> RuleResult:
    t = normalize(text)

    if len(t.split()) < MIN_TOKENS:
        return RuleResult("reject", "too_short", None)

    if QUESTION_END.search(t):
        return RuleResult("reject", "question", "?")

    m = QUESTION_TAIL.search(t)
    if m:
        return RuleResult("reject", "question", m.group(1))

    cue = _find_cue(t, QUESTION_CUES)
    if cue:
        return RuleResult("reject", "question", cue)

    cue = _find_cue(t, OPINION_CUES)
    if cue:
        return RuleResult("reject", "personal_opinion", cue)

    cue = _find_cue(t, SELF_CUES)
    if cue:
        return RuleResult("reject", "personal_diagnosis", cue)

    m = COMMAND_START.search(t)
    if m:
        return RuleResult("reject", "command", m.group(1))

    cue = _find_cue(t, COMMAND_CUES)
    if cue:
        return RuleResult("reject", "command", cue)

    return RuleResult("pass")
