"""Tầng 1 của Guard: rule-based filter cho claim tiếng Việt.

Loại sớm các input chắc chắn KHÔNG phải factual claim:
câu hỏi, cảm nhận cá nhân, chẩn đoán cá nhân, mệnh lệnh/yêu cầu, quá ngắn.
Trả về (verdict, reason): verdict "reject" hoặc "pass".
"""
import re
import unicodedata
from dataclasses import dataclass


def normalize(text: str) -> str:
    """Chuẩn hóa unicode NFC, lower, gọn khoảng trắng."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


# Từ/cụm để hỏi — xuất hiện ở đầu hoặc bất kỳ vị trí
QUESTION_CUES = [
    "có phải", "liệu rằng", "liệu có", "tại sao", "vì sao", "bao nhiêu",
    "như thế nào", "thế nào", "là gì", "có nên", "nên không", "được không",
    "có đúng không", "đúng không", "phải không", "hay không", "không ạ",
    "cho hỏi", "cho mình hỏi", "cho em hỏi", "xin hỏi",
]
QUESTION_END = re.compile(r"\?\s*$")

# Cảm nhận / ý kiến chủ quan
OPINION_CUES = [
    "tôi thấy", "mình thấy", "em thấy", "tôi nghĩ", "mình nghĩ", "em nghĩ",
    "theo tôi", "theo mình", "theo em", "tôi cảm thấy", "mình cảm thấy",
    "hình như", "có vẻ", "cảm giác", "tôi tin", "mình tin", "chắc là",
    "tôi cho rằng", "mình cho rằng",
]

# Chẩn đoán / tình trạng cá nhân
SELF_CUES = [
    "tôi bị", "mình bị", "em bị", "tôi đang bị", "mình đang bị",
    "tôi mắc", "mình mắc", "bác sĩ bảo tôi", "bác sĩ nói tôi",
    "kết quả của tôi", "xét nghiệm của tôi", "bệnh của tôi",
    "mẹ tôi bị", "bố tôi bị", "người nhà tôi",
]

# Mệnh lệnh / yêu cầu trợ giúp
COMMAND_CUES = [
    "hãy ", "giúp tôi", "giúp mình", "giúp em", "tư vấn", "khuyên tôi",
    "chỉ cho tôi", "hướng dẫn", "làm ơn", "vui lòng",
]

MIN_TOKENS = 4


@dataclass
class RuleResult:
    verdict: str          # "pass" | "reject"
    reason: str | None    # question | personal_opinion | personal_diagnosis | command | too_short
    matched: str | None   # cue khớp, để debug/giải thích


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
    if cue := _find_cue(t, QUESTION_CUES):
        return RuleResult("reject", "question", cue)

    if cue := _find_cue(t, OPINION_CUES):
        return RuleResult("reject", "personal_opinion", cue)

    if cue := _find_cue(t, SELF_CUES):
        return RuleResult("reject", "personal_diagnosis", cue)

    if cue := _find_cue(t, COMMAND_CUES):
        return RuleResult("reject", "command", cue)

    return RuleResult("pass", None, None)
