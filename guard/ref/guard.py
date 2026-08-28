"""Guard 3 tầng: rule-based -> gazetteer NER -> classifier fine-tuned.

Tầng 1 reject sớm các mẫu chắc chắn sai (câu hỏi, cảm nhận, chẩn đoán cá nhân).
Tầng 2 trích entity y khoa làm tín hiệu + feature.
Tầng 3 classifier quyết định cuối với input = text + feature NER.
"""
from dataclasses import dataclass, field

from .classifier import GuardClassifier
from .ner import Entity, GazetteerNER
from .rules import rule_filter

REJECT_MESSAGES = {
    "question": "Đây là câu hỏi, không phải một khẳng định (claim). "
                "Vui lòng diễn đạt lại thành một khẳng định, vd: 'Vitamin C chữa khỏi cảm cúm.'",
    "personal_opinion": "Đây là cảm nhận/ý kiến cá nhân, không kiểm chứng được. "
                        "Vui lòng đưa ra một khẳng định khách quan.",
    "personal_diagnosis": "Hệ thống không kiểm chứng tình trạng sức khỏe cá nhân. "
                          "Vui lòng đưa ra một khẳng định y khoa tổng quát.",
    "command": "Đây là một yêu cầu/mệnh lệnh, không phải claim. "
               "Vui lòng nhập một khẳng định cần kiểm chứng.",
    "too_short": "Câu quá ngắn để kiểm chứng. Vui lòng nhập một khẳng định đầy đủ.",
    "other": "Câu này không phải khẳng định y khoa thuộc phạm vi tim mạch. "
             "Vui lòng nhập một claim y khoa cụ thể.",
}


@dataclass
class GuardResult:
    valid: bool
    stage: str                       # rule | classifier
    reason: str | None = None        # nhãn lý do khi reject
    message: str | None = None       # thông báo cho người dùng
    confidence: float | None = None
    entities: list[Entity] = field(default_factory=list)
    matched_cue: str | None = None   # cue rule khớp (debug)


class Guard:
    def __init__(self, cfg: dict):
        gcfg = cfg["guard"]
        self.ner = GazetteerNER()
        self.classifier = GuardClassifier(
            model_path=gcfg.get("classifier_path"),
            base_model=gcfg.get("base_model", "vinai/phobert-base-v2"),
        )
        self.threshold = gcfg.get("threshold", 0.5)

    def check(self, text: str) -> GuardResult:
        # Tầng 1: rules
        rr = rule_filter(text)
        if rr.verdict == "reject":
            return GuardResult(
                valid=False, stage="rule", reason=rr.reason,
                message=REJECT_MESSAGES.get(rr.reason, REJECT_MESSAGES["other"]),
                matched_cue=rr.matched,
            )

        # Tầng 2: NER -> feature (không reject một mình)
        feats = self.ner.features(text)

        # Tầng 3: classifier quyết định cuối
        label, prob = self.classifier.predict(text, feats)
        if label == "factual_claim" and prob >= self.threshold:
            return GuardResult(
                valid=True, stage="classifier", confidence=prob,
                entities=feats["entities"],
            )
        return GuardResult(
            valid=False, stage="classifier", reason=label,
            message=REJECT_MESSAGES.get(label, REJECT_MESSAGES["other"]),
            confidence=prob, entities=feats["entities"],
        )
