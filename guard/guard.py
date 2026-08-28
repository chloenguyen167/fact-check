"""Guard 3 tầng (cascade): rule-based → gazetteer NER → classifier.

Tầng sau chỉ chạy nếu tầng trước cho qua.
Reject sớm trước khi nạp mô hình nặng (embedding / NLI / LLM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from guard.classifier import GuardClassifier
from guard.ner import Entity, GazetteerNER
from guard.rules import rule_filter

REJECT_MESSAGES = {
    "question": (
        "Đây là câu hỏi, không phải một khẳng định (claim). "
        "Vui lòng diễn đạt lại thành một khẳng định, ví dụ: "
        "'Vitamin C chữa khỏi cảm cúm.'"
    ),
    "personal_opinion": (
        "Đây là cảm nhận/ý kiến cá nhân, không kiểm chứng được. "
        "Vui lòng đưa ra một khẳng định khách quan, ví dụ: "
        "'Uống trà gừng giảm nguy cơ nhồi máu cơ tim.'"
    ),
    "personal_diagnosis": (
        "Hệ thống không kiểm chứng tình trạng sức khỏe cá nhân. "
        "Vui lòng đưa ra một khẳng định y khoa tổng quát, ví dụ: "
        "'Tăng huyết áp làm tăng nguy cơ đột quỵ.'"
    ),
    "command": (
        "Đây là một yêu cầu/mệnh lệnh, không phải claim. "
        "Vui lòng nhập một khẳng định cần kiểm chứng, ví dụ: "
        "'Ăn nhiều muối gây cao huyết áp.'"
    ),
    "too_short": (
        "Câu quá ngắn để kiểm chứng. "
        "Vui lòng nhập một khẳng định đầy đủ (ít nhất 4 từ)."
    ),
    "other": (
        "Câu này không phải khẳng định y khoa thuộc phạm vi hệ thống. "
        "Vui lòng nhập một claim y khoa cụ thể, ví dụ: "
        "'Vitamin C chữa khỏi cảm cúm.'"
    ),
    "factual_claim_low_conf": (
        "Độ tin cậy phân loại claim chưa đủ để kiểm chứng. "
        "Vui lòng diễn đạt rõ hơn thành một khẳng định y khoa."
    ),
}


@dataclass
class GuardResult:
    valid: bool
    stage: str | None = None
    reason: str | None = None
    message: str = ""
    confidence: float | None = None
    entities: list[Entity] = field(default_factory=list)
    matched_cue: str | None = None


class Guard:
    """Singleton Guard — cascade 3 tầng, không kéo embedding/NLI/LLM."""

    def __init__(
        self,
        classifier_path: str | None = None,
        base_model: str = "vinai/phobert-base-v2",
        threshold: float = 0.5,
        gazetteer_dir: str | None = None,
        ner_assist_min_conf: float = 0.35,
        ner_assist_min_entities: int = 1,
    ):
        if gazetteer_dir:
            self.ner = GazetteerNER(gazetteer_dir)
        else:
            self.ner = GazetteerNER()
        self.classifier = GuardClassifier(
            model_path=classifier_path,
            base_model=base_model,
        )
        self.threshold = threshold
        self.ner_assist_min_conf = ner_assist_min_conf
        self.ner_assist_min_entities = ner_assist_min_entities

    def _passes_classifier(
        self, label: str, prob: float, feats: dict[str, Any]
    ) -> bool:
        if label != "factual_claim":
            return False
        if prob >= self.threshold:
            return True
        return (
            feats["num_entities"] >= self.ner_assist_min_entities
            and prob >= self.ner_assist_min_conf
        )

    def check(self, text: str) -> GuardResult:
        # Tầng 1: rules
        rr = rule_filter(text)
        if rr.verdict == "reject":
            return GuardResult(
                valid=False,
                stage="rule",
                reason=rr.reason,
                message=REJECT_MESSAGES.get(
                    rr.reason or "", REJECT_MESSAGES["other"]
                ),
                matched_cue=rr.matched,
            )

        # Tầng 2: NER → feature (không reject một mình)
        feats = self.ner.features(text)

        # Tầng 3: classifier quyết định cuối
        label, prob = self.classifier.predict(text, feats)
        if self._passes_classifier(label, prob, feats):
            return GuardResult(
                valid=True,
                stage="classifier",
                confidence=prob,
                entities=feats["entities"],
            )

        reason = label if label != "factual_claim" else "factual_claim_low_conf"
        return GuardResult(
            valid=False,
            stage="classifier",
            reason=reason,
            message=REJECT_MESSAGES.get(reason, REJECT_MESSAGES["other"]),
            confidence=prob,
            entities=feats["entities"],
        )


@lru_cache(maxsize=1)
def get_guard() -> Guard:
    try:
        import config as cfg

        return Guard(
            classifier_path=getattr(cfg, "GUARD_CLASSIFIER_PATH", None),
            base_model=getattr(cfg, "GUARD_BASE_MODEL", "vinai/phobert-base-v2"),
            threshold=float(getattr(cfg, "GUARD_THRESHOLD", 0.5)),
            gazetteer_dir=getattr(cfg, "GUARD_GAZETTEER_DIR", None),
            ner_assist_min_conf=float(
                getattr(cfg, "GUARD_NER_ASSIST_MIN_CONF", 0.35)
            ),
            ner_assist_min_entities=int(
                getattr(cfg, "GUARD_NER_ASSIST_MIN_ENTITIES", 1)
            ),
        )
    except Exception:  # noqa: BLE001 — fallback cấu hình mặc định
        return Guard()


def check_claim(text: str) -> GuardResult:
    """API tiện dụng: kiểm tra claim qua Guard singleton."""
    return get_guard().check(text)
