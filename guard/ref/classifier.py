"""Tầng 3 của Guard: classifier PhoBERT fine-tuned phân loại claim.

Nhãn: factual_claim | question | personal_opinion | personal_diagnosis | other.
Feature NER được prepend vào input dạng token đặc biệt, vd:
    "[ENT=2] [DISEASE] [SUBSTANCE] ăn nhiều muối gây cao huyết áp"

Nếu chưa có model fine-tuned (models/guard_phobert chưa tồn tại),
dùng fallback heuristic dựa trên NER: có >=1 entity y khoa -> factual_claim.
"""
from pathlib import Path

LABELS = ["factual_claim", "question", "personal_opinion", "personal_diagnosis", "other"]


def build_input(text: str, ner_feats: dict) -> str:
    """Ghép feature NER vào text làm input cho classifier."""
    tags = [f"[ENT={ner_feats['num_entities']}]"]
    for key, tag in [
        ("has_disease", "[DISEASE]"),
        ("has_drug", "[DRUG]"),
        ("has_substance", "[SUBSTANCE]"),
        ("has_symptom_measure", "[SYMPTOM]"),
    ]:
        if ner_feats.get(key):
            tags.append(tag)
    return " ".join(tags) + " " + text


class GuardClassifier:
    def __init__(self, model_path: str | None, base_model: str = "vinai/phobert-base-v2"):
        self.model_path = model_path
        self.base_model = base_model
        self._pipe = None
        self.available = model_path is not None and Path(model_path).exists()

    def _load(self):
        if self._pipe is None:
            from pyvi import ViTokenizer
            from transformers import pipeline

            self._segment = ViTokenizer.tokenize  # PhoBERT cần input đã tách từ
            self._pipe = pipeline(
                "text-classification",
                model=self.model_path,
                tokenizer=self.model_path,
                top_k=None,
            )

    def predict(self, text: str, ner_feats: dict) -> tuple[str, float]:
        """Trả về (label, confidence)."""
        if self.available:
            self._load()
            inp = self._segment(build_input(text, ner_feats))
            scores = self._pipe(inp, truncation=True, max_length=256)[0]
            best = max(scores, key=lambda s: s["score"])
            return best["label"], best["score"]
        return self._fallback(ner_feats)

    @staticmethod
    def _fallback(ner_feats: dict) -> tuple[str, float]:
        """Heuristic khi chưa có model: dựa hoàn toàn vào NER.

        Rule tầng 1 đã loại question/opinion/diagnosis, nên đến đây
        chỉ cần phân biệt claim y khoa vs câu lan man ngoài domain.
        """
        n = ner_feats["num_entities"]
        if n >= 2:
            return "factual_claim", 0.85
        if n == 1:
            return "factual_claim", 0.65
        return "other", 0.6
