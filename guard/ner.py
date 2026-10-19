"""Tầng 2 của Guard: NER bằng gazetteer y khoa tiếng Việt.

Longest-match trên văn bản đã chuẩn hóa. Mỗi loại entity nạp từ một file
trong data/gazetteer/. Output là feature cho classifier tầng 3 —
KHÔNG tự mình reject claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from guard.rules import normalize

GAZETTEER_DIR = Path(__file__).resolve().parent.parent / "data" / "gazetteer"

LABEL_FILES = {
    "DISEASE": "diseases.txt",
    "DRUG": "drugs.txt",
    "SUBSTANCE": "substances.txt",
    "SYMPTOM_MEASURE": "symptoms_measures.txt",
}

# Từ khóa miền y tế tổng quát: chỉ đếm cho NER-fallback, không phải entity
MEDICAL_CUES_FILE = "medical_cues.txt"


def _load_terms(fpath: Path) -> list[str]:
    if not fpath.exists():
        return []
    terms = []
    for line in fpath.read_text(encoding="utf-8").splitlines():
        term = normalize(line.split("#", 1)[0])
        if term:
            terms.append(term)
    return terms


def _compile(terms: list[str]):
    if not terms:
        return None
    pattern = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    # \b kém ổn với tiếng Việt có dấu → lookaround
    return re.compile(rf"(?<![\wÀ-ỹ])({pattern})(?![\wÀ-ỹ])")


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int


class GazetteerNER:
    def __init__(self, gazetteer_dir: Path | str = GAZETTEER_DIR):
        gazetteer_dir = Path(gazetteer_dir)
        self.terms: list[tuple[str, str]] = []
        for label, fname in LABEL_FILES.items():
            for term in _load_terms(gazetteer_dir / fname):
                self.terms.append((term, label))

        self.terms.sort(key=lambda x: len(x[0]), reverse=True)
        self._label_of = {t: lab for t, lab in self.terms}
        self._regex = _compile([t for t, _ in self.terms])
        self._cue_regex = _compile(_load_terms(gazetteer_dir / MEDICAL_CUES_FILE))

    def extract(self, text: str) -> list[Entity]:
        if self._regex is None:
            return []
        t = normalize(text)
        entities = []
        for m in self._regex.finditer(t):
            term = m.group(1)
            entities.append(Entity(term, self._label_of[term], m.start(1), m.end(1)))
        return entities

    def features(self, text: str) -> dict:
        """Feature NER cho classifier tầng 3."""
        ents = self.extract(text)
        labels = {e.label for e in ents}
        return {
            "num_entities": len(ents),
            "has_disease": "DISEASE" in labels,
            "has_drug": "DRUG" in labels,
            "has_substance": "SUBSTANCE" in labels,
            "has_symptom_measure": "SYMPTOM_MEASURE" in labels,
            "num_medical_cues": (
                len(self._cue_regex.findall(normalize(text))) if self._cue_regex else 0
            ),
            "entities": ents,
        }
