"""Tests cho claim cache."""

import time

import claim_cache as cc


def test_cache_roundtrip(tmp_path):
    db = tmp_path / "cache.db"
    claim = "Ăn nhiều muối gây cao huyết áp"
    result = {
        "claim": claim,
        "verdict": "ĐÚNG",
        "verdict_score": 0.82,
        "evidence": "ev",
        "evidence_docs": [{"text": "t", "score": 0.9}],
        "explanation": "giải thích",
    }
    cc.save_cached(claim, result, db_path=db)
    hit = cc.get_cached(claim, db_path=db)
    assert hit is not None
    assert hit["from_cache"] is True
    assert hit["verdict"] == "ĐÚNG"


def test_same_claim_different_spacing(tmp_path):
    db = tmp_path / "cache.db"
    cc.save_cached(
        "Ăn   nhiều muối",
        {
            "claim": "x",
            "verdict": "SAI",
            "verdict_score": 0.5,
            "evidence": "",
            "evidence_docs": [],
            "explanation": "",
        },
        db_path=db,
    )
    hit = cc.get_cached("ăn nhiều muối", db_path=db)
    assert hit is not None
    assert hit["verdict"] == "SAI"


def test_ttl_expiry(tmp_path, monkeypatch):
    db = tmp_path / "cache.db"
    monkeypatch.setattr(cc, "_ttl_seconds", lambda: 1)

    cc.save_cached(
        "claim hết hạn",
        {
            "claim": "c",
            "verdict": "ĐÚNG",
            "verdict_score": 0.5,
            "evidence": "",
            "evidence_docs": [],
            "explanation": "",
        },
        db_path=db,
    )
    time.sleep(1.1)
    assert cc.get_cached("claim hết hạn", db_path=db) is None


def test_clear_cache(tmp_path):
    db = tmp_path / "cache.db"
    cc.save_cached(
        "test",
        {
            "claim": "t",
            "verdict": "ĐÚNG",
            "verdict_score": 0.5,
            "evidence": "",
            "evidence_docs": [],
            "explanation": "",
        },
        db_path=db,
    )
    n = cc.clear_cache(db_path=db)
    assert n == 1
    assert cc.get_cached("test", db_path=db) is None
