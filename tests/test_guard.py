from guard import check_claim, get_guard
from guard.ner import GazetteerNER
from guard.rules import rule_filter


def test_pass_factual_claim():
    r = check_claim("Vitamin C chữa khỏi cảm cúm")
    assert r.valid is True
    assert r.stage == "classifier"
    assert r.confidence is not None


def test_pass_claim_with_entities():
    r = check_claim("Ăn nhiều muối gây cao huyết áp")
    assert r.valid is True
    labels = {e.label for e in r.entities}
    assert "SUBSTANCE" in labels or "DISEASE" in labels or "SYMPTOM_MEASURE" in labels


def test_reject_question_mark():
    r = check_claim("Vitamin C có chữa khỏi cảm cúm không?")
    assert r.valid is False
    assert r.stage == "rule"
    assert r.reason == "question"


def test_reject_question_cue():
    r = check_claim("Cho em hỏi vitamin C có chữa cảm cúm")
    assert r.valid is False
    assert r.reason == "question"


def test_reject_opinion():
    r = check_claim("Theo tôi uống trà gừng giảm nhồi máu cơ tim")
    assert r.valid is False
    assert r.reason == "personal_opinion"


def test_reject_personal_diagnosis():
    r = check_claim("Tôi bị tăng huyết áp và hay đau đầu")
    assert r.valid is False
    assert r.reason == "personal_diagnosis"


def test_reject_command():
    r = check_claim("Giúp tôi tư vấn cách hạ huyết áp nhanh")
    assert r.valid is False
    assert r.reason == "command"


def test_reject_too_short():
    r = check_claim("muối huyết áp")
    assert r.valid is False
    assert r.reason == "too_short"


def test_pass_claim_with_negation():
    r = check_claim("ăn ngọt không gây mỡ máu")
    assert r.valid is True
    assert r.stage == "classifier"


def test_pass_herbal_tea_claim():
    r = check_claim("trà táo tàu hỗ trợ ngủ ngon")
    assert r.valid is True
    assert r.stage == "classifier"


def test_reject_off_topic_classifier():
    r = check_claim("Hôm nay trời đẹp và mọi người vui vẻ")
    assert r.valid is False
    assert r.stage == "classifier"
    assert r.reason == "other"


def test_reject_message_guides_user():
    r = check_claim("Tại sao uống nhiều muối gây cao huyết áp?")
    assert r.message
    assert "khẳng định" in r.message.lower() or "Vitamin C" in r.message


def test_ner_longest_match():
    ner = GazetteerNER()
    ents = ner.extract("Stress kéo dài gây nhồi máu cơ tim")
    texts = [e.text for e in ents]
    assert "nhồi máu cơ tim" in texts
    assert "cơ tim" not in texts  # không bị tách ngắn hơn


def test_rule_pass_then_guard_decides():
    rr = rule_filter("Ăn nhiều muối gây cao huyết áp")
    assert rr.verdict == "pass"
    g = get_guard().check("Ăn nhiều muối gây cao huyết áp")
    assert g.valid is True
