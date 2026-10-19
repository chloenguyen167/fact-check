from config import TOP_RERANK, VERDICT_STRATEGY
from retrieval import retrieve_with_metadata
from reranker import rerank_records, score_texts
from nli import predict_nli_ess, predict_nli_multi
from llm import classify_verdict_llm, explain
from claim_cache import get_cached, save_cached


def _enrich_link(doc: dict) -> dict:
    """Bổ sung link gốc nếu payload Qdrant thiếu (bài scrape_* còn trong SQLite)."""
    if doc.get("link"):
        return doc
    article_id = str(doc.get("article_id") or "")
    if not article_id.startswith("scrape_"):
        return doc
    try:
        from scraper import db

        db_id = int(article_id.removeprefix("scrape_"))
        row = db.get_article(db_id)
        if row and row.get("link"):
            doc = {**doc, "link": row["link"]}
    except Exception:  # noqa: BLE001 — không chặn pipeline vì lookup phụ
        pass
    return doc


def _build_evidence_bundle(claim):
    records = retrieve_with_metadata(claim)
    # top_n bắt buộc: mặc định rerank_records trả về toàn bộ 20 chunk
    ranked = rerank_records(claim, records, top_n=TOP_RERANK)
    evidence_docs = [_enrich_link(doc) for doc in ranked]
    evidence = "\n\n".join(doc["text"] for doc in evidence_docs)
    return evidence, evidence_docs


def verify_claim(claim, with_explanation=True, use_cache=True):
    if use_cache:
        cached = get_cached(claim)
        if cached is not None:
            return cached

    evidence, evidence_docs = _build_evidence_bundle(claim)

    key_sentences = []
    if VERDICT_STRATEGY == "ess":
        verdict, verdict_score, details = predict_nli_ess(claim, evidence_docs, score_texts)
        key_sentences = details.get("sentences", [])
    else:
        top_docs = [doc["text"] for doc in evidence_docs]
        verdict, verdict_score, _ = predict_nli_multi(
            claim, top_docs, strategy="weighted_max"
        )

    explanation = ""
    if with_explanation:
        explanation = explain(claim, evidence, verdict)

    result = {
        "claim": claim,
        "verdict": verdict,
        "verdict_score": float(verdict_score),
        "evidence": evidence,
        "evidence_docs": evidence_docs,
        "key_sentences": key_sentences,
        "explanation": explanation,
        "from_cache": False,
    }
    if use_cache:
        save_cached(claim, result)
    return result


def verify_claim_llm_only(claim, with_explanation=False):
    verdict, raw = classify_verdict_llm(claim)

    explanation = ""
    if with_explanation:
        explanation = explain(claim, "", verdict)

    return {
        "claim": claim,
        "verdict": verdict,
        "verdict_score": None,
        "llm_raw": raw,
        "evidence": "",
        "evidence_docs": [],
        "explanation": explanation,
        "method": "llm_only",
    }


def verify_claim_rag_llm(claim, with_explanation=False):
    evidence, evidence_docs = _build_evidence_bundle(claim)
    verdict, raw = classify_verdict_llm(claim, evidence=evidence)

    explanation = ""
    if with_explanation:
        explanation = explain(claim, evidence, verdict)

    return {
        "claim": claim,
        "verdict": verdict,
        "verdict_score": None,
        "llm_raw": raw,
        "evidence": evidence,
        "evidence_docs": evidence_docs,
        "explanation": explanation,
        "method": "rag_llm",
    }


if __name__ == "__main__":
    result = verify_claim("uống nước chanh chữa ung thư")
    print(f"Verdict: {result['verdict']} ({result['verdict_score']:.3f})")
    print(f"Evidence preview: {result['evidence'][:300]}...")
    print(f"Explanation: {result['explanation'][:300]}...")