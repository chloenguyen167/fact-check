import re

from evaluation.labels import VERDICTS, normalize_verdict
from evaluation.llm_utils import chat_completion


def parse_verdict_from_llm(text):
    normalized = normalize_verdict(text)
    if normalized:
        return normalized

    upper = str(text).upper()
    for verdict in VERDICTS:
        if verdict in upper:
            return verdict

    # Fallback: tìm token gần nhất trong câu trả lời
    if re.search(r"\bSAI\b", upper):
        return "SAI"
    if re.search(r"\bĐÚNG\b|\bDUNG\b", upper):
        return "ĐÚNG"

    return "KHÔNG ĐỦ BẰNG CHỨNG"


def classify_verdict_llm(claim, evidence=None, provider="auto"):
    """Phân loại claim bằng LLM (không qua NLI)."""
    if evidence:
        prompt = f"""
Bạn là hệ thống kiểm chứng thông tin y tế tiếng Việt.

Dựa trên bằng chứng dưới đây, phân loại claim thành MỘT trong ba nhãn:
- ĐÚNG: bằng chứng trực tiếp ủng hộ claim
- SAI: bằng chứng trực tiếp mâu thuẫn claim
- KHÔNG ĐỦ BẰNG CHỨNG: bằng chứng không đủ hoặc chỉ liên quan chủ đề

Claim:
{claim}

Bằng chứng:
{evidence}

Chỉ trả lời đúng một nhãn: ĐÚNG, SAI, hoặc KHÔNG ĐỦ BẰNG CHỨNG.
"""
    else:
        prompt = f"""
Bạn là hệ thống kiểm chứng thông tin y tế tiếng Việt.

Không có tài liệu tham chiếu. Dựa trên kiến thức y khoa phổ biến, phân loại claim:
- ĐÚNG
- SAI
- KHÔNG ĐỦ BẰNG CHỨNG

Claim:
{claim}

Chỉ trả lời đúng một nhãn: ĐÚNG, SAI, hoặc KHÔNG ĐỦ BẰNG CHỨNG.
"""

    raw = chat_completion(
        prompt,
        temperature=0,
        max_tokens=30,
        provider=provider,
    )
    return parse_verdict_from_llm(raw), raw.strip()


def explain(claim, evidence, verdict):
    prompt = f"""
Bạn là chuyên gia y khoa.

Nhiệm vụ:
Giải thích kết quả kiểm chứng thông tin y tế.

Claim:
{claim}

Verdict:
{verdict}

Evidence:
{evidence}

Yêu cầu:
- Trả lời bằng tiếng Việt.
- Giải thích ngắn gọn (3-5 câu).
- Nêu lý do vì sao kết luận như vậy.
- Dễ hiểu với người dân.
- Không viết phần suy nghĩ nội bộ; chỉ viết giải thích cuối cùng.
"""
    text = chat_completion(
        prompt,
        temperature=0.2,
        max_tokens=500,
        provider="auto",
        request_delay=0,
    )
    return (text or "").strip()
