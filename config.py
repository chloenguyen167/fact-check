# config.py
import os
from pathlib import Path

def _load_dotenv():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# Vector DB: Qdrant local file (embedded)
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "medical_articles"

EMBEDDING_MODEL = "BAAI/bge-m3"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

TOP_K = 20

TOP_RERANK = 3

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# OpenRouter

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

OPENROUTER_MODEL = "qwen/qwen3-next-80b-a3b-instruct:free"

# Danh sách model dự phòng: nếu model bị nghẽn (429) sẽ tự thử model kế tiếp
OPENROUTER_MODELS = [
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-120b:free",
    "z-ai/glm-4.5-air:free",
]

# Ollama (local, miễn phí — ưu tiên khi provider=auto)
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
# qwen3:4b hay trả content rỗng (thinking chiếm hết token) → ưu tiên 8b
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_ENABLED = True

# Guard 3 tầng (rule → NER → classifier). classifier_path=None hoặc chưa train → fallback NER.
GUARD_CLASSIFIER_PATH = "models/guard_phobert"  # train: python scripts/train_guard_classifier.py
GUARD_BASE_MODEL = "vinai/phobert-base-v2"
GUARD_THRESHOLD = 0.5
GUARD_NER_ASSIST_MIN_CONF = 0.35  # factual_claim + có entity y khoa
GUARD_NER_ASSIST_MIN_ENTITIES = 1
GUARD_GAZETTEER_DIR = "data/gazetteer"

# Bump khi đổi corpus/model pipeline để vô hiệu cache cũ
CLAIM_CACHE_VERSION = "2"
CLAIM_CACHE_TTL_SECONDS = 2 * 60 * 60  # 2 giờ

# Bộ ra kết luận (xem evaluation/ablation_cache.py + ablation_analyze.py)
#   "ess"         : đề xuất — weighted_max mức chunk quyết định có kết luận hay không,
#                   top-m câu bằng chứng (ESS) quyết định hướng ĐÚNG/SAI
#   "weighted_max": pipeline gốc (baseline)
VERDICT_STRATEGY = "ess"
ESS_TOP_M = 5  # chọn trên tập dev Viet-MedFact (None = mọi câu)
