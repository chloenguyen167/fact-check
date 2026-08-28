import os

# Thư mục dữ liệu và kết quả đánh giá
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EVAL_DIR, "data")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

VIET_MEDFACT_PATH = os.path.join(DATA_DIR, "viet_medfact.jsonl")
PUBHEALTH_VI_PATH = os.path.join(DATA_DIR, "pubhealth_vi.jsonl")
PUBHEALTH_RAW_PATH = os.path.join(DATA_DIR, "pubhealth_raw.jsonl")
VIFACTCHECK_HEALTH_PATH = os.path.join(DATA_DIR, "vifactcheck_health.jsonl")
VIFACTCHECK_BASELINE_RESULTS = os.path.join(
    RESULTS_DIR, "baseline_comparison_vifactcheck.json"
)

# Kích thước mẫu (theo từng nguồn — đảm bảo có cả SKDS và UMC)
VIET_MEDFACT_NUM_ARTICLES = 100
VIET_MEDFACT_NUM_SKDS = 50
VIET_MEDFACT_NUM_UMC = 50
VIET_MEDFACT_CLAIMS_PER_ARTICLE = 3

PUBHEALTH_SAMPLE_SIZE = 500
PUBHEALTH_LABELS = ["true", "false", "unproven"]

# Retrieval / Rerank
RETRIEVAL_K_VALUES = [5, 10, 20]
RERANK_K_VALUES = [1, 3]

# Hugging Face datasets
HEALTH_FACT_DATASET = "ImperialCollegeLondon/health_fact"
VIFACTCHECK_DATASET = "tranthaihoa/vifactcheck"
# Topic sức khỏe / y tế trong ViFactCheck (chuỗi gốc có nhiều biến thể hoa/thường)
VIFACTCHECK_HEALTH_TOPICS = ("sức khỏe", "sức khoẻ", "y tế")
# Dataset B mặc định cho NLI/LLM: "vifactcheck" | "pubhealth"
DATASET_B_DEFAULT = "vifactcheck"

# Giới hạn khi chạy thử nhanh (đặt None để dùng toàn bộ)
EVAL_MAX_SAMPLES = None

# LLM — tránh rate limit OpenRouter free tier (không áp dụng Ollama local)
LLM_REQUEST_DELAY = 0
LLM_MAX_RETRIES = 6

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
