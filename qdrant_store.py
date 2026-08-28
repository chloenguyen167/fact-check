import atexit
import os
import time
import threading
from contextlib import contextmanager

from qdrant_client import QdrantClient

from config import QDRANT_PATH

_client = None
_lock = threading.RLock()

# Local Qdrant chỉ cho 1 process. Chờ process khác (eval_e2e / streamlit) nhả lock.
QDRANT_LOCK_WAIT_SEC = int(os.environ.get("QDRANT_LOCK_WAIT_SEC", "1800"))
QDRANT_LOCK_RETRY_SEC = float(os.environ.get("QDRANT_LOCK_RETRY_SEC", "5"))


@contextmanager
def qdrant_lock():
    """Serialize local-mode Qdrant access across threads."""
    with _lock:
        yield


def _is_lock_error(exc):
    text = str(exc).lower()
    return (
        "already accessed" in text
        or "alreadylocked" in text
        or "resource temporarily unavailable" in text
    )


def _lock_help_message():
    return (
        f"Storage folder {QDRANT_PATH} đang bị process khác giữ lock "
        "(thường là streamlit / eval_e2e / ingest chạy song song).\n"
        "Qdrant local không hỗ trợ 2 client cùng lúc. Cách xử lý:\n"
        "  1) Đợi process kia chạy xong, hoặc\n"
        "  2) Tắt process đang giữ lock, rồi chạy lại.\n"
        "Kiểm tra: lsof +D ./qdrant_db | head\n"
        "Chỉ chạy baselines phần cần Qdrant sau khi rảnh:\n"
        "  python -m evaluation.eval_baselines --methods rag_llm rag_nli --max-samples 100"
    )


def get_qdrant_client(wait_sec=None, retry_sec=None):
    """
    Lấy singleton Qdrant client (local file mode).

    Nếu folder đang bị process khác lock, chờ tới `wait_sec` rồi mới báo lỗi.
    """
    global _client
    wait_sec = QDRANT_LOCK_WAIT_SEC if wait_sec is None else wait_sec
    retry_sec = QDRANT_LOCK_RETRY_SEC if retry_sec is None else retry_sec

    with _lock:
        if _client is not None:
            return _client

        deadline = time.monotonic() + max(0, wait_sec)
        attempt = 0
        last_exc = None

        while True:
            try:
                _client = QdrantClient(path=QDRANT_PATH)
                if attempt > 0:
                    print("[qdrant_store] Đã lấy được lock Qdrant.")
                return _client
            except Exception as exc:
                last_exc = exc
                if not _is_lock_error(exc):
                    raise

                attempt += 1
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(_lock_help_message()) from last_exc

                sleep_for = min(retry_sec, max(1.0, remaining))
                if attempt == 1 or attempt % 6 == 0:
                    print(
                        f"[qdrant_store] Qdrant đang bị process khác khóa — "
                        f"chờ {sleep_for:.0f}s rồi thử lại "
                        f"(còn ~{remaining:.0f}s; attempt {attempt})..."
                    )
                time.sleep(sleep_for)


def close_qdrant_client():
    global _client
    with _lock:
        if _client is None:
            return
        try:
            _client.close()
        except Exception:
            pass
        _client = None


atexit.register(close_qdrant_client)
