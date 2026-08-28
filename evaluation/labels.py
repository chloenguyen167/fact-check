"""Ánh xạ nhãn giữa các bộ dữ liệu và hệ thống."""

VERDICT_DUNG = "ĐÚNG"
VERDICT_SAI = "SAI"
VERDICT_KHONG_DU = "KHÔNG ĐỦ BẰNG CHỨNG"

VERDICTS = [VERDICT_DUNG, VERDICT_SAI, VERDICT_KHONG_DU]

PUBHEALTH_TO_VERDICT = {
    "true": VERDICT_DUNG,
    "false": VERDICT_SAI,
    "unproven": VERDICT_KHONG_DU,
    "mixture": VERDICT_KHONG_DU,
}

# ViFactCheck: 0=Supported, 1=Refuted, 2=NEI
VIFACTCHECK_ID_TO_LABEL = {
    0: "true",
    1: "false",
    2: "unproven",
}

VIFACTCHECK_TO_VERDICT = {
    0: VERDICT_DUNG,
    1: VERDICT_SAI,
    2: VERDICT_KHONG_DU,
    "0": VERDICT_DUNG,
    "1": VERDICT_SAI,
    "2": VERDICT_KHONG_DU,
    "supported": VERDICT_DUNG,
    "refuted": VERDICT_SAI,
    "nei": VERDICT_KHONG_DU,
}

VIET_MEDFACT_TO_VERDICT = {
    "dung": VERDICT_DUNG,
    "đúng": VERDICT_DUNG,
    "true": VERDICT_DUNG,
    "sai": VERDICT_SAI,
    "false": VERDICT_SAI,
    "khong_du": VERDICT_KHONG_DU,
    "không đủ bằng chứng": VERDICT_KHONG_DU,
    "khong du bang chung": VERDICT_KHONG_DU,
    "unproven": VERDICT_KHONG_DU,
    "neutral": VERDICT_KHONG_DU,
}


def normalize_verdict(label):
    if label is None or label == "":
        return None

    if isinstance(label, int):
        return VIFACTCHECK_TO_VERDICT.get(label)

    text = str(label).strip()
    upper = text.upper()

    if upper in VERDICTS:
        return upper

    if text in VIFACTCHECK_TO_VERDICT:
        return VIFACTCHECK_TO_VERDICT[text]

    key = text.lower()
    if key in VIET_MEDFACT_TO_VERDICT:
        return VIET_MEDFACT_TO_VERDICT[key]

    if key in PUBHEALTH_TO_VERDICT:
        return PUBHEALTH_TO_VERDICT[key]

    if key in VIFACTCHECK_TO_VERDICT:
        return VIFACTCHECK_TO_VERDICT[key]

    return None


def pubhealth_label_to_verdict(label):
    """Ánh xạ nhãn Dataset B (PUBHEALTH / ViFactCheck-Health) → verdict hệ thống."""
    return normalize_verdict(label)


def dataset_b_path(name=None):
    """Đường dẫn Dataset B theo tên nguồn."""
    from evaluation.config_eval import (
        DATASET_B_DEFAULT,
        PUBHEALTH_VI_PATH,
        VIFACTCHECK_HEALTH_PATH,
    )

    key = (name or DATASET_B_DEFAULT).lower()
    if key in ("vifactcheck", "vifactcheck_health", "vi"):
        return VIFACTCHECK_HEALTH_PATH
    if key in ("pubhealth", "pubhealth_vi", "b"):
        return PUBHEALTH_VI_PATH
    raise ValueError(f"Dataset B không hợp lệ: {name!r} (dùng vifactcheck|pubhealth)")
