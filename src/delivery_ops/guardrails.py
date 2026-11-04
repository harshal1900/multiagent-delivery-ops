"""Prompt-injection and noise-detection guardrails, plus the dedup and
event-consolidation helpers the preprocessor node runs before any LLM call.
"""

INJECTION_KEYWORDS = [
    "ignore previous",
    "ignore all previous",
    "ignore the above",
    "disregard the",
    "forget your instructions",
    "system prompt",
    "you are now",
    "new instructions",
    "reveal your",
    "jailbreak",
]

ROUTINE_STATUS_CODES = {"DELIVERED", "IN_TRANSIT", "OUT_FOR_DELIVERY", "SCANNED"}

NOISE_ANOMALY_INDICATORS = [
    "damage", "wrong", "suspicious", "overdue", "missing",
    "unexpected", "misroute", "lost", "stolen", "abandoned",
    "leak", "crush", "broke", "delay", "late", "fraud",
]


def scan_for_injection(text: str) -> bool:
    """Returns True if prompt injection keywords are detected in the text."""
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in INJECTION_KEYWORDS)


def scan_inputs_for_injection(consolidated: dict, raw_rows: list[dict]) -> bool:
    """Scan all free-text fields in delivery data for prompt injection."""
    texts = [consolidated["status_description"]]
    texts.extend(row.get("status_description", "") for row in raw_rows)
    return any(scan_for_injection(text) for text in texts)


def scan_chunks_for_injection(playbook_context: list[dict]) -> bool:
    """Scan retrieved RAG chunks for prompt injection."""
    return any(scan_for_injection(chunk.get("content", "")) for chunk in playbook_context)


def check_noise_override(consolidated: dict) -> bool:
    """Flag routine status codes with no anomaly indicators."""
    if consolidated["status_code"] not in ROUTINE_STATUS_CODES:
        return False
    desc = consolidated["status_description"].lower()
    return not any(indicator in desc for indicator in NOISE_ANOMALY_INDICATORS)


def deduplicate_rows(raw_rows: list[dict]) -> list[dict]:
    """Remove duplicate scan events."""
    return [r for r in raw_rows if r.get("is_duplicate_scan", "False") != "True"]


def consolidate_event(unique_rows: list[dict], raw_rows: list[dict]) -> dict:
    """Consolidate a multi-row shipment into a single event using the
    highest attempt number, carrying forward notes from earlier attempts."""
    if unique_rows:
        primary = max(unique_rows, key=lambda r: int(r.get("attempt_number", 0)))
    else:
        primary = raw_rows[0]

    prior_notes = [
        f"Attempt {r['attempt_number']}: {r['status_description']}"
        for r in unique_rows
        if r is not primary
    ]

    return {
        "shipment_id": primary["shipment_id"],
        "timestamp": primary["timestamp"],
        "status_code": primary["status_code"],
        "status_description": primary["status_description"],
        "customer_id": primary["customer_id"],
        "delivery_address": primary["delivery_address"],
        "package_type": primary["package_type"],
        "package_size": primary["package_size"],
        "attempt_number": int(primary["attempt_number"]),
        "prior_attempt_notes": prior_notes,
        "total_rows": len(raw_rows),
        "duplicates_removed": len(raw_rows) - len(unique_rows),
    }
