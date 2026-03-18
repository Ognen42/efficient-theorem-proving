"""Canonical evaluation protocol configuration (v2)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

PROTOCOL_NAME = "kimina_eval_v2"
PROTOCOL_VERSION = "2.0.0"
BASELINE_COMPAT_PROTOCOL_NAME = "kimina_eval_baseline_compat"
BASELINE_COMPAT_PROTOCOL_VERSION = "1.0.0"
NO_REASONING_PROTOCOL_NAME = "kimina_eval_no_reasoning"
NO_REASONING_PROTOCOL_VERSION = "1.0.0"

# Canonical defaults
DEFAULT_PASS_K = 4
KIMINA_TIMEOUT_SEC = 300
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 8096
DEFAULT_N_SAMPLES = 4
DEFAULT_KIMINA_URL = "http://localhost:8000"


def protocol_metadata(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build a protocol metadata record for persisted run outputs."""
    metadata: Dict[str, Any] = {
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "canonical_defaults": {
            "pass_k": DEFAULT_PASS_K,
            "kimina_timeout_sec": KIMINA_TIMEOUT_SEC,
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "n_samples": DEFAULT_N_SAMPLES,
            "kimina_url": DEFAULT_KIMINA_URL,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        metadata.update(extra)
    return metadata
