"""PII detection and redaction for agent tool outputs.

Detects and replaces sensitive data patterns with [REDACTED:type] tokens.
Skips content inside fenced code blocks (``` ... ```) to avoid false positives
in code, config examples, and documentation.
"""

from __future__ import annotations

import re

# ── Patterns ─────────────────────────────────────────────────────────

# Credit card numbers: 13-19 digits with optional spaces or dashes.
# Matches formats like 4111-1111-1111-1111, 4111 1111 1111 1111, 4111111111111111
_CC_RE = re.compile(
    r"\b"
    r"(?:\d[ -]?){12,18}\d"
    r"\b"
)

# Social Security Numbers: XXX-XX-XXXX
_SSN_RE = re.compile(
    r"\b"
    r"(?!000|666|9\d\d)"       # SSN area rules
    r"\d{3}-"
    r"(?!00)\d{2}-"
    r"(?!0000)\d{4}"
    r"\b"
)

# API keys with known prefixes.
# Each tuple: (compiled regex, redaction label)
_API_KEY_PATTERNS: list[tuple[re.Pattern, str]] = [
    # OpenAI / Stripe secret keys
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "api_key"),
    (re.compile(r"\bsk_live_[A-Za-z0-9_-]{20,}\b"), "api_key"),
    (re.compile(r"\bsk_test_[A-Za-z0-9_-]{20,}\b"), "api_key"),
    # Stripe publishable keys
    (re.compile(r"\bpk_live_[A-Za-z0-9_-]{20,}\b"), "api_key"),
    (re.compile(r"\bpk_test_[A-Za-z0-9_-]{20,}\b"), "api_key"),
    # GitHub tokens
    (re.compile(r"\bghp_[A-Za-z0-9]{36,}\b"), "api_key"),
    (re.compile(r"\bgho_[A-Za-z0-9]{36,}\b"), "api_key"),
    # AWS access keys: AKIA followed by 16 alphanumeric chars
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "aws_key"),
    # Slack tokens
    (re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}\b"), "api_key"),
    (re.compile(r"\bxoxp-[A-Za-z0-9-]{20,}\b"), "api_key"),
]

# Email addresses — intentionally conservative to avoid matching filenames
# like `user@v2.patch` or `name@sha256`. Requires a real-looking domain
# with at least two dot-separated labels before the TLD, OR a domain label
# of 3+ chars. This excludes patterns like `user@v2.patch` where the
# "domain" is a single short label + file extension.
#
# Common file-extension false positives are excluded via a negative lookahead
# on the final dot-segment.
_FILE_EXT_REJECT = (
    r"(?!\.(?:patch|diff|bak|tmp|log|old|orig|save|swp|swo|lock|pid|dat|bin|csv"
    r"|txt|json|xml|yaml|yml|toml|ini|cfg|conf|sql|db|md|rst|html|htm|css|js"
    r"|ts|py|rb|go|rs|c|h|cpp|hpp|java|sh|bash|zsh|fish)(?:\b|$))"
)
_EMAIL_RE = re.compile(
    r"\b"
    r"[A-Za-z0-9._%+-]+"
    r"@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"  # domain label
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*"  # sub-domains
    + _FILE_EXT_REJECT +
    r"\.[A-Za-z]{2,12}"  # TLD
    r"\b"
)

# ── Code block splitting ─────────────────────────────────────────────

# Matches fenced code blocks: ``` ... ```
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _is_plausible_cc(raw: str) -> bool:
    """Check if a digit sequence looks like a credit card number.

    Strips separators and checks length is in the 13-19 range used by
    major card networks.  Does NOT run Luhn (too many false positives
    with arbitrary numeric data); the length + context check is sufficient
    for defense-in-depth redaction.
    """
    digits = re.sub(r"[ -]", "", raw)
    return 13 <= len(digits) <= 19


def _redact_segment(text: str) -> str:
    """Apply all PII redaction patterns to a single non-code segment."""
    # Order matters: more specific patterns first to avoid partial matches.

    # API keys (before email, since some keys contain @ in URLs nearby)
    for pattern, label in _API_KEY_PATTERNS:
        text = pattern.sub(f"[REDACTED:{label}]", text)

    # SSN
    text = _SSN_RE.sub("[REDACTED:ssn]", text)

    # Credit cards — use a callback to validate digit count
    def _cc_replacer(m: re.Match) -> str:
        if _is_plausible_cc(m.group()):
            return "[REDACTED:credit_card]"
        return m.group()

    text = _CC_RE.sub(_cc_replacer, text)

    # Email addresses
    text = _EMAIL_RE.sub("[REDACTED:email]", text)

    return text


def redact_pii(text: str) -> str:
    """Replace detected PII with [REDACTED:type] tokens.

    Skips content inside fenced code blocks (``` ... ```) to reduce
    false positives in code samples, config examples, and documentation.

    Detected PII types:
    - credit_card: 13-19 digit card numbers (with optional spaces/dashes)
    - ssn: Social Security Numbers (XXX-XX-XXXX)
    - api_key: Keys with known prefixes (sk-, ghp_, xoxb-, etc.)
    - aws_key: AWS access key IDs (AKIA...)
    - email: Email addresses (conservative regex)
    """
    if not text:
        return text

    # Split text into code blocks and non-code segments.
    # We redact only the non-code segments.
    parts: list[str] = []
    last_end = 0

    for match in _CODE_FENCE_RE.finditer(text):
        # Redact the segment before this code block
        if match.start() > last_end:
            parts.append(_redact_segment(text[last_end:match.start()]))
        # Keep the code block as-is
        parts.append(match.group())
        last_end = match.end()

    # Redact the remaining text after the last code block
    if last_end < len(text):
        parts.append(_redact_segment(text[last_end:]))

    return "".join(parts)
