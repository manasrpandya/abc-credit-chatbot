"""Masking. Runs on anything that leaves the session: logs and text sent to the llm."""
import hashlib
import hmac
import os
import re

patterns = [
    (re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b"), "[pan]"),
    (re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"), "[aadhaar]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{9}(?!\d)"), "[phone]"),
]
# "my name is ravi kumar" style only, plain names can't be spotted reliably without context
name_hint = re.compile(r"\b(my name is|i am called|this is|naam)\s+([a-z]+(?: [a-z]+){0,2})", re.I)
six_digits = re.compile(r"(?<!\d)(\d{3})\d{3}(?!\d)")


def mask_text(text):
    if not text:
        return ""
    out = text
    for rx, label in patterns:
        out = rx.sub(label, out)
    out = six_digits.sub(r"\1***", out)
    return name_hint.sub(lambda m: m.group(1) + " [name]", out)


def mask_for_llm(text):
    """Same as mask_text except a six digit pincode survives, it's needed to fill the form
    and on its own doesn't identify anybody."""
    if not text:
        return ""
    out = text
    for rx, label in patterns:
        out = rx.sub(label, out)
    return name_hint.sub(lambda m: m.group(1) + " [name]", out)


def mask_pincode(pin):
    pin = str(pin or "")
    return pin[:3] + "***" if len(pin) == 6 else ""


def mask_phone(phone):
    digits = re.sub(r"\D", "", str(phone or ""))
    return digits[:2] + "******" + digits[-2:] if len(digits) >= 10 else ""


def token(value, secret=None):
    """Keyed hash, so the same number links across a session but can't be reversed."""
    key = (secret or os.environ.get("STATE_SECRET", "dev-only-secret")).encode()
    return hmac.new(key, str(value).encode(), hashlib.sha256).hexdigest()[:16]


def mask_answer(field, value):
    if field == "pincode":
        return mask_pincode(value)
    if field == "phone":
        return mask_phone(value)
    if isinstance(value, str):
        return mask_text(value)
    return value
