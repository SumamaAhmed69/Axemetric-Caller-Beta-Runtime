from __future__ import annotations

import re

# Public benchmark mirror of the action-integrity subset in the shipping
# engine/axemetric/voice_guard.py. Raw model output is still scored separately;
# this only models what Dialforge would actually allow to reach TTS.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_FUTURE_SUBJECT = r"(?:i|we)(?:(?:['’]\s*|\s+)(?:ll|will))\s+"
_PAST_SUBJECT = r"(?:i|we)(?:(?:['’]\s*|\s+)(?:ve|have))\s+"
_UNSUPPORTED_SEND_ACTION = re.compile(
    rf"\b{_FUTURE_SUBJECT}(?:send|email|text|message|follow\s+up)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_CALLBACK_ACTION = re.compile(
    rf"\b{_FUTURE_SUBJECT}(?:call|call\s+back)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_BOOKING_ACTION = re.compile(
    rf"\b{_FUTURE_SUBJECT}(?:book|schedule|reserve|set\s+up)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_PAST_ACTION = re.compile(
    rf"\b{_PAST_SUBJECT}(?:sent|emailed|texted|messaged|called)\b",
    re.IGNORECASE,
)
_BOOKING_CLAIM = re.compile(
    r"(?:\b(?:you(?:'re|’re|\s+are)|your\s+(?:meeting|call|appointment|consultation|demo|session)\s+is|i(?:'ve|’ve|\s+have)|we(?:'ve|’ve|\s+have))\s+(?:booked|scheduled|confirmed)\b|\b(?:meeting|appointment|call|consultation|demo|session)\s+(?:is|has\s+been)\s+(?:booked|scheduled|confirmed)\b)",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_scheduling_question(sentence: str) -> bool:
    lower = sentence.lower()
    return "?" in sentence and "time" in lower and any(
        token in lower for token in ("day", "date", "when", "works")
    )


def sanitize_spoken_action_integrity(
    text: str,
    *,
    allow_booking_claim: bool = False,
) -> str:
    """Mirror the shipping guard for unsupported external-action speech.

    Before a real tool succeeds, Dialforge may ask for exact scheduling details or
    note a follow-up request, but it may not promise that it will send, call, book,
    schedule, reserve, or confirm an external action. This function intentionally
    does not alter raw model scoring; it models only the final speech boundary.
    """
    value = _clean(text)
    if not value:
        return ""

    clean: list[str] = []
    for sentence in [part.strip() for part in _SENTENCE_SPLIT.split(value) if part.strip()]:
        if _UNSUPPORTED_SEND_ACTION.search(sentence) or _UNSUPPORTED_PAST_ACTION.search(sentence):
            safe = "I can note that request for the team."
        elif _UNSUPPORTED_CALLBACK_ACTION.search(sentence):
            safe = "What exact day and time works best?"
        elif not allow_booking_claim and _UNSUPPORTED_BOOKING_ACTION.search(sentence):
            safe = "What exact day and time works best?"
        elif not allow_booking_claim and _BOOKING_CLAIM.search(sentence):
            safe = "I can only confirm that after it is saved."
        else:
            safe = sentence

        if clean and _is_scheduling_question(clean[-1]) and _is_scheduling_question(safe):
            if len(safe) < len(clean[-1]):
                clean[-1] = safe
            continue
        if not clean or safe != clean[-1]:
            clean.append(safe)
        if len(clean) >= 3:
            break

    return " ".join(clean).strip()
