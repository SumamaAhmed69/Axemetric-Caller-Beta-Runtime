from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class RouteDecision:
    kind: str
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    spoken_reply: str | None = None
    reason: str = ""


_DNC = re.compile(
    r"\b(?:do\s*not|don't|dont|never)\s+(?:call|contact)\b|\bstop\s+(?:calling|contacting)\b|"
    r"\bremove\s+(?:me|us|this\s+business|this\s+number)\b.{0,40}\b(?:list|calls?|contact)\b|"
    r"\btake\s+(?:me|us|this\s+business|this\s+number)\s+off\b.{0,30}\blist\b|"
    r"\blose\s+my\s+number\b|\bno\s+further\s+contact\b", re.I,
)
_TEMPORAL_CALL_NEGATION = re.compile(
    r"\b(?:do\s*not|don't|dont)\s+call(?:\s+me|\s+us)?\s+(?:today|tomorrow|tonight|this\s+(?:morning|afternoon|evening|week)|next\s+week|before\b|after\b|until\b)", re.I,
)
_THIRD_PARTY_CONTACT_NEGATION = re.compile(
    r"\b(?:do\s*not|don't|dont)\s+(?:call|contact)\s+(?:the\s+)?(?:owner|manager|office|dispatcher|my\s+boss)\b", re.I,
)
_WRONG = re.compile(
    r"\bwrong\s+(?:number|person|business|company)\b|\b(?:you(?:'ve| have)\s+got|you\s+have)\s+the\s+wrong\s+(?:number|person|business|company)\b", re.I,
)
_VOICEMAIL = re.compile(
    r"\bleave\s+(?:a\s+)?message\b.{0,40}\b(?:tone|beep)\b|"
    r"\byou(?:'ve| have)\s+reached\b.{0,100}\b(?:leave\s+(?:a\s+)?message|can't\s+answer|cannot\s+answer)\b|"
    r"\bvoicemail\b|\banswering\s+machine\b", re.I,
)
_INFO = re.compile(
    r"\b(?:email|e-mail|text|send|message)\b.{0,50}\b(?:details?|info(?:rmation)?|summary|prices?|pricing|website\s+prices?|something)\b|"
    r"\b(?:have|get)\s+(?:a\s+)?(?:human|person|someone|your\s+team)\s+(?:follow\s*up|contact|email|text|call)\b|"
    r"\b(?:follow\s*up|follow-up)\s+(?:by|via)\s+(?:email|text|phone)\b", re.I,
)
_CLEAR_REFUSAL = re.compile(
    r"\bnot\s+interested\b|\bno\s+thanks\b|\bdon't\s+want\s+(?:the\s+)?pitch\b|"
    r"\bdo\s+not\s+want\s+(?:the\s+)?pitch\b|\bno\s+meeting\b|"
    r"\bdon't\s+want\s+(?:a\s+)?meeting\b|\bdo\s+not\s+want\s+(?:a\s+)?meeting\b", re.I,
)
_BOOKING_NEGATION = re.compile(
    r"\b(?:don't|do\s+not|dont)\s+book\b|\bnot\s+book\b|\bhold\s+off\b|\btentative\b|"
    r"\bprobably\s+(?:open|free|available)\b|\bmaybe\b|\bmight\b|\bcould\s+work\b|"
    r"\bconfirm\s+(?:it\s+)?later\b|\bi(?:'ll| will)\s+(?:confirm|let\s+you\s+know)\b|"
    r"\bjust\s+checking\s+availability\b|\bchecking\s+availability\b", re.I,
)
_BOOKING_CONSENT = re.compile(
    r"\b(?:book|schedule)\b|\bgo\s+ahead\b|\blet(?:'s| us)\s+do\s+it\b|"
    r"\bworks\s+for\s+me\b|\bthat\s+works\b|\bworks\.?\s*(?:book|schedule)?\b", re.I,
)
_SCHEDULING_SIGNAL = re.compile(r"\b(?:book|schedule|appointment|meeting|callback|call\s+me|reschedule)\b", re.I)

_MONTHS = {name: i for i, name in enumerate(
    ("january","february","march","april","may","june","july","august","september","october","november","december"), 1
)}
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(20\d{2})\b", re.I)
_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_TIME_12_RE = re.compile(r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(a\.?m\.?|p\.?m\.?)\b", re.I)
_TIME_24_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_TZ_RE = re.compile(
    r"\b(?:Eastern|Central|Mountain|Pacific|Atlantic|UTC|GMT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|ET|CT|MT|PT)\b|\b[A-Za-z]+/[A-Za-z_+-]+\b", re.I,
)

_PACKAGE_TOKEN = re.compile(r"\b(?:starter|basic|growth|premium|entry[-\s]?level)\b|[$£€]\s*\d[\d,]*(?:\.\d+)?", re.I)
_SCOPE_RELATION = re.compile(
    r"\b(?:include[sd]?|come[sd]?\s+with|add[sd]?|cover[sd]?|provide[sd]?|contain[sd]?|has|have|does\s+not\s+include|doesn't\s+include|without)\b", re.I,
)
_FEATURE_TERM = re.compile(
    r"\b(?:seo|search\s+engine\s+optimization|pages?|page\s+count|hosting|maintenance|support|revisions?|copy|content|google\s+ads?|paid\s+search|tracking|analytics|crm|automation|rankings?|local\s+seo|blog|e-?commerce|booking|forms?)\b", re.I,
)
_GUARANTEE = re.compile(
    r"\b(?:guarantee[sd]?|guaranteed|promise[sd]?)\b.{0,90}\b(?:rankings?|leads?|calls?|jobs?|revenue|results?|roi|savings?)\b|"
    r"\b(?:rankings?|leads?|calls?|jobs?|revenue|results?|roi|savings?)\b.{0,90}\b(?:guarantee[sd]?|guaranteed|promise[sd]?)\b", re.I,
)
_UNCERTAINTY = re.compile(
    r"\b(?:can't|cannot|can not|don't|do not|won't|will not|wouldn't|would not|not supplied|not provided|don't have|do not have|can't confirm|cannot confirm|don't know|do not know|no guarantee|not guaranteed|don't want to guess|do not want to guess)\b", re.I,
)
_NEGATIVE_SOURCE = re.compile(r"\b(?:no|not|never|cannot|can't|do not|don't|without|prohibited|unsupported|unavailable)\b", re.I)
_GUARANTEE_WORD = re.compile(r"\b(?:guarantee[sd]?|guaranteed|promise[sd]?)\b", re.I)
_CONTROL_LEAK = re.compile(
    r"\b(?:tool trigger|mandatory tool router|private call control rules|dialforge turn decision card|problem \+ impact|prospect asserts|call mark_do_not_call|action priority|sales priority|small-model mode)\b|<tool>|\brecord_outcome\s*\{|\bbook_meeting\s*\{", re.I,
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _date_parts(text: str):
    match = _ISO_DATE_RE.search(text)
    if match:
        year, month, day = map(int, match.groups())
    else:
        match = _MONTH_RE.search(text)
        if not match:
            return None
        month, day, year = _MONTHS[match.group(1).lower()], int(match.group(2)), int(match.group(3))
    try:
        datetime(year, month, day)
    except ValueError:
        return None
    return year, month, day


def _time_parts(text: str):
    match = _TIME_12_RE.search(text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2) or 0)
        ampm = re.sub(r"[^apm]", "", match.group(3).lower())
        if ampm.startswith("p") and hour != 12:
            hour += 12
        elif ampm.startswith("a") and hour == 12:
            hour = 0
        return hour, minute
    match = _TIME_24_RE.search(text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _timezone(text: str):
    match = _TZ_RE.search(text)
    return match.group(0).strip() if match else None


def _preferred_channel(text: str) -> str:
    lower = text.lower()
    if "email" in lower or "e-mail" in lower:
        return "email"
    if "text" in lower:
        return "text"
    if "call" in lower or "phone" in lower:
        return "phone"
    return ""


def booking_components(text: str) -> dict[str, Any]:
    value = _clean(text)
    date, clock, timezone = _date_parts(value), _time_parts(value), _timezone(value)
    negated = bool(_BOOKING_NEGATION.search(value))
    result = {"date":date,"time":clock,"timezone":timezone,"consent":bool(_BOOKING_CONSENT.search(value)) and not negated,"negated":negated}
    if date and clock and timezone:
        y,m,d = date
        h,minute = clock
        result["starts_at"] = f"{y:04d}-{m:02d}-{d:02d}T{h:02d}:{minute:02d}:00"
    return result


def route_turn(text: str) -> RouteDecision:
    value = _clean(text)
    if not value:
        return RouteDecision(kind="model", reason="empty")
    dnc = bool(_DNC.search(value))
    if dnc and not _TEMPORAL_CALL_NEGATION.search(value) and not _THIRD_PARTY_CONTACT_NEGATION.search(value):
        return RouteDecision("tool", "mark_do_not_call", {"reason":"prospect requested no further contact"}, reason="dnc")
    if _WRONG.search(value):
        return RouteDecision("tool", "record_outcome", {"outcome":"wrong_number"}, reason="wrong_number")
    if _VOICEMAIL.search(value):
        return RouteDecision("tool", "record_outcome", {"outcome":"voicemail"}, reason="voicemail")
    if _INFO.search(value):
        return RouteDecision("tool", "request_human_follow_up", {"reason":value[:500],"preferred_channel":_preferred_channel(value)}, reason="follow_up")
    if _CLEAR_REFUSAL.search(value):
        return RouteDecision("tool", "record_outcome", {"outcome":"not_interested"}, reason="not_interested")
    b = booking_components(value)
    if _SCHEDULING_SIGNAL.search(value) or any(b.get(k) for k in ("date","time","timezone")):
        if b["negated"]:
            return RouteDecision("spoken", spoken_reply="Okay. I won't book anything until you're ready.", reason="booking_not_authorized")
        missing = [name for name in ("date","time","timezone") if not b[name]]
        if not missing and b["consent"]:
            return RouteDecision("tool", "book_meeting", {"title":"Scheduled meeting","starts_at":b["starts_at"],"timezone":b["timezone"],"notes":"Booked from explicit prospect agreement."}, reason="exact_booking")
        if missing:
            prompts = {
                ("timezone",): "What timezone should I use?", ("time",): "What exact time works?", ("date",): "What exact date works?",
                ("date","time"): "What exact date and time works best?", ("time","timezone"): "What exact time and timezone should I use?",
                ("date","timezone"): "What exact date and timezone should I use?",
            }
            return RouteDecision("spoken", spoken_reply=prompts.get(tuple(missing), "What exact date, time, and timezone works best?"), reason="booking_incomplete")
        return RouteDecision("spoken", spoken_reply="I won't book it until you explicitly want me to.", reason="booking_consent_missing")
    return RouteDecision(kind="model", reason="sales_judgment")


def _positive_source_supports_guarantee(source: str) -> bool:
    for clause in [part.strip() for part in re.split(r"[.\n;]+", str(source or "")) if part.strip()]:
        if not _GUARANTEE_WORD.search(clause):
            continue
        if _NEGATIVE_SOURCE.search(clause):
            continue
        return True
    return False


def _scope_relation_supported(sentence: str, source: str) -> bool:
    package_match = _PACKAGE_TOKEN.search(sentence)
    features = {m.group(0).lower() for m in _FEATURE_TERM.finditer(sentence)}
    if not package_match or not features:
        return True
    package = package_match.group(0).lower().replace(",", "")
    source_value = _clean(source).lower().replace(",", "")
    for position in [m.start() for m in re.finditer(re.escape(package), source_value)]:
        window = source_value[max(0, position-180):position+320]
        if all(feature in window for feature in features) and not re.search(r"\b(?:not supplied|not provided|unknown|do not include exact|don't include exact)\b", window):
            return True
    return False


def sanitize_marketing_claim(text: str, source: str) -> str:
    value = _clean(text)
    if not value:
        return ""
    if _CONTROL_LEAK.search(value):
        return "I don't want to guess. I can answer from the information I have."
    uncertain = bool(_UNCERTAINTY.search(value))
    scope_claim = bool(_PACKAGE_TOKEN.search(value) and _SCOPE_RELATION.search(value) and _FEATURE_TERM.search(value))
    scope_bad = scope_claim and not uncertain and not _scope_relation_supported(value, source)
    guarantee_claim = bool(_GUARANTEE.search(value))
    guarantee_bad = guarantee_claim and not uncertain and not _positive_source_supports_guarantee(source)
    if scope_bad and guarantee_bad:
        return "I can't confirm those package features, and I can't guarantee that outcome."
    if scope_bad:
        return "I don't have the exact package feature breakdown, so I don't want to guess."
    if guarantee_bad:
        return "I can't guarantee that outcome."
    return value
