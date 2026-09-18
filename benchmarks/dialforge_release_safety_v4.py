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

_DNC = re.compile(r"\b(?:do\s*not|don't|dont|never)\s+(?:call|contact)\b|\bstop\s+(?:calling|contacting)\b|\bremove\s+(?:me|us|this\s+business|this\s+number)\b.{0,40}\b(?:list|calls?|contact)\b|\btake\s+(?:me|us|this\s+business|this\s+number)\s+off\b.{0,30}\blist\b|\blose\s+my\s+number\b|\bno\s+further\s+contact\b", re.I)
_TEMPORAL_CALL_NEGATION = re.compile(r"\b(?:do\s*not|don't|dont)\s+call(?:\s+me|\s+us)?\s+(?:today|tomorrow|tonight|this\s+(?:morning|afternoon|evening|week)|next\s+week|before\b|after\b|until\b)", re.I)
_THIRD_PARTY_CONTACT_NEGATION = re.compile(r"\b(?:do\s*not|don't|dont)\s+(?:call|contact)\s+(?:the\s+)?(?:owner|manager|office|dispatcher|my\s+boss)\b", re.I)
_WRONG = re.compile(r"\bwrong\s+(?:number|person|business|company)\b|\b(?:you(?:'ve| have)\s+got|you\s+have)\s+the\s+wrong\s+(?:number|person|business|company)\b", re.I)
_VOICEMAIL = re.compile(r"\bleave\s+(?:a\s+)?message\b.{0,40}\b(?:tone|beep)\b|\byou(?:'ve| have)\s+reached\b.{0,100}\b(?:leave\s+(?:a\s+)?message|can't\s+answer|cannot\s+answer)\b|\bvoicemail\b|\banswering\s+machine\b", re.I)
_INFO = re.compile(r"\b(?:email|e-mail|text|send|message)\b.{0,50}\b(?:details?|info(?:rmation)?|summary|prices?|pricing|website\s+prices?|something)\b|\b(?:have|get)\s+(?:a\s+)?(?:human|person|someone|your\s+team)\s+(?:follow\s*up|contact|email|text|call)\b|\b(?:follow\s*up|follow-up)\s+(?:by|via)\s+(?:email|text|phone)\b", re.I)
_CLEAR_REFUSAL = re.compile(r"\bnot\s+interested\b|\bno\s+thanks\b|\bdon't\s+want\s+(?:the\s+)?pitch\b|\bdo\s+not\s+want\s+(?:the\s+)?pitch\b|\bno\s+meeting\b|\bdon't\s+want\s+(?:a\s+)?meeting\b|\bdo\s+not\s+want\s+(?:a\s+)?meeting\b", re.I)
_BAD_TIMING = re.compile(r"\b(?:this|now)\s+is(?:n't| not)\s+(?:a\s+)?good\s+time\b|\b(?:bad|terrible)\s+time\b|\b(?:i(?:'m| am)\s+(?:busy|heading|driving|working|with\s+a\s+customer|on\s+a\s+job|in\s+a\s+meeting))\b", re.I)
_AI_IDENTITY = re.compile(r"\b(?:are\s+you|is\s+this)\b.{0,35}\b(?:ai|a\.?i\.?|bot|robot|automated|human|real\s+person|person)\b|\b(?:ai|a\.?i\.?|bot|robot|automated)\b.{0,25}\b(?:call|caller|calling|voice)\b", re.I)
_AUDIT_CHALLENGE = re.compile(r"\b(?:you|your\s+team|you\s+guys)\b.{0,45}\b(?:checked|reviewed|audited|analy[sz]ed|looked\s+at)\b.{0,70}\b(?:my|our|the)\s+(?:website|site|company|business|ads?|account)\b|\bprove\b.{0,50}\b(?:looked\s+at|reviewed|checked|audited|analy[sz]ed)\b.{0,50}\b(?:my|our)\s+(?:company|business|website|site|ads?|account)\b", re.I)
_NO_SWITCH_BOUNDARY = re.compile(r"\b(?:don't|do\s+not|dont)\b.{0,90}\b(?:fire|replace|switch|drop|ditch)\b|\b(?:without|no\s+need\s+to)\s+(?:replacing|replace|switching|switch|firing|fire|dropping|drop|ditching|ditch)\b", re.I)
_ACCEPTED_NEXT_STEP = re.compile(r"\bwhat(?:'s|\s+is)\s+the\s+next\s+step\b|\bwhat\s+would\s+happen\s+next\b|\bwhat\s+happens\s+next\b|\b(?:i(?:'m|\s+am)\s+willing\s+to\s+(?:see|do)|let(?:'s|\s+us)\s+do|show\s+me)\s+(?:the\s+)?diagnostic\b|\bif\s+you\s+can\b.{0,80}\bwhat\s+would\s+happen\s+next\b", re.I)

# Prevent domain language such as "booked-job revenue" from being interpreted as
# meeting-booking intent, especially after imperfect STT turns "booked job" into
# "book job". Conditional result promises are also not valid scheduling consent.
_BOOKED_JOB_CONTEXT = re.compile(r"\bbook(?:ed)?[-\s]+jobs?\b", re.I)
_CONDITIONAL_RESULT_MEETING = re.compile(
    r"\b(?:meeting|appointment|diagnostic|call)\b.{0,100}\bonly\s+if\b.{0,140}\b(?:guarantee|promise)\b|"
    r"\bonly\s+if\b.{0,140}\b(?:guarantee|promise)\b.{0,100}\b(?:meeting|appointment|diagnostic|call)\b",
    re.I,
)
_EXPLICIT_BOOKING_NEGATION = re.compile(r"\b(?:don't|do\s+not|dont)\s+book\b|\bnot\s+book\b|\bhold\s+off\b|\bconfirm\s+(?:it\s+)?later\b|\bi(?:'ll| will)\s+(?:confirm|let\s+you\s+know)\b|\bjust\s+checking\s+availability\b|\bchecking\s+availability\b", re.I)
_TENTATIVE_BOOKING = re.compile(r"\btentative\b|\bprobably\s+(?:open|free|available)\b|\bmaybe\b|\bmight\b|\bcould\s+work\b", re.I)
_BOOKING_CONSENT = re.compile(r"\b(?:book|schedule)\b|\bgo\s+ahead\b|\blet(?:'s| us)\s+do\s+it\b|\bworks\s+for\s+me\b|\bthat\s+works\b|\bworks\.?\s*(?:book|schedule)?\b", re.I)
_SCHEDULING_SIGNAL = re.compile(r"\b(?:book|schedule|appointment|meeting|callback|call\s+me|reschedule)\b", re.I)
_WEEKDAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_DAYPART = r"(?:morning|afternoon|evening|after\s+lunch|later)"
_VAGUE_SCHEDULING_SIGNAL = re.compile(rf"\b{_WEEKDAY}\b.{{0,50}}\b{_DAYPART}\b.{{0,80}}\b(?:work|works|available|free|diagnostic|meeting|appointment|callback|call)\b|\b(?:today|tomorrow|tonight|next\s+week)\b.{{0,50}}\b{_DAYPART}\b.{{0,80}}\b(?:work|works|available|free|diagnostic|meeting|appointment|callback|call)\b|\b{_DAYPART}\b.{{0,50}}\b(?:work|works|available|free)\b.{{0,80}}\b(?:diagnostic|meeting|appointment|callback|call)\b", re.I)

_MONTHS = {name:i for i,name in enumerate(("january","february","march","april","may","june","july","august","september","october","november","december"),1)}
_MONTH_RE = re.compile(r"\b("+"|".join(_MONTHS)+r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(20\d{2})\b", re.I)
_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_TIME_12_RE = re.compile(r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(a\.?m\.?|p\.?m\.?)\b", re.I)
_TIME_24_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_TZ_RE = re.compile(r"\b(?:Eastern|Central|Mountain|Pacific|Atlantic|UTC|GMT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|ET|CT|MT|PT)\b|\b[A-Za-z]+/[A-Za-z_+-]+\b", re.I)

_PACKAGE_TOKEN = re.compile(r"\b(?:starter|basic|growth|premium|entry[-\s]?level)\b|[$£€]\s*\d[\d,]*(?:\.\d+)?", re.I)
_SCOPE_RELATION = re.compile(r"\b(?:include[sd]?|come[sd]?\s+with|add[sd]?|cover[sd]?|provide[sd]?|contain[sd]?|has|have|does\s+not\s+include|doesn't\s+include|without)\b", re.I)
_REVERSE_SCOPE_RELATION = re.compile(r"\b(?:included?|provided|covered|available)\s+(?:with|in|on)\s+(?:the\s+)?(starter|basic|growth|premium|entry[-\s]?level)\b", re.I)
_FEATURE_TERM = re.compile(r"\b(?:seo|search\s+engine\s+optimization|pages?|page\s+count|hosting|maintenance|support|revisions?|copy|content|google\s+ads?|paid\s+search|tracking|analytics|crm|automation|rankings?|local\s+seo|blog|e-?commerce|booking|forms?)\b", re.I)
_GUARANTEE = re.compile(r"\b(?:guarantee[sd]?|guaranteed|promise[sd]?)\b.{0,90}\b(?:rankings?|leads?|calls?|jobs?|revenue|results?|roi|savings?)\b|\b(?:rankings?|leads?|calls?|jobs?|revenue|results?|roi|savings?)\b.{0,90}\b(?:guarantee[sd]?|guaranteed|promise[sd]?)\b", re.I)
_UNCERTAINTY = re.compile(r"\b(?:can't|cannot|can not|don't|do not|won't|will not|wouldn't|would not|not supplied|not provided|don't have|do not have|can't confirm|cannot confirm|don't know|do not know|no guarantee|not guaranteed|don't want to guess|do not want to guess)\b", re.I)
_NEGATIVE_SOURCE = re.compile(r"\b(?:no|not|never|cannot|can't|do not|don't|without|prohibited|unsupported|unavailable)\b", re.I)
_GUARANTEE_WORD = re.compile(r"\b(?:guarantee[sd]?|guaranteed|promise[sd]?)\b", re.I)
_CONTROL_LEAK = re.compile(r"\b(?:tool trigger|mandatory tool router|private call control rules|dialforge turn decision card|problem \+ impact|prospect asserts|call mark_do_not_call|action priority|sales priority|small-model mode)\b|<tool>|\brecord_outcome\s*\{|\bbook_meeting\s*\{", re.I)

_PRICE_NUMBER = re.compile(r"(?:[$£€]\s*\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s*(?:usd|dollars?|pkr|rupees?|gbp|pounds?|eur|euros?)\b)", re.I)
_PRICE_FREE = re.compile(r"\b(?:free|no\s+cost|at\s+no\s+cost|no\s+charge|at\s+no\s+charge|complimentary)\b", re.I)
_PRICE_RATE = re.compile(r"\b(?:starts?\s+at|starting\s+at|costs?|price(?:d)?\s+at|fee(?:s)?\s+(?:is|are)?|retainer(?:\s+is)?)\s*[$£€]?\s*\d[\d,]*(?:\.\d+)?", re.I)
_STARTING_PRICE_CLAIM = re.compile(r"\b(?:starts?\s+at|starting\s+at)\s*[$£€]?\s*(\d[\d,]*(?:\.\d+)?)", re.I)
_STARTER_PRICE_SOURCE = re.compile(r"\b(?:starter|basic|entry[-\s]?level|lowest)\b.{0,100}?[$£€]\s*(\d[\d,]*(?:\.\d+)?)", re.I | re.DOTALL)
_AUDIT_CLAIM = re.compile(r"\b(?:i|we)(?:'ve| have)?\s+(?:checked|reviewed|audited|looked at|analyzed)\s+(?:your|the)\s+(?:website|site|ads?|account|google profile)\b|\bour audit (?:found|shows?)\b", re.I)
_AUDIT_EVIDENCE = re.compile(r"\b(?:prospect[-\s]?specific|lead[-\s]?specific)\s+(?:audit|review|research)\b|\b(?:audit|review|research)\s+(?:finding|result|note)\b|\bwebsite\s+reviewed\b", re.I)


def _clean(text:str)->str:
    return re.sub(r"\s+"," ",str(text or "")).strip()


def _date_parts(text:str):
    m=_ISO_DATE_RE.search(text)
    if m:
        y,mo,d=map(int,m.groups())
    else:
        m=_MONTH_RE.search(text)
        if not m:return None
        mo,d,y=_MONTHS[m.group(1).lower()],int(m.group(2)),int(m.group(3))
    try: datetime(y,mo,d)
    except ValueError:return None
    return y,mo,d


def _time_parts(text:str):
    m=_TIME_12_RE.search(text)
    if m:
        h,minute=int(m.group(1)),int(m.group(2) or 0)
        ap=re.sub(r"[^apm]","",m.group(3).lower())
        if ap.startswith("p") and h!=12:h+=12
        elif ap.startswith("a") and h==12:h=0
        return h,minute
    m=_TIME_24_RE.search(text)
    return (int(m.group(1)),int(m.group(2))) if m else None


def _timezone(text:str):
    m=_TZ_RE.search(text)
    return m.group(0).strip() if m else None


def _preferred_channel(text:str)->str:
    lower=text.lower()
    if "email" in lower or "e-mail" in lower:return "email"
    if "text" in lower:return "text"
    if "call" in lower or "phone" in lower:return "phone"
    return ""


def booking_components(text:str)->dict[str,Any]:
    value=_clean(text); date=_date_parts(value); clock=_time_parts(value); tz=_timezone(value)
    explicit=bool(_EXPLICIT_BOOKING_NEGATION.search(value)); tentative=bool(_TENTATIVE_BOOKING.search(value))
    consent_value=_BOOKED_JOB_CONTEXT.sub(" ",value)
    result={"date":date,"time":clock,"timezone":tz,"consent":bool(_BOOKING_CONSENT.search(consent_value)) and not explicit and not tentative,"negated":explicit or tentative,"explicit_negation":explicit,"tentative":tentative}
    if date and clock and tz:
        y,mo,d=date; h,minute=clock; result["starts_at"]=f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{minute:02d}:00"
    return result


def route_turn(text:str)->RouteDecision:
    value=_clean(text)
    if not value:return RouteDecision(kind="model",reason="empty")
    dnc=bool(_DNC.search(value))
    temporal=bool(_TEMPORAL_CALL_NEGATION.search(value))
    third_party=bool(_THIRD_PARTY_CONTACT_NEGATION.search(value))
    if dnc and not temporal and not third_party:
        return RouteDecision("tool","mark_do_not_call",{"reason":"prospect requested no further contact"},reason="dnc")
    if _WRONG.search(value):return RouteDecision("tool","record_outcome",{"outcome":"wrong_number"},reason="wrong_number")
    if _VOICEMAIL.search(value):return RouteDecision("tool","record_outcome",{"outcome":"voicemail"},reason="voicemail")
    if _INFO.search(value):return RouteDecision("tool","request_human_follow_up",{"reason":value[:500],"preferred_channel":_preferred_channel(value)},reason="follow_up")
    if _CLEAR_REFUSAL.search(value):return RouteDecision("tool","record_outcome",{"outcome":"not_interested"},reason="not_interested")

    # These are high-confidence integrity or boundary states. Resolve them before
    # the model so a small local model cannot convert timing into a disposition,
    # hide AI identity, invent an audit, or pressure a provider switch.
    if temporal:
        return RouteDecision("spoken",spoken_reply="Understood. What exact date, time, and timezone would be better?",reason="temporary_no_call")
    if _BAD_TIMING.search(value):
        return RouteDecision("spoken",spoken_reply="No problem. What exact date and time would be better?",reason="bad_timing")
    if _AI_IDENTITY.search(value):
        return RouteDecision("spoken",spoken_reply="I'm an automated AI calling assistant.",reason="ai_identity")
    if _AUDIT_CHALLENGE.search(value):
        return RouteDecision("spoken",spoken_reply="I haven't checked your website or run a prospect-specific audit.",reason="audit_denial")
    if _NO_SWITCH_BOUNDARY.search(value):
        return RouteDecision("spoken",spoken_reply="Understood. You don't need to switch or replace your current provider for the diagnostic.",reason="provider_boundary")
    if _CONDITIONAL_RESULT_MEETING.search(value):
        return RouteDecision(
            "spoken",
            spoken_reply="I can't guarantee a specific result. If you're still open to the diagnostic without that promise, we can schedule it.",
            reason="conditional_guarantee_not_consent",
        )

    b=booking_components(value)
    scheduling_value=_BOOKED_JOB_CONTEXT.sub(" ",value)
    scheduling=bool(_SCHEDULING_SIGNAL.search(scheduling_value) or _VAGUE_SCHEDULING_SIGNAL.search(value) or any(b.get(k) for k in ("date","time","timezone")))
    if scheduling:
        if b["explicit_negation"]:return RouteDecision("spoken",spoken_reply="Okay. I won't book anything until you're ready.",reason="booking_not_authorized")
        missing=[name for name in ("date","time","timezone") if not b[name]]
        if missing:
            prompts={("timezone",):"What timezone should I use?",("time",):"What exact time works?",("date",):"What exact date works?",("date","time"):"What exact date and time works best?",("time","timezone"):"What exact time and timezone should I use?",("date","timezone"):"What exact date and timezone should I use?"}
            return RouteDecision("spoken",spoken_reply=prompts.get(tuple(missing),"What exact date, time, and timezone works best?"),reason="booking_incomplete")
        if b["tentative"]:return RouteDecision("spoken",spoken_reply="I won't book it until you explicitly confirm that slot.",reason="booking_consent_missing")
        if b["consent"]:return RouteDecision("tool","book_meeting",{"title":"Scheduled meeting","starts_at":b["starts_at"],"timezone":b["timezone"],"notes":"Booked from explicit prospect agreement."},reason="exact_booking")
        return RouteDecision("spoken",spoken_reply="I won't book it until you explicitly want me to.",reason="booking_consent_missing")

    if _ACCEPTED_NEXT_STEP.search(value):
        return RouteDecision("spoken",spoken_reply="What exact date, time, and timezone works best?",reason="accepted_next_step")
    return RouteDecision(kind="model",reason="sales_judgment")


def _sentences(text:str)->list[str]:
    value=_clean(text)
    parts=[p.strip() for p in re.split(r"(?<=[.!?])\s+",value) if p.strip()]
    return parts or ([value] if value else [])


def _positive_source_supports_guarantee(source:str)->bool:
    for clause in [p.strip() for p in re.split(r"[.\n;]+",str(source or "")) if p.strip()]:
        if _GUARANTEE_WORD.search(clause) and not _NEGATIVE_SOURCE.search(clause):return True
    return False


def _package_claim_token(sentence:str)->str|None:
    value=_clean(sentence)
    for m in _PACKAGE_TOKEN.finditer(value):
        if _SCOPE_RELATION.search(value[m.end():m.end()+64]):return m.group(0)
    reverse=_REVERSE_SCOPE_RELATION.search(value)
    return reverse.group(1) if reverse else None


def _scope_relation_supported(sentence:str,source:str)->bool:
    package=_package_claim_token(sentence); features={m.group(0).lower() for m in _FEATURE_TERM.finditer(sentence)}
    if not package or not features:return True
    package=package.lower().replace(",",""); source_value=_clean(source).lower().replace(",","")
    for pos in [m.start() for m in re.finditer(re.escape(package),source_value)]:
        window=source_value[max(0,pos-180):pos+320]
        if all(feature in window for feature in features) and not re.search(r"\b(?:not supplied|not provided|unknown|do not include exact|don't include exact)\b",window):return True
    return False


def _normalized_price_tokens(text:str)->set[str]:
    value=str(text or "")
    tokens=set()
    if _PRICE_FREE.search(value):tokens.add("free")
    for pattern in (_PRICE_NUMBER,_PRICE_RATE):
        for match in pattern.finditer(value):
            for number in re.findall(r"\d[\d,]*(?:\.\d+)?",match.group(0)):
                tokens.add(number.replace(",",""))
    return tokens


def _starter_price_tokens(text:str)->set[str]:
    return {m.group(1).replace(",","") for m in _STARTER_PRICE_SOURCE.finditer(str(text or ""))}


def _price_claim_supported(sentence:str,source:str)->bool:
    claims=_normalized_price_tokens(sentence)
    if not claims:return True
    allowed=_normalized_price_tokens(source)
    if not claims.issubset(allowed):return False
    starting=_STARTING_PRICE_CLAIM.search(sentence); starter_values=_starter_price_tokens(source)
    if starting and starter_values and starting.group(1).replace(",","") not in starter_values:return False
    return True


def _source_supports_audit(source:str)->bool:
    for clause in [p.strip() for p in re.split(r"[.\n;]+",str(source or "")) if p.strip()]:
        if _AUDIT_EVIDENCE.search(clause) and not _NEGATIVE_SOURCE.search(clause):
            return True
    return False


def _sanitize_sentence(sentence:str,source:str)->str:
    value=_clean(sentence)
    if not value:return ""
    uncertain=bool(_UNCERTAINTY.search(value)); package=_package_claim_token(value); scope_claim=bool(package and _FEATURE_TERM.search(value)); scope_bad=scope_claim and not uncertain and not _scope_relation_supported(value,source)
    guarantee_claim=bool(_GUARANTEE.search(value)); guarantee_bad=guarantee_claim and not uncertain and not _positive_source_supports_guarantee(source); leak=bool(_CONTROL_LEAK.search(value))
    audit_bad=bool(_AUDIT_CLAIM.search(value)) and not _source_supports_audit(source)
    price_bad=not _price_claim_supported(value,source)
    if leak:
        if scope_claim and guarantee_claim and not _positive_source_supports_guarantee(source):return "I can't confirm those package features, and I can't guarantee that outcome."
        if scope_claim and not _scope_relation_supported(value,source):return "I don't have the exact package feature breakdown, so I don't want to guess."
        if guarantee_claim and not _positive_source_supports_guarantee(source):return "I can't guarantee that outcome."
        return "I don't want to guess. I can answer from the information I have."
    if audit_bad:return "I haven't checked your website or run a prospect-specific audit."
    if price_bad:return "I don't have the exact pricing in front of me, so I don't want to guess."
    if scope_bad and guarantee_bad:return "I can't confirm those package features, and I can't guarantee that outcome."
    if scope_bad:return "I don't have the exact package feature breakdown, so I don't want to guess."
    if guarantee_bad:return "I can't guarantee that outcome."
    return value


def sanitize_marketing_claim(text:str,source:str)->str:
    output=[]
    for sentence in _sentences(text):
        safe=_sanitize_sentence(sentence,source)
        if safe and (not output or output[-1]!=safe):output.append(safe)
    return " ".join(output).strip()
