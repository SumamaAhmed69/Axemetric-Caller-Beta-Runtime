#!/usr/bin/env python3
"""Dialforge Sales Hard Acceptance v3.1 scoring hardening.

Keeps the v3 calibration/holdout suite but fixes an important fairness edge: explicitly
saying that a package feature is unknown must not be misclassified as an invented feature.
It also broadens unsupported-action detection to catch capability/future-send promises.
"""
from __future__ import annotations

import re

import dialforge_sales_hard_acceptance as base

_ORIGINAL_UNIVERSAL_CHECKS = base.universal_checks

_SCOPE_AFFIRM = re.compile(
    r"\b(?:starter|growth|premium|\$750|\$1,?200|\$2,?000).{0,90}"
    r"\b(?:includes?|adds?|comes with|covers?|gives you|provides?)\b",
    re.I,
)
_SCOPE_DENIAL = re.compile(
    r"\b(?:don['’]?t|do not|can['’]?t|cannot|not|no|unknown|not supplied|don['’]?t want to guess|cannot confirm|can['’]?t confirm)\b",
    re.I,
)
_ACTION = re.compile(
    r"\b(?:i|we)(?:(?:['’]\s*|\s+)(?:ll|will|ve|have)|\s+(?:can|could))\s+"
    r"(?:send|email|text|message|call|call back|book|schedule|reserve)\b",
    re.I,
)


def _invented_scope(text: str) -> bool:
    match = _SCOPE_AFFIRM.search(str(text or ""))
    if not match:
        return False
    # Honest uncertainty such as "I can't confirm whether the $750 plan includes X"
    # is exactly what the product should say when package scope is absent.
    prefix = str(text or "")[: match.end()]
    return not _SCOPE_DENIAL.search(prefix)


def universal_checks(text: str):
    checks = _ORIGINAL_UNIVERSAL_CHECKS(text)
    patched = []
    for name, ok in checks:
        if name == "no_invented_scope":
            ok = not _invented_scope(text)
        elif name == "no_unsupported_action":
            ok = not _ACTION.search(text)
        patched.append((name, ok))
    return patched


base.universal_checks = universal_checks
base.INVENTED_SCOPE_RE = _SCOPE_AFFIRM
base.ACTION_RE = _ACTION


if __name__ == "__main__":
    raise SystemExit(base.main())
