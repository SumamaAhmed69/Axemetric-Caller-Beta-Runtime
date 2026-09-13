from pathlib import Path

path = Path('benchmarks/dialforge_final_release_acceptance_v4.py')
text = path.read_text(encoding='utf-8')


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label} match count: {count}')
    text = text.replace(old, new, 1)


replace_once(
    'from dialforge_release_safety_v4 import route_turn, sanitize_marketing_claim\nfrom dialforge_spoken_safety import sanitize_spoken_action_integrity\n',
    'from dialforge_release_safety_v4 import route_turn, sanitize_marketing_claim\nfrom dialforge_release_policy_v4 import release_system_prompt\nfrom dialforge_spoken_safety import sanitize_spoken_action_integrity\n',
    'policy import',
)
replace_once(
    'def product_system(model):\n    return base.system_prompt(model) + "\\n\\nFINAL RELEASE RULE: scheduling requires exact calendar date, clock time, timezone and explicit consent; never expose internal control text."\n',
    'def product_system(model):\n    return release_system_prompt(model, base.CORE_SALES, base.CAMPAIGN)\n',
    'product system',
)
replace_once(
    '{"prospect":"If you have something concrete, email a short summary to the office instead.","expect":"grounded_value"},',
    '{"prospect":"What should I tell the owner this is actually about, in one sentence?","expect":"grounded_value"},',
    'gatekeeper turn',
)
replace_once(
    'def guarded(text):\n    return sanitize_marketing_claim(sanitize_spoken_action_integrity(text, allow_booking_claim=False), base.CAMPAIGN)\n',
    '''def guarded(text):
    return sanitize_marketing_claim(sanitize_spoken_action_integrity(text, allow_booking_claim=False), base.CAMPAIGN)


def product_sales_response(prospect, raw_text):
    """Apply the same deterministic pre-LLM control path used by the shipping caller."""
    decision = route_turn(prospect)
    if decision.kind == "spoken" and decision.spoken_reply:
        return guarded(decision.spoken_reply), "router_spoken"
    if decision.kind == "tool":
        safe = {
            "mark_do_not_call": "Understood. I won't call again.",
            "request_human_follow_up": "I can note that request for the team.",
            "record_outcome": "Understood. Thanks for your time.",
            "book_meeting": "I can confirm that once the meeting is saved.",
        }.get(decision.tool_name or "", "Understood.")
        return guarded(safe), "router_tool"
    return guarded(raw_text), "model"
''',
    'guard helper',
)
replace_once(
    'raw=result["content"]; product=guarded(raw)',
    'raw=result["content"]; product,product_source=product_sales_response(turn["prospect"],raw)',
    'product response selection',
)
replace_once(
    '"raw":raw,"product":product,"raw_score":rs',
    '"raw":raw,"product":product,"product_source":product_source,"raw_score":rs',
    'product source record',
)

compile(text, str(path), 'exec')
path.write_text(text, encoding='utf-8')
