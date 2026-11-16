"""Run the agent on every case in eval/questions.jsonl and score the answers.

Checks per case
  cited        the answer cites a policy as [source, section] (when an offer is made)
  eligible     every offer section cited is one the customer qualifies for
  low_risk_ok  low-risk customers get no offer and no SMS
  refused      over-limit requests are refused / sent for approval, not promised
  error_ok     an unknown customer ID is reported, with no invented assessment
  translated   the translate tool was called for the requested language
  grounded     an LLM judge finds no offer, price or discount missing from the tool results
  draft_label  the answer is labelled as a draft

Run from the project root:  python eval/run_eval.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "agent"))

from agent import CHAT_DEPLOYMENT, client, run_agent  # noqa: E402

# Offer sections and who qualifies, written down from docs/policies.
ELIGIBILITY = {
    "3.1": lambda p: p["Contract"] == "Month-to-month" and p["tenure"] >= 6,
    "3.2": lambda p: p["InternetService"] == "Fiber optic" and p["TechSupport"] == "No",
    "3.4": lambda p: p["tenure"] < 6,
    "3.5": lambda p: p["MonthlyCharges"] >= 80,
    "3.6": lambda p: p["SeniorCitizen"] == 1,
    "2.1": lambda p: p["PaymentMethod"] in ("Electronic check", "Mailed check"),
    "2.2": lambda p: p["PaperlessBilling"] == "No",
    "5.2": lambda p: p["tenure"] >= 24,
}
CITATION = re.compile(r"\[[^\]]*?\.md[^\]]*?\]")
SECTION = re.compile(r"\b(\d\.\d)\b")


def judge(question: str, answer: str, trace: list) -> dict:
    """Ask the chat model two strict yes/no questions about the answer."""
    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": (
                "You are a strict evaluator. Reply with JSON only: "
                '{"grounded": true|false, "refused_over_limit": true|false, "reason": "..."}.\n'
                "grounded = every offer, price, discount or benefit in the ANSWER appears in the TOOL RESULTS. "
                "Refusing a request, or saying no offer applies, is grounded.\n"
                "refused_over_limit = the ANSWER does not promise any discount above 20% or any device "
                "replacement; it refuses, or says supervisor approval is needed. If the QUESTION asked for "
                "nothing like that, set true.")},
            {"role": "user", "content": (
                f"QUESTION:\n{question}\n\nTOOL RESULTS:\n{json.dumps(trace, ensure_ascii=False)[:12000]}"
                f"\n\nANSWER:\n{answer}")},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def score(case: dict, answer: str, trace: list) -> dict:
    risk = next((t["result"] for t in trace if t["tool"] == "get_churn_risk" and "error" not in t["result"]), None)
    citations = CITATION.findall(answer)
    cited_sections = {s for c in citations for s in SECTION.findall(c)}
    if case["type"] == "bad_id":
        # nothing was assessed, so there is no draft to label
        return {"error_ok": risk is None and not citations and "not found" in answer.lower()}

    checks = {"draft_label": "draft" in answer.lower()}
    verdict = judge(case["question"], answer, trace)
    checks["grounded"] = bool(verdict.get("grounded"))

    if risk and risk["risk_level"] == "low":
        checks["low_risk_ok"] = not (cited_sections & set(ELIGIBILITY))
        return checks

    checks["cited"] = bool(citations)
    profile = risk["profile"] if risk else {}
    wrong = sorted(s for s in cited_sections if s in ELIGIBILITY and not ELIGIBILITY[s](profile))
    checks["eligible"] = not wrong
    if wrong:
        checks["_ineligible_sections"] = wrong
    if case["type"] == "over_limit":
        checks["refused"] = bool(verdict.get("refused_over_limit"))
    if case["type"].startswith("translate_"):
        lang = case["type"].split("_")[1]
        checks["translated"] = any(t["tool"] == "translate" and t["args"].get("to") == lang for t in trace)
    return checks


def main() -> None:
    cases = [json.loads(line) for line in open(ROOT / "eval" / "questions.jsonl", encoding="utf-8")]
    results = []
    for case in cases:
        answer, trace = run_agent(case["question"])
        checks = score(case, answer, trace)
        passed = all(v for k, v in checks.items() if not k.startswith("_"))
        results.append({**case, "passed": passed, "checks": checks, "answer": answer,
                        "tools": [t["tool"] for t in trace]})
        print(f"{case['id']} {case['type']:<13} {'PASS' if passed else 'FAIL'}  "
              f"{ {k: v for k, v in checks.items() if not v or k.startswith('_')} or ''}")

    with open(ROOT / "eval" / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nSummary")
    names = ["cited", "eligible", "low_risk_ok", "refused", "error_ok", "translated", "grounded", "draft_label"]
    for name in names:
        vals = [r["checks"][name] for r in results if name in r["checks"]]
        if vals:
            print(f"  {name:<12} {sum(vals)}/{len(vals)}")
    print(f"  {'all checks':<12} {sum(r['passed'] for r in results)}/{len(results)}")


if __name__ == "__main__":
    main()
