"""Eval for tone-agent: 10 fixed messages, scored on the AGENT, not the classifier.

Gold labels are the classifier's own labels for these messages. That is
deliberate and stated plainly: this eval measures whether the agent uses its
tools, names the right tone, and uses what it fetched — it does NOT measure
whether the classifier itself is right (that was measured separately: 86% on
a 79-sample self-labeled validation set, TF-IDF baseline 71%).

Per case, 3 points:
  (a) the agent called classify_tone
  (b) the final answer names the gold tone
  (c) the final answer references the guidance/memory it fetched
      (shares a content word with a fetched observation)

Run:
    python eval.py            (real brain — needs GROQ_API_KEY etc.)
    python eval.py --stub     (deterministic fake brain — proves the plumbing)
"""

import argparse
import json
import os
import re
import sys

from agent import StubBrain, make_brain, run_agent
from tools import classify_tone

CASES = [
    {"id": "affection-1", "text": "I love how you always make me laugh", "note": "affection"},
    {"id": "bid-1", "text": "Did you pick up milk on the way home?", "note": "bid"},
    {"id": "criticism-1", "text": "You never listen to me, you always ignore what I say", "note": "criticism"},
    {"id": "contempt-1", "text": "Oh great, another brilliant idea from you", "note": "contempt"},
    {"id": "defensiveness-1", "text": "It's not my fault, you never told me the plan changed", "note": "defensiveness"},
    {"id": "stonewalling-1", "text": "fine. whatever.", "note": "stonewalling"},
    {"id": "repair-1", "text": "I'm sorry, I was wrong, can we try again?", "note": "repair"},
    {"id": "tricky-sarcasm", "text": "Wow, thanks for the help. Really. So helpful.", "note": "sarcasm — classifier reads it as contempt"},
    {"id": "tricky-repair", "text": "Can we start over? I don't want to fight about this", "note": "repair attempt"},
    {"id": "tricky-neutral", "text": "The meeting is at 3pm tomorrow in room 204", "note": "neutral factual — classifier is unsure (low confidence)"},
]

WORD_RE = re.compile(r"[a-z]{4,}")


def content_words(text):
    return set(WORD_RE.findall(text.lower()))


def score_case(case, final_answer, trace):
    """Score one run. Returns (points_dict, total)."""
    gold = case["gold"]
    called_classify = any(t.get("action") == "classify_tone" for t in trace)
    names_tone = gold.lower() in final_answer.lower()
    fetched = " ".join(
        t.get("observation", "")
        for t in trace
        if t.get("action") in ("tone_guidance", "search_memory")
    )
    references = bool(content_words(fetched) & content_words(final_answer))
    points = {
        "called_classify": called_classify,
        "names_right_tone": names_tone,
        "references_guidance": references,
    }
    return points, sum(points.values())


def main():
    parser = argparse.ArgumentParser(description="Run the tone-agent eval (10 cases).")
    parser.add_argument("--stub", action="store_true", help="use the deterministic fake brain")
    args = parser.parse_args()

    brain_name = "stub" if args.stub else "real"
    print("Gold labels are the classifier's own labels for these messages.\n"
          "This eval scores the AGENT (tool use, right tone named, guidance used),\n"
          "not the classifier.\n")

    # Gold labels first, so a failed agent run still leaves the golds visible.
    for case in CASES:
        gold = classify_tone(case["text"])
        case["gold"] = gold["label"]
        case["gold_confidence"] = gold["confidence"]

    results = []
    for case in CASES:
        brain = StubBrain() if args.stub else make_brain()
        try:
            final_answer, trace = run_agent(case["text"], brain, verbose=False)
            points, total = score_case(case, final_answer, trace)
            error = None
        except Exception as exc:
            final_answer, trace, points, total, error = "", [], {}, 0, str(exc)
        results.append(
            {
                "id": case["id"],
                "text": case["text"],
                "gold": case["gold"],
                "gold_confidence": case["gold_confidence"],
                "final_answer": final_answer,
                "points": points,
                "score": total,
                "error": error,
            }
        )

    print(f"{'case':<18} {'gold':<15} {'classify':<9} {'tone':<6} {'ref':<5} score")
    for r in results:
        p = r["points"]
        print(
            f"{r['id']:<18} {r['gold']:<15} "
            f"{'Y' if p.get('called_classify') else 'N':<9} "
            f"{'Y' if p.get('names_right_tone') else 'N':<6} "
            f"{'Y' if p.get('references_guidance') else 'N':<5} {r['score']}/3"
            + (f"  ERROR: {r['error']}" if r["error"] else "")
        )
    total = sum(r["score"] for r in results)
    print(f"\nTotal: {total}/{len(results) * 3}  ({total / (len(results) * 3):.0%})")

    out = {
        "brain": brain_name,
        "note": "Gold labels are the classifier's own labels. This scores the agent, not the classifier.",
        "results": results,
        "total": total,
        "max": len(results) * 3,
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved results.json")


if __name__ == "__main__":
    main()
