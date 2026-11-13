"""Tools for the tone agent.

Three real tools, no stubs:
- classify_tone: Tobi's fine-tuned DistilBERT classifier (7 tones).
- search_memory: keyword retrieval over a small Gottman knowledge base.
- tone_guidance: short sourced guidance for one tone label.
"""

import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

FALLBACK_LABELS = [
    "affection",
    "bid",
    "criticism",
    "contempt",
    "defensiveness",
    "stonewalling",
    "repair",
]

_model = None
_tokenizer = None
_labels = None


def _load_json(name):
    with open(os.path.join(REPO_ROOT, name), encoding="utf-8") as f:
        return json.load(f)


def _default_model_path():
    return os.path.join(REPO_ROOT, "models", "distilbert-v3")


def classify_tone(text):
    """Classify one chat message with the fine-tuned DistilBERT model.

    Loads the weights lazily from CLASSIFIER_MODEL_PATH, or from
    models/distilbert-v3/ inside this repo. The weights are NOT in the repo
    (250MB) — see the README for the download link.
    """
    global _model, _tokenizer, _labels
    if _model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = os.environ.get("CLASSIFIER_MODEL_PATH") or _default_model_path()
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Classifier weights not found at {path}. Download model_bert_v3.zip from "
                "https://drive.google.com/file/d/1B53Utf5ngEhOVxGaPTEjjDY4GsURVus5/view "
                "and unzip it into models/ so the folder is models/distilbert-v3/."
            )
        _tokenizer = AutoTokenizer.from_pretrained(path)
        _model = AutoModelForSequenceClassification.from_pretrained(path)
        _model.eval()
        labels_path = os.path.join(path, "labels.json")
        _labels = json.load(open(labels_path)) if os.path.exists(labels_path) else FALLBACK_LABELS

    import torch

    enc = _tokenizer(text, truncation=True, max_length=32, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(_model(**enc).logits, dim=1)[0]
    best = int(probs.argmax())
    top3 = sorted(
        ((_labels[i], round(float(probs[i]), 3)) for i in range(len(_labels))),
        key=lambda x: -x[1],
    )[:3]
    return {
        "label": _labels[best],
        "confidence": round(float(probs[best]), 3),
        "top3": [{"label": label, "probability": p} for label, p in top3],
    }


def search_memory(query, top_n=3):
    """Keyword retrieval over knowledge.json (Gottman ideas).

    Scores each entry by word overlap with the query, weighted x3 for
    matches against the entry's keyword list. Returns the best matches.
    """
    entries = _load_json("knowledge.json")["entries"]
    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for entry in entries:
        text_words = set(
            re.findall(r"[a-z0-9]+", (entry["title"] + " " + entry["text"]).lower())
        )
        keywords = set(k.lower() for k in entry.get("keywords", []))
        score = len(query_words & text_words) + 3 * len(query_words & keywords)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: -x[0])
    return [
        {"title": e["title"], "text": e["text"], "score": s}
        for s, e in scored[:top_n]
    ]


def tone_guidance(tone):
    """Guidance for one tone label: what it means, what a good reply does."""
    data = _load_json("guidance.json")
    tone = tone.lower().strip()
    if tone not in data:
        raise ValueError(
            f"No guidance for tone '{tone}'. Known tones: {', '.join(sorted(data))}"
        )
    return data[tone]


TOOLS = {
    "classify_tone": {
        "fn": classify_tone,
        "desc": "Classify a chat message into one of affection, bid, criticism, contempt, defensiveness, stonewalling, repair. Args: {\"text\": \"<the message>\"}. Returns label, confidence, top 3 labels.",
    },
    "search_memory": {
        "fn": search_memory,
        "desc": "Keyword search over a small knowledge base of relationship ideas (Gottman bids, the four horsemen, repair attempts, perceived responsiveness). Args: {\"query\": \"<keywords>\"}. Returns matching entries.",
    },
    "tone_guidance": {
        "fn": tone_guidance,
        "desc": "Guidance for one tone label: what it means and what a good reply does. Args: {\"tone\": \"<one of the 7 labels>\"}. Returns meaning, do, dont.",
    },
}


def run_tool(name, args):
    """Run a tool by name with a dict of arguments. Raises on unknown tools."""
    if name not in TOOLS:
        raise ValueError(f"Unknown tool '{name}'. Known tools: {', '.join(sorted(TOOLS))}")
    if not isinstance(args, dict):
        raise ValueError(f"Action Input for '{name}' must be a JSON object, got: {args!r}")
    return TOOLS[name]["fn"](**args)
