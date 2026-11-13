# tone-agent

A small ReAct tool-calling agent that reads a chat message and recommends how
to reply. Built as a real, runnable demo of agent work — not a wrapper around
a chat API.

## What it is

Give it a message like "You never listen to me" and it:

1. **Classifies the tone** with a fine-tuned DistilBERT model (7 classes:
   affection, bid, criticism, contempt, defensiveness, stonewalling, repair).
2. **Looks up guidance** for that tone, and **searches a small knowledge
   base** of relationship ideas (Gottman bids, the four horsemen, repair
   attempts, perceived responsiveness).
3. **Recommends a reply** based on what the tools returned — never invented.

## Architecture

```
                 +------------------+
                 |  Brain (LLM)     |   Groq (free) / Kimi / OpenAI
                 |  ReAct loop      |   via OpenAI-compatible API
                 +--------+---------+
                          | Thought / Action / Action Input / Observation
          +---------------+---------------+
          |               |               |
   classify_tone   search_memory   tone_guidance
   (DistilBERT,    (keyword        (per-tone
    fine-tuned)     retrieval,      guidance,
                    knowledge.json) guidance.json)
```

The loop is hand-rolled (no LangGraph): at most 6 steps, strict output
parsing, one retry on a malformed step, then a clear error. Every step is
printed to the console.

## How to run

```bash
pip install -r requirements.txt   # CPU torch, no GPU needed

# 1. Get the classifier weights (250MB, not in this repo):
#    download model_bert_v3.zip from
#    https://drive.google.com/file/d/1B53Utf5ngEhOVxGaPTEjjDY4GsURVus5/view
#    and unzip it into models/ so the folder is models/distilbert-v3/
#    (or point CLASSIFIER_MODEL_PATH at an existing copy)

# 2. Get a free Groq key: https://console.groq.com/keys
export GROQ_API_KEY=your_key_here

# 3. Run it
python agent.py "You never listen to me, you always ignore what I say"
```

Other brains: set `MOONSHOT_API_KEY` (Kimi, model `kimi-k3`) or
`OPENAI_API_KEY` (with optional `OPENAI_BASE_URL` / `OPENAI_MODEL`).
Without any key the program exits with a message telling you how to get one
— it never fakes a reply.

Test the loop without a key:

```bash
CLASSIFIER_MODEL_PATH=/path/to/model_bert_v3 python agent.py --stub "fine. whatever."
```

## Eval

`eval.py` runs 10 fixed messages — one per tone class, plus 3 tricky ones
(sarcasm, a repair attempt, a neutral factual message). Gold labels are the
classifier's own labels for those messages, stated plainly: this eval scores
the **agent** (did it call `classify_tone`, name the right tone, and use the
guidance/memory it fetched), not the classifier.

| case | what it tests | score |
|---|---|---|
| affection-1 | affection | pending |
| bid-1 | bid | pending |
| criticism-1 | criticism | pending |
| contempt-1 | contempt | pending |
| defensiveness-1 | defensiveness | pending |
| stonewalling-1 | stonewalling | pending |
| repair-1 | repair | pending |
| tricky-sarcasm | sarcasm (classifier reads it as contempt) | pending |
| tricky-repair | repair attempt | pending |
| tricky-neutral | neutral factual message (classifier is unsure) | pending |

**Eval status: pending — run `python eval.py` on your laptop with a Groq
key.** The table above has no numbers because no real-brain eval has been
run yet. The eval harness itself was verified end-to-end with the stub brain
(`python eval.py --stub`): the plumbing, scoring, and output format work.

```bash
export GROQ_API_KEY=your_key_here
python eval.py          # prints a table, saves results.json
```

## Honest limits

- The classifier is a DistilBERT fine-tuned on 405 hand-labeled messages,
  **86% on a 79-sample self-labeled validation set** (TF-IDF baseline 71%
  on the same split). Small data, self-labeled — treat its labels as a
  signal, not ground truth. The `repair` class is weakest (only 30 examples).
- `search_memory` is keyword retrieval, not embeddings — no semantic search.
- The agent only knows what its tools return. If the classifier is wrong,
  the recommendation follows the wrong tone.
- Runs on CPU. A single message takes a few seconds, mostly model load.
