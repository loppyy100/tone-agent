"""tone-agent: a ReAct tool-calling agent that recommends how to reply to a chat message.

Usage:
    python agent.py "your message here"     (needs GROQ_API_KEY, MOONSHOT_API_KEY, or OPENAI_API_KEY)
    python agent.py --stub "your message"   (deterministic fake brain, for testing the loop)

The brain talks to any OpenAI-compatible API. The tools are real: a fine-tuned
DistilBERT tone classifier, keyword retrieval over a small Gottman knowledge
base, and per-tone guidance.
"""

import argparse
import json
import os
import re
import sys

from tools import TOOLS, run_tool

MAX_STEPS = 6

SYSTEM_PROMPT = """You are a helpful assistant that recommends how to reply to a chat message.
You work in a strict ReAct loop. At each step output exactly one of:

Thought: <one short sentence about what to do next>
Action: <one tool name>
Action Input: <JSON object with the tool's arguments>

or, when you are done:

Final Answer: <your recommendation>

Tools available:
- classify_tone: classify a chat message into one of affection, bid, criticism, contempt, defensiveness, stonewalling, repair. Arguments: {"text": "<the message>"}. Returns the label, confidence, and top 3 labels.
- search_memory: keyword search over a small knowledge base of relationship ideas (Gottman bids, the four horsemen, repair attempts, perceived responsiveness). Arguments: {"query": "<keywords>"}. Returns matching entries.
- tone_guidance: guidance for one tone label: what it means and what a good reply does. Arguments: {"tone": "<one of the 7 labels>"}. Returns meaning, do, dont.

Rules:
- Always call classify_tone first, before any other tool.
- After classifying, call tone_guidance for the label you got, and search_memory for the relevant idea.
- Your Final Answer must name the tone, say what a good reply does (using the guidance you fetched), and keep it to 2-3 sentences.
- Never invent a label or a confidence number. Only use what the tools returned.
"""

FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)
ACTION_RE = re.compile(r"Action:\s*([a-z_]+)", re.IGNORECASE)
INPUT_RE = re.compile(r"Action Input:\s*(\{.*\})", re.DOTALL)
THOUGHT_RE = re.compile(r"Thought:\s*(.*)", re.DOTALL)
LABEL_RE = re.compile(r'"label":\s*"([a-z]+)"')
DO_RE = re.compile(r'"do":\s*"([^"]+)"')


class OpenAIBrain:
    """A brain that talks to any OpenAI-compatible chat API (Groq, Kimi, OpenAI)."""

    def __init__(self, base_url, api_key, model):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def __call__(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.2
        )
        return resp.choices[0].message.content


class StubBrain:
    """Deterministic fake brain for testing the loop without an API key.

    Step 1 calls classify_tone, step 2 calls tone_guidance for the label the
    classifier returned, step 3 gives a Final Answer built from that guidance.
    """

    def __init__(self):
        self.step = 0

    def __call__(self, messages):
        self.step += 1
        history = "\n".join(m["content"] for m in messages)
        if self.step == 1:
            text = re.search(r'Chat message to respond to: "(.*)"', history).group(1)
            return (
                "Thought: I need to know the tone of this message first.\n"
                f"Action: classify_tone\nAction Input: {json.dumps({'text': text})}"
            )
        label = LABEL_RE.search(history).group(1)
        if self.step == 2:
            return (
                "Thought: Now I need guidance for this tone.\n"
                f"Action: tone_guidance\nAction Input: {json.dumps({'tone': label})}"
            )
        do = DO_RE.search(history).group(1)
        return (
            f"Final Answer: The tone is {label}. {do} "
            "Reply in that spirit, in your own words."
        )


def make_brain():
    """Pick a brain from env vars. Groq first (free), then Kimi, then OpenAI."""
    if os.environ.get("GROQ_API_KEY"):
        return OpenAIBrain(
            "https://api.groq.com/openai/v1",
            os.environ["GROQ_API_KEY"],
            os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        )
    if os.environ.get("MOONSHOT_API_KEY"):
        return OpenAIBrain(
            os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
            os.environ["MOONSHOT_API_KEY"],
            os.environ.get("MOONSHOT_MODEL", "kimi-k3"),
        )
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIBrain(
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            os.environ["OPENAI_API_KEY"],
            os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        )
    print(
        "No API key found.\n\n"
        "This agent needs a free Groq key (or Kimi / OpenAI):\n"
        "  1. Create a free account at https://console.groq.com\n"
        "  2. Make a key at https://console.groq.com/keys\n"
        "  3. Run:  export GROQ_API_KEY=your_key_here\n"
        "  4. Then run:  python agent.py \"your message here\"\n\n"
        "Alternatively set MOONSHOT_API_KEY (Kimi) or OPENAI_API_KEY.",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_step(text):
    """Parse one brain reply. Returns (kind, data) or ("bad", None)."""
    final = FINAL_RE.search(text)
    if final:
        return ("final", final.group(1).strip())
    action = ACTION_RE.search(text)
    inp = INPUT_RE.search(text)
    if action and inp:
        try:
            args = json.loads(inp.group(1))
        except json.JSONDecodeError:
            return ("bad", None)
        thought = THOUGHT_RE.search(text)
        return ("action", (thought.group(1).strip() if thought else "", action.group(1), args))
    return ("bad", None)


def run_agent(message, brain, verbose=True):
    """Run the ReAct loop. Returns (final_answer, trace)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Chat message to respond to: "{message}"'},
    ]
    trace = []
    retries = 0
    for step in range(1, MAX_STEPS + 1):
        raw = brain(messages)
        kind, data = parse_step(raw)
        if kind == "final":
            if verbose:
                print(f"\n--- step {step}: final ---")
                print(data)
            trace.append({"step": step, "final_answer": data})
            return data, trace
        if kind == "action":
            thought, name, args = data
            if verbose:
                print(f"\n--- step {step} ---")
                print(f"Thought: {thought}")
                print(f"Action: {name} {json.dumps(args)}")
            try:
                result = run_tool(name, args)
                observation = json.dumps(result, ensure_ascii=False)
            except Exception as exc:  # tool errors become observations, not crashes
                observation = f"Error: {exc}"
            if verbose:
                print(f"Observation: {observation}")
            trace.append(
                {"step": step, "thought": thought, "action": name, "args": args, "observation": observation}
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            continue
        # malformed step: one retry with a format reminder, then stop
        if retries < 1:
            retries += 1
            if verbose:
                print(f"\n--- step {step}: malformed output, retrying once ---")
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": "Your last reply did not follow the format. Reply with exactly one Thought + Action + Action Input, or a Final Answer.",
                }
            )
            continue
        raise RuntimeError("Brain returned malformed output twice; stopping.")
    raise RuntimeError(f"Reached {MAX_STEPS} steps without a Final Answer.")


def main():
    parser = argparse.ArgumentParser(description="tone-agent: recommend a reply to a chat message.")
    parser.add_argument("message", help="the chat message to respond to")
    parser.add_argument("--stub", action="store_true", help="use the deterministic fake brain (no API key)")
    args = parser.parse_args()
    brain = StubBrain() if args.stub else make_brain()
    try:
        answer, _ = run_agent(args.message, brain)
    except RuntimeError as exc:
        print(f"\nAgent failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"\nFinal recommendation: {answer}")


if __name__ == "__main__":
    main()
